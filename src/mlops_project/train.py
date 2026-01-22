import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss

import mlops_project.data
from mlops_project.data import ArxivPapersDataset, ensure_data_exists
from mlops_project.evaluate import create_ir_evaluator
from mlops_project.model import instantiate_sentence_transformer as get_model
from mlops_project.utils import build_output_dir_name

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("train.log", level="DEBUG", rotation="5 MB")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_config")
def train(config):
    logger.debug(f"Training config:\n{config}")

    if config.meta.require_cuda:
        assert torch.cuda.is_available(), (
            "CUDA is required for this run. Set meta.require_cuda=false to allow CPU/debug runs."
        )
    elif not torch.cuda.is_available():
        logger.warning("CUDA not available. Continuing on CPU.")

    if config.meta.use_gcs:
        data_dir = Path(f"/gcs/{config.meta.bucket_name}/data")
    else:
        data_dir = Path(f"{get_original_cwd()}/data")
    ensure_data_exists(data_dir, config)

    test_dataset = ArxivPapersDataset("test", data_dir=data_dir).dataset
    model = get_model(model_name=config.train.model, cache_dir=f"{get_original_cwd()}/models/cache/")

    # Create/load training/eval pairs
    if config.meta.use_gcs:
        train_pairs_path = Path(f"/gcs/{config.meta.bucket_name}/data/train_pairs")
        eval_pairs_path = Path(f"/gcs/{config.meta.bucket_name}/data/eval_pairs")
    else:
        train_pairs_path = data_dir / "train_pairs"
        eval_pairs_path = data_dir / "eval_pairs"

    train_pairs = mlops_project.data.load_pairs(train_pairs_path)
    logger.info(f"Training pairs: {len(train_pairs)}")

    eval_pairs = mlops_project.data.load_pairs(eval_pairs_path)
    logger.info(f"Evaluation pairs: {len(eval_pairs)}")

    # Test dataset for evaluation
    # Create the IR evaluator for precision@k
    ir_evaluator = create_ir_evaluator(test_dataset, sample_size=5000)
    logger.info("IR Evaluator created for precision@k metrics")

    if not config.wandb.enabled:
        logger.info("W&B logging disabled (config.wandb.enabled=false).")

    # Save model to GCS or locally
    if config.meta.use_gcs:
        logger.info("Using GCS for data storage (config.meta.use_gcs=true).")
        output_dir = f"/gcs/{config.meta.bucket_name}"
    else:
        output_dir = f"{get_original_cwd()}"

    # Build output directory name from config
    output_dir_name = build_output_dir_name(
        model=config.train.model,
        loss=config.train.loss,
        num_pairs=len(train_pairs),
        balanced=config.pairs.balanced,
    )

    # Define training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"{output_dir}/models/{output_dir_name}",
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        learning_rate=2e-5,
        warmup_ratio=config.train.warmup_ratio,
        seed=config.train.seed,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps" if config.meta.save_model else "no",
        save_steps=500,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="wandb" if config.wandb.enabled else "none",
    )

    # Initialize loss function
    match config.train.loss:
        case "ContrastiveLoss":
            loss = ContrastiveLoss(model=model)
        case "MultipleNegativesRankingLoss":
            loss = MultipleNegativesRankingLoss(model=model)
        case _:
            logger.error(f"Unsupported loss type from config: {config.train.loss}")
            raise ValueError(f"Unsupported loss type: {config.train.loss}")

    # Create trainer with IR evaluator for precision@k
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_pairs,
        eval_dataset=eval_pairs,
        loss=loss,
        evaluator=ir_evaluator,
    )
    logger.info("Trainer initialized with precision@k evaluator. Ready to train!")

    # Start training
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    train()
