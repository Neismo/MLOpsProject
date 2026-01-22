import mlops_project.data
from mlops_project.model import instantiate_sentence_transformer as get_model
from mlops_project.data import ArxivPapersDataset, preprocess, LossType
import torch
from loguru import logger
import hydra
from omegaconf import OmegaConf
import sys
from pathlib import Path
from hydra.utils import get_original_cwd
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("train.log", level="DEBUG", rotation="5 MB")


def create_ir_evaluator(dataset, sample_size: int = 5000, name: str = "arxiv-retrieval"):
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)

    # Split indices: first 20% for queries, rest for corpus (no overlap)
    num_queries = sample_size // 5
    query_indices = indices[:num_queries]
    corpus_indices = indices[num_queries:]

    queries = {}
    corpus = {}
    relevant_docs = {}
    subject_to_corpus_ids = defaultdict(set)

    # Build corpus from corpus_indices only
    for i, idx in enumerate(corpus_indices):
        idx = int(idx)
        corpus_id = f"doc_{i}"
        corpus[corpus_id] = dataset[idx]["abstract"]
        subject = dataset[idx]["primary_subject"]
        subject_to_corpus_ids[subject].add(corpus_id)

    # Build queries from query_indices (no overlap with corpus)
    for i, idx in enumerate(query_indices):
        idx = int(idx)
        query_id = f"query_{i}"
        queries[query_id] = dataset[idx]["abstract"]
        subject = dataset[idx]["primary_subject"]
        # Relevant docs are corpus docs with same subject
        relevant_docs[query_id] = subject_to_corpus_ids[subject].copy()

    # Filter out queries with no relevant docs
    queries = {qid: q for qid, q in queries.items() if len(relevant_docs.get(qid, set())) > 0}
    relevant_docs = {qid: docs for qid, docs in relevant_docs.items() if qid in queries}

    logger.info(f"IR Evaluator: {len(queries)} queries, {len(corpus)} corpus docs (no overlap)")

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        precision_recall_at_k=[1, 5, 10],
        show_progress_bar=True,
    )


def ensure_data_exists(data_dir: Path) -> None:
    """Run preprocessing if required data doesn't exist."""
    train_pairs_path = data_dir / "train_pairs"
    eval_pairs_path = data_dir / "eval_pairs"
    test_path = data_dir / "test"

    if train_pairs_path.exists() and eval_pairs_path.exists() and test_path.exists():
        logger.info("Data already exists, skipping preprocessing.")
        return

    logger.info("Data not found, running preprocessing...")
    config_path = Path(__file__).parent.parent.parent / "configs" / "dataset.yaml"
    dataset_config = OmegaConf.load(config_path)

    loss_str = dataset_config.pairs.loss
    loss = (
        LossType.MultipleNegativesRankingLoss
        if loss_str == "MultipleNegativesRankingLoss"
        else LossType.ContrastiveLoss
    )

    preprocess(
        loss=loss,
        output_folder=data_dir,
        test_size=dataset_config.splits.test_size,
        number_of_pairs=dataset_config.pairs.num_train,
        number_of_eval_pairs=dataset_config.pairs.num_eval,
        seed=dataset_config.splits.seed,
        source=dataset_config.source,
        columns=list(dataset_config.columns),
        text_field=dataset_config.pairs.text_field,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_config")
def train(config):
    logger.debug(f"Training config:\n{config}")

    if config.meta.require_cuda:
        assert torch.cuda.is_available(), (
            "CUDA is required for this run. Set meta.require_cuda=false to allow CPU/debug runs."
        )
    elif not torch.cuda.is_available():
        logger.warning("CUDA not available. Continuing on CPU.")

    data_dir = Path(f"{get_original_cwd()}/data")
    ensure_data_exists(data_dir)

    test_dataset = ArxivPapersDataset("test", data_dir=data_dir).dataset
    model = get_model(cache_dir=f"{get_original_cwd()}/models/cache/")

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

    # Define training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"{get_original_cwd()}/models/contrastive-minilm",
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        learning_rate=2e-5,
        warmup_ratio=config.train.warmup_ratio,
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
