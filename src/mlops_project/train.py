import sys
import hydra
from pathlib import Path
from hydra.utils import get_original_cwd
from loguru import logger
from sentence_transformers import losses
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers
from mlops_project.model import instantiate_sentence_transformer as get_model
from mlops_project.data import ArxivPapersDataset

# setup loggers
logger.remove()
logger.add(sys.stdout, level="DEBUG")

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_config")
def train(config):
    logger.info("Starting training process...")
    logger.debug(f"Training configuration: {config}")

    train_dataset = ArxivPapersDataset("train", data_dir=Path(f"{get_original_cwd()}/data")).dataset
    model = get_model(cache_dir=f"{get_original_cwd()}/models/cache/")

    loss = losses.MultipleNegativesRankingLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"{get_original_cwd()}/models/ArXiv-embeddings-model",
        # Optional training parameters:
        num_train_epochs=config.train.epochs,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        logging_steps=100,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    model.save_pretrained(f"{get_original_cwd()}/models/ArXiv-embeddings-model")

if __name__ == "__main__":
    train()
