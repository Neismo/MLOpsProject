import sys
import hydra
from pathlib import Path
from hydra.utils import get_original_cwd
from loguru import logger
from mlops_project.model import instantiate_sentence_transformer as get_model
from mlops_project.data import ArxivPapersDataset

# setup loggers
logger.remove()
logger.add(sys.stdout, level="DEBUG")

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_config")
def train(config):
    logger.info("Starting training process...")
    logger.debug(f"Training configuration: {config}")

    train_dataset = ArxivPapersDataset("train", data_dir=Path(f"{get_original_cwd()}/data"))
    model = get_model(cache_dir=f"{get_original_cwd()}/models/cache/")

    # rest of training loop here!

if __name__ == "__main__":
    train()
