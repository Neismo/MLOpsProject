from pathlib import Path
import typing
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk
from collections import defaultdict
import random
from enum import Enum
import hydra
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf, ListConfig


class LossType(str, Enum):
    ContrastiveLoss = "ContrastiveLoss"
    MultipleNegativesRankingLoss = "MultipleNegativesRankingLoss"


class ArxivPapersDataset:
    """Arxiv papers dataset from Hugging Face."""

    def __init__(self, split: str = "train", data_dir: Path = Path("data")) -> None:
        path = data_dir / split
        if path.exists():
            self.dataset = load_from_disk(str(path))
        else:
            raise FileNotFoundError(f"Dataset not found at {path}. Run preprocessing first.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """Return a given sample from the dataset."""
        return self.dataset[index]


def create_contrastive_pairs(
    dataset, num_pairs: int = 100000, text_field: str = "abstract", seed: int = 42, balanced: bool = True
):
    """
    Create positive and negative pairs for ContrastiveLoss.

    Returns a dataset with columns: sentence1, sentence2, label
    - label=1.0 for positive pairs (same subject)
    - label=0.0 for negative pairs (different subjects)

    Args:
        balanced: If True, each subject has equal probability of being chosen.
                  If False, subjects are weighted by their frequency in the dataset.
    """
    random.seed(seed)
    np.random.seed(seed)

    subject_to_indices = defaultdict(list)
    for idx, subject in enumerate(dataset["primary_subject"]):
        subject_to_indices[subject].append(idx)

    subjects = [s for s in subject_to_indices.keys() if len(subject_to_indices[s]) >= 2]
    subject_weights = [len(subject_to_indices[s]) for s in subjects]
    logger.info(f"Found {len(subjects)} subjects with 2+ samples")

    pairs: dict[str, list[str | float]] = {"sentence1": [], "sentence2": [], "label": []}

    # Need at least 2 subjects to create negative pairs
    if len(subjects) < 2:
        num_positive = num_pairs
        num_negative = 0
        logger.warning(f"Only {len(subjects)} subjects with 2+ samples, creating only positive pairs")
    else:
        num_positive = num_pairs // 2
        num_negative = num_pairs - num_positive

    logger.info(f"Creating {num_positive} positive pairs (balanced={balanced})...")
    for _ in range(num_positive):
        if balanced:
            subject = random.choice(subjects)
        else:
            subject = random.choices(subjects, weights=subject_weights, k=1)[0]
        idx1, idx2 = random.sample(subject_to_indices[subject], 2)
        pairs["sentence1"].append(dataset[idx1][text_field])
        pairs["sentence2"].append(dataset[idx2][text_field])
        pairs["label"].append(1.0)

    logger.info(f"Creating {num_negative} negative pairs...")
    for _ in range(num_negative):
        subj1, subj2 = random.sample(subjects, 2)
        idx1 = random.choice(subject_to_indices[subj1])
        idx2 = random.choice(subject_to_indices[subj2])
        pairs["sentence1"].append(dataset[idx1][text_field])
        pairs["sentence2"].append(dataset[idx2][text_field])
        pairs["label"].append(0.0)

    return Dataset.from_dict(pairs)


def create_positive_pairs(
    dataset, num_pairs: int = 100000, text_field: str = "abstract", seed: int = 42, balanced: bool = True
):
    """
    Create positive pairs for MultipleNegativesRankingLoss.

    Returns a dataset with columns: anchor, positive
    Each pair contains two abstracts from papers with the same primary_subject.
    MNRL will use in-batch negatives automatically.

    Args:
        balanced: If True, each subject has equal probability of being chosen.
                  If False, subjects are weighted by their frequency in the dataset.
    """
    random.seed(seed)
    np.random.seed(seed)

    subject_to_indices = defaultdict(list)
    for idx, subject in enumerate(dataset["primary_subject"]):
        subject_to_indices[subject].append(idx)

    subjects = [s for s in subject_to_indices.keys() if len(subject_to_indices[s]) >= 2]
    subject_weights = [len(subject_to_indices[s]) for s in subjects]
    logger.info(f"Found {len(subjects)} subjects with 2+ samples")

    pairs: dict[str, list[str]] = {"anchor": [], "positive": []}

    logger.info(f"Creating {num_pairs} positive pairs (balanced={balanced})...")
    for _ in range(num_pairs):
        if balanced:
            subject = random.choice(subjects)
        else:
            subject = random.choices(subjects, weights=subject_weights, k=1)[0]
        idx1, idx2 = random.sample(subject_to_indices[subject], 2)
        pairs["anchor"].append(dataset[idx1][text_field])
        pairs["positive"].append(dataset[idx2][text_field])

    return Dataset.from_dict(pairs)


def create_pairs(
    dataset,
    pair_fn: typing.Callable,
    save_path: Path,
    num_pairs: int,
    text_field: str = "abstract",
    seed: int = 42,
    balanced: bool = True,
) -> Dataset:
    """Create and save pairs to disk."""
    pairs = pair_fn(dataset, num_pairs=num_pairs, text_field=text_field, seed=seed, balanced=balanced)
    pairs.save_to_disk(str(save_path))
    return pairs


def load_pairs(load_path: Path) -> Dataset:
    """Load pairs from disk."""
    if not load_path.exists():
        raise FileNotFoundError(f"Pairs file not found at {load_path}. Run preprocessing first.")
    logger.info(f"Loading pairs from {load_path}")
    return load_from_disk(str(load_path))


def preprocess(
    loss: LossType = LossType.MultipleNegativesRankingLoss,
    output_folder: Path = Path("data"),
    test_size: float = 0.2,
    number_of_pairs: int = 1_000_000,
    number_of_eval_pairs: int = 10_000,
    seed: int = 42,
    source: str = "nick007x/arxiv-papers",
    columns: list[str] | None = None,
    text_field: str = "abstract",
    balanced: bool = True,
) -> None:
    """Download and preprocess the arxiv papers dataset."""
    if columns is None:
        columns = ["primary_subject", "subjects", "abstract", "title"]

    logger.info(f"Downloading dataset: {source}")
    dataset = load_dataset(source, split="train")

    logger.info(f"Selecting columns: {columns}")
    dataset = dataset.select_columns(columns)

    logger.info(f"Splitting dataset (test_size={test_size})...")
    splits_train_test = dataset.train_test_split(test_size=test_size, seed=seed)

    output_folder.mkdir(parents=True, exist_ok=True)

    train_path = output_folder / "train"
    eval_path = output_folder / "eval"
    test_path = output_folder / "test"

    splits_train_eval = splits_train_test["train"].train_test_split(test_size=test_size, seed=seed)

    logger.info(f"Saving train split ({len(splits_train_eval['train'])} samples) to {train_path}")
    splits_train_eval["train"].save_to_disk(str(train_path))

    logger.info(f"Saving eval split ({len(splits_train_eval['test'])} samples) to {eval_path}")
    splits_train_eval["test"].save_to_disk(str(eval_path))

    logger.info(f"Saving test split ({len(splits_train_test['test'])} samples) to {test_path}")
    splits_train_test["test"].save_to_disk(str(test_path))

    train_dataset = load_from_disk(str(train_path))
    eval_dataset = load_from_disk(str(eval_path))

    logger.info(f"Creating pairs using {loss.value}...")
    match loss:
        case LossType.ContrastiveLoss:
            pair_fn = create_contrastive_pairs
        case LossType.MultipleNegativesRankingLoss:
            pair_fn = create_positive_pairs
        case _:
            raise ValueError(f"Unsupported loss type: {loss}")

    create_pairs(
        dataset=train_dataset,
        pair_fn=pair_fn,
        save_path=output_folder / "train_pairs",
        num_pairs=number_of_pairs,
        text_field=text_field,
        seed=seed,
        balanced=balanced,
    )

    create_pairs(
        dataset=eval_dataset,
        pair_fn=pair_fn,
        save_path=output_folder / "eval_pairs",
        num_pairs=number_of_eval_pairs,
        text_field=text_field,
        seed=seed,
        balanced=balanced,
    )

    # Save config used for preprocessing
    used_config = {
        "loss": loss.value,
        "test_size": test_size,
        "number_of_pairs": number_of_pairs,
        "number_of_eval_pairs": number_of_eval_pairs,
        "seed": seed,
        "source": source,
        "columns": columns,
        "text_field": text_field,
        "balanced": balanced,
    }
    OmegaConf.save(OmegaConf.create(used_config), output_folder / "preprocess_config.yaml")
    logger.info(f"Saved preprocessing config to {output_folder / 'preprocess_config.yaml'}")

    logger.info("Done!")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="dataset")
def preprocess_hydra(config: DictConfig | ListConfig) -> None:
    """Hydra entry point for preprocessing."""
    output_folder = Path(get_original_cwd()) / config.output_dir

    loss_str = config.pairs.loss
    loss = (
        LossType.MultipleNegativesRankingLoss
        if loss_str == "MultipleNegativesRankingLoss"
        else LossType.ContrastiveLoss
    )

    preprocess(
        loss=loss,
        output_folder=output_folder,
        test_size=config.splits.test_size,
        number_of_pairs=config.pairs.num_train,
        number_of_eval_pairs=config.pairs.num_eval,
        seed=config.splits.seed,
        source=config.source,
        columns=list(config.columns),
        text_field=config.pairs.text_field,
        balanced=config.pairs.balanced,
    )


def _build_config_dict(dataset_config: DictConfig | ListConfig) -> dict:
    """Build a config dict from dataset config for comparison."""
    return {
        "loss": dataset_config.pairs.loss,
        "test_size": dataset_config.splits.test_size,
        "number_of_pairs": dataset_config.pairs.num_train,
        "number_of_eval_pairs": dataset_config.pairs.num_eval,
        "seed": dataset_config.splits.seed,
        "source": dataset_config.source,
        "columns": list(dataset_config.columns),
        "text_field": dataset_config.pairs.text_field,
        "balanced": dataset_config.pairs.balanced,
    }


def ensure_data_exists(data_dir: Path, dataset_config: DictConfig | ListConfig | None = None) -> None:
    """Run preprocessing if required data doesn't exist or config has changed."""

    logger.info(f"Ensuring data exists at {data_dir}...")

    train_pairs_path = data_dir / "train_pairs"
    eval_pairs_path = data_dir / "eval_pairs"
    test_path = data_dir / "test"
    saved_config_path = data_dir / "preprocess_config.yaml"

    if dataset_config is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "dataset.yaml"
        dataset_config = OmegaConf.load(config_path)

    current_config = _build_config_dict(dataset_config)

    # Check if data exists and config matches
    if train_pairs_path.exists() and eval_pairs_path.exists() and test_path.exists():
        if saved_config_path.exists():
            saved_config = OmegaConf.to_container(OmegaConf.load(saved_config_path))
            if saved_config == current_config:
                logger.info("Data already exists with matching config, skipping preprocessing.")
                return
            logger.info("Config has changed, re-running preprocessing...")
        else:
            logger.info("No saved config found, re-running preprocessing to ensure consistency...")
    else:
        logger.info("Data not found, running preprocessing...")

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
        balanced=dataset_config.pairs.balanced,
    )


if __name__ == "__main__":  # pragma: no cover
    preprocess_hydra()
