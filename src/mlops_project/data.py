from pathlib import Path
import typing
import numpy as np
import typer
from datasets import load_dataset
from collections import defaultdict
from datasets import Dataset
from datasets import load_from_disk
import random
from enum import Enum


class LossType(str, Enum):
    ContrastiveLoss = "ContrastiveLoss"
    MultipleNegativesRankingLoss = "MultipleNegativesRankingLoss"


COLUMNS = ["primary_subject", "subjects", "abstract", "title"]


class ArxivPapersDataset(Dataset):
    """Arxiv papers dataset from Hugging Face."""

    def __init__(self, split: str = "train", data_dir: Path = Path("data")) -> None:
        path = data_dir / split
        if path.exists():
            from datasets import load_from_disk

            self.dataset = load_from_disk(str(path))
        else:
            raise FileNotFoundError(f"Dataset not found at {path}. Run preprocessing first.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """Return a given sample from the dataset."""
        return self.dataset[index]


def create_contrastive_pairs(dataset, num_pairs: int = 100000, text_field: str = "abstract", seed: int = 42):
    """
    Create balanced positive and negative pairs for ContrastiveLoss.

    Returns a dataset with columns: sentence1, sentence2, label
    - label=1.0 for positive pairs (same subject)
    - label=0.0 for negative pairs (different subjects)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group indices by subject
    subject_to_indices = defaultdict(list)
    for idx, subject in enumerate(dataset["primary_subject"]):
        subject_to_indices[subject].append(idx)

    subjects = list(subject_to_indices.keys())
    print(f"Found {len(subjects)} unique subjects")

    pairs: dict[str, list[str | float]] = {"sentence1": [], "sentence2": [], "label": []}
    num_positive = num_pairs // 2
    num_negative = num_pairs - num_positive

    # Create positive pairs (same subject)
    print(f"Creating {num_positive} positive pairs...")
    for _ in range(num_positive):
        subject = random.choice([s for s in subjects if len(subject_to_indices[s]) >= 2])
        idx1, idx2 = random.sample(subject_to_indices[subject], 2)
        pairs["sentence1"].append(dataset[idx1][text_field])
        pairs["sentence2"].append(dataset[idx2][text_field])
        pairs["label"].append(1.0)

    # Create negative pairs (different subjects)
    print(f"Creating {num_negative} negative pairs...")
    for _ in range(num_negative):
        subj1, subj2 = random.sample(subjects, 2)
        idx1 = random.choice(subject_to_indices[subj1])
        idx2 = random.choice(subject_to_indices[subj2])
        pairs["sentence1"].append(dataset[idx1][text_field])
        pairs["sentence2"].append(dataset[idx2][text_field])
        pairs["label"].append(0.0)

    return Dataset.from_dict(pairs)


def create_positive_pairs(dataset, num_pairs: int = 100000, text_field: str = "abstract", seed: int = 42):
    """
    Create positive pairs for MultipleNegativesRankingLoss.

    Returns a dataset with columns: anchor, positive
    Each pair contains two abstracts from papers with the same primary_subject.
    MNRL will use in-batch negatives automatically.
    """
    random.seed(seed)
    np.random.seed(seed)

    subject_to_indices = defaultdict(list)
    for idx, subject in enumerate(dataset["primary_subject"]):
        subject_to_indices[subject].append(idx)

    subjects = [s for s in subject_to_indices.keys() if len(subject_to_indices[s]) >= 2]
    print(f"Found {len(subjects)} subjects with 2+ samples")

    pairs: dict[str, list[str]] = {"anchor": [], "positive": []}

    print(f"Creating {num_pairs} positive pairs...")
    for _ in range(num_pairs):
        subject = random.choice(subjects)
        idx1, idx2 = random.sample(subject_to_indices[subject], 2)
        pairs["anchor"].append(dataset[idx1][text_field])
        pairs["positive"].append(dataset[idx2][text_field])

    return Dataset.from_dict(pairs)


def create_pairs(
    dataset, pair_fn: typing.Callable, save_path: Path, num_pairs: int, text_field: str = "abstract", seed: int = 42
) -> Dataset:
    """Create and save pairs to disk."""
    pairs = pair_fn(dataset, num_pairs=num_pairs, text_field=text_field, seed=seed)
    pairs.save_to_disk(str(save_path))
    return pairs


def load_pairs(load_path: Path) -> Dataset:
    """Load pairs from disk."""
    if not load_path.exists():
        raise FileNotFoundError(f"Pairs file not found at {load_path}. Run preprocessing first.")
    print(f"Loading pairs from {load_path}")
    return load_from_disk(str(load_path))


def preprocess(
    loss: LossType = LossType.MultipleNegativesRankingLoss,
    output_folder: Path = Path("data"),
    test_size: float = 0.2,
    number_of_pairs: int = 1_000_000,
    seed: int = 42,
) -> None:
    """Download and preprocess the arxiv papers dataset."""
    print("Downloading arxiv-papers dataset...")
    dataset = load_dataset("nick007x/arxiv-papers", split="train")

    print(f"Selecting columns: {COLUMNS}")
    dataset = dataset.select_columns(COLUMNS)

    print(f"Splitting dataset (test_size={test_size})...")
    splits_train_test = dataset.train_test_split(test_size=test_size, seed=seed)

    output_folder.mkdir(parents=True, exist_ok=True)

    train_path = output_folder / "train"
    eval_path = output_folder / "eval"
    test_path = output_folder / "test"

    splits_train_eval = splits_train_test["train"].train_test_split(test_size=test_size, seed=seed)

    print(f"Saving train split ({len(splits_train_eval['train'])} samples) to {train_path}")
    splits_train_eval["train"].save_to_disk(str(train_path))

    print(f"Saving eval split ({len(splits_train_eval['test'])} samples) to {eval_path}")
    splits_train_eval["test"].save_to_disk(str(eval_path))

    print(f"Saving test split ({len(splits_train_test['test'])} samples) to {test_path}")
    splits_train_test["test"].save_to_disk(str(test_path))

    train_dataset = load_from_disk(str(output_folder / "train"))
    eval_dataset = load_from_disk(str(output_folder / "eval"))
    test_dataset = load_from_disk(str(output_folder / "test"))

    print(f"Creating training, evaluation and test pairs using {loss.value}...")
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
        text_field="abstract",
        seed=seed,
    )

    create_pairs(
        dataset=eval_dataset,
        pair_fn=pair_fn,
        save_path=output_folder / "eval_pairs",
        num_pairs=number_of_pairs // 100,
        text_field="abstract",
        seed=seed,
    )
    create_pairs(
        dataset=test_dataset,
        pair_fn=pair_fn,
        save_path=output_folder / "test_pairs",
        num_pairs=number_of_pairs // 100,
        text_field="abstract",
        seed=seed,
    )

    print("Done!")


if __name__ == "__main__":  # pragma: no cover
    typer.run(preprocess)
