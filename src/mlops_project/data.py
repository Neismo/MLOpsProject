from pathlib import Path

import typer
from datasets import load_dataset
from torch.utils.data import Dataset

COLUMNS = ["primary_subject", "subjects", "abstract", "title"]
COLUMNS_TRAIN = ["abstract", "primary_subject"]


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

        # Force (anchor, positive) tuple format for SentenceTransformer for now!
        return self.dataset[index]


def preprocess(output_folder: Path = Path("data"), test_size: float = 0.2, seed: int = 42) -> None:
    """Download and preprocess the arxiv papers dataset."""
    print("Downloading arxiv-papers dataset...")
    dataset = load_dataset("nick007x/arxiv-papers", split="train")

    print(f"Selecting columns: {COLUMNS_TRAIN}")
    dataset = dataset.select_columns(COLUMNS_TRAIN)

    print(f"Splitting dataset (test_size={test_size})...")
    splits = dataset.train_test_split(test_size=test_size, seed=seed)

    output_folder.mkdir(parents=True, exist_ok=True)

    train_path = output_folder / "train"
    test_path = output_folder / "test"

    print(f"Saving train split ({len(splits['train'])} samples) to {train_path}")
    splits["train"].save_to_disk(str(train_path))

    print(f"Saving test split ({len(splits['test'])} samples) to {test_path}")
    splits["test"].save_to_disk(str(test_path))

    print("Done!")


if __name__ == "__main__":
    typer.run(preprocess)
