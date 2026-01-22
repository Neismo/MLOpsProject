import pytest

from tests import N_TRAIN
import mlops_project.data
from datasets import Dataset


def test_positive_pairs_dataset_shapes_and_length_and_types(mock_positive_dataset):
    train_dataset = mlops_project.data.load_pairs("data/train_pairs")

    assert len(train_dataset) == N_TRAIN, "Train dataset should not be empty."
    assert isinstance(train_dataset[0], dict), "Each item in the dataset should be a dictionary."
    assert isinstance(train_dataset[0]["anchor"], str) and train_dataset[0]["anchor"] is not None, (
        "anchor should be a string."
    )
    assert isinstance(train_dataset[0]["positive"], str) and train_dataset[0]["positive"] is not None, (
        "positive should be a string."
    )


def test_create_positive_pairs_logic(mock_data):
    """Test to verify that every pair generated shares the same primary_subject."""
    num_pairs = 50
    result = mlops_project.data.create_positive_pairs(mock_data, num_pairs=num_pairs, text_field="abstract", seed=42)

    text_to_subject = {row["abstract"]: row["primary_subject"] for row in mock_data}

    assert len(result) == num_pairs

    for row in result:
        anchor_subj = text_to_subject[row["anchor"]]
        pos_subj = text_to_subject[row["positive"]]

        assert anchor_subj == pos_subj
        assert row["anchor"] != row["positive"]


def test_positive_pairs_reproducibility(mock_data):
    """Ensure that the internal seeding logic works as expected."""

    run1 = mlops_project.data.create_positive_pairs(dataset=mock_data, num_pairs=10, seed=99)
    run2 = mlops_project.data.create_positive_pairs(dataset=mock_data, num_pairs=10, seed=99)

    assert run1["anchor"] == run2["anchor"]
    assert run1["positive"] == run2["positive"]


@pytest.mark.parametrize("requested_pairs", [1, 5, 100])
def test_positive_pairs_dynamic_output_sizes(mock_data, requested_pairs):
    """Verify the function respects the num_pairs argument."""
    result = mlops_project.data.create_positive_pairs(mock_data, num_pairs=requested_pairs)
    assert len(result) == requested_pairs


def test_positive_pairs_subject_filtering():
    """Verify subjects with < 2 samples are excluded from being chosen."""
    data = {"primary_subject": ["Physics", "Physics", "Solo_Subject"], "abstract": ["P1", "P2", "S1"]}
    ds = Dataset.from_dict(data)

    # If the logic is correct, 'S1' should never appear because 'Solo_Subject'
    # has only 1 sample and is filtered out of the 'subjects' list.
    result = mlops_project.data.create_positive_pairs(ds, num_pairs=20)

    for row in result:
        assert row["anchor"] != "S1"
        assert row["positive"] != "S1"


# Tests for create_contrastive_pairs
def test_contrastive_dataset_shapes_and_length_and_types(mock_contrastive_dataset):
    train_dataset = mlops_project.data.load_pairs("data/train_pairs")

    assert len(train_dataset) == N_TRAIN, "Train dataset should not be empty."
    assert isinstance(train_dataset[0], dict), "Each item in the dataset should be a dictionary."
    assert isinstance(train_dataset[0]["sentence1"], str) and train_dataset[0]["sentence1"] is not None, (
        "sentence1 should be a string."
    )
    assert isinstance(train_dataset[0]["sentence2"], str) and train_dataset[0]["sentence2"] is not None, (
        "sentence2 should be a string."
    )
    assert isinstance(train_dataset[0]["label"], float), "label should be a float."
    assert train_dataset[0]["label"] in [0.0, 1.0], "label should be either 0.0 or 1.0."


def test_create_contrastive_pairs_logic(mock_data):
    """Test to verify that positive pairs share the same subject and negative pairs don't."""
    num_pairs = 50
    result = mlops_project.data.create_contrastive_pairs(mock_data, num_pairs=num_pairs, text_field="abstract", seed=42)

    text_to_subject = {row["abstract"]: row["primary_subject"] for row in mock_data}

    assert len(result) == num_pairs

    num_positive = 0
    num_negative = 0

    for row in result:
        sent1_subj = text_to_subject[row["sentence1"]]
        sent2_subj = text_to_subject[row["sentence2"]]

        if row["label"] == 1.0:
            # Positive pairs should have the same subject
            assert sent1_subj == sent2_subj
            assert row["sentence1"] != row["sentence2"]
            num_positive += 1
        elif row["label"] == 0.0:
            # Negative pairs should have different subjects
            assert sent1_subj != sent2_subj
            num_negative += 1
        else:
            pytest.fail(f"Invalid label: {row['label']}")

    # Check that we have roughly balanced positive and negative pairs
    assert num_positive == num_pairs // 2
    assert num_negative == num_pairs - num_pairs // 2


def test_contrastive_pairs_reproducibility(mock_data):
    """Ensure that the internal seeding logic works as expected."""

    run1 = mlops_project.data.create_contrastive_pairs(dataset=mock_data, num_pairs=10, seed=99)
    run2 = mlops_project.data.create_contrastive_pairs(dataset=mock_data, num_pairs=10, seed=99)

    assert run1["sentence1"] == run2["sentence1"]
    assert run1["sentence2"] == run2["sentence2"]
    assert run1["label"] == run2["label"]


@pytest.mark.parametrize("requested_pairs", [2, 10, 100])
def test_contrastive_pairs_dynamic_output_sizes(mock_data, requested_pairs):
    """Verify the function respects the num_pairs argument."""
    result = mlops_project.data.create_contrastive_pairs(mock_data, num_pairs=requested_pairs)
    assert len(result) == requested_pairs


def test_contrastive_pairs_subject_filtering():
    """Verify subjects with < 2 samples can only appear in negative pairs."""
    data = {"primary_subject": ["Physics", "Physics", "Solo_Subject"], "abstract": ["P1", "P2", "S1"]}
    ds = Dataset.from_dict(data)

    result = mlops_project.data.create_contrastive_pairs(ds, num_pairs=20)

    for row in result:
        # For positive pairs (label=1.0), 'S1' should never appear
        # because 'Solo_Subject' has only 1 sample
        if row["label"] == 1.0:
            assert row["sentence1"] != "S1"
            assert row["sentence2"] != "S1"


def test_contrastive_pairs_balance(mock_data):
    """Verify that positive and negative pairs are properly balanced."""
    num_pairs = 100
    result = mlops_project.data.create_contrastive_pairs(mock_data, num_pairs=num_pairs)

    positive_count = sum(1 for row in result if row["label"] == 1.0)
    negative_count = sum(1 for row in result if row["label"] == 0.0)

    assert positive_count == num_pairs // 2
    assert negative_count == num_pairs - num_pairs // 2
    assert positive_count + negative_count == num_pairs


# Tests for load_pairs
def test_load_pairs_success(mock_data, tmp_path):
    """Test that load_pairs successfully loads pairs from disk."""

    # Create pairs and save them
    pairs = mlops_project.data.create_positive_pairs(mock_data, num_pairs=10)
    save_path = tmp_path / "test_pairs"
    pairs.save_to_disk(str(save_path))

    # Load the pairs
    loaded_pairs = mlops_project.data.load_pairs(save_path)

    assert len(loaded_pairs) == 10
    assert "anchor" in loaded_pairs.column_names
    assert "positive" in loaded_pairs.column_names


def test_load_pairs_file_not_found(tmp_path):
    """Test that load_pairs raises FileNotFoundError when path doesn't exist."""

    non_existent_path = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Pairs file not found"):
        mlops_project.data.load_pairs(non_existent_path)


# Tests for create_pairs
def test_create_pairs_positive_pairs(mock_data, tmp_path):
    """Test that create_pairs successfully creates and saves positive pairs."""

    save_path = tmp_path / "created_pairs"
    num_pairs = 20

    result = mlops_project.data.create_pairs(
        dataset=mock_data,
        pair_fn=mlops_project.data.create_positive_pairs,
        save_path=save_path,
        num_pairs=num_pairs,
        text_field="abstract",
        seed=42,
    )

    # Check the returned dataset
    assert len(result) == num_pairs
    assert "anchor" in result.column_names
    assert "positive" in result.column_names

    # Check that it was saved to disk
    assert save_path.exists()

    # Load and verify
    loaded = mlops_project.data.load_pairs(save_path)
    assert len(loaded) == num_pairs


def test_create_pairs_contrastive_pairs(mock_data, tmp_path):
    """Test that create_pairs successfully creates and saves contrastive pairs."""

    save_path = tmp_path / "created_contrastive_pairs"
    num_pairs = 30

    result = mlops_project.data.create_pairs(
        dataset=mock_data,
        pair_fn=mlops_project.data.create_contrastive_pairs,
        save_path=save_path,
        num_pairs=num_pairs,
        text_field="abstract",
        seed=99,
    )

    # Check the returned dataset
    assert len(result) == num_pairs
    assert "sentence1" in result.column_names
    assert "sentence2" in result.column_names
    assert "label" in result.column_names

    # Check that it was saved to disk
    assert save_path.exists()

    # Load and verify
    loaded = mlops_project.data.load_pairs(save_path)
    assert len(loaded) == num_pairs


def test_create_pairs_to_nonexistent_parent_directory(mock_data, tmp_path):
    """Test that create_pairs works even when parent directories don't exist."""

    # Create a nested path that doesn't exist yet
    save_path = tmp_path / "nested" / "directories" / "pairs"
    assert not save_path.parent.exists()

    # save_to_disk should handle creating parent directories
    result = mlops_project.data.create_pairs(
        dataset=mock_data, pair_fn=mlops_project.data.create_positive_pairs, save_path=save_path, num_pairs=5, seed=42
    )

    # Verify it was created successfully
    assert save_path.exists()
    assert len(result) == 5


# Tests for preprocess
@pytest.mark.parametrize(
    "loss_type,expected_columns",
    [
        (mlops_project.data.LossType.MultipleNegativesRankingLoss, ["anchor", "positive"]),
        (mlops_project.data.LossType.ContrastiveLoss, ["sentence1", "sentence2", "label"]),
    ],
    ids=["MultipleNegativesRankingLoss", "ContrastiveLoss"],
)
def test_preprocess(mock_raw_data, tmp_path, monkeypatch, loss_type, expected_columns):
    """Test that preprocess creates all expected files and directories for different loss types."""
    from unittest.mock import MagicMock

    # Mock load_dataset to return our mock_raw_data instead of downloading
    mock_load_dataset = MagicMock(return_value=mock_raw_data)
    monkeypatch.setattr("mlops_project.data.load_dataset", mock_load_dataset)

    output_folder = tmp_path / "test_data"
    num_pairs = 50000
    test_size = 0.2
    seed = 123

    # Run preprocess with the specified loss type
    mlops_project.data.preprocess(
        loss=loss_type,
        output_folder=output_folder,
        test_size=test_size,
        number_of_pairs=num_pairs,
        number_of_eval_pairs=num_pairs // 100,
        seed=seed,
    )

    # Verify that load_dataset was called
    mock_load_dataset.assert_called_once_with("nick007x/arxiv-papers", split="train")

    # Verify directory structure
    assert output_folder.exists()
    assert (output_folder / "train").exists()
    assert (output_folder / "eval").exists()
    assert (output_folder / "test").exists()
    assert (output_folder / "train_pairs").exists()
    assert (output_folder / "eval_pairs").exists()
    # Note: test_pairs is not created - test split is used raw for evaluation

    # Load and verify the splits
    train_split = mlops_project.data.load_pairs(output_folder / "train_pairs")
    eval_split = mlops_project.data.load_pairs(output_folder / "eval_pairs")

    # Verify train_pairs has the correct structure
    for col in expected_columns:
        assert col in train_split.column_names
    assert len(train_split) == num_pairs

    # Verify eval_pairs has the expected number (num_pairs // 100)
    assert len(eval_split) == num_pairs // 100

    # Additional validation for ContrastiveLoss
    if loss_type == mlops_project.data.LossType.ContrastiveLoss:
        for row in train_split:
            assert row["label"] in [0.0, 1.0]


def test_init_success(mock_data, tmp_path):
    """Test that ArxivPapersDataset initializes correctly when data exists."""
    # Save mock data to disk in the expected location
    split_name = "train"
    data_dir = tmp_path / "data"
    split_path = data_dir / split_name
    split_path.mkdir(parents=True)
    mock_data.save_to_disk(str(split_path))

    # Create dataset instance
    dataset = mlops_project.data.ArxivPapersDataset(split=split_name, data_dir=data_dir)

    assert dataset is not None
    assert hasattr(dataset, "dataset")


def test_init_file_not_found(tmp_path):
    """Test that ArxivPapersDataset raises FileNotFoundError when data doesn't exist."""
    data_dir = tmp_path / "nonexistent_data"

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        mlops_project.data.ArxivPapersDataset(split="train", data_dir=data_dir)


def test_len(mock_data, tmp_path):
    """Test that __len__ returns the correct length of the dataset."""
    split_name = "train"
    data_dir = tmp_path / "data"
    split_path = data_dir / split_name
    split_path.mkdir(parents=True)
    mock_data.save_to_disk(str(split_path))

    dataset = mlops_project.data.ArxivPapersDataset(split=split_name, data_dir=data_dir)

    assert len(dataset) == len(mock_data)


def test_getitem(mock_data, tmp_path):
    """Test that __getitem__ returns the correct item from the dataset."""
    split_name = "train"
    data_dir = tmp_path / "data"
    split_path = data_dir / split_name
    split_path.mkdir(parents=True)
    mock_data.save_to_disk(str(split_path))

    dataset = mlops_project.data.ArxivPapersDataset(split=split_name, data_dir=data_dir)

    # Test getting first item
    item = dataset[0]
    assert isinstance(item, dict)
    assert "primary_subject" in item
    assert "abstract" in item


def test_getitem_all_indices(mock_data, tmp_path):
    """Test that __getitem__ works for all valid indices."""
    split_name = "train"
    data_dir = tmp_path / "data"
    split_path = data_dir / split_name
    split_path.mkdir(parents=True)
    mock_data.save_to_disk(str(split_path))

    dataset = mlops_project.data.ArxivPapersDataset(split=split_name, data_dir=data_dir)

    # Test that all indices are accessible
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)


@pytest.mark.parametrize("split", ["train", "eval", "test"])
def test_different_splits(mock_data, tmp_path, split):
    """Test that ArxivPapersDataset works with different split names."""
    data_dir = tmp_path / "data"

    split_path = data_dir / split
    split_path.mkdir(parents=True, exist_ok=True)
    mock_data.save_to_disk(str(split_path))

    dataset = mlops_project.data.ArxivPapersDataset(split=split, data_dir=data_dir)
    assert len(dataset) == len(mock_data)
