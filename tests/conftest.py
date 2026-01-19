import pytest
import omegaconf
import random
import numpy as np
import sys
import types
import mlops_project.data
import mlops_project.model
from tests import N_TRAIN

from datasets import Dataset

SUBJECT_ABSTRACT_MAP = {
    "Condensed Matter": [
        "We study the low-temperature behavior of glassy systems on random graphs.",
        "Anharmonicities in single-site potentials lead to frustration in structural glasses.",
        "The distribution of localization lengths is analyzed in chemical gels.",
    ],
    "Network Theory": [
        "Conductivity of random resistor networks is calculated on a Bethe lattice.",
        "Percolation thresholds are determined for anisotropic regular lattices.",
        "Power-law expansions are used to approximate coordination numbers in networks.",
    ],
    "Machine Learning": [
        "Contrastive learning objective functions improve semantic representation.",
        "Transformers utilize self-attention mechanisms to capture global dependencies.",
        "We propose a new regularization technique for deep neural networks.",
    ],
}


def _generate_mock_contrastive_dataset(num_pairs: int = 64) -> Dataset:
    """Creates a mock dataset formatted for ContrastiveLoss."""

    data: dict[str, list[str | float]] = {"sentence1": [], "sentence2": [], "label": []}
    subject_names = list(SUBJECT_ABSTRACT_MAP.keys())

    num_positive = num_pairs // 2
    num_negative = num_pairs - num_positive

    for _ in range(num_positive):
        sub = random.choice(subject_names)
        pair = random.sample(SUBJECT_ABSTRACT_MAP[sub], k=2)

        data["sentence1"].append(f"{pair[0]} (Ref: {random.randint(2010, 2024)})")
        data["sentence2"].append(f"{pair[1]} (Ref: {random.randint(2010, 2024)})")
        data["label"].append(1.0)

    # 2. Generate Negative Pairs (Different subjects, Label 0.0)
    for _ in range(num_negative):
        # Pick two distinct subjects
        sub1, sub2 = random.sample(subject_names, k=2)

        sent1 = random.choice(SUBJECT_ABSTRACT_MAP[sub1])
        sent2 = random.choice(SUBJECT_ABSTRACT_MAP[sub2])

        data["sentence1"].append(f"{sent1} (Ref: {random.randint(2010, 2024)})")
        data["sentence2"].append(f"{sent2} (Ref: {random.randint(2010, 2024)})")
        data["label"].append(0.0)

    return Dataset.from_dict(data)


def _generate_mock_positive_pairs_dataset(num_pairs: int = 64) -> Dataset:
    """Creates a mock dataset formatted for MultipleNegativesRankingLoss."""

    data: dict[str, list[str]] = {"anchor": [], "positive": []}
    subject_names = list(SUBJECT_ABSTRACT_MAP.keys())

    for _ in range(num_pairs):
        sub = random.choice(subject_names)

        pair = random.sample(SUBJECT_ABSTRACT_MAP[sub], k=2)

        anchor_text = f"{pair[0]} Investigation conducted in {random.randint(2010, 2024)}."
        positive_text = f"{pair[1]} Research published in {random.randint(2010, 2024)}."

        data["anchor"].append(anchor_text)
        data["positive"].append(positive_text)

    return Dataset.from_dict(data)


@pytest.fixture(scope="function")
def mock_sentence_transformer_model(monkeypatch, tmp_path):
    """Provide a fake SentenceTransformer to avoid downloads during tests."""

    call_info: dict[str, str] = {}

    class _DummySentenceTransformer:
        def __init__(self, model_name: str, cache_folder: str):
            call_info["model_name"] = model_name
            call_info["cache_folder"] = cache_folder

        def __call__(self, sentences: list[str]):
            return np.zeros((len(sentences), 384))

    monkeypatch.setattr(mlops_project.model, "SentenceTransformer", _DummySentenceTransformer)

    cache_dir = tmp_path / "cache"
    model = mlops_project.model.instantiate_sentence_transformer(cache_dir=str(cache_dir))
    return model, call_info, cache_dir


@pytest.fixture(scope="function")
def mock_onnx_model(monkeypatch, tmp_path):
    """Provide a fake onnxruntime session to avoid loading a real ONNX file."""

    call_info: dict[str, object] = {}

    class _DummyInferenceSession:
        def __init__(self, onnx_path: str, providers: list[str]):
            call_info["onnx_path"] = onnx_path
            call_info["providers"] = providers

        def run(self, output_names, input_feed):
            batch_size = len(next(iter(input_feed.values())))
            return [np.zeros((batch_size, 384), dtype=np.float32)]

    dummy_module = types.SimpleNamespace(InferenceSession=_DummyInferenceSession)

    monkeypatch.setitem(sys.modules, "onnxruntime", dummy_module)
    monkeypatch.setattr(mlops_project.model.os.path, "exists", lambda path: True)

    onnx_path = tmp_path / "dummy.onnx"
    model = mlops_project.model.instantiate_onnx_model(str(onnx_path))
    return model, call_info, onnx_path


@pytest.fixture(autouse=False, scope="function")
def config_base() -> omegaconf.DictConfig:
    """Configuration with wandb disabled and no saving!"""
    return omegaconf.OmegaConf.create(
        {
            "meta": {"save_model": False},
            "wandb": {"enabled": False},
        }
    )


@pytest.fixture(autouse=False, scope="function")
def mock_positive_dataset(monkeypatch) -> None:
    """Override the data loading to some random data instead of real data.
    This assumes the positive pairs approach."""

    dataset = _generate_mock_positive_pairs_dataset(num_pairs=N_TRAIN)

    def _mock_load(load_path) -> Dataset:
        print(f"Mock loading pairs from {load_path}")
        return dataset

    monkeypatch.setattr(mlops_project.data, "load_pairs", _mock_load)


@pytest.fixture(autouse=False, scope="function")
def mock_contrastive_dataset(monkeypatch) -> None:
    """Override the data loading to some random data instead of real data.
    This assumes the contrastive pairs approach."""

    dataset = _generate_mock_contrastive_dataset(num_pairs=N_TRAIN)

    def _mock_load(load_path) -> Dataset:
        print(f"Mock loading pairs from {load_path}")
        return dataset

    monkeypatch.setattr(mlops_project.data, "load_pairs", _mock_load)


@pytest.fixture(autouse=False, scope="function")
def mock_data() -> Dataset:
    """Generates the dataset based on the provided physics and ML categories."""
    # Flatten the map into a format compatible with the datasets library
    data: dict[str, list[str]] = {"primary_subject": [], "abstract": []}
    for subject, abstracts in SUBJECT_ABSTRACT_MAP.items():
        for abs_text in abstracts:
            data["primary_subject"].append(subject)
            data["abstract"].append(abs_text)

    return Dataset.from_dict(data)


@pytest.fixture(autouse=False, scope="function")
def mock_raw_data() -> Dataset:
    """Generates raw dataset with all columns expected by preprocess function.

    Returns a dataset with columns: ["primary_subject", "subjects", "abstract", "title"]
    Generates 1000 samples with randomized variations to ensure uniqueness.
    """
    data: dict[str, list[str | list[str]]] = {"primary_subject": [], "subjects": [], "abstract": [], "title": []}

    title_templates = {
        "Condensed Matter": [
            "Low-Temperature Dynamics in Random Systems",
            "Frustration Effects in Structural Glasses",
            "Localization in Chemical Gel Networks",
        ],
        "Network Theory": [
            "Conductivity Analysis on Bethe Lattices",
            "Percolation in Anisotropic Lattices",
            "Power-Law Networks and Coordination",
        ],
        "Machine Learning": [
            "Contrastive Learning for Semantic Representation",
            "Self-Attention in Transformer Architectures",
            "Novel Regularization for Deep Networks",
        ],
    }

    semi_random_suffixes = [
        "with experimental validation",
        "using Monte Carlo simulations",
        "through numerical analysis",
        "via theoretical framework",
        "by computational methods",
        "with statistical inference",
        "using variational principles",
        "through systematic investigation",
    ]

    num_samples = 1000
    subjects_list = list(SUBJECT_ABSTRACT_MAP.keys())

    for i in range(num_samples):
        # Cycle through subjects to maintain balance
        subject = subjects_list[i % len(subjects_list)]
        abstracts = SUBJECT_ABSTRACT_MAP[subject]
        titles = title_templates[subject]

        # Cycle through the abstracts for each subject
        base_abstract = abstracts[i % len(abstracts)]
        base_title = titles[i % len(titles)]

        # Add random variations to make each sample unique
        year = random.randint(2010, 2024)
        value = random.uniform(0.1, 10.0)
        suffix = random.choice(semi_random_suffixes)

        # Create unique abstract with random additions
        unique_abstract = f"{base_abstract} Study {i + 1} ({year}), coefficient={value:.2f}, {suffix}."
        unique_title = f"{base_title} - Part {i + 1}"

        data["primary_subject"].append(subject)
        data["subjects"].append([subject, "General Physics"])
        data["abstract"].append(unique_abstract)
        data["title"].append(unique_title)

    return Dataset.from_dict(data)
