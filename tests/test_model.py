import numpy as np
import pytest

from mlops_project.model import instantiate_sentence_transformer


def test_instantiate_uses_cache_dir_and_model_name(mock_sentence_transformer_model):
    model, call_info, cache_dir = mock_sentence_transformer_model

    assert cache_dir.exists()
    assert call_info["cache_folder"] == str(cache_dir)
    assert call_info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert model is not None


def test_model_call_returns_expected_shape(mock_sentence_transformer_model):
    model, _, _ = mock_sentence_transformer_model

    sentences = ["short text", "another one", "final sample"]
    embeddings = model(sentences)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(sentences), 384)


def test_from_checkpoint_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        instantiate_sentence_transformer(from_checkpoint=True, cache_dir=str(tmp_path))
