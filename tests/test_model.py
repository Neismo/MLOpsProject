import numpy as np
import pytest

from mlops_project.model import instantiate_sentence_transformer, instantiate_onnx_model


def test_instantiate_uses_cache_dir_and_model_name(mock_sentence_transformer_model):
    model, call_info, cache_dir = mock_sentence_transformer_model

    assert cache_dir.exists()
    assert call_info["cache_folder"] == str(cache_dir)
    assert call_info["model_name"] == "all-MiniLM-L6-v2"
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


def test_instantiate_onnx_model_uses_path_and_cpu_provider(mock_onnx_model):
    model, call_info, onnx_path = mock_onnx_model

    assert call_info["onnx_path"] == str(onnx_path)
    assert call_info["providers"] == ["CPUExecutionProvider"]
    assert model is not None


def test_onnx_output_matches_sentence_transformer(mock_sentence_transformer_model, mock_onnx_model):
    st_model, _, _ = mock_sentence_transformer_model
    onnx_model, _, _ = mock_onnx_model

    sentences = ["short text", "another one", "final sample"]
    st_embeddings = st_model(sentences)

    input_feed = {"input_ids": np.zeros((len(sentences), 8), dtype=np.int64)}
    onnx_embeddings = onnx_model.run(None, input_feed)[0]

    assert isinstance(onnx_embeddings, np.ndarray)
    assert onnx_embeddings.shape == st_embeddings.shape
    assert np.allclose(onnx_embeddings, st_embeddings)


def test_instantiate_onnx_model_file_not_found(tmp_path):
    missing_path = tmp_path / "missing.onnx"

    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        instantiate_onnx_model(str(missing_path))
