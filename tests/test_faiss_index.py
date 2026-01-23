"""Tests for FAISS index functionality."""

import sys

import numpy as np
import pytest

from mlops_project.faiss_index import FAISSIndex, SearchResult


@pytest.mark.skipif(sys.platform == "darwin", reason="FAISS IVF training segfaults on macOS CI")
class TestFAISSIndex:
    """Tests for the FAISSIndex class."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample normalized embeddings."""
        np.random.seed(42)
        n_samples = 1000
        dim = 384
        embeddings = np.random.randn(n_samples, dim).astype(np.float32)
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    @pytest.fixture
    def sample_metadata(self):
        """Generate sample metadata matching embeddings."""
        subjects = ["Machine Learning", "Physics", "Biology"]
        return [
            {
                "title": f"Paper {i}",
                "abstract": f"Abstract content for paper {i}",
                "primary_subject": subjects[i % len(subjects)],
                "subjects": [subjects[i % len(subjects)], "General Science"],
            }
            for i in range(1000)
        ]

    @pytest.fixture
    def faiss_index(self, sample_embeddings, sample_metadata):
        """Build a FAISS index for testing."""
        return FAISSIndex.build(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            nlist=32,  # Use smaller nlist for faster tests
            nprobe=8,
        )

    def test_build_index(self, sample_embeddings, sample_metadata):
        """Test that build() creates a valid index."""
        index = FAISSIndex.build(
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            nlist=32,
            nprobe=8,
        )

        assert index.index is not None
        assert index.index.ntotal == len(sample_embeddings)
        assert len(index.metadata) == len(sample_metadata)
        assert index.nprobe == 8

    def test_build_index_metadata_mismatch(self, sample_embeddings):
        """Test that build() raises on size mismatch."""
        wrong_metadata = [{"title": "Paper", "abstract": "x", "primary_subject": "ML", "subjects": []}]
        with pytest.raises(ValueError, match="size mismatch"):
            FAISSIndex.build(sample_embeddings, wrong_metadata)

    def test_search_returns_results(self, faiss_index, sample_embeddings):
        """Test that search returns expected number of results."""
        query = sample_embeddings[0]
        results = faiss_index.search(query, k=5)

        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_result_structure(self, faiss_index, sample_embeddings):
        """Test that search results have correct structure."""
        query = sample_embeddings[0]
        results = faiss_index.search(query, k=3)

        for result in results:
            assert isinstance(result.score, float)
            assert isinstance(result.title, str)
            assert isinstance(result.abstract, str)
            assert isinstance(result.primary_subject, str)
            assert isinstance(result.subjects, list)

    def test_search_scores_descending(self, faiss_index, sample_embeddings):
        """Test that search results are sorted by descending score."""
        query = sample_embeddings[0]
        results = faiss_index.search(query, k=10)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_self_highest_score(self, faiss_index, sample_embeddings):
        """Test that querying with an indexed vector returns itself first."""
        query = sample_embeddings[42]
        results = faiss_index.search(query, k=1)

        assert len(results) == 1
        # The score should be close to 1.0 for the exact match
        assert results[0].score > 0.99
        assert results[0].title == "Paper 42"

    def test_search_handles_2d_query(self, faiss_index, sample_embeddings):
        """Test that search handles both 1D and 2D query arrays."""
        query_1d = sample_embeddings[0]
        query_2d = sample_embeddings[0].reshape(1, -1)

        results_1d = faiss_index.search(query_1d, k=5)
        results_2d = faiss_index.search(query_2d, k=5)

        assert len(results_1d) == len(results_2d)
        for r1, r2 in zip(results_1d, results_2d):
            assert r1.title == r2.title

    def test_search_without_index_raises(self):
        """Test that search without loaded index raises error."""
        empty_index = FAISSIndex()
        query = np.random.randn(384).astype(np.float32)

        with pytest.raises(RuntimeError, match="Index not loaded"):
            empty_index.search(query, k=5)

    def test_save_and_load(self, faiss_index, sample_embeddings, tmp_path):
        """Test saving and loading the index."""
        save_dir = tmp_path / "faiss_index"
        faiss_index.save(save_dir)

        # Check files exist
        assert (save_dir / "index.faiss").exists()
        assert (save_dir / "metadata.json").exists()

        # Load and verify
        loaded_index = FAISSIndex.load(save_dir)
        assert loaded_index.index.ntotal == faiss_index.index.ntotal
        assert len(loaded_index.metadata) == len(faiss_index.metadata)
        assert loaded_index.nprobe == faiss_index.nprobe

        # Verify search produces same results
        query = sample_embeddings[0]
        original_results = faiss_index.search(query, k=5)
        loaded_results = loaded_index.search(query, k=5)

        for orig, loaded in zip(original_results, loaded_results):
            assert orig.title == loaded.title
            assert abs(orig.score - loaded.score) < 1e-5

    def test_load_nonexistent_index(self, tmp_path):
        """Test that loading non-existent index raises error."""
        with pytest.raises(FileNotFoundError, match="Index file not found"):
            FAISSIndex.load(tmp_path / "nonexistent")

    def test_load_missing_metadata(self, faiss_index, tmp_path):
        """Test that loading with missing metadata raises error."""
        save_dir = tmp_path / "partial_index"
        faiss_index.save(save_dir)

        # Remove metadata file
        (save_dir / "metadata.json").unlink()

        with pytest.raises(FileNotFoundError, match="Neither .* nor .* found"):
            FAISSIndex.load(save_dir)

    def test_load_with_nprobe_override(self, faiss_index, tmp_path):
        """Test loading with nprobe override."""
        save_dir = tmp_path / "faiss_index"
        faiss_index.save(save_dir)

        loaded_index = FAISSIndex.load(save_dir, nprobe=16)
        assert loaded_index.nprobe == 16


@pytest.mark.skipif(sys.platform == "darwin", reason="FAISS IVF training segfaults on macOS CI")
class TestSearchAPIEndpoint:
    """Tests for the /search API endpoint."""

    @pytest.fixture
    def api_client_with_faiss(self, monkeypatch, tmp_path):
        """TestClient with mocked FAISS index for API tests."""

        # Create mock embeddings and metadata
        np.random.seed(123)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        metadata = [
            {
                "title": f"Test Paper {i}",
                "abstract": f"Abstract for test paper {i}",
                "primary_subject": "Test Subject",
                "subjects": ["Test Subject", "Other"],
            }
            for i in range(100)
        ]

        # Build and save test index
        index_dir = tmp_path / "faiss"
        index_dir.mkdir()
        test_index = FAISSIndex.build(embeddings, metadata, nlist=8, nprobe=4)
        test_index.save(index_dir)

        # Mock model
        class _MockModel:
            def __init__(self, model_path, device=None):
                pass

            def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
                np.random.seed(hash(text) % (2**32))
                embedding = np.random.randn(384).astype(np.float32)
                return embedding / np.linalg.norm(embedding)

        monkeypatch.setattr("sentence_transformers.SentenceTransformer", _MockModel)

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        monkeypatch.setenv("MODEL_PATH", str(model_dir))
        monkeypatch.setenv("INDEX_PATH", str(index_dir))

        import importlib
        import mlops_project.api

        importlib.reload(mlops_project.api)

        from fastapi.testclient import TestClient

        return TestClient(mlops_project.api.app)

    @pytest.fixture
    def api_client_no_faiss(self, monkeypatch, tmp_path):
        """TestClient without FAISS index for 503 tests."""

        class _MockModel:
            def __init__(self, model_path, device=None):
                pass

            def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
                embedding = np.random.randn(384).astype(np.float32)
                return embedding / np.linalg.norm(embedding)

        monkeypatch.setattr("sentence_transformers.SentenceTransformer", _MockModel)

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        monkeypatch.setenv("MODEL_PATH", str(model_dir))
        # Ensure INDEX_PATH is not set
        monkeypatch.delenv("INDEX_PATH", raising=False)

        import importlib
        import mlops_project.api

        importlib.reload(mlops_project.api)

        from fastapi.testclient import TestClient

        return TestClient(mlops_project.api.app)

    def test_search_returns_200(self, api_client_with_faiss):
        """Test that search endpoint returns 200 with valid request."""
        response = api_client_with_faiss.post("/search", json={"abstract": "Test query about neural networks", "k": 5})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 5

    def test_search_result_structure(self, api_client_with_faiss):
        """Test that search results have expected structure."""
        response = api_client_with_faiss.post("/search", json={"abstract": "Test query", "k": 3})
        assert response.status_code == 200

        results = response.json()["results"]
        for result in results:
            assert "score" in result
            assert "title" in result
            assert "abstract" in result
            assert "primary_subject" in result
            assert "subjects" in result

    def test_search_empty_abstract_returns_400(self, api_client_with_faiss):
        """Test that empty abstract returns 400."""
        response = api_client_with_faiss.post("/search", json={"abstract": "", "k": 5})
        assert response.status_code == 400

    def test_search_whitespace_abstract_returns_400(self, api_client_with_faiss):
        """Test that whitespace-only abstract returns 400."""
        response = api_client_with_faiss.post("/search", json={"abstract": "   ", "k": 5})
        assert response.status_code == 400

    def test_search_missing_abstract_returns_422(self, api_client_with_faiss):
        """Test that missing abstract field returns 422."""
        response = api_client_with_faiss.post("/search", json={"k": 5})
        assert response.status_code == 422

    def test_search_no_index_returns_503(self, api_client_no_faiss):
        """Test that search without index returns 503."""
        response = api_client_no_faiss.post("/search", json={"abstract": "Test query", "k": 5})
        assert response.status_code == 503
        assert "FAISS index not loaded" in response.json()["detail"]

    def test_search_default_k(self, api_client_with_faiss):
        """Test that k defaults to 10."""
        response = api_client_with_faiss.post("/search", json={"abstract": "Test query"})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 10

    def test_search_with_nprobe(self, api_client_with_faiss):
        """Test that nprobe parameter is accepted."""
        response = api_client_with_faiss.post("/search", json={"abstract": "Test query", "k": 5, "nprobe": 16})
        assert response.status_code == 200

    def test_search_invalid_k_returns_422(self, api_client_with_faiss):
        """Test that invalid k values return 422."""
        response = api_client_with_faiss.post("/search", json={"abstract": "Test", "k": 0})
        assert response.status_code == 422

        response = api_client_with_faiss.post("/search", json={"abstract": "Test", "k": 101})
        assert response.status_code == 422


@pytest.mark.skipif(sys.platform == "darwin", reason="FAISS IVF training segfaults on macOS CI")
class TestMetricAPIEndpoint:
    """Tests for Prometheus metrics endpoint."""

    @pytest.fixture
    def api_client_with_faiss(self, monkeypatch, tmp_path):
        """TestClient with mocked FAISS index for API tests."""

        # Create mock embeddings and metadata
        np.random.seed(123)
        embeddings = np.random.randn(100, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        metadata = [
            {
                "title": f"Test Paper {i}",
                "abstract": f"Abstract for test paper {i}",
                "primary_subject": "Test Subject",
                "subjects": ["Test Subject", "Other"],
            }
            for i in range(100)
        ]

        # Build and save test index
        index_dir = tmp_path / "faiss"
        index_dir.mkdir()
        test_index = FAISSIndex.build(embeddings, metadata, nlist=8, nprobe=4)
        test_index.save(index_dir)

        # Mock model
        class _MockModel:
            def __init__(self, model_path, device=None):
                pass

            def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
                np.random.seed(hash(text) % (2**32))
                embedding = np.random.randn(384).astype(np.float32)
                return embedding / np.linalg.norm(embedding)

        monkeypatch.setattr("sentence_transformers.SentenceTransformer", _MockModel)

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        monkeypatch.setenv("MODEL_PATH", str(model_dir))
        monkeypatch.setenv("INDEX_PATH", str(index_dir))

        import importlib
        import mlops_project.api

        importlib.reload(mlops_project.api)

        from fastapi.testclient import TestClient

        return TestClient(mlops_project.api.app)

    def test_metrics_endpoint(self, api_client_with_faiss):
        """Test that /metrics endpoint returns Prometheus metrics."""
        response = api_client_with_faiss.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "api_request_count" in content
        assert "Total number of API requests" in content
