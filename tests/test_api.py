import numpy as np


class TestHealthEndpoint:
    def test_health_returns_ok(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["device"] in ["cpu", "cuda"]


class TestEmbedEndpoint:
    def test_embed_valid_abstract(self, api_client):
        response = api_client.post("/embed", json={"abstract": "Test abstract"})
        assert response.status_code == 200
        data = response.json()
        assert "embedding_dim" in data
        assert "embedding" in data
        assert len(data["embedding"]) == data["embedding_dim"]

    def test_embed_empty_returns_400(self, api_client):
        response = api_client.post("/embed", json={"abstract": ""})
        assert response.status_code == 400

    def test_embed_whitespace_returns_400(self, api_client):
        response = api_client.post("/embed", json={"abstract": "   "})
        assert response.status_code == 400

    def test_embed_missing_field_returns_422(self, api_client):
        response = api_client.post("/embed", json={})
        assert response.status_code == 422

    def test_embedding_is_normalized(self, api_client):
        response = api_client.post("/embed", json={"abstract": "Test"})
        embedding = np.array(response.json()["embedding"])
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
