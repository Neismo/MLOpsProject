# API and deployment

## FastAPI service
The API serves normalized embeddings for abstracts.

Endpoints:
- `GET /health` returns service status and device.
- `POST /embed` accepts `{ "abstract": "..." }` and returns embedding data.

Example request:
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d "{\"abstract\": \"Graph neural networks for molecule property prediction.\"}"
```

## Running locally
The API loads a model from `MODEL_PATH` or defaults to `models/contrastive-minilm/`.
```bash
MODEL_PATH="models/all-MiniLM-L6-v2-mnrl-100k-balanced" uv run uvicorn src.mlops_project.api:app \
  --host 0.0.0.0 --port 8000
```
The model directory must exist and contain a SentenceTransformer model. GPU is used when available.

## Docker image
Build and run the inference container:
```bash
docker build -f dockerfiles/api.dockerfile -t mlops-api:latest .
docker run --rm -p 8000:8000 -e MODEL_PATH=/app/models/contrastive-minilm mlops-api:latest
```
