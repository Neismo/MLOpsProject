# API module notes

The [FastAPI](https://fastapi.tiangolo.com/) app lives in `src/mlops_project/api.py` and loads a
[SentenceTransformer](https://www.sbert.net/) model at import time.
That behavior is not friendly to auto documentation, so this module is documented manually.

## Entry points
The API reads configuration from the CLI flag `--model-path`, then falls back to the `MODEL_PATH` environment
variable, and finally defaults to `models/contrastive-minilm/`.

## Endpoints
The service exposes `GET /health` to report status and device, and `POST /embed` to return normalized embeddings
for an abstract.

## Runtime notes
The model loads once on startup and normalizes embeddings to unit length. GPU is used when available. ONNX export
and inference are supported via `mlops_project.model` and [ONNX Runtime](https://onnxruntime.ai/docs/).

See [API and deployment](../api.md) for usage and deployment details.
