# API module notes

The [FastAPI](https://fastapi.tiangolo.com/) app lives in `src/mlops_project/api.py` and loads a
[SentenceTransformer](https://www.sbert.net/) model at import time.
That behavior is not friendly to auto documentation, so this module is documented manually.

## Entry points
- CLI flag: `--model-path`
- Environment: `MODEL_PATH`
- Default: `models/contrastive-minilm/`

## Endpoints
- `GET /health` returns status and device.
- `POST /embed` returns normalized embeddings for an abstract.

## Runtime notes
- The model is loaded once on startup.
- Embeddings are normalized to unit length.
- GPU is used when available.
- ONNX export and inference are supported via `mlops_project.model` and [ONNX Runtime](https://onnxruntime.ai/docs/).

See [API and deployment](../api.md) for usage and deployment details.
