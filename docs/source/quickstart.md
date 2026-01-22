# Quickstart

## Setup and data
This repository uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for package and project management.

**Install dependencies**
```bash
uv sync --dev
```

**Preprocess data**
```bash
uv run python src/mlops_project/data.py
```

This writes the dataset splits and pair datasets under `data/`.

## Train and serve
**Train a model**
```bash
uv run python src/mlops_project/train.py
```

To run on CPU or tweak a parameter:
```bash
uv run python src/mlops_project/train.py meta.require_cuda=false train.batch_size=64 train.epochs=1
```

**Run the embedding API**
Run the embedding API with [Uvicorn](https://www.uvicorn.org/) serving [FastAPI](https://fastapi.tiangolo.com/):
```bash
MODEL_PATH="models/all-MiniLM-L6-v2-mnrl-100k-balanced" uv run uvicorn src.mlops_project.api:app \
  --host 0.0.0.0 --port 8000
```

## Validate and iterate
**Run tests**
Run the test suite with [pytest](https://docs.pytest.org/):
```bash
uv run pytest tests/
```

**Serve docs**
Serve docs locally with [MkDocs](https://www.mkdocs.org/):
```bash
uv run mkdocs serve -f mkdocs.yaml
```
