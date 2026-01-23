# Quickstart

## Setup and data
This repository uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for package and project management.
Start by syncing dependencies:
```bash
uv sync --dev
```

Then preprocess the data to create dataset splits and pair datasets under `data/`:
```bash
uv run python src/mlops_project/data.py
```

## Train and serve
Train a model with the default config:
```bash
uv run python src/mlops_project/train.py
```

If you want to run on CPU or tweak a parameter, override the config on the CLI:
```bash
uv run python src/mlops_project/train.py meta.require_cuda=false train.batch_size=64 train.epochs=1
```

Run the embedding API with [Uvicorn](https://www.uvicorn.org/) serving [FastAPI](https://fastapi.tiangolo.com/):
```bash
MODEL_PATH="models/all-MiniLM-L6-v2-mnrl-100k-balanced" uv run uvicorn src.mlops_project.api:app \
  --host 0.0.0.0 --port 8000
```

## Validate and iterate
Run the test suite with [pytest](https://docs.pytest.org/):
```bash
uv run pytest tests/
```

Serve docs locally with [MkDocs](https://www.mkdocs.org/):
```bash
uv run mkdocs serve -f mkdocs.yaml
```
