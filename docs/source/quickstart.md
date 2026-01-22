# Quickstart

## Setup and data
**Install dependencies**
```bash
uv sync --dev
```

**Preprocess data**
```bash
uv run inv preprocess
```

This writes the dataset splits and pair datasets under `data/`.

## Train and serve
**Train a model**
```bash
uv run inv train
```

To run on CPU or tweak a parameter:
```bash
uv run python src/mlops_project/train.py meta.require_cuda=false train.batch_size=64 train.epochs=1
```

**Run the embedding API**
```bash
MODEL_PATH="models/all-MiniLM-L6-v2-mnrl-100k-balanced" uv run uvicorn src.mlops_project.api:app \
  --host 0.0.0.0 --port 8000
```

## Validate and iterate
**Run tests**
```bash
uv run inv test
```

**Serve docs**
```bash
uv run mkdocs serve -f mkdocs.yaml
```
