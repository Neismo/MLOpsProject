# Overview

## Goals and scope
This project fine tunes sentence transformer models on arXiv titles and abstracts so that papers from the same
primary subject are close in embedding space, while unrelated subjects move apart. The learned embeddings are
used for semantic retrieval and serve as a foundation for lightweight classification experiments.

Implemented today:
- Preprocessing and pair generation pipeline.
- Training with retrieval evaluation during training.
- API service with `/health` and `/embed` endpoints.
- Dockerfiles for training and inference, CI pipeline for lint and tests.

Planned next steps:
- Embedding based classifier (logistic regression) and similarity search integration.
- ONNX inference path and accelerated retrieval backends like FAISS.
- Model comparisons with larger transformer backbones and TF-IDF baselines.

## Inputs, outputs, and components
Inputs:
- `title`, `abstract`, `primary_subject`, `subjects`

Outputs:
- trained embedding models
- pair datasets for contrastive learning
- retrieval metrics and API ready artifacts

Core components:
- Data preprocessing that splits the dataset and builds contrastive or positive pairs.
- Training with `SentenceTransformerTrainer` and losses such as `MultipleNegativesRankingLoss` or `ContrastiveLoss`.
- Evaluation using `InformationRetrievalEvaluator` to report precision at k.
- FastAPI service that returns normalized embeddings for any abstract.

## Technology stack and dependencies
Core runtime:
- `torch` and `accelerate` for model training and runtime acceleration.
- `transformers` and `sentence-transformers` for embedding models, losses, and trainers.
- `datasets` for dataset loading, slicing, and disk caching.
- `hydra-core` for config composition and CLI overrides.
- `loguru` for structured logging to console and files.
- `fastapi` and `uvicorn` for the embedding API service.
- `wandb` for experiment tracking and metrics.
- `dvc[gs]` and `dvc-gdrive` for data versioning and remote storage.
- `requests` for HTTP clients and integrations.
- `invoke` for project task automation.
- `typer` for lightweight CLI tooling and future command interfaces.
- `jupyter` for exploratory notebooks.

Dev, quality, and docs:
- `pytest` and `coverage` for tests and coverage reporting.
- `ruff` and `mypy` for linting, formatting, and typing.
- `pre-commit` for consistent local checks.
- `mkdocs`, `mkdocs-material`, and `mkdocstrings-python` for documentation.

Tooling:
- `uv` is used as the package manager and command runner in all examples.
