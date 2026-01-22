# Overview

## Goal
This project fine tunes sentence transformer models on arXiv titles and abstracts so that papers from the same
primary subject are close in embedding space, while unrelated subjects move apart. The learned embeddings are
used for semantic retrieval and serve as a foundation for lightweight classification experiments.

## Inputs and outputs
- Inputs: `title`, `abstract`, `primary_subject`, `subjects`
- Outputs: trained embedding models, pair datasets for contrastive learning, retrieval metrics, API ready artifacts

## Core components
- Data preprocessing that splits the dataset and builds contrastive or positive pairs.
- Training with `SentenceTransformerTrainer` and losses such as `MultipleNegativesRankingLoss` or `ContrastiveLoss`.
- Evaluation using `InformationRetrievalEvaluator` to report precision at k.
- FastAPI service that returns normalized embeddings for any abstract.

## Technology stack
- Python 3.12, PyTorch, Sentence Transformers
- Hydra for configuration, W&B for experiment tracking
- DVC for data versioning, Docker for reproducible containers
- GitHub Actions and Google Cloud Build for CI and builds

## Status and scope
Implemented today:
- Preprocessing and pair generation pipeline.
- Training with retrieval evaluation during training.
- API service with `/health` and `/embed` endpoints.
- Dockerfiles for training and inference, CI pipeline for lint and tests.

Planned next steps:
- Embedding based classifier and similarity search integration.
- ONNX inference path and accelerated retrieval backends like FAISS.
