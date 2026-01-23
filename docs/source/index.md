# ArXiv Contrastive Embeddings

Fine tune [Sentence Transformers](https://www.sbert.net/) models on arXiv titles and abstracts to produce embeddings
that cluster papers by primary subject and power similarity search.

## Project summary
We build pair datasets from arXiv metadata for contrastive learning, fine tune sentence transformer models across
multiple backbones, and evaluate semantic retrieval alongside embedding-based classification. The system builds a
[FAISS](https://faiss.ai/) similarity search index, supports [ONNX Runtime](https://onnxruntime.ai/docs/) for faster
inference paths, and serves embeddings behind a FastAPI endpoint for downstream apps and retrieval services.

The pipeline starts by downloading and splitting the arXiv dataset, then builds contrastive pairs from primary
subjects. We train sentence transformers with the selected loss, evaluate precision@k alongside classifier metrics,
compare TF-IDF baselines, export ONNX models, and build a FAISS index before serving embeddings through the API.

## Get started
This repository uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for package and project management.
```bash
uv sync --dev
uv run python src/mlops_project/data.py
uv run python src/mlops_project/train.py
```

## Key docs
Start with [Data and preprocessing](data.md), then follow [Training](training.md) and [Evaluation](evaluation.md).
Deployment details live in [API and deployment](api.md), and configuration lives in [Configuration](configuration.md).
