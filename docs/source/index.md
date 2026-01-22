# ArXiv Contrastive Embeddings

Fine tune [Sentence Transformers](https://www.sbert.net/) models on arXiv titles and abstracts to produce embeddings
that cluster papers by primary subject and power similarity search.

## Project summary
We build pair datasets from arXiv metadata for contrastive learning, fine tune sentence transformer models with
[Sentence Transformers](https://www.sbert.net/) across multiple backbones, and evaluate semantic retrieval alongside
embedding-based classification using [scikit-learn](https://scikit-learn.org/stable/). The system builds a
[FAISS](https://faiss.ai/) similarity search index, supports [ONNX Runtime](https://onnxruntime.ai/docs/) for faster
inference paths, and serves embeddings behind a [FastAPI](https://fastapi.tiangolo.com/) endpoint for downstream
apps and retrieval services.

**Pipeline steps**
1. Download and split the arXiv dataset into train, eval, and test sets.
2. Build positive or contrastive pairs based on primary subject labels.
3. Train sentence transformers with the chosen contrastive objective and model backbone.
4. Evaluate retrieval quality with precision at 1, 5, and 10, plus classifier metrics.
5. Compare to TF-IDF baselines with [scikit-learn](https://scikit-learn.org/stable/) and export models to ONNX.
6. Build a [FAISS](https://faiss.ai/) similarity search index and deploy the embedding API for downstream use.

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
