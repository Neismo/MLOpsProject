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

## Stack
A quick snapshot of the tooling that anchors data, training, automation, and serving across the project.

<div class="stack-figure" role="img" aria-label="Stack diagram showing data, training, cloud automation, and serving tools.">
  <div class="stack-strip">
    <div class="stack-group">
      <div class="stack-group-title">Data + versioning</div>
      <div class="stack-items">
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/huggingface" alt="Hugging Face logo">Hugging Face Datasets</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/dvc" alt="DVC logo">DVC</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/googlecloud" alt="Google Cloud logo">GCS</span>
      </div>
    </div>
    <div class="stack-group">
      <div class="stack-group-title">Training + tracking</div>
      <div class="stack-items">
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/pytorch" alt="PyTorch logo">PyTorch + Sentence Transformers</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/weightsandbiases" alt="Weights and Biases logo">Weights & Biases</span>
        <span class="stack-tool"><img src="logos/logo.svg" alt="Hydra logo">Hydra</span>
      </div>
    </div>
    <div class="stack-group">
      <div class="stack-group-title">Source + automation</div>
      <div class="stack-items">
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/github" alt="GitHub logo">GitHub</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/githubactions" alt="GitHub Actions logo">GitHub Actions</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/docker" alt="Docker logo">Docker</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/googlecloud" alt="Google Cloud logo">Cloud Build</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/googlecloud" alt="Google Cloud logo">Vertex AI</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/googlecloud" alt="Google Cloud logo">Artifact Registry / Container Registry</span>
      </div>
    </div>
    <div class="stack-group">
      <div class="stack-group-title">Serving + retrieval</div>
      <div class="stack-items">
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/googlecloud" alt="Google Cloud logo">Cloud Run</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/fastapi" alt="FastAPI logo">FastAPI</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/onnx" alt="ONNX logo">ONNX Runtime</span>
        <span class="stack-tool"><img src="https://cdn.simpleicons.org/meta" alt="Meta logo">FAISS</span>
      </div>
    </div>
  </div>
</div>

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
