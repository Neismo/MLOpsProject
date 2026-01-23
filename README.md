# Machine Learning Operations Project - 02476 @ DTU

<p align="center">
  <img src="figs/embedding_landscape.png" alt="Embedding landscape">
</p>

## Project Description

The goal of this project is to finetune text embedding models on scientific paper titles and abstracts from arXiv in order to learn embeddings that correspond to the papers subject categories. Papers belonging to the same category have similar embeddings (based on cosine similarity), while papers from different categories are farther apart. This is achieved using a contrastive learning objective (MultipleNegativesRankingLoss), where positive pairs are formed from papers sharing the same subject label and negatives are implicitly sampled from other categories within a batch.

The quality of embeddings is evaluated using information retrieval metrics including precision@k, recall@k, MRR (Mean Reciprocal Rank), MAP (Mean Average Precision), and NDCG (Normalized Discounted Cumulative Gain). The embeddings are used to perform similarity search, enabling the retrieval of the most semantically similar papers given an abstract. This demonstrates that the learned representations are useful for content-based paper discovery.

For deployment, we use a FastAPI application running in a Docker container. Similarity search is powered by FAISS (Facebook AI Similarity Search) with an IVF index for efficient nearest-neighbor lookup across all indexed papers.

## Documentation

Full documentation is available at [https://neismo.github.io/MLOpsProject/](https://neismo.github.io/MLOpsProject/)

The documentation includes:
- Project overview and architecture
- Data preprocessing and model training guides
- API documentation and deployment instructions
- Configuration management with Hydra
- MLOps stack and tools

### Building documentation locally

To build and serve the documentation locally:
```bash
# Build the documentation
uv run mkdocs build -f mkdocs.yaml

# Serve locally with live reload
uv run mkdocs serve -f mkdocs.yaml
```

The documentation will be available at `http://127.0.0.1:8000/`

## Frameworks

The project is implemented in `Python` using `PyTorch` as the deep learning backend. We use the `sentence-transformers` library to handle transformer-based embedding models, loss functions for contrastive learning, and evaluation utilities. Experiment tracking and metric logging is handled with `Weights & Biases (WandB)` to enable comparison between different model configurations and training runs. `Hydra` is used for configuration management, allowing us to define datasets, model parameters, and training settings in a modular and reproducible way.

## Data

We use the arXiv papers [dataset](https://huggingface.co/datasets/nick007x/arxiv-papers), using the title, abstract, and subject (primary category) fields. The subject labels are used for contrastive embedding training, where papers from the same category form positive pairs. The dataset is balanced by limiting samples per category to reduce class imbalance.

### To download and create dataset

```bash
uv run python -m mlops_project.data
```

### Build FAISS index

After training a model, build a FAISS index for similarity search:

```bash
# Build index with default settings
uv run python -m mlops_project.build_index

# Custom model and output paths
uv run python -m mlops_project.build_index \
  --model-path models/your-model \
  --output-dir data/faiss \
  --batch-size 256
```

Options:
- `--model-path`, `-m`: Path to trained SentenceTransformer model
- `--output-dir`, `-o`: Output directory for index files (default: `data/faiss`)
- `--batch-size`, `-b`: Batch size for encoding (default: 256)
- `--splits`, `-s`: Dataset splits to index (default: train, eval, test)
- `--nlist`: Number of IVF clusters (default: 4096)
- `--nprobe`: Clusters to probe at search time (default: 64)
- `--compile`: Use torch.compile() for faster encoding

### Train with Docker + CUDA

Build the CUDA training image (fixes Ubuntu base for Python 3.12):

```bash
docker build -f dockerfiles/train.dockerfile -t mlops-train:cuda .
```

Run training on GPU, mounting data/models so downloads and checkpoints persist:

```bash
docker run --rm --gpus all \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  mlops-train:cuda uv run python -m mlops_project.train
```

Disable Weights & Biases logging (optional):

```bash
docker run --rm --gpus all \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  mlops-train:cuda uv run python -m mlops_project.train wandb.enabled=false
```

A docker image is automatically built when pushed to the `build` branch if `format + lint` and `tests` actions pass via a GitHub action workflow using the Google Cloud Build. A branch was specifically set up for this to avoid wasting many minutes, as each build can, without optimization of any sorts, take up towards 20 minutes before completion.

## API

The project includes a FastAPI application for generating embeddings and performing similarity search.

### Running the API locally

```bash
MODEL_PATH=./models/your-model INDEX_PATH=./data/faiss uv run uvicorn mlops_project.api:app
```

### Running with Docker

```bash
# Build the API image
docker build -f dockerfiles/api.dockerfile -t mlops-api .

# Run with model and index mounted
docker run --rm -p 8000:8000 \
  -v ${PWD}/models/your-model:/data/model \
  -v ${PWD}/data/faiss:/data/index \
  mlops-api
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns status and device (cpu/cuda) |
| `/embed` | POST | Generate embedding for an abstract |
| `/search` | POST | Find similar papers using FAISS index |
| `/metrics` | GET | Prometheus metrics |
| `/` | GET | Web UI for interactive search |

### Example requests

```bash
# Health check
curl http://localhost:8000/health

# Generate embedding
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"abstract": "We study neural network optimization..."}'

# Search for similar papers (requires FAISS index)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"abstract": "We study neural network optimization...", "k": 5}'
```

## Models

We use the embedding model [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a lightweight text embedding model with 22M parameters that offers a good balance between performance and training efficiency. We also experimented with [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) to compare how model scale impacts embedding quality and downstream performance.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   └── workflows/
│       ├── docs.yml
│       ├── lint-test-build.yaml
│       └── pre-commit-update.yaml
├── configs/                  # Configuration files
│   ├── dataset.yaml
│   ├── gpu_train_vertex.yaml
│   └── train_config.yaml
├── data/                     # Data directory (DVC tracked)
│   ├── train_pairs/
│   ├── eval_pairs/
│   └── test/
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── README.md
│   └── source/
│       ├── api.md
│       ├── configuration.md
│       ├── data.md
│       ├── evaluation.md
│       ├── index.md
│       ├── ops.md
│       ├── overview.md
│       ├── project-structure.md
│       ├── quickstart.md
│       ├── reference.md
│       ├── tags.md
│       └── training.md
├── figs/                     # Figures for README
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   ├── README.md
│   └── report.py
├── scripts/                  # Utility scripts
│   ├── load_test.py
│   └── load_test_health.py
├── src/                      # Source code
│   └── mlops_project/
│       ├── __init__.py
│       ├── api.py
│       ├── build_index.py
│       ├── data.py
│       ├── evaluate.py
│       ├── faiss_index.py
│       ├── metrics.py
│       ├── model.py
│       ├── train.py
│       ├── utils.py
│       ├── visualize.py
│       └── static/
│           └── index.html
├── tests/                    # Tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_faiss_index.py
│   └── test_model.py
├── .dockerignore
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── cloudbuild.yaml           # Google Cloud Build config
├── README.md                 # Project README
├── data.dvc                  # DVC file for data tracking
├── mkdocs.yaml               # MkDocs configuration
├── uv.lock                   # uv lockfile
└── tasks.py                  # Invoke tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps). Modified as needed
