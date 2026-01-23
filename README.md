# Machine Learning Operations Project

ML Ops Project for 02476 @DTU

<p align="center">
  <img src="figs/embedding_landscape.png" alt="Embedding landscape">
</p>

## Project Description

The goal of this project is to finetune text embedding models on scientific paper titles and abstracts from arXiv in order to learn embeddings that correspond to the papers subject categories. The goal is for papers belonging to the same category to have similar embeddings (based on cosine similarity), while papers from different categories are farther apart. This will be done using a contrastive learning objective, such as ContrastiveLoss or MultipleNegativesRankingLoss, where positive pairs are formed from papers sharing the same subject label and negatives are implicitly sampled from other categories within a batch. Based on the embeddings we can evaluate how many of the closest neighbours are from the same category (precision@k).

Based on the learned embeddings, we will train a lightweight classifier (logistic regression) to predict the subject category of a paper. This allows us to evaluate the quality of the embeddings quantitatively through classification accuracy and related metrics. In addition, the embeddings will be used to perform similarity search using cosine similarity, enabling the retrieval of the most semantically similar papers given a title or abstract. Together, these tasks allow us to assess whether the learned representations are useful both for categorization and for content-based paper discovery.

For deployment we plan to make a FastAPI endpoint running in a docker container. We plan to experiment using ONNX as the runtime. We are also interested in using a specific library for similarity search across all our samples, such as FAISS.

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

The project will be implemented in `Python` using `PyTorch` as the deep learning backend. We will use the `sentence-transformers` library to handle transformer-based embedding models, loss functions for contrastive learning, and evaluation utilities. Experiment tracking and metric logging will be handled with `Weights & Biases (WandB)` to enable comparison between different model configurations and training runs. `Hydra` will be used for configuration management, allowing us to define datasets, model parameters, and training settings in a modular and reproducible way.

## Data

We will use the arXiv papers [dataset](https://huggingface.co/datasets/nick007x/arxiv-papers), using the title, abstract, and subject (primary category) fields. The subject labels will be used for both the contrastive embedding training and the classification task. For early experiments, we may restrict the dataset to a subset of high-level categories or limit the number of samples per category to reduce computational cost and class imbalance. The same dataset will be used for two main tasks:

1. Predicting the subject category of a paper based on its text
2. Retrieving semantically similar papers using embedding-based similarity search

### To download and create dataset

```bash
uv run python -m mlops_project.data
```

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
  mlops-train:cuda uv run python src/mlops_project/train.py
```

Disable Weights & Biases logging (optional):

```bash
docker run --rm --gpus all \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  mlops-train:cuda uv run python src/mlops_project/train.py wandb.enabled=false
```

A docker image is automatically built when pushed to the `build` branch if `format + lint` and `tests` actions pass via a GitHub action workflow using the Google Cloud Build. A branch was specifically set up for this to avoid wasting many minutes, as each build can, without optimization of any sorts, take up towards 20 minutes before completion.

## Models

We plan to use the embedding model [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a lightweight text embedding model with 22M parameters that offers a good balance between performance and training efficiency. Depending on available training time and resources, we will also experiment with larger and more expressive models, such as [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), to compare how model scale impacts embedding quality and downstream performance.

As a baseline we are also considering TF-IDF with linear classification.

## Project structure

The directory structure of the project looks like this:
```txt
├── .dvc/
│   ├── config
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       ├── docs.yml
│       ├── lint-test-build.yml
│       └── pre-commit-update.yml
├── configs/                  # Configuration files
│   ├── dataset.yaml
│   ├── gpu_train_vertex.yaml
│   └── train_config.yaml
├── data/                     # Data directory
│   ├── train_pairs/
│   ├── eval_pairs/
│   └── test/
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── README.md
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   ├── figures/
│   ├── README.md
│   └── report.py
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   ├── utils.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .dockerignore
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── coudbuild.yaml
├── README.md                 # Project README
├── data.dvc
├── mkdocs.yaml
├── uv.lock
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps). Modified as needed
