# Machine Learning Operations Project

ML Ops Project for 02476 @DTU

## Project Description

The goal of this project is to finetune text embedding models on scientific paper titles and abstracts from arXiv in order to learn embeddings that correspond to the papers subject categories. The goal is for papers belonging to the same category to have similar embeddings (based on cosine similarity), while papers from different categories are farther apart. This will be done using a contrastive learning objective, such as ContrastiveLoss or MultipleNegativesRankingLoss, where positive pairs are formed from papers sharing the same subject label and negatives are implicitly sampled from other categories within a batch. Based on the embeddings we can evaluate how many of the closest neighbours are from the same category (precision@k).

Based on the learned embeddings, we will train a lightweight classifier (logistic regression) to predict the subject category of a paper. This allows us to evaluate the quality of the embeddings quantitatively through classification accuracy and related metrics. In addition, the embeddings will be used to perform similarity search using cosine similarity, enabling the retrieval of the most semantically similar papers given a title or abstract. Together, these tasks allow us to assess whether the learned representations are useful both for categorization and for content-based paper discovery.

For deployment we plan to make a FastAPI endpoint running in a docker container. We plan to experiment using ONNX as the runtime. We are also interested in using a specific library for similarity search across all our samples, such as FAISS.

## Frameworks

The project will be implemented in `Python` using `PyTorch` as the deep learning backend. We will use the `sentence-transformers` library to handle transformer-based embedding models, loss functions for contrastive learning, and evaluation utilities. Experiment tracking and metric logging will be handled with `Weights & Biases (WandB)` to enable comparison between different model configurations and training runs. `Hydra` will be used for configuration management, allowing us to define datasets, model parameters, and training settings in a modular and reproducible way.

## Data

We will use the arXiv papers [dataset](https://huggingface.co/datasets/nick007x/arxiv-papers), using the title, abstract, and subject (primary category) fields. The subject labels will be used for both the contrastive embedding training and the classification task. For early experiments, we may restrict the dataset to a subset of high-level categories or limit the number of samples per category to reduce computational cost and class imbalance. The same dataset will be used for two main tasks:

1. Predicting the subject category of a paper based on its text
2. Retrieving semantically similar papers using embedding-based similarity search

## Models

We plan to use the embedding model [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a lightweight text embedding model with 22M parameters that offers a good balance between performance and training efficiency. Depending on available training time and resources, we will also experiment with larger and more expressive models, such as [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), to compare how model scale impacts embedding quality and downstream performance.

As a baseline we are also considering TF-IDF with linear classification.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
