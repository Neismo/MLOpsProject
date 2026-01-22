# Project structure

```text
.
|-- .dvc/                       # DVC settings and remotes
|-- .github/                    # GitHub Actions workflows and Dependabot
|   |-- dependabot.yaml
|   `-- workflows/
|       |-- lint-test-build.yaml
|       `-- pre-commit-update.yaml
|-- configs/                    # Hydra config files
|   |-- dataset.yaml
|   |-- gpu_train_vertex.yaml
|   `-- train_config.yaml
|-- data/                       # Versioned datasets (DVC)
|-- dockerfiles/                # Training and API containers
|   |-- api.dockerfile
|   `-- train.dockerfile
|-- docs/                       # Documentation sources
|   |-- README.md
|   `-- source/
|       |-- api.md
|       |-- configuration.md
|       |-- data.md
|       |-- evaluation.md
|       |-- index.md
|       |-- ops.md
|       |-- overview.md
|       |-- project-structure.md
|       |-- quickstart.md
|       |-- reference.md
|       |-- tags.md
|       |-- training.md
|       |-- reference/
|       |   `-- api.md
|       `-- stylesheets/
|           `-- extra.css
|-- models/                     # Trained model artifacts
|-- notebooks/                  # Exploration notebooks
|-- outputs/                    # Training outputs and logs
|-- reports/                    # Reports and figures
|-- src/
|   `-- mlops_project/
|       |-- __init__.py
|       |-- api.py
|       |-- data.py
|       |-- evaluate.py
|       |-- model.py
|       |-- train.py
|       |-- utils.py
|       `-- visualize.py
|-- tests/                      # Unit tests
|   |-- conftest.py
|   |-- test_api.py
|   |-- test_data.py
|   `-- test_model.py
|-- cloudbuild.yaml
|-- data.dvc
|-- mkdocs.yaml
|-- pyproject.toml
|-- README.md
|-- tasks.py
`-- uv.lock
```
