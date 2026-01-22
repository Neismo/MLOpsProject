# MLOps stack

## Data versioning
The dataset is tracked with DVC. `data.dvc` pins the dataset state and `.dvc/config` points to the GCS remote.

## Experiment tracking
W&B logging is controlled by `wandb.enabled`. Disable it for offline runs or CI.

## Docker and containers
Two Dockerfiles provide reproducible environments:
- `dockerfiles/train.dockerfile` for training.
- `dockerfiles/api.dockerfile` for inference.

## CI and automation
GitHub Actions runs formatting, linting, type checking, and tests on every push to `master`.
The build job submits `cloudbuild.yaml` to Google Cloud Build to publish the training image.

## Cloud training
`configs/gpu_train_vertex.yaml` describes a Vertex AI custom job for GPU training.
Combine it with the Cloud Build image to run managed training jobs.

## On this page
- [Data versioning](#data-versioning)
- [Experiment tracking](#experiment-tracking)
- [Docker and containers](#docker-and-containers)
- [CI and automation](#ci-and-automation)
- [Cloud training](#cloud-training)
