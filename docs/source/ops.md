# MLOps stack
This page covers the data, experiment, and deployment tooling that keeps the project reproducible.

## Ops stack
[DVC](https://dvc.org/doc) tracks datasets, `data.dvc` pins state, and `.dvc/config` points to the GCS remote. Experiment
tracking is handled through [Weights & Biases](https://docs.wandb.ai/) and controlled by `wandb.enabled`, which can be
disabled for offline runs or CI. We ship [Docker](https://docs.docker.com/) images for training and inference using
`dockerfiles/train.dockerfile` and `dockerfiles/api.dockerfile`.

## Automation and cloud
[GitHub Actions](https://docs.github.com/actions) runs formatting, linting, type checking, and tests on every push to
`master`. The build job submits `cloudbuild.yaml` to
[Cloud Build](https://cloud.google.com/build/docs) to publish the training image.

`configs/gpu_train_vertex.yaml` describes a [Vertex AI](https://cloud.google.com/vertex-ai/docs) custom job for GPU
training. Combine it with the Cloud Build image to run managed training jobs.
