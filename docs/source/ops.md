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

While the project can run end to end on a laptop, the cloud path is designed to be the durable, repeatable route for
large runs. Dataset state is tracked with DVC and the configured remote in `.dvc/config` points to a GCS bucket
(`gs://mlops-proj`), so the exact same dataset snapshot can be pulled into local experiments or cloud training without
recreating preprocessing steps.

The training environment is built by Cloud Build from `cloudbuild.yaml`, which pushes a GPU-ready image to Artifact
Registry in the `europe-west1` region. That same image URI is referenced in `configs/gpu_train_vertex.yaml`, so a Vertex
AI custom job executes the identical container that you run locally with Docker. If you reuse this setup in another GCP
project or region, update the registry path and bucket name so the cloud jobs can resolve the image and storage
locations correctly.
