# MLOps stack

## Ops stack
- Data versioning: [DVC](https://dvc.org/doc) tracks datasets, `data.dvc` pins state, and `.dvc/config` points to the GCS remote.
- Experiment tracking: [Weights & Biases](https://docs.wandb.ai/) logging is controlled by `wandb.enabled` and can be disabled for offline runs or CI.
- Containers: [Docker](https://docs.docker.com/) images in `dockerfiles/train.dockerfile` for training and `dockerfiles/api.dockerfile` for inference.

## Automation and cloud
[GitHub Actions](https://docs.github.com/actions) runs formatting, linting, type checking, and tests on every push to
`master`. The build job submits `cloudbuild.yaml` to
[Cloud Build](https://cloud.google.com/build/docs) to publish the training image.

`configs/gpu_train_vertex.yaml` describes a [Vertex AI](https://cloud.google.com/vertex-ai/docs) custom job for GPU
training. Combine it with the Cloud Build image to run managed training jobs.
