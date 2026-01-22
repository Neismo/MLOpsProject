# MLOps stack

## Ops stack
- Data versioning: DVC tracks datasets, `data.dvc` pins state, and `.dvc/config` points to the GCS remote.
- Experiment tracking: W&B logging is controlled by `wandb.enabled` and can be disabled for offline runs or CI.
- Containers: `dockerfiles/train.dockerfile` for training and `dockerfiles/api.dockerfile` for inference.

## Automation and cloud
GitHub Actions runs formatting, linting, type checking, and tests on every push to `master`. The build job submits
`cloudbuild.yaml` to Google Cloud Build to publish the training image.

`configs/gpu_train_vertex.yaml` describes a Vertex AI custom job for GPU training. Combine it with the Cloud Build
image to run managed training jobs.
