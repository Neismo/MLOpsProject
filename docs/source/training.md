# Training

## Training pipeline
Training runs from `src/mlops_project/train.py` and is configured via [Hydra](https://hydra.cc/docs/intro/) using
`configs/train_config.yaml`.

The training run validates CUDA availability when `meta.require_cuda=true`, ensures the dataset and pairs exist (and
preprocesses if needed), loads training and evaluation pairs, instantiates the SentenceTransformer model, and then
trains with `SentenceTransformerTrainer` while evaluating precision@k during training.

Key configuration options include `train.model` (default `all-MiniLM-L6-v2`), `train.loss` set to
`MultipleNegativesRankingLoss` or `ContrastiveLoss`, and the usual training controls like `train.batch_size`,
`train.epochs`, `train.warmup_ratio`, and `train.seed`. Use `meta.save_model` to toggle checkpoint saving, and
`meta.use_gcs` plus `meta.bucket_name` to load and save to GCS paths. Disable
[Weights & Biases](https://docs.wandb.ai/) logging with `wandb.enabled=false` for offline runs.

## Run training and outputs
```bash
uv run python src/mlops_project/train.py
```

Override parameters:
```bash
uv run python src/mlops_project/train.py train.loss=ContrastiveLoss train.epochs=3 wandb.enabled=false
```

## Docker + CUDA
Requires [Docker](https://docs.docker.com/) with NVIDIA GPU support.

Build the CUDA training image:
```bash
docker build -f dockerfiles/train.dockerfile -t mlops-train:cuda .
```

Run training on GPU with mounted volumes:
```bash
docker run --rm --gpus all \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  mlops-train:cuda uv run python src/mlops_project/train.py
```

Disable W&B logging:
```bash
docker run --rm --gpus all \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  mlops-train:cuda uv run python src/mlops_project/train.py wandb.enabled=false
```

Models are saved under `models/<model>-<loss>-<pairs>-<balanced>` when `meta.save_model=true`.
Logs stream to stdout and `train.log` via [Loguru](https://github.com/Delgan/loguru).
