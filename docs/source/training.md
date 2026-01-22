# Training

## Training pipeline
Training runs from `src/mlops_project/train.py` and is configured via [Hydra](https://hydra.cc/docs/intro/) using
`configs/train_config.yaml`.

Flow:
1. Validate CUDA availability when `meta.require_cuda=true`.
2. Ensure data exists (preprocess if needed).
3. Load training and evaluation pairs.
4. Instantiate the [SentenceTransformer](https://www.sbert.net/) model.
5. Train with `SentenceTransformerTrainer` from
   [Sentence Transformers](https://www.sbert.net/docs/training/overview.html) and evaluate precision at k.

Key configuration options:
- `train.model`: base model (default `all-MiniLM-L6-v2`).
- `train.loss`: `MultipleNegativesRankingLoss` or `ContrastiveLoss`.
- `train.batch_size`, `train.epochs`, `train.warmup_ratio`, `train.seed`.
- `meta.save_model`: toggle checkpoint saving.
- `meta.use_gcs` and `meta.bucket_name`: load and save to GCS paths.
- `wandb.enabled`: disable [Weights & Biases](https://docs.wandb.ai/) logging for offline runs.

## Run training and outputs
```bash
uv run python src/mlops_project/train.py
```

Override parameters:
```bash
uv run python src/mlops_project/train.py train.loss=ContrastiveLoss train.epochs=3 wandb.enabled=false
```

**Docker + CUDA**
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
