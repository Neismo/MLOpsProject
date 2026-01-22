# Training

## Entry point
Training runs from `src/mlops_project/train.py` and is configured via Hydra using `configs/train_config.yaml`.

## Training flow
1. Validate CUDA availability when `meta.require_cuda=true`.
2. Ensure data exists (preprocess if needed).
3. Load training and evaluation pairs.
4. Instantiate the sentence transformer model.
5. Train with `SentenceTransformerTrainer` and evaluate precision at k.

## Key configuration options
- `train.model`: base model (default `all-MiniLM-L6-v2`).
- `train.loss`: `MultipleNegativesRankingLoss` or `ContrastiveLoss`.
- `train.batch_size`, `train.epochs`, `train.warmup_ratio`, `train.seed`.
- `meta.save_model`: toggle checkpoint saving.
- `meta.use_gcs` and `meta.bucket_name`: load and save to GCS paths.
- `wandb.enabled`: disable logging for offline runs.

## Run training
```bash
uv run inv train
```

Override parameters:
```bash
uv run python src/mlops_project/train.py train.loss=ContrastiveLoss train.epochs=3 wandb.enabled=false
```

## Outputs and logs
Models are saved under `models/<model>-<loss>-<pairs>-<balanced>` when `meta.save_model=true`.
Logs stream to stdout and `train.log` via loguru.
