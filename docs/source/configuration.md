# Configuration

## Hydra config layout
Training and preprocessing use Hydra config files in `configs/`.

Key files:
- `configs/dataset.yaml`
- `configs/train_config.yaml`
- `configs/gpu_train_vertex.yaml` (Vertex AI custom job spec)

## Dataset configuration
```yaml
# configs/dataset.yaml
source: nick007x/arxiv-papers
splits:
  test_size: 0.2
  seed: 42
pairs:
  num_train: 100000
  num_eval: 10000
  loss: MultipleNegativesRankingLoss
  balanced: true
  text_field: abstract
```

## Training configuration
```yaml
# configs/train_config.yaml
meta:
  save_model: true
  use_gcs: false
  bucket_name: mlops-proj
  require_cuda: true
train:
  epochs: 1
  batch_size: 256
  loss: MultipleNegativesRankingLoss
  model: all-MiniLM-L6-v2
wandb:
  enabled: true
```

## Override examples
```bash
uv run python src/mlops_project/train.py train.batch_size=128 train.loss=ContrastiveLoss
uv run python src/mlops_project/data.py pairs.num_train=50000 pairs.balanced=false
```
