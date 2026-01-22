# Data and preprocessing

## Dataset and splits
The project uses the Hugging Face dataset `nick007x/arxiv-papers`. We rely on the following fields:
- `primary_subject` for contrastive labels
- `subjects` for metadata
- `abstract` and `title` as text inputs

`preprocess` creates train, eval, and test splits and stores them under `data/` in Hugging Face datasets format.
The split ratios and seed live in `configs/dataset.yaml`.

## Pairs and preprocessing workflow
Two pair building strategies are supported:
- `create_positive_pairs` for `MultipleNegativesRankingLoss` (anchor/positive pairs).
- `create_contrastive_pairs` for `ContrastiveLoss` (sentence1/sentence2 with labels).

The `balanced` flag controls whether subjects are sampled uniformly or weighted by frequency.

**Run preprocessing**
```bash
uv run inv preprocess
```

Override defaults with Hydra:
```bash
uv run python src/mlops_project/data.py pairs.loss=ContrastiveLoss pairs.num_train=200000 pairs.balanced=false
```

Alternative module entrypoint:
```bash
uv run python -m mlops_project.data
```

**Output artifacts**
After preprocessing you should see:
- `data/train`, `data/eval`, `data/test`
- `data/train_pairs`, `data/eval_pairs`
- `data/preprocess_config.yaml` for reproducibility checks

`ensure_data_exists` will re run preprocessing if the saved config no longer matches the requested config.
