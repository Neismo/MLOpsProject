# Data and preprocessing

## Dataset and splits
The project uses the Hugging Face dataset [`nick007x/arxiv-papers`](https://huggingface.co/datasets/nick007x/arxiv-papers)
and loads it with [datasets](https://huggingface.co/docs/datasets/). We rely on the following fields:
- `primary_subject` for contrastive labels
- `subjects` for metadata
- `abstract` and `title` as text inputs

`preprocess` creates train, eval, and test splits and stores them under `data/` in datasets format.
The split ratios and seed live in `configs/dataset.yaml` and are managed via [Hydra](https://hydra.cc/docs/intro/).

## Pairs and preprocessing workflow
Two pair building strategies are supported:
- `create_positive_pairs` for `MultipleNegativesRankingLoss` (anchor/positive pairs).
- `create_contrastive_pairs` for `ContrastiveLoss` (sentence1/sentence2 with labels).

The `balanced` flag controls whether subjects are sampled uniformly or weighted by frequency.

## Preprocessing flow
```mermaid
flowchart TD
  A[Load dataset] --> B[Select columns]
  B --> C[Split train eval test]
  C --> D[Save splits to data]
  C --> E[Build train pairs]
  C --> F[Build eval pairs]
  E --> G[Save train pairs]
  F --> H[Save eval pairs]
  D --> I[Save preprocess config]
  G --> I
  H --> I
```

**Run preprocessing**
```bash
uv run python src/mlops_project/data.py
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

`ensure_data_exists` will re-run preprocessing if the saved config no longer matches the requested config.
