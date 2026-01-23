# Evaluation
This page summarizes how we evaluate retrieval, classification, baselines, and similarity search.

## Retrieval metrics
Evaluation uses `InformationRetrievalEvaluator` with precision at k (1, 5, 10). The evaluator samples a subset of the
test dataset (default 5000), uses 20 percent of samples as queries and the rest as corpus, and computes precision at
k on subject-aligned relevance sets.

Training evaluates every 500 steps and logs metrics to stdout and
[Weights & Biases](https://docs.wandb.ai/) when enabled.

## Embedding classifier
We train a lightweight logistic regression classifier on frozen embeddings with
[scikit-learn](https://scikit-learn.org/stable/). Report accuracy, macro F1, and per class metrics to quantify how
well embeddings separate subject categories.

## Baselines and model comparisons
Baseline comparisons include TF-IDF features with scikit-learn plus logistic regression. We also compare multiple
transformer backbones to quantify the tradeoffs between model size, training time, and retrieval/classification quality.

## Similarity search
We build a [FAISS](https://faiss.ai/) index over normalized embeddings to support fast similarity search. Cosine
similarity is implemented as inner product on unit normalized vectors, which aligns with the training objective.

## Custom evaluation
You can build an evaluator directly:
```python
from mlops_project.evaluate import create_ir_evaluator
from mlops_project.data import ArxivPapersDataset

dataset = ArxivPapersDataset("test").dataset
evaluator = create_ir_evaluator(dataset, sample_size=2000)
```
