# Evaluation

## Retrieval metrics
Evaluation uses `InformationRetrievalEvaluator` with precision at k (1, 5, 10). The evaluator:
- Samples a subset of the test dataset (default 5000).
- Uses 20 percent of samples as queries and the rest as corpus.
- Computes precision at k on subject aligned relevance sets.

## During training
Training evaluates every 500 steps and logs metrics to stdout and W&B when enabled.

## Custom evaluation
You can build an evaluator directly:
```python
from mlops_project.evaluate import create_ir_evaluator
from mlops_project.data import ArxivPapersDataset

dataset = ArxivPapersDataset("test").dataset
evaluator = create_ir_evaluator(dataset, sample_size=2000)
```
