# ArXiv Contrastive Embeddings

Fine tune sentence transformer models on arXiv titles and abstracts to produce embeddings that cluster papers
by primary subject and power similarity search.

## Project summary
- Builds pair datasets from arXiv metadata for contrastive learning.
- Fine tunes sentence transformer models with Sentence Transformers trainers.
- Evaluates semantic retrieval with precision at k metrics.
- Serves embeddings behind a FastAPI endpoint for downstream apps.

**Pipeline steps**
1. Download and split the arXiv dataset into train, eval, and test sets.
2. Build positive or contrastive pairs based on primary subject labels.
3. Train sentence transformers with the chosen contrastive objective.
4. Evaluate retrieval quality with precision at 1, 5, and 10.
5. Deploy an embedding API for similarity search and downstream experiments.

## Get started
```bash
uv sync --dev
uv run inv preprocess
uv run inv train
```

## Key docs
- [Data and preprocessing](data.md)
- [Training](training.md)
- [Evaluation](evaluation.md)
- [API and deployment](api.md)
- [Configuration](configuration.md)
