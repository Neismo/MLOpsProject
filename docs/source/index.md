<div class="hero">
  <div class="hero-inner">
    <p class="hero-tag">MLOps Project</p>
    <h1 class="hero-title">ArXiv Contrastive Embeddings</h1>
    <p class="hero-lead">
      Fine tune sentence transformer models on arXiv titles and abstracts to produce
      embeddings that cluster papers by primary subject and power similarity search.
    </p>
    <div class="hero-actions">
      <a class="md-button md-button--primary" href="overview/">Read the overview</a>
      <a class="md-button" href="quickstart/">Quickstart</a>
    </div>
    <div class="hero-meta">
      <span class="hero-pill">Dataset: nick007x/arxiv-papers</span>
      <span class="hero-pill">Loss: MultipleNegativesRankingLoss</span>
      <span class="hero-pill">Metric: precision@k</span>
    </div>
  </div>
  <div class="hero-card">
    <div class="stat">
      <div class="stat-label">Pairs</div>
      <div class="stat-value">100k train / 10k eval</div>
    </div>
    <div class="stat">
      <div class="stat-label">Model</div>
      <div class="stat-value">all-MiniLM-L6-v2</div>
    </div>
    <div class="stat">
      <div class="stat-label">Output</div>
      <div class="stat-value">Normalized embeddings</div>
    </div>
  </div>
</div>

## What this project does
- Builds pair datasets from arXiv metadata for contrastive learning.
- Fine tunes sentence transformer models with Sentence Transformers trainers.
- Evaluates semantic retrieval with precision at k metrics.
- Serves embeddings behind a FastAPI endpoint for downstream apps.

## Pipeline at a glance
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
- [Data and preprocessing](data/)
- [Training](training/)
- [Evaluation](evaluation/)
- [API and deployment](api/)
- [Configuration](configuration/)

## On this page
- [What this project does](#what-this-project-does)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Get started](#get-started)
- [Key docs](#key-docs)
