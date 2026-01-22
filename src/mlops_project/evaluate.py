from collections import defaultdict

import numpy as np
from loguru import logger
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def create_ir_evaluator(dataset, sample_size: int = 5000, name: str = "arxiv-retrieval"):
    """Create an Information Retrieval evaluator for precision@k metrics."""
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)

    # Split indices: first 20% for queries, rest for corpus (no overlap)
    num_queries = sample_size // 5
    query_indices = indices[:num_queries]
    corpus_indices = indices[num_queries:]

    queries = {}
    corpus = {}
    relevant_docs = {}
    subject_to_corpus_ids = defaultdict(set)

    # Build corpus from corpus_indices only
    for i, idx in enumerate(corpus_indices):
        idx = int(idx)
        corpus_id = f"doc_{i}"
        corpus[corpus_id] = dataset[idx]["abstract"]
        subject = dataset[idx]["primary_subject"]
        subject_to_corpus_ids[subject].add(corpus_id)

    # Build queries from query_indices (no overlap with corpus)
    for i, idx in enumerate(query_indices):
        idx = int(idx)
        query_id = f"query_{i}"
        queries[query_id] = dataset[idx]["abstract"]
        subject = dataset[idx]["primary_subject"]
        # Relevant docs are corpus docs with same subject
        relevant_docs[query_id] = subject_to_corpus_ids[subject].copy()

    # Filter out queries with no relevant docs
    queries = {qid: q for qid, q in queries.items() if len(relevant_docs.get(qid, set())) > 0}
    relevant_docs = {qid: docs for qid, docs in relevant_docs.items() if qid in queries}

    logger.info(f"IR Evaluator: {len(queries)} queries, {len(corpus)} corpus docs (no overlap)")

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        precision_recall_at_k=[1, 5, 10],
        show_progress_bar=True,
    )
