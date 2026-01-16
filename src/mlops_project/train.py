import os
import random
import mlops_project.data
from mlops_project.model import instantiate_sentence_transformer as get_model
from mlops_project.data import ArxivPapersDataset
import torch
from loguru import logger
import hydra
import sys
from pathlib import Path
from hydra.utils import get_original_cwd
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss

REQUIRE_CUDA = os.getenv("REQUIRE_CUDA", "1") != "0"
WANDB_DISABLED = os.getenv("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
if REQUIRE_CUDA:
    assert torch.cuda.is_available(), "CUDA is required for this run. Set REQUIRE_CUDA=0 to allow CPU/debug runs."
elif not torch.cuda.is_available():
    logger.warning("CUDA not available; continuing on CPU because REQUIRE_CUDA=0.")

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("train.log", level="DEBUG", rotation="5 MB")


def create_ir_evaluator(dataset, sample_size: int = 5000, name: str = "arxiv-retrieval"):
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


def create_stratified_ir_evaluator(dataset, samples_per_subject: int = 50, name: str = "arxiv-stratified"):
    """
    Creates an IR Evaluator that guarantees coverage of all subjects.

    Caps large subjects to prevent dominance and ensures rare subjects are represented.

    Args:
        dataset: The dataset to evaluate on.
        samples_per_subject: Max samples per subject (split between query/corpus).
        name: Name for the evaluator.
    """
    random.seed(42)

    subject_to_indices = defaultdict(list)
    for idx, subject in enumerate(dataset["primary_subject"]):
        subject_to_indices[subject].append(idx)

    queries = {}
    corpus = {}
    relevant_docs = {}
    subject_to_corpus_ids = defaultdict(set)

    logger.info(f"Stratifying across {len(subject_to_indices)} subjects...")

    for subject, indices in subject_to_indices.items():
        if len(indices) < 2:
            continue

        random.shuffle(indices)
        n_samples = min(len(indices), samples_per_subject)
        selected_indices = indices[:n_samples]

        # Split: 20% queries (min 1), rest corpus
        n_queries = max(1, int(n_samples * 0.2))
        query_idxs = selected_indices[:n_queries]
        corpus_idxs = selected_indices[n_queries:]

        for idx in corpus_idxs:
            corpus_id = f"doc_{idx}"
            corpus[corpus_id] = dataset[int(idx)]["abstract"]
            subject_to_corpus_ids[subject].add(corpus_id)

        for idx in query_idxs:
            query_id = f"query_{idx}"
            queries[query_id] = dataset[int(idx)]["abstract"]
            queries[query_id + "_subj"] = subject

    # Link queries to relevant docs
    final_queries = {}
    for q_key, q_text in queries.items():
        if "_subj" in q_key:
            continue
        subject = queries[q_key + "_subj"]
        rel_docs = subject_to_corpus_ids[subject]
        if len(rel_docs) > 0:
            final_queries[q_key] = q_text
            relevant_docs[q_key] = rel_docs

    logger.info(f"Stratified Evaluator: {len(final_queries)} queries, {len(corpus)} corpus docs")
    logger.info(f"Coverage: {len(subject_to_corpus_ids)} subjects represented")

    return InformationRetrievalEvaluator(
        queries=final_queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        precision_recall_at_k=[1, 5, 10],
        show_progress_bar=True,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_config")
def train(config):
    logger.debug(f"Training config:\n{config}")

    test_dataset = ArxivPapersDataset("test", data_dir=Path(f"{get_original_cwd()}/data")).dataset
    model = get_model(
        model_name=config.model.name,
        max_seq_length=config.model.max_seq_length,
        cache_dir=f"{get_original_cwd()}/models/cache/",
    )

    # Create/load training/eval pairs
    if config.meta.use_gcs:
        train_pairs_path = Path(f"/gcs/{config.meta.bucket_name}/data/train_pairs")
        eval_pairs_path = Path(f"/gcs/{config.meta.bucket_name}/data/eval_pairs")
    else:
        train_pairs_path = Path(f"{get_original_cwd()}/data/train_pairs")
        eval_pairs_path = Path(f"{get_original_cwd()}/data/eval_pairs")

    train_pairs = mlops_project.data.load_pairs(train_pairs_path)
    logger.info(f"Training pairs: {len(train_pairs)}")

    eval_pairs = mlops_project.data.load_pairs(eval_pairs_path)
    logger.info(f"Evaluation pairs: {len(eval_pairs)}")

    # Create the IR evaluator for precision@k
    if config.eval.stratified:
        ir_evaluator = create_stratified_ir_evaluator(
            test_dataset, samples_per_subject=config.eval.samples_per_subject
        )
    else:
        ir_evaluator = create_ir_evaluator(test_dataset, sample_size=config.eval.sample_size)
    logger.info("IR Evaluator created for precision@k metrics")

    wandb_enabled = config.wandb.enabled and not WANDB_DISABLED
    if not wandb_enabled:
        logger.info(
            f"W&B logging disabled (config.wandb.enabled={config.wandb.enabled}, WANDB_DISABLED={WANDB_DISABLED})"
        )

    if config.meta.use_gcs:
        model_dir = Path(f"/gcs/{config.meta.bucket_name}/models/contrastive-minilm")
    else:
        model_dir = Path(f"{get_original_cwd()}/models/contrastive-minilm")

    # create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Define training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        learning_rate=config.train.learning_rate,
        warmup_ratio=config.train.warmup_ratio,
        eval_strategy="steps",
        eval_steps=config.train.eval_steps,
        eval_on_start=config.train.eval_on_start,
        save_strategy="steps" if config.meta.save_model else "no",
        save_steps=config.train.save_steps,
        logging_steps=config.train.logging_steps,
        torch_compile=config.train.torch_compile and torch.cuda.is_available(),
        fp16=torch.cuda.is_available() and not config.train.bf16,
        bf16=config.train.bf16 and torch.cuda.is_available(),
        tf32=config.train.tf32 and torch.cuda.is_available(),
        report_to="wandb" if wandb_enabled else "none",
    )

    # Initialize loss function
    match config.train.loss:
        case "ContrastiveLoss":
            loss = ContrastiveLoss(model=model)
        case "MultipleNegativesRankingLoss":
            loss = MultipleNegativesRankingLoss(model=model)
        case _:
            logger.error(f"Unsupported loss type from config: {config.train.loss}")
            raise ValueError(f"Unsupported loss type: {config.train.loss}")

    # Create trainer with IR evaluator for precision@k
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_pairs,
        eval_dataset=eval_pairs,
        loss=loss,
        evaluator=ir_evaluator,
    )
    logger.info("Trainer initialized with precision@k evaluator. Ready to train!")

    # Start training
    trainer.train()


if __name__ == "__main__":  # pragma: no cover
    train()
