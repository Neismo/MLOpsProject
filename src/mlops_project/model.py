import os
from sentence_transformers import SentenceTransformer


def instantiate_sentence_transformer(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_seq_length: int = 256,
    from_checkpoint: bool = False,
    cache_dir: str = "models/cache/",
):
    """Instantiate a SentenceTransformer model.

    Args:
        model_name: HuggingFace model name or path.
        max_seq_length: Maximum sequence length for tokenization.
        from_checkpoint: Load from a saved checkpoint (not implemented).
        cache_dir: Directory to cache downloaded models.

    Returns:
        SentenceTransformer model instance.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if from_checkpoint:
        raise NotImplementedError("Loading from checkpoint is not implemented yet.")

    # ModernBERT requires special handling with custom Transformer + Pooling
    if "ModernBERT" in model_name:
        from sentence_transformers import models

        word_model = models.Transformer(
            model_name,
            max_seq_length=max_seq_length,
            model_args={
                "attn_implementation": "sdpa",
                "torch_dtype": "bfloat16",
                "device_map": "auto",
            },
        )
        pooling = models.Pooling(word_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_model, pooling])
    else:
        model = SentenceTransformer(model_name, cache_folder=cache_dir)

    return model


if __name__ == "__main__":  # pragma: no cover
    model = instantiate_sentence_transformer()
    x = ["This is a test sentence.", "This is another sentence."]
    print(f"Output shape of model: {model.encode(x).shape}")
