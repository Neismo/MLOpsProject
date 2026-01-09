import os
from sentence_transformers import SentenceTransformer

def instantiate_sentence_transformer(from_checkpoint: bool = False, cache_dir: str = 'models/cache/'):
    os.makedirs(cache_dir, exist_ok=True)
    if from_checkpoint:
        raise NotImplementedError("Loading from checkpoint is not implemented yet.")
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
    return model

if __name__ == "__main__":
    model = instantiate_sentence_transformer()
    x = ["This is a test sentence.", "This is another sentence."]
    print(f"Output shape of model: {model(x).shape}")
