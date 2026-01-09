import os
from sentence_transformers import SentenceTransformer
class Model:
    def __init__(self, from_checkpoint: bool = False, cache_dir: str = 'models/cache/'):
        os.makedirs(cache_dir, exist_ok=True)
        if from_checkpoint:
            raise NotImplementedError("Loading from checkpoint is not implemented yet.")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
    def __call__(self, x):
        return self.model.encode(x)

if __name__ == "__main__":
    model = Model()
    x = ["This is a test sentence.", "This is another sentence."]
    print(f"Output shape of model: {model(x).shape}")
