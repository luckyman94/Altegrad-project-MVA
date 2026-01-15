import faiss
import torch
import numpy as np

class GraphRetriever:
    def __init__(self, index_path, texts_path):
        self.index = faiss.read_index(index_path)
        self.texts = torch.load(texts_path)

    def search(self, z_query: torch.Tensor, k: int = 3):
        z = z_query.cpu().numpy()
        z = z / np.linalg.norm(z, axis=1, keepdims=True)

        _, idx = self.index.search(z.astype("float32"), k)

        return [
            [self.texts[j] for j in row]
            for row in idx
        ]
