import os
import faiss
import numpy as np


class FaissIndexer:
    def __init__(self, dimension: int, index_path: str):
        self.dimension = dimension
        self.index_path = index_path
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            flat = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(flat)

    def add(self, ids: np.ndarray, embeddings: np.ndarray) -> None:
        assert ids.dtype == np.int64
        assert embeddings.dtype == np.float32
        self.index.add_with_ids(embeddings, ids)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
