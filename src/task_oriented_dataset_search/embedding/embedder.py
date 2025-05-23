from typing import List

import numpy as np
from task_oriented_dataset_search.embedding.interface import BaseEmbedder
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        embs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        return embs.astype("float32")
