import logging
import time
from typing import List

import numpy as np
from task_oriented_dataset_search.embedding.interface import BaseEmbedder
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        logger.info(
            f"Initializing SentenceTransformerEmbedder with model: {model_name}"
        )
        logger.debug(f"Device: {device or 'auto'}, Batch size: {batch_size}")
        try:
            self._model = SentenceTransformer(model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self.batch_size = batch_size
            logger.info(
                f"SentenceTransformer model loaded successfully. Dimension: {self._dimension}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model {model_name}: {e}",
                exc_info=True,
            )
            raise

    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("Embed called with an empty list of texts.")
            return np.array([], dtype="float32").reshape(0, self._dimension)

        logger.debug(f"Embedding {len(texts)} texts...")
        start_time = time.time()

        try:
            embs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            end_time = time.time()
            logger.info(
                f"Finished embedding {len(texts)} texts in {end_time - start_time:.2f} seconds."
            )
            return embs.astype("float32")
        except Exception as e:
            logger.error(f"Failed during text embedding: {e}", exc_info=True)
            raise
