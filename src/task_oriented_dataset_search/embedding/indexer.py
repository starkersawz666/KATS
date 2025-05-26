import logging
import os
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndexer:
    def __init__(self, dimension: int, index_path: str):
        self.dimension = dimension
        self.index_path = index_path
        logger.info(
            f"Initializing FaissIndexer with dimension {dimension} and path {index_path}"
        )
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))

    @staticmethod
    def hex_to_int64(hex_id: str) -> np.int64:
        return np.int64(int(hex_id, 16) & ((1 << 63) - 1))

    def add(self, ids: np.ndarray, embeddings: np.ndarray) -> None:
        try:
            assert ids.dtype == np.int64
            assert embeddings.dtype == np.float32
            self.index.add_with_ids(embeddings, ids)
            logger.debug(
                f"Successfully added embeddings. Index now contains {self.index.ntotal} vectors."
            )
        except Exception as e:
            logger.error(f"Failed to add embeddings to Faiss index: {e}", exc_info=True)
            raise

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.debug(
                f"Faiss index saved successfully. ({self.index.ntotal} vectors)"
            )
        except Exception as e:
            logger.error(f"Failed to save Faiss index: {e}", exc_info=True)
            raise
