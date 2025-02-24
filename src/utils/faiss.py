import faiss
from bson import ObjectId
import numpy as np


class FaissService:

    def __init__():
        pass

    @staticmethod
    def task_similarity(
        self,
        faiss_index: faiss.IndexIDMap,
        task1: str,
        task2: str,
        task_similarity_threshold: float = 0.7,
        keyword_overlap_threshold: float = 0.7,
        weak_similarity_threshold: float = 0.4,
    ) -> float:
        pass

    @staticmethod
    def get_faiss_id_from_mongo_id(mongo_id: str | ObjectId) -> int:
        return (
            int(str(mongo_id)[-8:], 16)
            if isinstance(mongo_id, ObjectId)
            else int(mongo_id[-8:], 16)
        )

    @staticmethod
    def get_dimension() -> int:
        return 384

    @staticmethod
    def get_vectors_by_ids(
        index: faiss.IndexIDMap, faiss_ids: np.array
    ) -> np.vstack | None:
        stored_ids = np.array([index.id_map.at(i) for i in range(index.ntotal)])
        valid_ids = [faiss_id for faiss_id in faiss_ids if faiss_id in stored_ids]
        positions = [np.where(stored_ids == faiss_id)[0][0] for faiss_id in valid_ids]
        vectors = [index.index.reconstruct(int(pos)) for pos in positions]
        return np.vstack(vectors) if vectors else None
