import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tinydb import TinyDB
from task_oriented_dataset_search.embedding.embedder import SentenceTransformerEmbedder
from task_oriented_dataset_search.embedding.indexer import FaissIndexer
from task_oriented_dataset_search.embedding.interface import BaseEmbedder

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(
        self,
        embedder: BaseEmbedder,
        task_index_path: str,
        dataset_index_path: str,
        task_parquet_path: str,
        dataset_parquet_path: str,
    ):
        logger.info("Initializing EmbeddingPipeline...")
        logger.debug(
            f"Task Index: {task_index_path}, Dataset Index: {dataset_index_path}"
        )
        logger.debug(
            f"Task Parquet: {task_parquet_path}, Dataset Parquet: {dataset_parquet_path}"
        )
        self.embedder = embedder
        self.task_indexer = FaissIndexer(embedder.dimension(), task_index_path)
        self.dataset_indexer = FaissIndexer(embedder.dimension(), dataset_index_path)
        self.task_parquet_path = task_parquet_path
        self.dataset_parquet_path = dataset_parquet_path
        logger.info("EmbeddingPipeline initialized.")

    def load_existing_ids(self, parquet_path: str) -> set:
        logger.debug(f"Loading existing IDs from: {parquet_path}")
        if not os.path.exists(parquet_path):
            return set()
        try:
            df = pd.read_parquet(parquet_path, columns=["hex_id"])
            return set(df["hex_id"])
        except Exception as e:
            logger.warning(
                f"Failed to read parquet file {parquet_path}: {e}. A null set will be returned."
            )
            return set()

    def append_to_parquet(self, parquet_path: str, new_data_df: pd.DataFrame):
        if os.path.exists(parquet_path):
            try:
                existing_df = pd.read_parquet(parquet_path)
                combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
                combined_df.to_parquet(parquet_path, index=False)
            except Exception as e:
                logger.error(f"Failed to update parquet file {parquet_path}: {e}.")
                raise
        else:
            new_data_df.to_parquet(parquet_path, index=False)

    def index_tasks(self, tasks: List[Dict]):
        logger.info(f"Starting task indexing for {len(tasks)} tasks.")
        existing_ids = self.load_existing_ids(self.task_parquet_path)
        new_ids_hex = []
        new_ids_int64 = []
        new_texts = []

        for t in tasks:
            tid = t.get("id")
            desc = t.get("task_description", "")
            if not tid or not desc:
                continue
            if tid not in existing_ids:
                new_ids_hex.append(tid)
                new_ids_int64.append(self.task_indexer.hex_to_int64(tid))
                new_texts.append(desc)

        if not new_texts:
            logger.debug("No new tasks found to index.")
            return

        embs = self.embedder.embed(new_texts)
        id_arr = np.array(new_ids_int64, dtype=np.int64)

        new_tasks_df = pd.DataFrame(
            {
                "hex_id": new_ids_hex,
                "int64_id": new_ids_int64,
                "text": new_texts,
                "embedding": list(embs),
            }
        )
        self.append_to_parquet(self.task_parquet_path, new_tasks_df)

        self.task_indexer.add(id_arr, embs)
        self.task_indexer.save()
        logger.info(f"Finished indexing {len(new_texts)} new tasks.")

    def index_datasets(self, datasets: List[Dict]):
        logger.info(f"Starting dataset indexing for {len(datasets)} datasets.")
        existing_ids = self.load_existing_ids(self.dataset_parquet_path)
        new_ids_hex = []
        new_ids_int64 = []
        new_texts = []

        for d in datasets:
            did = d.get("id")
            title = d.get("title", "")
            desc = d.get("description", "")
            if not did or not (title or desc):
                continue
            if did not in existing_ids:
                new_ids_hex.append(did)
                new_ids_int64.append(self.dataset_indexer.hex_to_int64(did))
                new_texts.append(f"{title} {desc}".strip())

        if not new_texts:
            logger.debug("No new datasets found to index.")
            return

        embs = self.embedder.embed(new_texts)
        id_arr = np.array(new_ids_int64, dtype=np.int64)
        new_datasets_df = pd.DataFrame(
            {
                "hex_id": new_ids_hex,
                "int64_id": new_ids_int64,
                "text": new_texts,
                "embedding": list(embs),
            }
        )
        self.append_to_parquet(self.dataset_parquet_path, new_datasets_df)

        self.dataset_indexer.add(id_arr, embs)
        self.dataset_indexer.save()
        logger.info(f"Finished indexing {len(new_texts)} new datasets.")

    def embed_all(
        self,
        db_path: str,
    ):
        logger.info(f"Starting embed_all process from DB: {db_path}")
        try:
            db = TinyDB(db_path)
            tasks = db.table("tasks").all()
            datasets = db.table("datasets").all()
            if tasks:
                self.index_tasks(tasks)
            else:
                logger.warning("No tasks found in DB to index.")
            if datasets:
                self.index_datasets(datasets)
            else:
                logger.warning("No datasets found in DB to index.")
        except Exception as e:
            logger.error(f"Error during embed_all process: {e}", exc_info=True)
            raise
