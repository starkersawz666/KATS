import logging
import os

import faiss
import networkx as nx
import numpy as np
import pandas as pd
from tinydb import Query, TinyDB

from task_oriented_dataset_search.graph.builder import GraphBuilder


logger = logging.getLogger(__name__)


class TaskMerger:
    def __init__(
        self,
        db_path: str,
        graph_path: str,
        graph_processed_path: str,
        task_faiss_path: str,
        task_parquet_path: str,
        strong_similarity_threshold: float = 0.8,
        keyword_overlap_threshold: float = 0.7,
        weak_similarity_threshold: float = 0.6,
        max_merge: int = 10,
    ):
        self.db_path = db_path
        self.graph_path = graph_path
        self.graph_processed_path = graph_processed_path
        self.task_faiss_path = task_faiss_path
        self.task_parquet_path = task_parquet_path
        self.strong_similarity_threshold = strong_similarity_threshold
        self.keyword_overlap_threshold = keyword_overlap_threshold
        self.weak_similarity_threshold = weak_similarity_threshold
        self.max_merge = max_merge

        self.graph_builder = GraphBuilder(db_path, graph_path)
        self.graph = self.graph_builder.get_graph()
        self.faiss_index = self._load_faiss_index()
        self.task_df, self.int64_to_hex_map, self.hex_to_keywords_map = (
            self._load_data()
        )

    def _load_faiss_index(self) -> faiss.Index:
        if not os.path.exists(self.task_faiss_path):
            raise FileNotFoundError(f"Faiss index not found: {self.task_faiss_path}")
        return faiss.read_index(self.task_faiss_path)

    def _load_data(self):
        if not os.path.exists(self.task_parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {self.task_parquet_path}")
        df = pd.read_parquet(self.task_parquet_path)
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype="float32"))

        db = TinyDB(self.db_path)
        tasks_tbl = db.table("tasks")
        hex_to_keywords = {
            task["id"]: set(task.get("keywords", [])) for task in tasks_tbl.all()
        }

        int64_to_hex = pd.Series(df.hex_id.values, index=df.int64_id).to_dict()

        return df, int64_to_hex, hex_to_keywords

    def _calculate_keyword_overlap(self, keywords1: set, keywords2: set) -> float:
        if not keywords1 or not keywords2:
            return 0.0

        common_keywords = keywords1.intersection(keywords2)
        min_len = min(len(keywords1), len(keywords2))

        return len(common_keywords) / min_len if min_len > 0 else 0.0

    def save_graph(self):
        self.graph_builder.save_graph(self.graph_processed_path)

    def get_graph(self) -> nx.Graph:
        return self.graph_builder.get_graph()

    def merge_tasks(self):
        if self.task_df.empty:
            logger.warning("No tasks found to merge.")
            return
        all_vectors = np.stack(self.task_df["embedding"].tolist())
        all_int64_ids = self.task_df["int64_id"].values

        if not self.faiss_index or self.faiss_index.ntotal == 0:
            logger.warning("Faiss index not found or empty.")
            return

        k = min(self.max_merge, self.faiss_index.ntotal)
        D, I = self.faiss_index.search(all_vectors, k)

        merged_count = 0
        added_edges = set()

        for idx, int64_id_1 in enumerate(all_int64_ids):
            hex_id_1 = self.int64_to_hex_map.get(int64_id_1)
            if not hex_id_1 or not self.graph.has_node(hex_id_1):
                continue

            keywords1 = self.hex_to_keywords_map.get(hex_id_1, set())

            for j, int64_id_2 in enumerate(I[idx]):
                similarity = D[idx, j]
                if int64_id_2 == -1 or similarity < self.weak_similarity_threshold:
                    continue

                hex_id_2 = self.int64_to_hex_map.get(int64_id_2)
                if (
                    not hex_id_2
                    or hex_id_1 == hex_id_2
                    or not self.graph.has_node(hex_id_2)
                ):
                    continue

                edge_tuple = tuple(sorted((hex_id_1, hex_id_2)))
                if edge_tuple in added_edges or self.graph.has_edge(hex_id_1, hex_id_2):
                    continue

                keywords2 = self.hex_to_keywords_map.get(hex_id_2, set())

                add_edge = False
                weight = similarity

                if similarity > self.strong_similarity_threshold:
                    add_edge = True
                    weight = 1.0
                else:
                    overlap = self._calculate_keyword_overlap(keywords1, keywords2)
                    if overlap >= self.keyword_overlap_threshold:
                        add_edge = True
                        weight = 1.0
                    else:
                        add_edge = True
                        weight = similarity

                if add_edge:
                    self.graph.add_edge(
                        hex_id_1, hex_id_2, weight=weight, type="similar_task"
                    )
                    added_edges.add(edge_tuple)
                    merged_count += 1

        logger.info(f"Merged {merged_count} tasks based on similarity.")
