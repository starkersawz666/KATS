import logging
import os

import faiss
import numpy as np
import networkx as nx
import pandas as pd
from tinydb import Query, TinyDB
from sklearn.metrics.pairwise import cosine_similarity
from task_oriented_dataset_search.embedding.interface import BaseEmbedder


logger = logging.getLogger(__name__)


class Searcher:
    def __init__(
        self,
        embedder: BaseEmbedder,
        db_path: str,
        faiss_tasks_index_path: str,
        task_parquet_path: str,
        graph_processed_path: str,
        graph_tasks_path: str,
    ):
        self.embedder = embedder
        if not os.path.exists(db_path):
            logger.error(f"TinyDB database not found at {db_path}")
            raise FileNotFoundError(f"TinyDB database not found at {db_path}")

        self.db = TinyDB(db_path)
        self.tasks_tbl = self.db.table("tasks")
        self.datasets_tbl = self.db.table("datasets")
        self.TaskQ = Query()
        self.DatasetQ = Query()

        self.task_faiss_index = self._load_faiss_index(faiss_tasks_index_path)
        self.task_data_df, self.int64_to_hex_map_tasks = self._load_task_data(
            task_parquet_path
        )

        main_graph_path = graph_processed_path
        if not os.path.exists(main_graph_path):
            logger.error(f"Main processed graph not found at {main_graph_path}")
            raise FileNotFoundError(
                f"Main processed graph not found at {main_graph_path}"
            )
        self.main_graph = self._load_graph(main_graph_path)
        self.task_similarity_graph = self._load_graph(graph_tasks_path)

    def _load_faiss_index(self, path: str) -> faiss.Index:
        if not os.path.exists(path):
            logger.error(f"FAISS index file not found at {path}")
            raise FileNotFoundError(f"FAISS index file not found at {path}")
        return faiss.read_index(path)

    def _load_task_data(self, path: str) -> tuple[pd.DataFrame, dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task Parquet file not found at {path}")

        df = pd.read_parquet(path)
        if not {"hex_id", "int64_id", "embedding"}.issubset(df.columns):
            logger.error(
                "Task Parquet file is missing required columns (hex_id, int64_id, embedding)"
            )
            raise ValueError(
                "Task Parquet file is missing required columns (hex_id, int64_id, embedding)"
            )

        df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype="float32"))

        int64_to_hex_map = pd.Series(df.hex_id.values, index=df.int64_id).to_dict()

        return df.set_index("hex_id"), int64_to_hex_map

    def _load_graph(self, path: str) -> nx.Graph:
        return nx.read_graphml(path)

    def search(
        self,
        task_description: str,
        top_k_datasets: int = 5,
        initial_faiss_k: int = 2,
        similarity_threshold: float = 0.85,
        min_seed_similarity: float = 0.6,
        pagerank_alpha: float = 0.85,
    ) -> list:
        query_vector = self.embedder.embed([task_description])[0].reshape(1, -1)
        actual_k_for_faiss = min(initial_faiss_k, self.task_faiss_index.ntotal)
        faiss_scores, faiss_int64_ids_flat = self.task_faiss_index.search(
            query_vector, actual_k_for_faiss
        )
        faiss_int64_ids = faiss_int64_ids_flat[0]
        seed_task_hex_ids = []
        candidate_similarities = []

        for i, int_id in enumerate(faiss_int64_ids):
            if int_id == -1:
                continue
            hex_id = self.int64_to_hex_map_tasks.get(int_id)
            if hex_id and hex_id in self.task_data_df.index:
                task_embedding = self.task_data_df.loc[hex_id, "embedding"].reshape(
                    1, -1
                )
                sim = cosine_similarity(query_vector, task_embedding)[0][0]
                candidate_similarities.append((hex_id, sim))
                if sim >= similarity_threshold:
                    seed_task_hex_ids.append(hex_id)

        if not seed_task_hex_ids and candidate_similarities:
            candidate_similarities.sort(key=lambda x: x[1], reverse=True)
            for hex_id, sim in candidate_similarities:
                if sim >= min_seed_similarity:
                    seed_task_hex_ids.append(hex_id)
                if len(seed_task_hex_ids) >= max(1, initial_faiss_k // 3):
                    break

        pagerank_scores = {}
        if (
            self.task_similarity_graph
            and self.task_similarity_graph.number_of_nodes() > 0
        ):
            personalization_vector = None
            if seed_task_hex_ids:
                valid_seed_nodes = [
                    node
                    for node in seed_task_hex_ids
                    if node in self.task_similarity_graph
                ]
                if valid_seed_nodes:
                    personalization_vector = {
                        node: 1.0 if node in valid_seed_nodes else 0.001
                        for node in self.task_similarity_graph.nodes()
                    }

            try:
                pagerank_scores = nx.pagerank(
                    self.task_similarity_graph,
                    alpha=pagerank_alpha,
                    personalization=personalization_vector,
                    weight="weight",
                )
            except nx.PowerIterationFailedConvergence as e:
                logger.warning(f"PageRank not converged: {e}")
                pagerank_scores = e.args[0]
            except Exception as e:
                logger.error(f"PageRank failed: {e}")
        else:
            logger.error("Pagerank skipped: No task similarity graph found.")

        num_pagerank_candidates_to_consider = top_k_datasets * 3
        related_task_hex_ids = {
            node_id
            for node_id, score in sorted(
                pagerank_scores.items(), key=lambda x: x[1], reverse=True
            )[:num_pagerank_candidates_to_consider]
        }
        related_task_hex_ids.update(seed_task_hex_ids)

        if not related_task_hex_ids:
            logger.warning("No related tasks found.")
            return []

        related_dataset_hex_ids = set()
        for task_hex_id in related_task_hex_ids:
            if self.main_graph.has_node(task_hex_id):
                for neighbor_hex_id in self.main_graph.neighbors(task_hex_id):
                    if self.main_graph.has_node(neighbor_hex_id):
                        node_attrs = self.main_graph.nodes[neighbor_hex_id]
                        if node_attrs.get("type") == "dataset":
                            related_dataset_hex_ids.add(neighbor_hex_id)

        if not related_dataset_hex_ids:
            logger.warning("No related datasets found.")
            return []

        dataset_aggregated_scores = {ds_id: 0.0 for ds_id in related_dataset_hex_ids}
        for task_hex_id in related_task_hex_ids:
            if self.main_graph.has_node(task_hex_id):
                task_score = pagerank_scores.get(task_hex_id, 0.0)
                for neighbor_hex_id in self.main_graph.neighbors(task_hex_id):
                    if neighbor_hex_id in dataset_aggregated_scores:
                        dataset_aggregated_scores[neighbor_hex_id] += 0.1 + task_score

        sorted_dataset_ids_by_score = sorted(
            dataset_aggregated_scores.keys(),
            key=lambda ds_id: dataset_aggregated_scores[ds_id],
            reverse=True,
        )
        final_dataset_ids_to_fetch = sorted_dataset_ids_by_score[:top_k_datasets]
        if not final_dataset_ids_to_fetch:
            logger.warning("No final datasets found.")
            return []
        final_dataset_docs = self.datasets_tbl.search(
            self.DatasetQ.id.one_of(list(final_dataset_ids_to_fetch))
        )
        ordered_final_docs = sorted(
            final_dataset_docs,
            key=lambda doc: final_dataset_ids_to_fetch.index(doc["id"]),
        )
        return ordered_final_docs
