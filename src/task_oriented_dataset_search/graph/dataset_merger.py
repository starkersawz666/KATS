import json
import logging
import os
import re
import time
from typing import Dict, List, Tuple

from disjoint_set import DisjointSet
import faiss
import numpy as np
import pandas as pd
import networkx as nx
from tinydb import Query, TinyDB

from task_oriented_dataset_search.extraction.client import BaseLLMClient
from task_oriented_dataset_search.utils.cache import CacheManager


logger = logging.getLogger(__name__)


class DatasetMerger:
    def __init__(
        self,
        db_path: str,
        graph_path: str,
        graph_processed_path: str,
        dataset_faiss_path: str,
        dataset_parquet_path: str,
        llm_client: BaseLLMClient,
        cache_manager: CacheManager,
        similarity_threshold: float = 0.7,
        k_neighbors: int = 10,
        llm_retries: int = 3,
        llm_retry_delay: float = 10.0,
        alias_dict_name: str = "dataset_alias.json",
    ):
        self.db_path = db_path
        self.graph_path = graph_path
        self.graph_processed_path = graph_processed_path
        self.dataset_faiss_path = dataset_faiss_path
        self.dataset_parquet_path = dataset_parquet_path
        self.llm_client = llm_client
        self.cache = cache_manager
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.llm_retries = llm_retries
        self.llm_retry_delay = llm_retry_delay
        self.alias_dict_name = alias_dict_name

        logger.info("Initializing DatasetMerger...")
        logger.debug(
            f"Params: SimThreshold={similarity_threshold}, K={k_neighbors}, LLMRetries={llm_retries}"
        )

        self.db = TinyDB(self.db_path)
        self.datasets_tbl = self.db.table("datasets")
        self.DatasetQ = Query()

        self.graph = self._load_graph()
        self.faiss_index = self._load_faiss_index()
        self.dataset_df, self.int64_to_hex_map, self.dataset_details = self._load_data()
        self.alias_dict = self._load_alias_dict()
        self._prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        self.dataset_ids = list(self.dataset_details.keys())
        self.dsu = DisjointSet()
        for item in self.dataset_ids:
            self.dsu.find(item)

        logger.info(
            f"DatasetMerger initialized. Found {len(self.dataset_ids)} datasets."
        )

    def _load_faiss_index(self) -> faiss.Index:
        return faiss.read_index(self.dataset_faiss_path)

    def _load_data(self) -> Tuple[pd.DataFrame, Dict[int, str], Dict[str, Dict]]:
        df = pd.read_parquet(self.dataset_parquet_path)
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype="float32"))
        int64_to_hex = pd.Series(df.hex_id.values, index=df.int64_id).to_dict()
        all_datasets = self.datasets_tbl.all()
        dataset_details = {
            ds["id"]: {
                "title": ds.get("title", ""),
                "description": ds.get("description", ""),
                "link": ds.get("link", "None"),
            }
            for ds in all_datasets
        }
        return df, int64_to_hex, dataset_details

    def _load_graph(self) -> nx.Graph:
        logger.info(f"Loading graph for dataset merging from: {self.graph_path}")
        return nx.read_graphml(self.graph_path)

    def _load_alias_dict(self) -> Dict[str, str]:
        logger.debug(f"Loading alias dictionary ({self.alias_dict_name})...")

        def loader(path: str) -> Dict:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {}

        path = self.cache.get_cache_path(
            "aliases", "dataset_alias_fingerprint", ".json"
        )
        data = self.cache.load_if_exists(
            "aliases", "dataset_alias_fingerprint", loader, ".json"
        )

        return data if data is not None else {}

    def _save_alias_dict(self) -> None:
        logger.debug(f"Saving alias dictionary with {len(self.alias_dict)} entries...")

        def saver(path: str, data: Dict):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        self.cache.save(
            "aliases", "dataset_alias_fingerprint", self.alias_dict, saver, ".json"
        )

    def _read_prompt(self, relpath: str) -> str:
        path = os.path.join(self._prompts_dir, os.path.basename(relpath))
        logger.debug(f"Reading prompt file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _normalize_title(self, title: str) -> str:
        text = title.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"\b(a|the|an|dataset|corpus|data)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        current = text
        while current in self.alias_dict and self.alias_dict[current] != current:
            current = self.alias_dict[current]
        return current

    def _call_llm_judge(self, ds1_id: str, ds2_id: str) -> bool:
        logger.debug(f"Calling LLM to judge if {ds1_id} and {ds2_id} are the same.")
        ds1 = self.dataset_details[ds1_id]
        ds2 = self.dataset_details[ds2_id]
        prompt_template = self._read_prompt("dataset_identity_prompt.txt")

        prompt = prompt_template.format(
            title1=ds1["title"],
            description1=ds1["description"],
            link1=ds1["link"],
            title2=ds2["title"],
            description2=ds2["description"],
            link2=ds2["link"],
        )

        messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.llm_retries):
            try:
                response = self.llm_client.chat(messages)  #
                content = response.choices[0].message.content  #
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    json_response = json.loads(match.group(0))
                    is_same = json_response.get("is_same", False)
                    return is_same

            except Exception as e:
                logger.warning(
                    f"LLM call failed (attempt {attempt+1}/{self.llm_retries}) for {ds1_id}/{ds2_id}: {e}"
                )
                if attempt < self.llm_retries - 1:
                    time.sleep(self.llm_retry_delay)

        logger.error(
            f"LLM call failed after {self.llm_retries} retries for {ds1_id}/{ds2_id}."
        )
        return False

    def _generate_candidates(self) -> List[Tuple[str, str, float]]:
        logger.info("Generating dataset merge candidates using Faiss.")
        if (
            self.dataset_df.empty
            or not self.faiss_index
            or self.faiss_index.ntotal == 0
        ):
            logger.warning(
                "Dataset data or Faiss index is empty. No candidates generated."
            )
            return []

        all_vectors = np.stack(self.dataset_df["embedding"].tolist())
        all_int64_ids = self.dataset_df["int64_id"].values

        k = min(self.k_neighbors, self.faiss_index.ntotal)
        D, I = self.faiss_index.search(all_vectors, k)  #
        candidates = []
        seen_pairs = set()

        for idx, int64_id_1 in enumerate(all_int64_ids):
            hex_id_1 = self.int64_to_hex_map.get(int64_id_1)
            if not hex_id_1:
                continue
            for j, int64_id_2 in enumerate(I[idx]):
                similarity = D[idx, j]
                if int64_id_2 == -1 or similarity < self.similarity_threshold:
                    continue
                hex_id_2 = self.int64_to_hex_map.get(int64_id_2)
                if not hex_id_2 or hex_id_1 == hex_id_2:
                    continue
                pair = tuple(sorted((hex_id_1, hex_id_2)))
                if pair not in seen_pairs:
                    candidates.append((hex_id_1, hex_id_2, similarity))
                    seen_pairs.add(pair)
        logger.info(f"Generated {len(candidates)} dataset merge candidates.")
        return candidates

    def merge_datasets(self):
        logger.info("Starting dataset merging process...")
        candidates = self._generate_candidates()

        for ds1_id, ds2_id, similarity in candidates:
            if self.dsu.connected(ds1_id, ds2_id):
                continue

            norm_title1 = self._normalize_title(self.dataset_details[ds1_id]["title"])
            norm_title2 = self._normalize_title(self.dataset_details[ds2_id]["title"])

            if norm_title1 == norm_title2 and norm_title1:
                self.dsu.union(ds1_id, ds2_id)
                continue

            if self._call_llm_judge(ds1_id, ds2_id):
                self.dsu.union(ds1_id, ds2_id)
                root = self.dsu.find(ds1_id)
                self.alias_dict[norm_title1] = root
                self.alias_dict[norm_title2] = root

        logger.info(f"Dataset merging check finished.")
        self._save_alias_dict()
        self._merge_graph_nodes()

    def _merge_graph_nodes(self):
        logger.info("Merging dataset nodes in the graph based on DSU sets.")
        groups = list(self.dsu.itersets())
        merged_count = 0

        for group_set in groups:
            group = list(group_set)
            if len(group) > 1:
                representative = min(group)
                aliases = [node_id for node_id in group if node_id != representative]
                if not self.graph.has_node(representative):
                    logger.warning(
                        f"Representative node {representative} not found in graph. Skipping merge for this group."
                    )
                    continue

                for alias_node in aliases:
                    if self.graph.has_node(alias_node):
                        for neighbor in list(self.graph.neighbors(alias_node)):
                            if neighbor != representative:
                                edge_data = self.graph.get_edge_data(
                                    alias_node, neighbor
                                )
                                if not self.graph.has_edge(representative, neighbor):
                                    self.graph.add_edge(
                                        representative, neighbor, **edge_data
                                    )

                        if "aliases" not in self.graph.nodes[representative]:
                            self.graph.nodes[representative]["aliases"] = []
                        self.graph.nodes[representative]["aliases"].append(alias_node)

                        self.graph.remove_node(alias_node)
                        merged_count += 1
                    else:
                        logger.warning(f"Alias node {alias_node} not found in graph.")
        logger.info(f"Merged {merged_count} dataset nodes in graph.")

    def save_graph(self):
        logger.info(f"Saving merged dataset graph to {self.graph_processed_path}...")
        graph_to_save = self.graph.copy()
        for node_id, data in graph_to_save.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = ",".join(map(str, value))

        for u, v, data in graph_to_save.edges(data=True):
            for key, value in list(data.items()):
                if isinstance(value, list):
                    data[key] = ",".join(map(str, value))

        os.makedirs(os.path.dirname(self.graph_processed_path) or ".", exist_ok=True)
        nx.write_graphml(graph_to_save, self.graph_processed_path)
        logger.info("Merged dataset graph saved successfully.")

    def get_graph(self) -> nx.Graph:
        return self.graph
