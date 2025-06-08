from collections import defaultdict
import logging
import os
from typing import Dict, Set
import networkx as nx

from tinydb import Query, TinyDB


logger = logging.getLogger(__name__)


class DatasetStatistics:
    def __init__(self, db_path: str, graph_processed_path: str):
        self.db_path = db_path
        self.graph_processed_path = graph_processed_path

        if not os.path.exists(db_path):
            logger.error(f"TinyDB database not found at {db_path}")
            raise FileNotFoundError(f"TinyDB database not found at {db_path}")
        self.db = TinyDB(self.db_path)
        self.datasets_tbl = self.db.table("datasets")
        self.DatasetQ = Query()

        self.graph = self._load_graph()

    def _load_graph(self) -> nx.Graph | None:
        if not os.path.exists(self.graph_processed_path):
            logger.warning(
                f"Processed graph not found at {self.graph_processed_path}. "
                "Counts will not reflect dataset merges."
            )
            return None
        try:
            graph = nx.read_graphml(self.graph_processed_path)
            for node_id, data in graph.nodes(data=True):
                if "aliases" in data and isinstance(data["aliases"], str):
                    data["aliases"] = [
                        alias.strip()
                        for alias in data["aliases"].split(",")
                        if alias.strip()
                    ]
            logger.info(
                f"Successfully loaded processed graph from {self.graph_processed_path}"
            )
            return graph
        except Exception as e:
            logger.error(f"Failed to load graph: {e}. Counts will not reflect merges.")
            return None

    def _build_representative_map(self) -> Dict[str, str]:
        id_to_rep: Dict[str, str] = {}
        if not self.graph:
            return id_to_rep

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "dataset":
                aliases = data.get("aliases")
                if aliases and isinstance(aliases, list):
                    id_to_rep[node_id] = node_id
                    for alias in aliases:
                        id_to_rep[alias] = node_id
                elif node_id not in id_to_rep:
                    id_to_rep[node_id] = node_id

        logger.debug(f"Built representative map with {len(id_to_rep)} entries.")
        return id_to_rep

    def calculate_dataset_document_counts(self) -> Dict[str, Dict[str, int]]:
        logger.info("Calculating dataset document counts...")
        all_datasets = self.datasets_tbl.all()

        id_to_rep_map = self._build_representative_map()

        rep_to_docs: Dict[str, Set[str]] = defaultdict(set)
        ds_id_to_title: Dict[str, str] = {}
        doc_to_ds_ids: Dict[str, Set[str]] = defaultdict(set)

        for ds in all_datasets:
            ds_id = ds.get("id")
            doc_id = ds.get("document_id")
            title = ds.get("title", "N/A")

            if not ds_id or not doc_id:
                continue

            ds_id_to_title[ds_id] = title
            doc_to_ds_ids[doc_id].add(ds_id)

            rep_id = id_to_rep_map.get(ds_id, ds_id)
            rep_to_docs[rep_id].add(doc_id)

        rep_counts = {rep_id: len(docs) for rep_id, docs in rep_to_docs.items()}
        logger.debug(
            f"Calculated counts for {len(rep_counts)} representative datasets."
        )

        doc_stats: Dict[str, Dict[str, int]] = defaultdict(dict)
        for doc_id, dataset_ids in doc_to_ds_ids.items():
            for ds_id in dataset_ids:
                rep_id = id_to_rep_map.get(ds_id, ds_id)
                count = rep_counts.get(rep_id, 1)
                title = ds_id_to_title.get(ds_id, "N/A")
                doc_stats[doc_id][title] = count

        logger.info("Finished calculating dataset document counts.")
        return dict(doc_stats)

    def get_document_dataset_counts(self) -> Dict[str, Dict[str, int]]:
        return self.calculate_dataset_document_counts()
