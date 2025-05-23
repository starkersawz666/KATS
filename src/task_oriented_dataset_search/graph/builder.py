import logging
import os
import networkx as nx
from tinydb import TinyDB

logger = logging.getLogger(__name__)


class GraphBuilder:

    def __init__(self, db_path: str, graph_path: str):
        self.db_path = db_path
        self.graph_path = graph_path
        self.db = TinyDB(self.db_path)
        self.graph = self._load_or_create_graph()

    def _load_or_create_graph(self) -> nx.Graph:
        if os.path.exists(self.graph_path):
            try:
                return nx.read_graphml(self.graph_path)
            except Exception as e:
                return nx.Graph()
        else:
            return nx.Graph()

    def _add_node_if_not_exists(self, node_id: str, **attrs):
        if node_id and not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **attrs)

    def get_graph(self) -> nx.Graph:
        return self.graph

    def build_graph(self):
        documents = self.db.table("documents").all()
        datasets = self.db.table("datasets").all()
        tasks = self.db.table("tasks").all()

        for doc in documents:
            doc_id = doc.get("id")
            self._add_node_if_not_exists(doc_id, type="document")

        for ds in datasets:
            ds_id = ds.get("id")
            doc_id = ds.get("document_id")
            self._add_node_if_not_exists(ds_id, type="dataset")
            if (
                doc_id
                and ds_id
                and self.graph.has_node(doc_id)
                and self.graph.has_node(ds_id)
            ):
                if not self.graph.has_edge(doc_id, ds_id):
                    self.graph.add_edge(doc_id, ds_id, type="contains_dataset")

        for task in tasks:
            task_id = task.get("id")
            ds_id = task.get("dataset_id")
            self._add_node_if_not_exists(task_id, type="task")
            if (
                ds_id
                and task_id
                and self.graph.has_node(ds_id)
                and self.graph.has_node(task_id)
            ):
                if not self.graph.has_edge(ds_id, task_id):
                    self.graph.add_edge(ds_id, task_id, type="used_for_task")

    def save_graph(self):
        try:
            os.makedirs(os.path.dirname(self.graph_path) or ".", exist_ok=True)
            nx.write_graphml(self.graph, self.graph_path)
        except Exception as e:
            logger.error(f"Failed to save graph to {self.graph_path}: {e}")
