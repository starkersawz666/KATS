import logging
import os
import networkx as nx
from tinydb import TinyDB

logger = logging.getLogger(__name__)


class GraphBuilder:

    def __init__(self, db_path: str, graph_path: str, save_path: str | None = None):
        self.db_path = db_path
        self.graph_path = graph_path
        self.save_path = save_path or graph_path
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

    def build_basic_graph(self):
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

    def save_graph(self, save_path: str = None):
        graph_save_path = save_path or self.save_path
        try:
            os.makedirs(os.path.dirname(graph_save_path) or ".", exist_ok=True)
            nx.write_graphml(self.graph, graph_save_path)
        except Exception as e:
            logger.error(f"Failed to save graph to {graph_save_path}: {e}")

    def build_and_save_task_similarity_graph(self):
        task_sim_graph = nx.Graph()
        task_node_ids = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "task":
                task_sim_graph.add_node(node_id, **attrs)
                task_node_ids.append(node_id)

        for u, v, attrs in self.graph.edges(data=True):
            if (
                u in task_node_ids
                and v in task_node_ids
                and attrs.get("type") == "similar_task"
            ):
                weight = attrs.get("weight", 0.0)
                if weight > 0:
                    task_sim_graph.add_edge(u, v, **attrs)

        self.graph = task_sim_graph
        self.save_graph()
