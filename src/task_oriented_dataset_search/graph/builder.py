import logging
import os
import networkx as nx
from tinydb import Query, TinyDB

logger = logging.getLogger(__name__)


class GraphBuilder:

    def __init__(self, db_path: str, graph_path: str, save_path: str | None = None):
        self.db_path = db_path
        self.graph_path = graph_path
        self.save_path = save_path or graph_path
        logger.info(
            f"Initializing GraphBuilder. DB: {db_path}, Graph Path: {graph_path}, Save Path: {self.save_path}"
        )
        self.db = TinyDB(self.db_path)
        self.graph = self._load_or_create_graph()

    def _load_or_create_graph(self) -> nx.Graph:
        if os.path.exists(self.graph_path):
            try:
                graph = nx.read_graphml(self.graph_path)
                logger.info(
                    f"Successfully loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
                )
                return graph
            except Exception as e:
                logger.error(
                    f"Failed to load graph from {self.graph_path}: {e}. Creating a new graph.",
                    exc_info=True,
                )
                return nx.Graph()
        else:
            logger.info(
                f"Graph file not found at {self.graph_path}. Creating a new graph."
            )
            return nx.Graph()

    def _add_node_if_not_exists(self, node_id: str, **attrs):
        if node_id and not self.graph.has_node(node_id):
            logger.debug(f"Adding node: {node_id} with attrs: {attrs}")
            self.graph.add_node(node_id, **attrs)

    def get_graph(self) -> nx.Graph:
        return self.graph

    def build_basic_graph(self):
        logger.info("Starting to build the basic graph...")
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
        logger.info(f"Finished building basic graph.")

    def save_graph(self, save_path: str = None):
        graph_save_path = save_path or self.save_path
        logger.info(f"Saving graph to: {graph_save_path}...")
        try:
            os.makedirs(os.path.dirname(graph_save_path) or ".", exist_ok=True)
            nx.write_graphml(self.graph, graph_save_path)
            logger.info(f"Graph saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save graph to {graph_save_path}: {e}")
            raise

    def build_and_save_task_similarity_graph(self):
        logger.info("Starting to build task similarity graph...")
        task_sim_graph = nx.Graph()
        task_node_ids = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("type") == "task":
                task_sim_graph.add_node(node_id, **attrs)
                task_node_ids.append(node_id)
        logger.info(f"Found {len(task_node_ids)} task nodes.")

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
        logger.info("Finished building and saving task similarity graph.")

    def update_basic_graph(self, new_document_fingerprints: list[str]):
        logger.info(f"Incrementally updating basic graph with {len(new_document_fingerprints)} new documents...")
        if not new_document_fingerprints:
            logger.info("No new documents to add to the graph. Skipping.")
            return

        # The graph is already loaded in the constructor.
        # Query for new entities related to the new documents.
        documents_tbl = self.db.table("documents")
        datasets_tbl = self.db.table("datasets")
        tasks_tbl = self.db.table("tasks")
        DocQ, DatasetQ, TaskQ = Query(), Query(), Query()

        new_docs = documents_tbl.search(DocQ.id.one_of(new_document_fingerprints))
        new_datasets = datasets_tbl.search(DatasetQ.document_id.one_of(new_document_fingerprints))
        new_dataset_ids = [ds.get("id") for ds in new_datasets if ds.get("id")]
        new_tasks = tasks_tbl.search(TaskQ.dataset_id.one_of(new_dataset_ids))

        logger.info(f"Found {len(new_docs)} new docs, {len(new_datasets)} new datasets, {len(new_tasks)} new tasks.")

        # Add new nodes and edges
        for doc in new_docs:
            doc_id = doc.get("id")
            self._add_node_if_not_exists(doc_id, type="document")

        for ds in new_datasets:
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

        for task in new_tasks:
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
        
        logger.info("Basic graph updated with new entities.")