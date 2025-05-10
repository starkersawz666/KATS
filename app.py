from pathlib import Path
import pandas as pd
import streamlit as st
import networkx as nx
from src.pages.baseline import BaselinePage
from src.utils.faiss import FaissService
from src.utils.graph import GraphService
from src.utils.cache_utils import load_model, get_mongo_client
from src.pages.search import SearchPage
from src.pages.importation import ImportPage
from src.pages.manage import ManagePage
from src.pages.similar_datasets import SimilarDatasetsPage
from src.pages.benchmark import BenchmarkPage
import math

# set streamlit page config as the first statement
st.set_page_config(page_title="Task-Oriented Dataset Search", layout="wide")

model = load_model()

client = get_mongo_client()
db = client["paperdatabase"]
collection_regular_paper = db["regular_papers"]
collection_dataset_paper = db["dataset_papers"]
collection_nodes_papers = db["papers"]
collection_nodes_datasets = db["datasets"]
collection_nodes_tasks = db["tasks"]

faiss_index_paths = FaissService.faiss_init()

faiss_index_datasets = FaissService.load_faiss_index(faiss_index_paths["datasets"])
faiss_index_task_descriptions = FaissService.load_faiss_index(
    faiss_index_paths["task_descriptions"]
)
faiss_index_task_keywords = FaissService.load_faiss_index(
    faiss_index_paths["task_keywords"]
)

graph_path = GraphService.graph_init()

graph = GraphService.load_graph(graph_path)


task_nodes = [str(doc["_id"]) for doc in collection_nodes_tasks.find({}, {"_id": 1})]
task_graph = graph.subgraph(task_nodes).copy()
task_graph_neglog = nx.Graph()
for u, v, data in task_graph.edges(data=True):
    weight = data.get("weight", 0)
    if weight > 0:
        task_graph_neglog.add_edge(u, v, weight=-math.log(weight))


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Your Page",
    ["Import", "Manage DB", "Search", "Similar Datasets", "Benchmark", "Baseline"],
)

# Import Page
if page == "Import":
    ImportPage.page_import(collection_regular_paper, collection_dataset_paper)

elif page == "Manage DB":
    ManagePage.page_manage(
        collection_regular_paper,
        collection_dataset_paper,
        collection_nodes_papers,
        collection_nodes_datasets,
        collection_nodes_tasks,
        model,
        faiss_index_datasets,
        faiss_index_task_descriptions,
        faiss_index_task_keywords,
        faiss_index_paths,
        graph_path,
        graph,
    )
elif page == "Search":
    SearchPage.page_search(
        faiss_index_task_descriptions,
        graph,
        task_graph_neglog,
        collection_nodes_tasks,
        collection_nodes_datasets,
        model,
    )
elif page == "Similar Datasets":
    SimilarDatasetsPage.page_similar_datasets(
        collection_nodes_papers, collection_nodes_datasets, faiss_index_datasets, graph
    )
elif page == "Benchmark":
    BenchmarkPage.page_benchmark(
        collection_nodes_datasets,
        collection_nodes_tasks,
        faiss_index_task_descriptions,
        faiss_index_datasets,
        graph,
        model,
    )
elif page == "Baseline":
    BaselinePage.page_baseline(
        collection_nodes_datasets,
        collection_nodes_tasks,
        faiss_index_task_descriptions,
        faiss_index_datasets,
        graph,
        model,
        1,
    )
