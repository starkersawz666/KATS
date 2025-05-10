from pathlib import Path
import pandas as pd
import pymongo
import streamlit as st
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer

from src.pages.search import SearchPage
from src.utils.faiss import FaissService


class BaselinePage:

    @staticmethod
    def baseline(
        collection_nodes_datasets: pymongo.collection.Collection,
        collection_nodes_tasks: pymongo.collection.Collection,
        faiss_index_task_descriptions: faiss.IndexIDMap,
        faiss_index_datasets: faiss.IndexIDMap,
        graph: nx.graph,
        model: SentenceTransformer,
        top_k: int,
    ):
        csv_path = Path("./test") / "baseline.csv"
        test_csv = pd.read_csv(csv_path)

        dataset_nodes = [
            str(doc["_id"]) for doc in collection_nodes_datasets.find({}, {"_id": 1})
        ]
        task_nodes = [
            str(doc["_id"]) for doc in collection_nodes_tasks.find({}, {"_id": 1})
        ]
        similar_pairs, num_similar = FaissService.get_similar_datasets(
            collection_nodes_datasets,
            faiss_index_datasets,
        )
        graph_copy = graph.copy()
        faiss_index_copy = faiss.clone_index(faiss_index_task_descriptions)
        for u, v in similar_pairs:
            if u not in graph_copy or v not in graph_copy:
                continue
            neighbors_v = list(graph_copy.neighbors(v))
            for neighbor in neighbors_v:
                if neighbor == u:
                    continue
                weight_vn = graph_copy[v][neighbor].get("weight", 1)
                if graph_copy.has_edge(u, neighbor):
                    weight_un = graph_copy[u][neighbor].get("weight", 1)
                    graph_copy[u][neighbor]["weight"] = max(weight_un, weight_vn)
                else:
                    graph_copy.add_edge(u, neighbor, weight=weight_vn)
            graph_copy.remove_node(v)

        benchmark_task_nodes_raw = [
            str(doc["_id"]) for doc in collection_nodes_tasks.find({}, {"_id": 1})
        ]
        benchmark_task_nodes = [
            id for id in benchmark_task_nodes_raw if id in graph_copy
        ]
        benchmark_task_graph = graph_copy.subgraph(benchmark_task_nodes).copy()

        cnt = 0
        for idx, row in test_csv.iterrows():
            query = row["query"]
            title = row["title"]
            link = row["link"]
            top_k_search = top_k - 1
            while True:
                results = SearchPage.search_pagerank(
                    query,
                    faiss_index_copy,
                    graph_copy,
                    benchmark_task_graph,
                    collection_nodes_tasks,
                    collection_nodes_datasets,
                    model,
                    top_k_search,
                )
                if len(results) < top_k:
                    top_k_search += 1
                else:
                    break
            results_titles = [str(result["title"]) for result in results]
            results_links = [str(result["link"]) for result in results]
            for i in range(len(results_titles)):
                if results_titles[i] == title or (
                    results_links[i] == link and results_links[i] is not None
                ):
                    cnt += 1

        print(f"Accuracy for top_k = {top_k}: {cnt / len(test_csv)}")

    @staticmethod
    def page_baseline(
        collection_nodes_datasets: pymongo.collection.Collection,
        collection_nodes_tasks: pymongo.collection.Collection,
        faiss_index_task_descriptions: faiss.IndexIDMap,
        faiss_index_datasets: faiss.IndexIDMap,
        graph: nx.graph,
        model: SentenceTransformer,
        top_k: int,
    ):
        if st.button("Run Baseline"):
            BaselinePage.baseline(
                collection_nodes_datasets,
                collection_nodes_tasks,
                faiss_index_task_descriptions,
                faiss_index_datasets,
                graph,
                model,
                top_k,
            )
