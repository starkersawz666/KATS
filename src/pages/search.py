import streamlit as st
import math
import networkx as nx
from src.utils.bert import BertService
from src.utils.faiss import FaissService
import faiss
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId
from sentence_transformers import SentenceTransformer


class SearchPage:

    @staticmethod
    def perform_search(
        task_description: str,
        faiss_index: faiss.IndexIDMap,
        graph: nx.Graph,
        graph_tasks: nx.Graph,
        collection_nodes_tasks: pymongo.collection.Collection,
        collection_nodes_datasets: pymongo.collection.Collection,
        model: SentenceTransformer,
        top_k: int = 5,
    ) -> list:
        query_vector = BertService.text_to_vector(model, task_description)
        _, indices = faiss_index.search(query_vector.reshape(1, -1), 1)
        closest_faiss_id = indices[0][0]
        matched_task_nodes = []
        for task in collection_nodes_tasks.find():
            task_id_hex = int(str(task["_id"])[-8:], 16)
            if task_id_hex == closest_faiss_id:
                bert_vector = BertService.text_to_vector(
                    model, task.get("task_description", "")
                ).reshape(1, -1)
                faiss_vector = FaissService.get_vectors_by_ids(
                    faiss_index, [task_id_hex]
                )[0].reshape(1, -1)
                similarity = cosine_similarity(bert_vector, faiss_vector)
                if similarity > 0.95:
                    matched_task_nodes.append(str(task["_id"]))
        related_task_nodes = set()
        for task_node in matched_task_nodes:
            related_task_nodes.update([task_node])
            if top_k > 0:
                if task_node in graph_tasks:
                    paths = nx.single_source_dijkstra_path_length(
                        graph_tasks, task_node
                    )
                    sorted_paths = sorted(
                        paths.items(), key=lambda x: math.exp(-x[1]), reverse=True
                    )
                    related_task_nodes.update(
                        [node for node, weight in sorted_paths[:top_k]]
                    )
        related_datasets = set()
        dataset_nodes = set(map(str, collection_nodes_datasets.distinct("_id")))
        for task_node in related_task_nodes:
            for neighbor in graph.neighbors(task_node):
                if neighbor in dataset_nodes:
                    related_datasets.add(neighbor)
        dataset_results = list(
            collection_nodes_datasets.find(
                {"_id": {"$in": [ObjectId(d) for d in related_datasets]}}
            )
        )
        return dataset_results

    @staticmethod
    def page_search(
        faiss_index_task_descriptions: faiss.IndexIDMap,
        graph: nx.Graph,
        graph_tasks: nx.Graph,
        collection_node_tasks: pymongo.collection.Collection,
        collection_node_datasets: pymongo.collection.Collection,
        model: SentenceTransformer,
    ):
        st.title("Search for Datasets")
        dataset_query = st.text_input("Enter your search query:", "")
        if st.button("Search"):
            if dataset_query.strip():
                st.session_state["search_results"] = SearchPage.perform_search(
                    dataset_query,
                    faiss_index_task_descriptions,
                    graph,
                    graph_tasks,
                    collection_node_tasks,
                    collection_node_datasets,
                    model,
                )
                st.write(st.session_state["search_results"])
            else:
                st.warning("Please enter a search query.")
