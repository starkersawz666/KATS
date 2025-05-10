import streamlit as st
import faiss
from src.utils.faiss import FaissService
import pymongo
import networkx as nx
import random
from bson import ObjectId
import numpy as np
import math
from src.pages.search import SearchPage
from sentence_transformers import SentenceTransformer


class BenchmarkPage:

    @staticmethod
    def benchmark(
        collection_nodes_datasets: pymongo.collection.Collection,
        collection_nodes_tasks: pymongo.collection.Collection,
        faiss_index_task_descriptions: faiss.IndexIDMap,
        faiss_index_datasets: faiss.IndexIDMap,
        graph: nx.graph,
        model: SentenceTransformer,
    ):
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
        print(f"Original Graph: {graph.number_of_nodes()} nodes")
        print(f"Benchmark Graph: {graph_copy.number_of_nodes()} nodes")

        benchmark_pairs = []
        for _ in range(100):
            candidate_nodes = []
            for node in dataset_nodes:
                if (
                    node in graph_copy
                    and len(set(graph_copy[node]) & set(task_nodes)) >= 5
                ):
                    candidate_nodes.append(node)
            if not candidate_nodes:
                print(
                    f"No more dataset nodes with 2 or more tasks; Current nodes: {_ + 1}"
                )
                break
            random.shuffle(candidate_nodes)
            second_candidate_nodes = []
            candidate_node_titles = set()
            for candidate_node in candidate_nodes:
                node_title = collection_nodes_datasets.find_one(
                    {"_id": ObjectId(candidate_node)}
                )["title"]
                if node_title not in candidate_node_titles:
                    candidate_node_titles.add(node_title)
                    second_candidate_nodes.append(candidate_node)
            random_dataset_node = random.choice(second_candidate_nodes)
            node_document = collection_nodes_datasets.find_one(
                {"_id": ObjectId(random_dataset_node)}
            )
            print(node_document["title"])
            connected_task_nodes = list(
                set(graph_copy[random_dataset_node]) & set(task_nodes)
            )
            selected_task_node = random.choice(connected_task_nodes)
            selected_task_node_objid = ObjectId(selected_task_node)
            task_info = collection_nodes_tasks.find_one(
                {"_id": selected_task_node_objid}, {"task_description": 1}
            )
            if task_info and "task_description" in task_info:
                task_description = task_info["task_description"]
                benchmark_pairs.append(
                    (task_description, random_dataset_node, selected_task_node)
                )
                graph_copy.remove_node(selected_task_node)
                faiss_id = FaissService.get_faiss_id_from_mongo_id(selected_task_node)
                faiss_index_copy.remove_ids(np.array([faiss_id], dtype=np.int64))

        # print(f"{len(benchmark_pairs)} pairs of (task_description, dataset_node)")

        benchmark_task_nodes_raw = [
            str(doc["_id"]) for doc in collection_nodes_tasks.find({}, {"_id": 1})
        ]
        benchmark_task_nodes = [
            id for id in benchmark_task_nodes_raw if id in graph_copy
        ]
        benchmark_task_graph = graph_copy.subgraph(benchmark_task_nodes).copy()
        benchmark_task_graph_neglog = nx.Graph()
        for u, v, data in benchmark_task_graph.edges(data=True):
            weight = data.get("weight", 0)
            if weight > 0:
                benchmark_task_graph_neglog.add_edge(u, v, weight=-math.log(weight))
        num_pairs = len(benchmark_pairs)
        cnt_pass_results = 0

        deleted_tasks = [item[2] for item in benchmark_pairs]

        for benchmark_pair in benchmark_pairs:
            task_des = benchmark_pair[0]
            baseline = benchmark_pair[1]
            results = SearchPage.search_pagerank(
                task_des,
                faiss_index_copy,
                graph_copy,
                benchmark_task_graph,
                collection_nodes_tasks,
                collection_nodes_datasets,
                model,
                top_k=9,
            )
            results_ids = [str(result["_id"]) for result in results]
            # print("Results:", results_ids)
            if baseline in results_ids:
                cnt_pass_results += 1
        print(f"Accuracy: {cnt_pass_results / num_pairs}")

    @staticmethod
    def page_benchmark(
        collection_nodes_datasets: pymongo.collection.Collection,
        collection_nodes_tasks: pymongo.collection.Collection,
        faiss_index_task_descriptions: faiss.IndexIDMap,
        faiss_index_datasets: faiss.IndexIDMap,
        graph: nx.graph,
        model: SentenceTransformer,
    ):
        if st.button("Run Benchmark"):
            BenchmarkPage.benchmark(
                collection_nodes_datasets,
                collection_nodes_tasks,
                faiss_index_task_descriptions,
                faiss_index_datasets,
                graph,
                model,
            )
