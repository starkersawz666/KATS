import streamlit as st
import numpy as np
from bson import ObjectId
import pymongo
import faiss
import networkx as nx
from src.utils.faiss import FaissService
from collections import Counter


class SimilarDatasetsPage:

    @staticmethod
    def page_similar_datasets(
        collection_nodes_papers: pymongo.collection.Collection,
        collection_nodes_datasets: pymongo.collection.Collection,
        faiss_index_datasets: faiss.IndexIDMap,
        graph: nx.graph,
    ):
        if st.button("Get Similar Datasets"):
            similar_datasets, cnt = FaissService.get_similar_datasets(
                collection_nodes_datasets,
                faiss_index_datasets,
            )
            # print(similar_datasets)
            print(cnt)
            similar_datasets_in_paper = {}
            paper_nodes = set(map(str, collection_nodes_papers.distinct("_id")))
            for similar_pair in similar_datasets:
                # task1_id = FaissService.get_faiss_id_from_mongo_id(similar_pair[0])
                # task2_id = FaissService.get_faiss_id_from_mongo_id(similar_pair[1])
                task1_id = similar_pair[0]
                task2_id = similar_pair[1]
                for task_id in [task1_id, task2_id]:
                    neighbors = set(graph.neighbors(task_id))
                    paper_neighbors = [
                        node for node in neighbors if node in paper_nodes
                    ]
                    if len(paper_neighbors) == 1:
                        if paper_neighbors[0] in similar_datasets_in_paper:
                            similar_datasets_in_paper[paper_neighbors[0]].append(
                                task_id
                            )
                        else:
                            similar_datasets_in_paper[paper_neighbors[0]] = [task_id]
            # print(similar_datasets_in_paper)
            text_similar_datasets_in_paper = {}
            for key, value in similar_datasets_in_paper.items():
                paper_title = collection_nodes_papers.find_one({"_id": ObjectId(key)})[
                    "title"
                ]
                dataset_names = []
                for dataset_id in value:
                    dataset_title = collection_nodes_datasets.find_one(
                        {"_id": ObjectId(dataset_id)}
                    )["title"]
                    if dataset_title not in dataset_names:
                        dataset_names.append(dataset_title)
                text_similar_datasets_in_paper[paper_title] = dataset_names

            all_values = []
            for lst in text_similar_datasets_in_paper.values():
                all_values.extend(lst)

            counts = Counter(all_values)

            new_dict = {}
            for key, lst in text_similar_datasets_in_paper.items():
                new_lst = [f"{s} ({counts[s]})" for s in lst]
                new_dict[key] = new_lst

            for item in new_dict:
                st.write(item, new_dict[item])
