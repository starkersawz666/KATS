from .faiss import FaissService
import numpy as np
from bson import ObjectId
from sentence_transformers import SentenceTransformer
import networkx as nx
import pymongo
import faiss


class PaperProcessor:

    @staticmethod
    def process_regular_paper(
        bert_model: SentenceTransformer,
        paper_json: dict,
        collection_nodes_papers: pymongo.collection.Collection,
        collection_nodes_datasets: pymongo.collection.Collection,
        collection_nodes_tasks: pymongo.collection.Collection,
        faiss_index_datasets: faiss.IndexIDMap,
        faiss_index_task_descriptions: faiss.IndexIDMap,
        faiss_index_task_keywords: faiss.IndexIDMap,
        graph: nx.Graph,
    ):
        paper_json = paper_json["content"]
        paper_id = str(
            collection_nodes_papers.insert_one(
                {
                    "title": paper_json.get("title", "Unknown"),
                    "authors": paper_json.get("authors", "Unknown"),
                    "year": paper_json.get("year", "Unknown"),
                    "url": paper_json.get("url", "Unknown"),
                }
            ).inserted_id
        )

        pass_task_obj_ids = []
        for dataset in paper_json.get("datasets", []):
            dataset_obj = {
                "title": dataset.get("title", "Unknown"),
                "description": dataset.get("description", "No description available"),
                "link": dataset.get("link", "No link available"),
                "reference": dataset.get("reference", "No reference available"),
            }
            dataset_id = str(
                collection_nodes_datasets.insert_one(dataset_obj).inserted_id
            )
            numeric_dataset_id = FaissService.get_faiss_id_from_mongo_id(
                dataset_id[-8:]
            )
            dataset_embedding = bert_model.encode(
                dataset["title"] + " " + dataset["description"]
            ).astype("float32")
            faiss_index_datasets.add_with_ids(
                np.array([dataset_embedding]), numeric_dataset_id
            )

            task_obj = {
                "task": dataset["task"],
                "task_description": dataset["task_description"],
            }
            same_task_flag = False
            for id in pass_task_obj_ids:
                pass_task_obj = collection_nodes_tasks.find_one({"_id": ObjectId(id)})
                task = pass_task_obj.get("task", ["Unknown"])
                task_description = pass_task_obj.get(
                    "task_description", "No description available"
                )
                if (
                    set(task) == set(dataset["task"])
                    or task_description == dataset["task_description"]
                ):
                    task_id = id
                    same_task_flag = True
                    break
            if not same_task_flag:
                task_id = str(collection_nodes_tasks.insert_one(task_obj).inserted_id)
                pass_task_obj_ids.append(task_id)
            numeric_task_id = FaissService.get_faiss_id_from_mongo_id(task_id)
            if not same_task_flag:
                collection_nodes_tasks.update_one(
                    {"_id": task_id}, {"$set": {"faiss_index_id": numeric_task_id}}
                )
                task_description_embedding = bert_model.encode(
                    dataset["task_description"]
                ).astype("float32")
                faiss_index_task_descriptions.add_with_ids(
                    np.array([task_description_embedding]), numeric_task_id
                )
                task_keywords_embeddidng = bert_model.encode(
                    ", ".join(dataset["task"])
                ).astype("float32")
                faiss_index_task_keywords.add_with_ids(
                    np.array([task_keywords_embeddidng]), numeric_task_id
                )

            graph.add_edge(dataset_id, task_id)
            graph.add_edge(paper_id, dataset_id)

        return f"Paper `{paper_json['title']}` processed successfully!"
