import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pymongo import MongoClient
import zipfile
import json
import networkx as nx
import os
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId
from src.utils.faiss import FaissService
from src.utils.graph import GraphService
from src.utils.cache_utils import load_model, get_mongo_client
from src.utils.paper_processor import PaperProcessor
from src.utils.bert import BertService
from src.pages.search import SearchPage

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


# Merge tasks
def merge_tasks(
    strong_similarity_threshold: float = 0.8,
    keyword_overlap_threshold: float = 0.7,
    weak_similarity_threshold: float = 0.6,
    max_merge: int = 10,
):
    task_data = list(collection_nodes_tasks.find({}, {"_id": 1, "task": 1}))
    task_id_map = {
        str(doc["_id"]): FaissService.get_faiss_id_from_mongo_id(doc["_id"])
        for doc in task_data
    }
    task_keyword_map = {str(doc["_id"]): set(doc.get("task", [])) for doc in task_data}
    task_nodes = list(task_id_map.keys())
    faiss_ids = np.array(list(task_id_map.values()), dtype=int)

    all_vectors = FaissService.get_vectors_by_ids(
        faiss_index_task_descriptions, faiss_ids
    )
    k = min(max_merge, len(task_nodes))
    D, I = faiss_index_task_descriptions.search(all_vectors, k)

    cnt = 0
    for idx, task1_id in enumerate(task_nodes):
        task1_keywords = task_keyword_map.get(task1_id, set())
        for j, faiss_idx in enumerate(I[idx]):
            if faiss_idx == -1 or D[idx, j] < weak_similarity_threshold:
                continue

            task2_id = next(
                (key for key, value in task_id_map.items() if value == faiss_idx), None
            )
            if not task2_id or task1_id == task2_id:
                continue
            similarity = D[idx, j]

            if similarity > strong_similarity_threshold:
                graph.add_edge(task1_id, task2_id, weight=1)
                cnt += 1
            else:
                task2_keywords = task_keyword_map.get(task2_id, set())
                common_keywords = task1_keywords.intersection(task2_keywords)
                min_keywords = (
                    min(len(task1_keywords), len(task2_keywords))
                    if min(len(task1_keywords), len(task2_keywords)) > 0
                    else 1
                )

                if len(common_keywords) >= keyword_overlap_threshold * min_keywords:
                    graph.add_edge(task1_id, task2_id, weight=1)
                    cnt += 1
                else:
                    graph.add_edge(task1_id, task2_id, weight=similarity)
                    cnt += 1

    print(f"{cnt} pairs of tasks merged among {len(task_nodes)} tasks")
    GraphService.save_graph(graph, graph_path)
    return "Merge tasks complete"


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Your Page", ["Import", "Manage DB", "Search"])

# Import Page
if page == "Import":
    st.title("Import JSON Data for Papers")
    select_paper_type_cols = st.columns([1, 1, 1, 0.8])
    with select_paper_type_cols[0]:
        paper_type = st.selectbox(
            "Select Paper Type", ["", "Regular Paper", "Dataset Paper"]
        )
    if paper_type:
        # Upload ZIP file
        uploaded_file = st.file_uploader(
            "Upload a ZIP file containing JSON files", type=["zip"]
        )

        if uploaded_file is not None:
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                json_files = [f for f in zip_ref.namelist() if f.endswith(".json")]

                if not json_files:
                    st.error("[Error] No JSON files found in the ZIP!")
                else:
                    st.success(
                        f"[Success] Found {len(json_files)} JSON files, processing..."
                    )
                    result_message = st.empty()
                    progress_bar = st.progress(0)
                    new_data = []
                    for i, json_filename in enumerate(json_files):
                        with zip_ref.open(json_filename) as json_file:
                            try:
                                data = json.load(json_file)
                                # Filter out empty JSON or `{}`
                                if data and data != {}:
                                    new_data.append({"content": data})
                                    st.success(
                                        f"[Success] Processing `{json_filename}`."
                                    )
                                else:
                                    st.warning(f"[Skipped]`{json_filename}` is empty.")

                            except json.JSONDecodeError:
                                st.warning(
                                    f"[Skipped]`{json_filename}` is not a valid JSON file."
                                )

                        # Update progress bar
                        progress_bar.progress((i + 1) / len(json_files))

                    # Insert valid JSON into MongoDB
                    if new_data:
                        if paper_type == "Regular Paper":
                            collection_regular_paper.insert_many(new_data)
                        elif paper_type == "Dataset Paper":
                            collection_dataset_paper.insert_many(new_data)
                        result_message.success(
                            f"[Success] Successfully imported {len(new_data)} JSON files into MongoDB."
                        )
                    else:
                        result_message.error(
                            "[Error] No valid JSON data found for import."
                        )

elif page == "Manage DB":
    st.title("Manage MongoDB Content")

    select_paper_type_cols = st.columns([1, 1, 1, 0.8])
    with select_paper_type_cols[0]:
        paper_type = st.selectbox(
            "Select Paper Type", ["", "Regular Paper", "Dataset Paper"]
        )

    if paper_type:
        selected_collection = (
            collection_regular_paper
            if paper_type == "Regular Paper"
            else collection_dataset_paper
        )
        col1, col2, col3, col4 = st.columns(4)
        result_area = st.empty()
        if "result_log" not in st.session_state:
            st.session_state["result_log"] = []
        # Fetch all documents with _id
        with col1:
            if st.button("Show Database Content"):
                documents = list(selected_collection.find({}, {"_id": 1, "content": 1}))
                if documents:
                    st.session_state["result_log"] = []
                    for doc in documents:
                        result_log_id = f"**ID:** {doc['_id']}"
                        result_log_json = doc["content"]
                        st.session_state["result_log"].append(
                            {"text": result_log_id, "json": result_log_json}
                        )
                    with result_area.container():
                        for entry in st.session_state["result_log"]:
                            st.write(entry["text"])
                            st.json(entry["json"])
                else:
                    result_area.warning("No documents found in the database.")
        # Delete all documents in the collection
        with col2:
            if st.button("Clear All Content"):
                collection_regular_paper.delete_many({})
                collection_dataset_paper.delete_many({})
                collection_nodes_papers.delete_many({})
                collection_nodes_datasets.delete_many({})
                collection_nodes_tasks.delete_many({})
                result_area.success(
                    "All documents have been deleted from the database."
                )

        with col3:
            if st.button("Process Papers"):
                papers = list(collection_regular_paper.find({}))
                for paper in papers:
                    result_area.write(
                        PaperProcessor.process_regular_paper(
                            model,
                            paper,
                            collection_nodes_papers,
                            collection_nodes_datasets,
                            collection_nodes_tasks,
                            faiss_index_datasets,
                            faiss_index_task_descriptions,
                            faiss_index_task_keywords,
                            graph,
                        )
                    )
                merge_tasks()
                FaissService.save_faiss_index(
                    faiss_index_paths,
                    faiss_index_datasets,
                    faiss_index_task_descriptions,
                    faiss_index_task_keywords,
                )
                GraphService.save_graph(graph, graph_path)
                result_area.success(
                    "All papers processed, indexed, and graph updated successfully!"
                )

        with col4:
            if st.button("Show Graph Visualization"):
                # Load the graph from GraphML
                graph = GraphService.load_graph(graph_path)
                node_colors = []
                color_map = {
                    "papers": "lightblue",
                    "datasets": "lightgreen",
                    "tasks": "salmon",
                }
                object_id_papers = set(
                    map(str, collection_nodes_papers.distinct("_id"))
                )
                object_id_datasets = set(
                    map(str, collection_nodes_datasets.distinct("_id"))
                )
                object_id_tasks = set(map(str, collection_nodes_tasks.distinct("_id")))
                for node in graph.nodes():
                    if node in object_id_papers:
                        node_colors.append(color_map["papers"])
                    elif node in object_id_datasets:
                        node_colors.append(color_map["datasets"])
                    elif node in object_id_tasks:
                        node_colors.append(color_map["tasks"])
                    else:
                        node_colors.append("gray")  # Default color
                labels = {node: node[-3:] for node in graph.nodes()}
                # Draw the graph
                fig, ax = plt.subplots(figsize=(40, 27))
                nx.draw(
                    graph,
                    with_labels=True,
                    labels=labels,
                    node_color=node_colors,
                    edge_color="gray",
                    node_size=500,
                    font_size=10,
                    ax=ax,
                )

                # Display in Streamlit
                result_area.pyplot(fig)
elif page == "Search":
    SearchPage.page_search(
        faiss_index_task_descriptions,
        graph,
        collection_nodes_tasks,
        collection_nodes_datasets,
        model,
    )
