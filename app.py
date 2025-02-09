import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pymongo import MongoClient
import zipfile
import json

# set streamlit page config as the first statement
st.set_page_config(page_title="Task-Oriented Dataset Search", layout="wide")

# Load BERT Transformer
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = load_model()

# Load MongoDB Connection
@st.cache_resource
def get_mongo_client():
    return MongoClient("mongodb://tods_mongodb:27017/")
client = get_mongo_client()
db = client["paperdatabase"]
collection_regular_paper = db["regular_papers"]
collection_dataset_paper = db["dataset_papers"]

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Your Page", ["Import", "Manage DB"])

# Import Page
if page == "Import":
    st.title("Import JSON Data for Papers")
    select_paper_type_cols = st.columns([1, 1, 1, 0.8])
    with select_paper_type_cols[0]:
        paper_type = st.selectbox("Select Paper Type", ["", "Regular Paper", "Dataset Paper"])
    if paper_type:
        # Upload ZIP file
        uploaded_file = st.file_uploader("Upload a ZIP file containing JSON files", type=["zip"])

        if uploaded_file is not None:
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                json_files = [f for f in zip_ref.namelist() if f.endswith(".json")]

                if not json_files:
                    st.error("[Error] No JSON files found in the ZIP!")
                else:
                    st.success(f"[Success] Found {len(json_files)} JSON files, processing...")
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
                                    st.success(f"[Success] Processing `{json_filename}`.")
                                else:
                                    st.warning(f"[Skipped]`{json_filename}` is empty.")

                            except json.JSONDecodeError:
                                st.warning(f"[Skipped]`{json_filename}` is not a valid JSON file.")

                        # Update progress bar
                        progress_bar.progress((i + 1) / len(json_files))

                    # Insert valid JSON into MongoDB
                    if new_data:
                        if paper_type == "Regular Paper":
                            collection_regular_paper.insert_many(new_data)
                        elif paper_type == "Dataset Paper":
                            collection_dataset_paper.insert_many(new_data)
                        result_message.success(f"[Success] Successfully imported {len(new_data)} JSON files into MongoDB.")
                    else:
                        result_message.error("[Error] No valid JSON data found for import.")

elif page == "Manage DB":
    st.title("Manage MongoDB Content")

    select_paper_type_cols = st.columns([1, 1, 1, 0.8])
    with select_paper_type_cols[0]:
        paper_type = st.selectbox("Select Paper Type", ["", "Regular Paper", "Dataset Paper"])
    
    if paper_type:
        selected_collection = collection_regular_paper if paper_type == "Regular Papers" else collection_dataset_paper
        col1, col2 = st.columns(2)
        
        # Fetch all documents with _id
        with col1:
            if st.button("Show Database Content"):
                documents = list(selected_collection.find({}, {"_id": 1, "content": 1}))  
                if documents:
                    for doc in documents:
                        st.write(f"**ID:** {doc['_id']}")
                        st.json(doc["content"])
                else:
                    st.warning("No documents found in the database.")
        # Delete all documents in the collection
        with col2:
            if st.button("Clear All Content"):
                selected_collection.delete_many({})  
                st.success("All documents have been deleted from the database.")