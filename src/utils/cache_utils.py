import streamlit as st
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_mongo_client():
    return MongoClient("mongodb://tods_mongodb:27017/")
