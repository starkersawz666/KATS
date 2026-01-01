# KATS: Knowledge graph-Augmented Task-oriented dataset Search

Welcome to the documentation for **KATS**, the system presented in our paper: _"Revisiting Task-Oriented Dataset Search in the Era of Large Language Models: Challenges, Benchmark, and Solution"_.

---

## 📖 Introduction

Finding the right dataset is often the first and most critical step in data-driven research. However, researchers often struggle to find datasets based on high-level **task descriptions** due to the semantic gap between user intent and keyword-based search engines.

**KATS** is an end-to-end system designed to bridge this gap. It leverages Large Language Models (LLMs) and Knowledge Graphs (KGs) to:

1.  **Extract** structured knowledge (tasks & datasets) from unstructured scientific literature.
2.  **Resolve** entity ambiguities (e.g., "COCO" and "MS-COCO").
3.  **Retrieve** highly relevant datasets using a hybrid engine combining vector similarity and graph-based reasoning.

## ✨ Key Features

- **Task-Oriented Discovery**: Move beyond keywords. Search for datasets using natural language descriptions of your research task.
- **Knowledge Graph Augmented**: Built upon a high-quality Task-Dataset Knowledge Graph that captures relationships between research tasks and datasets.
- **Hybrid Search Engine**: A retrieval pipeline that combines dense vector retrieval with graph-based re-ranking.
- **Incremental Updates**: Efficiently integrate new papers into the knowledge base without costly full rebuilds.
- **CS-TDS Benchmark**: Includes the **Computer Science Task Dataset Search (CS-TDS)** benchmark suite for standardized evaluation.

## 📚 Documentation Guide

Explore our documentation to get started:

- **[Installation](installation.md)**: Setup guide, prerequisites, and environment configuration.
- **[Quick Start](quick_start.md)**: Run your first query with KATS in minutes.
- **[Configuration](configuration.md)**: detailed reference for `PipelineConfig` parameters.
- **[Reproduction](reproduction.md)**: Step-by-step guide to reproduce the experiments and results from our VLDB paper.
