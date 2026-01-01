# Quick Start 🚀

## 1. Prepare Your Data

KATS is designed to extract knowledge from unstructured scientific papers (PDFs or pure text files).

1.  Create a directory for your documents (e.g., `my_papers/`).
2.  Place a few papers inside this directory.

## 2. Basic Usage

The easiest way to use KATS is via the `TodsEngine` Python API.

### Step 1: Import and Configure

Firstly, configure the pipeline. You need to specify where your input documents are and provide your API Key.

```python
import os
from task_oriented_dataset_search.pipeline import TodsEngine, PipelineConfig

# 1. Setup Configuration
config = PipelineConfig(
    input_folder="my_papers",           # Path to your PDF directory
    api_key="your_qpi_key",                   # Your OpenAI API Key

    # Optional: Adjust for performance/cost
    model="gpt-4o-mini",                # Model for extraction
    qa_model="gpt-4o",                  # Model for final QA
    extract_workers=5,                  # Parallel extraction processing count
    cache_root=".kats_cache"            # Where to store the database and index
)

# 2. Initialize the Engine
engine = TodsEngine(config=config)
```

### Step 2: Build the Knowledge Base (Offline Phase)

Before you can search, KATS needs to process the documents to build the Knowledge Graph.

```python
engine.build()
```

Once the build is complete, the data is persisted in the `cache_root`.

### Step 3: Search and QA (Online Phase)

#### Dataset Search

Retrieve the most relevant datasets for a specific task.

```python
query = "Training a model to detect pedestrians in urban environments"

results = engine.search(
    task=query,
    top_k_datasets=3
)

print(f"Found {len(results)} datasets:")
for res in results:
    print(f"- {res['dataset_name']}: {res['description']}")
```

#### Question Answering

Get a synthesized natural language answer.

```python
answer = engine.qa(task_description=query)

print(answer)
```

## 3. Incremental Updates

One of KATS's key features is handling new data efficiently without rebuilding the entire graph. If you add new PDFs to your `my_papers/` folder, simply call update():

```python
engine.update(input_folder="my_papers")
```
