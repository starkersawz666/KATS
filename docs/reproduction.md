# Experiment Reproduction Guide 🧪

This guide details how to reproduce the experimental results reported in our VLDB paper using the **CS-TDS** benchmark suite.

## The Benchmark Suite (CS-TDS)

We provide two benchmark subsets located in `tests/benchmark/`:

- **CS-TDS_M**:
  - Corpus: 628 papers.
  - Test Queries: 47 task descriptions.
- **CS-TDS_L**:
  - Corpus: 2,101 papers.
  - Test Queries: 204 task descriptions.

Each subset contains two files:

1.  `*_Build_Data.csv`: Contains the text corpus used to construct the Knowledge Base.
2.  `*_Test_Data.csv`: Contains the query tasks and ground truth dataset labels.

## Reproduction Script

We provide a comprehensive Python script to automate the reproduction process. This script performs three steps:

1.  **Data Preparation**: Download papers from `Build_Data.csv` corpus.
2.  **Offline Build**: Runs the KATS pipeline to extract entities and build the Knowledge Graph.
3.  **Inference**: Executes search for all queries in Test_Data.csv and saves the results to a new CSV file.

Create a python file with the code below:

```python
import os
import csv
import time
import shutil
import requests
import pandas as pd
from pathlib import Path
from task_oriented_dataset_search.pipeline import TodsEngine, PipelineConfig

# Configuration
BENCHMARK_NAME = "CS-TDS_M" # or CS-TDS_L
BASE_DIR = f"tests/benchmark/{BENCHMARK_NAME}"
BUILD_CSV = f"{BASE_DIR}/{BENCHMARK_NAME}_Build_Data.csv"
TEST_CSV = f"{BASE_DIR}/{BENCHMARK_NAME}_Test_Data.csv"

TEMP_INPUT_DIR = f"temp_{BENCHMARK_NAME}_corpus"
CACHE_ROOT = f".cache_{BENCHMARK_NAME}"
API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable.")

# Data Preparation
def prepare_data():
    """Download PDFs from URLs in Build CSV to local directory"""
    if os.path.exists(TEMP_INPUT_DIR) and len(os.listdir(TEMP_INPUT_DIR)) > 0:
        print(f"Directory {TEMP_INPUT_DIR} exists and is not empty. Skipping download.")
        return

    os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
    df = pd.read_csv(BUILD_CSV)

    success_count = 0
    total_count = len(df)

    print(f"Found {total_count} papers to download")

    for index, row in df.iterrows():
        url = row.get('url')
        title = row.get('title', f'doc_{index}')

        if not url or pd.isna(url):
            continue

        safe_title = "".join([c for c in str(title) if c.isalnum() or c in (' ', '_', '-')]).strip()
        safe_title = safe_title.replace(" ", "_")[:50]
        filename = f"{index}_{safe_title}.pdf"
        filepath = os.path.join(TEMP_INPUT_DIR, filename)

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                success_count += 1
            else:
                print(f"Failed to download {url}: Status {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
        time.sleep(0.2)

    print(f"Successfully downloaded {success_count}/{total_count} files")

# Build Knowledge Base
def run_build():
    config = PipelineConfig(
        input_folder=TEMP_INPUT_DIR,
        api_key=API_KEY,
        cache_root=CACHE_ROOT,
        model="gpt-4o-mini",
        qa_model="gpt-4o",
        extract_workers=4
    )
    engine = TodsEngine(config=config)

    print("Starting Offline Knowledge Base Construction")
    engine.build()
    print("Build Complete")
    return engine

# Generate Answers
def generate_predictions(engine):
    print(f"Generating predictions for {TEST_CSV}")
    if not os.path.exists(TEST_CSV):
        print(f"Test Data not found at {TEST_CSV}")
        return

    df_test = pd.read_csv(TEST_CSV)
    results_data = []

    total = len(df_test)
    for index, row in df_test.iterrows():
        query = row.get('query')
        ground_truth = row.get('ground_truth')

        if pd.isna(query):
            continue

        print(f"Processing Query {index + 1}/{total}")
        try:
            answer = engine.qa(query, top_k_datasets=5)
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = "ERROR"
        results_data.append({
            "query_id": index,
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer
        })

    result_df = pd.DataFrame(results_data)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"QA Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()
    engine = run_build()
    generate_predictions(engine)
```

Note: Minor variances are expected due to the probabilistic nature of LLMs and updates to LLM versions over time.
