# Configuration Reference ⚙️

KATS is highly configurable to adapt to different scales of data and hardware environments. The central configuration class is `PipelineConfig`, which controls everything from file paths to knowledge graph construction hyperparameters.

## Usage

You can pass these parameters when initializing the `PipelineConfig` object:

```python
from task_oriented_dataset_search.pipeline import PipelineConfig

config = PipelineConfig(
    input_folder="./my_papers",
    api_key="...",
    model="...",
    # more parameters
)
```

## Core & I/O Parameters

These parameters define the basic input/output behavior and authentication.

| Parameter      | Type  | Default      | Description                                                                    |
| :------------- | :---- | :----------- | :----------------------------------------------------------------------------- |
| `input_folder` | `str` | **Required** | Path to the directory containing papers.                                       |
| `api_key`      | `str` | **Required** | Your API Key.                                                                  |
| `cache_root`   | `str` | `".cache"`   | Root directory where KATS stores the database, indices, and graph files.       |
| `db_path`      | `str` | `None`       | Custom path for the TinyDB JSON file. Defaults to `{cache_root}/tiny_db.json`. |

## LLM & Model Parameters

Control which Large Language Models are used for different stages of the pipeline. Separation of extraction and QA models allows for cost optimization.

| Parameter        | Type    | Default         | Description                                                                                      |
| :--------------- | :------ | :-------------- | :----------------------------------------------------------------------------------------------- |
| `model`          | `str`   | `"gpt-4o-mini"` | The model used for **Information Extraction** and **Entity Resolution** (Offline Phase).         |
| `qa_model`       | `str`   | `"gpt-4o"`      | The model used for **Answer Generation** (Online Phase).                                         |
| `temperature`    | `float` | `0.1`           | Temperature for extraction tasks. Lower values ensure consistent structured outputs.             |
| `qa_temperature` | `float` | `1.0`           | Temperature for the final QA response. Higher values allow for more natural language generation. |
| `api_base`       | `str`   | `None`          | Custom base URL for the OpenAI API (e.g., for Azure OpenAI or local proxies).                    |

## KG Construction Hyperparameters

These parameters control the logic for building and refining the Task-Dataset Knowledge Graph.

| Parameter                            | Default | Description                                                                                           |
| :----------------------------------- | :------ | :---------------------------------------------------------------------------------------------------- |
| `strong_similarity_threshold`        | `0.8`   | If vector similarity > this value, tasks are considered highly similar and may be merged.             |
| `weak_similarity_threshold`          | `0.6`   | If vector similarity > this value, a weighted edge is created between tasks, but they are not merged. |
| `keyword_overlap_threshold`          | `0.7`   | Alternative threshold based on Jaccard similarity of extracted keywords.                              |
| `task_max_merge`                     | `10`    | Maximum number of tasks that can be merged into a single node cluster.                                |
| `dataset_merge_k_neighbors`          | `10`    | Number of candidate datasets to retrieve from the vector index for comparison.                        |
| `dataset_merge_similarity_threshold` | `0.7`   | Minimum similarity score required to trigger the LLM-based verification step.                         |

## System & Performance

Parameters to optimize the speed of the offline build process.

| Parameter            | Type  | Default | Description                                                                                                 |
| :------------------- | :---- | :------ | :---------------------------------------------------------------------------------------------------------- |
| `preprocess_workers` | `int` | `1`     | Number of parallel threads for possible PDF parsing. Increase this for large corpora.                       |
| `extract_workers`    | `int` | `1`     | Number of parallel threads for LLM extraction. **Warning**: Increasing this increases API rate limit usage. |
| `retry_limit`        | `int` | `1`     | Number of retries for processing a file if it fails.                                                        |
| `llm_retries`        | `int` | `3`     | Number of retries for LLM API calls in case of network errors or rate limits.                               |
