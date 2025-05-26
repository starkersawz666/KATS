# Task Oriented Dataset Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Task Oriented Dataset Search (TODS) is a system designed to help researchers and developers discover relevant datasets by describing their specific tasks. It uses Large Language Models, vector embeddings, and knowledge graphs to automatically process a corpus of documents (like academic papers), extract information about datasets, and build a searchable index. Users can then query this index using natural language task descriptions to find suitable datasets.

## Installation

### Prerequisites

- Python 3.11 or higher.
- An OpenAI API Key. You need to set this as an environment variable:
  ```bash
  export OPENAI_API_KEY="your_openai_api_key_here"
  ```

### From Release

1.  Navigate to the [**Releases**](https://github.com/starkersawz666/task-oriented-dataset-search/releases) page of this repository.
2.  Download the latest `.whl` (wheel) file appropriate for your system.
3.  Install the wheel file using `pip`:
    ```bash
    pip install path/to/downloaded_wheel_file.whl
    ```
    This will install the `task_oriented_dataset_search` package and its dependencies.

## Usage

You can use the TODS pipeline programmatically as shown below. For a more detailed walkthrough, please refer to the example notebook: [`examples/demo.ipynb`](https://github.com/starkersawz666/task-oriented-dataset-search/blob/pipeline/examples/demo.ipynb).

Make sure you have set your `OPENAI_API_KEY` environment variable before running the code.

## License

This project is licensed under the [MIT License](LICENSE).
