# Installation Guide

## 📋 Prerequisites

Before installing KATS, ensure your system meets the following requirements:

- **Python**: Version **3.11** or higher.
- **API Key**: An active API Key (support for `gpt-4o` models required).

## 🛠️ Method 1: Installation from Source

This method is good for researchers and developers who want to inspect the source code and reproduce experiments. We use [Poetry](https://python-poetry.org/) for dependency management.

### 1. Clone the Repository

```bash
git clone [https://github.com/starkersawz666/KATS.git](https://github.com/starkersawz666/KATS.git)
cd KATS
```

### 2. Set up the Environment

You can install dependencies using Poetry.

If you don't have Poetry installed, install it first (refer to [Poetry Documentation](https://python-poetry.org/docs/)).

```bash
poetry install
```

To activate the virtual environment:

```bash
poetry shell
```

📦 Method 2: Installation from Wheel

This method is suitable if you only want to use the KATS library in your own scripts.

1. Go to the Releases page.
2. Download the latest `.whl` file
3. Install it via pip:

```bash
pip install path/to/kats.whl
```

## 🔑 Configuration

KATS relies on large language models (embedding and generation). You need to configure your API key before running the system.

### Environment Variable

Set the `OPENAI_API_KEY` environment variable in your terminal:

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."
# Windows
$env:OPENAI_API_KEY="sk-..."
```

### Verify Configuration

To verify that the installation and configuration are correct, you can run a simple Python snippet:

```python
import os
from task_oriented_dataset_search.pipeline import PipelineConfig

api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print("✅ OpenAI API Key found.")
    config = PipelineConfig(input_folder="./data", api_key=api_key)
    print("✅ KATS Configuration initialized successfully.")
else:
    print("❌ Error: OPENAI_API_KEY not set.")
```
