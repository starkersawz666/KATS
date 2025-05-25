from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

from tinydb import Query, TinyDB

from task_oriented_dataset_search.embedding.embedder import SentenceTransformerEmbedder
from task_oriented_dataset_search.embedding.pipeline import EmbeddingPipeline
from task_oriented_dataset_search.extraction.client import OpenAIClient
from task_oriented_dataset_search.extraction.extractor import StandardExtractor
from task_oriented_dataset_search.extraction.file_extractor import extract_file
from task_oriented_dataset_search.graph.builder import GraphBuilder
from task_oriented_dataset_search.graph.merger import TaskMerger
from task_oriented_dataset_search.importer.db_importer import TinyDBImporter
from task_oriented_dataset_search.preprocessing.processor import preprocess
from task_oriented_dataset_search.search.searcher import Searcher
from task_oriented_dataset_search.utils.cache import CacheManager


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    input_folder: str
    api_key: str

    cache_root: str = ".cache"
    preprocess_workers: int = 1
    extract_workers: int = 1
    retry_limit: int = 1
    api_base: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1

    qa_api_key: str | None = None
    qa_api_base: str | None = None
    qa_model: str = "gpt-4o"
    qa_temperature: float = 1

    db_path: str | None = None
    faiss_tasks_index_path: str | None = None
    faiss_datasets_index_path: str | None = None
    task_parquet_path: str | None = None
    dataset_parquet_path: str | None = None
    graph_path: str | None = None
    graph_processed_path: str | None = None
    graph_tasks_path: str | None = None
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 30.0
    strong_similarity_threshold: float = 0.8
    keyword_overlap_threshold: float = 0.7
    weak_similarity_threshold: float = 0.6
    task_max_merge: int = 10

    def __post_init__(self):
        if self.db_path is None:
            self.db_path = os.path.join(self.cache_root, "tiny_db.json")
        if self.faiss_tasks_index_path is None:
            self.faiss_tasks_index_path = os.path.join(
                self.cache_root, "faiss_tasks.index"
            )
        if self.faiss_datasets_index_path is None:
            self.faiss_datasets_index_path = os.path.join(
                self.cache_root, "faiss_datasets.index"
            )
        if self.task_parquet_path is None:
            self.task_parquet_path = os.path.join(self.cache_root, "tasks.parquet")
        if self.dataset_parquet_path is None:
            self.dataset_parquet_path = os.path.join(
                self.cache_root, "datasets.parquet"
            )
        if self.graph_path is None:
            self.graph_path = os.path.join(self.cache_root, "original_kg.graphml")
        if self.graph_processed_path is None:
            self.graph_processed_path = os.path.join(
                self.cache_root, "processed_kg.graphml"
            )
        if self.graph_tasks_path is None:
            self.graph_tasks_path = os.path.join(self.cache_root, "tasks_kg.graphml")
        if self.qa_api_key is None:
            self.qa_api_key = self.api_key
        if self.qa_api_base is None:
            self.qa_api_base = self.api_base


class TodsEngine:
    def __init__(self, config: PipelineConfig = None, **kwargs):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = PipelineConfig(**kwargs)

        self.llm_client = OpenAIClient(
            api_key=self.cfg.api_key,
            model=self.cfg.model,
            base_url=self.cfg.api_base,
            temperature=self.cfg.temperature,
        )
        self.qa_client = OpenAIClient(
            api_key=self.cfg.qa_api_key,
            model=self.cfg.qa_model,
            base_url=self.cfg.qa_api_base,
            temperature=self.cfg.qa_temperature,
        )
        self.searcher_instance = None

    def _get_searcher(self) -> Searcher:
        if self.searcher_instance is None:
            embedder = SentenceTransformerEmbedder()
            self.searcher_instance = Searcher(
                embedder=embedder,
                db_path=self.cfg.db_path,
                faiss_tasks_index_path=self.cfg.faiss_tasks_index_path,
                task_parquet_path=self.cfg.task_parquet_path,
                graph_processed_path=self.cfg.graph_processed_path,
                graph_tasks_path=self.cfg.graph_tasks_path,
            )
        return self.searcher_instance

    def build(self):
        cfg = self.cfg
        if not cfg.input_folder or not cfg.api_key:
            raise ValueError("Missing arguments: input_folder or api_key")

        cache = CacheManager(cfg.cache_root)

        # STEP 1: Preprocessing
        files = list(Path(cfg.input_folder).rglob("*"))
        with ThreadPoolExecutor(cfg.preprocess_workers) as exe:
            futures = {exe.submit(preprocess, str(f)): f for f in files if f.is_file()}
            for fut in as_completed(futures):
                try:
                    doc = fut.result()
                    print(doc)
                except Exception as e:
                    pass

        # STEP 2: Extraction
        extractor = StandardExtractor(self.llm_client)
        pre_dir = Path(cfg.cache_root) / "preprocessing"
        txts = list(pre_dir.glob("*.txt"))

        def worker(txt_path: Path):
            fp = txt_path.stem
            last_exc = None
            delay = cfg.retry_initial_delay
            for attempt in range(1, cfg.retry_limit + 1):
                try:
                    res = extract_file(str(txt_path), extractor, cache)
                    return fp, res
                except Exception as e:
                    last_exc = e
                    if attempt == cfg.retry_limit:
                        break
                    logger.warning(
                        f"Fail to extract {fp} with exception: {e}, retrying in {delay} seconds for the {attempt}/{cfg.retry_limit - 1} time"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, cfg.retry_max_delay)
            raise last_exc

        with ThreadPoolExecutor(cfg.extract_workers) as exe:
            futures = {exe.submit(worker, txt): txt for txt in txts}
            for fut in as_completed(futures):
                try:
                    fp, res = fut.result()
                except Exception as e:
                    pass

        # STEP 3: Import into TinyDB
        ext_dir = Path(cfg.cache_root) / "extraction"
        importer = TinyDBImporter(db_path=cfg.db_path)
        importer.import_all(ext_dir)

        # STEP 4: Vector Embedding and Indexing
        embedder = SentenceTransformerEmbedder()
        embedding_pipeline = EmbeddingPipeline(
            embedder,
            task_index_path=cfg.faiss_tasks_index_path,
            dataset_index_path=cfg.faiss_datasets_index_path,
            task_parquet_path=cfg.task_parquet_path,
            dataset_parquet_path=cfg.dataset_parquet_path,
        )
        embedding_pipeline.embed_all(db_path=cfg.db_path)

        # STEP 5: Build Knowledge Graph
        graph_builder = GraphBuilder(db_path=cfg.db_path, graph_path=cfg.graph_path)
        graph_builder.build_basic_graph()
        graph_builder.save_graph()

        # STEP 6: Merge Tasks
        task_merger = TaskMerger(
            db_path=cfg.db_path,
            graph_path=cfg.graph_path,
            graph_processed_path=cfg.graph_processed_path,
            task_faiss_path=cfg.faiss_tasks_index_path,
            task_parquet_path=cfg.task_parquet_path,
            strong_similarity_threshold=cfg.strong_similarity_threshold,
            keyword_overlap_threshold=cfg.keyword_overlap_threshold,
            weak_similarity_threshold=cfg.weak_similarity_threshold,
            max_merge=cfg.task_max_merge,
        )
        task_merger.merge_tasks()
        task_merger.save_graph()
        merged_graph_builder = GraphBuilder(
            db_path=cfg.db_path,
            graph_path=cfg.graph_processed_path,
            save_path=cfg.graph_tasks_path,
        )
        merged_graph_builder.build_and_save_task_similarity_graph()

    def search(self, task: str):
        searcher = self._get_searcher()
        results = searcher.search(task)
        return results

    def qa(self, task_description: str) -> str:
        searcher = self._get_searcher()
        search_results = searcher.search(task_description)

        if not search_results:
            return "Sorry, I cannot find any related dataset for your question."

        db = TinyDB(self.cfg.db_path)
        tasks_tbl = db.table("tasks")
        TaskQ = Query()

        context_list = []
        for i, ds in enumerate(search_results):
            ds_id = ds.get("id")
            title = ds.get("title", "N/A")
            description = ds.get("description", "N/A")
            link = ds.get("link", "N/A")

            associated_tasks = tasks_tbl.search(TaskQ.dataset_id == ds_id)
            tasks_info = (
                "; ".join([t.get("task_description", "N/A") for t in associated_tasks])
                if associated_tasks
                else "N/A"
            )

            context_list.append(
                f"{i+1}. **Dataset**: {title}\n"
                f"   **Description**: {description}\n"
                f"   **Related Tasks**: {tasks_info}\n"
                f"   **Link**: {link}"
            )

        context_str = "\n\n".join(context_list)

        prompt = f"""
You are an AI assistant specializing in dataset discovery. Several relevant datasets are found based on their described task. Your goal is to formulate a concise and helpful response Based *only* on the context provided below.

Follow these instructions:
- Your answer must be *strictly* based on the provided context information.
- *All the datasets* provided in the context information &must be included in your answer&.
- If there are significantly duplicate datasets in the context, you can select one of them to include in your answer.
- Do *not* invent or include any information beyond the provided context.
- Present the information clearly, highlighting the dataset titles and briefly explaining why they might be relevant to the user's task.

---
Context Information (Datasets found):
{context_str}
---

User Task: {task_description}

Answer:
"""

        # Call LLM
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.qa_client.chat(messages)
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            logger.error(f"QA LLM call failed: {e}")
            return "Sorry, an error occurred while generating the answer."
