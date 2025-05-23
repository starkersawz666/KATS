from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

from task_oriented_dataset_search.embedding.embedder import SentenceTransformerEmbedder
from task_oriented_dataset_search.embedding.pipeline import EmbeddingPipeline
from task_oriented_dataset_search.extraction.client import OpenAIClient
from task_oriented_dataset_search.extraction.extractor import StandardExtractor
from task_oriented_dataset_search.extraction.file_extractor import extract_file
from task_oriented_dataset_search.graph.builder import GraphBuilder
from task_oriented_dataset_search.graph.merger import TaskMerger
from task_oriented_dataset_search.importer.db_importer import TinyDBImporter
from task_oriented_dataset_search.preprocessing.processor import preprocess
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
    db_path: str | None = None
    faiss_tasks_index_path: str | None = None
    faiss_datasets_index_path: str | None = None
    task_parquet_path: str | None = None
    dataset_parquet_path: str | None = None
    graph_path: str | None = None
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 30.0
    temperature: float = 0.1
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
            self.graph_path = os.path.join(self.cache_root, "knowledge_graph.graphml")


class TodsBuilder:
    def __init__(self, config: PipelineConfig = None, **kwargs):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = PipelineConfig(**kwargs)

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
        client = OpenAIClient(
            api_key=cfg.api_key,
            model=cfg.model,
            base_url=cfg.api_base,
            temperature=cfg.temperature,
        )
        extractor = StandardExtractor(client)
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
        graph_builder.build_graph()
        graph_builder.save_graph()

        # STEP 6: Merge Tasks
        task_merger = TaskMerger(
            db_path=cfg.db_path,
            graph_path=cfg.graph_path,
            task_faiss_path=cfg.faiss_tasks_index_path,
            task_parquet_path=cfg.task_parquet_path,
            strong_similarity_threshold=cfg.strong_similarity_threshold,
            keyword_overlap_threshold=cfg.keyword_overlap_threshold,
            weak_similarity_threshold=cfg.weak_similarity_threshold,
            max_merge=cfg.task_max_merge,
        )
        task_merger.merge_tasks()
        task_merger.save_graph()
