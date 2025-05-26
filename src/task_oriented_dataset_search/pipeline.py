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
from task_oriented_dataset_search.graph.dataset_merger import DatasetMerger
from task_oriented_dataset_search.graph.task_merger import TaskMerger
from task_oriented_dataset_search.importer.db_importer import TinyDBImporter
from task_oriented_dataset_search.preprocessing.processor import preprocess
from task_oriented_dataset_search.search.qa import QAEngine
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

    dataset_merge_k_neighbors: int = 10
    dataset_merge_similarity_threshold: float = 0.7
    llm_retries: int = 3
    llm_retry_delay: float = 10.0

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

        logger.info("Initializing TodsEngine...")
        logger.debug(f"Configuration: {self.cfg}")

        self.llm_client = OpenAIClient(
            api_key=self.cfg.api_key,
            model=self.cfg.model,
            base_url=self.cfg.api_base,
            temperature=self.cfg.temperature,
        )
        logger.info(f"Initialized extraction LLM client with model: {self.cfg.model}")
        self.qa_client = OpenAIClient(
            api_key=self.cfg.qa_api_key,
            model=self.cfg.qa_model,
            base_url=self.cfg.qa_api_base,
            temperature=self.cfg.qa_temperature,
        )
        logger.info(f"Initialized QA LLM client with model: {self.cfg.qa_model}")

        self.searcher_instance = None
        self.qa_engine_instance = None
        logger.info("TodsEngine initialized.")

    def _get_searcher(self) -> Searcher:
        if self.searcher_instance is None:
            logger.info("Initializing Searcher...")
            embedder = SentenceTransformerEmbedder()
            self.searcher_instance = Searcher(
                embedder=embedder,
                db_path=self.cfg.db_path,
                faiss_tasks_index_path=self.cfg.faiss_tasks_index_path,
                task_parquet_path=self.cfg.task_parquet_path,
                graph_processed_path=self.cfg.graph_processed_path,
                graph_tasks_path=self.cfg.graph_tasks_path,
            )
            logger.info("Searcher initialized.")
        return self.searcher_instance

    def _get_qa_engine(self) -> QAEngine:
        if self.qa_engine_instance is None:
            logger.info("Initializing QAEngine...")
            searcher = self._get_searcher()
            self.qa_engine_instance = QAEngine(
                qa_client=self.qa_client, searcher=searcher, db_path=self.cfg.db_path
            )
            logger.info("QAEngine initialized.")
        return self.qa_engine_instance

    def build(self):
        cfg = self.cfg
        logger.info("Starting build process...")
        if not cfg.input_folder or not cfg.api_key:
            logger.error("Missing arguments: input_folder or api_key")
            raise ValueError("Missing arguments: input_folder or api_key")

        cache = CacheManager(cfg.cache_root)
        logger.info(f"Using cache root: {cfg.cache_root}")

        # STEP 1: Preprocessing
        logger.info("--- STEP 1: Starting Preprocessing ---")
        files = list(Path(cfg.input_folder).rglob("*"))
        logger.info(f"Found {len(files)} potential files in {cfg.input_folder}.")
        processed_count = 0
        failed_count = 0
        with ThreadPoolExecutor(cfg.preprocess_workers) as exe:
            futures = {exe.submit(preprocess, str(f)): f for f in files if f.is_file()}
            logger.info(
                f"Submitting {len(futures)} files for preprocessing with {cfg.preprocess_workers} workers."
            )
            for fut in as_completed(futures):
                file_path = futures[fut]
                try:
                    doc = fut.result()
                    logger.debug(f"Successfully preprocessed: {file_path}")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to preprocess {file_path}: {e}")
                    failed_count += 1
        logger.info(
            f"--- STEP 1: Preprocessing Finished. Processed: {processed_count}, Failed: {failed_count} ---"
        )

        # STEP 2: Extraction
        logger.info("--- STEP 2: Starting Extraction ---")
        extractor = StandardExtractor(self.llm_client)
        pre_dir = Path(cfg.cache_root) / "preprocessing"
        txts = list(pre_dir.glob("*.txt"))
        logger.info(f"Found {len(txts)} preprocessed text files for extraction.")

        extracted_count = 0
        failed_count = 0

        def worker(txt_path: Path):
            fp = txt_path.stem
            last_exc = None
            delay = cfg.retry_initial_delay
            for attempt in range(1, cfg.retry_limit + 1):
                try:
                    logger.debug(f"Attempt {attempt}/{cfg.retry_limit} to extract {fp}")
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
            logger.error(
                f"Extraction failed for {fp} after {cfg.retry_limit} attempts: {last_exc}"
            )
            raise last_exc

        with ThreadPoolExecutor(cfg.extract_workers) as exe:
            futures = {exe.submit(worker, txt): txt for txt in txts}
            logger.info(
                f"Submitting {len(futures)} files for extraction with {cfg.extract_workers} workers."
            )
            for fut in as_completed(futures):
                txt_file = futures[fut]
                try:
                    fp, res = fut.result()
                    logger.debug(
                        f"Successfully extracted: {txt_file.name} -> {fp}.json"
                    )
                    extracted_count += 1
                except Exception as e:
                    logger.error(
                        f"Extraction ultimately failed for {txt_file.name}: {e}"
                    )
                    failed_count += 1
        logger.info(
            f"--- STEP 2: Extraction Finished. Extracted: {extracted_count}, Failed: {failed_count} ---"
        )

        # STEP 3: Import into TinyDB
        logger.info("--- STEP 3: Starting TinyDB Import ---")
        ext_dir = Path(cfg.cache_root) / "extraction"
        importer = TinyDBImporter(db_path=cfg.db_path)
        importer.import_all(ext_dir)
        logger.info(f"--- STEP 3: TinyDB Import Finished. DB at: {cfg.db_path} ---")

        # STEP 4: Vector Embedding and Indexing
        logger.info("--- STEP 4: Starting Vector Embedding and Indexing ---")
        embedder = SentenceTransformerEmbedder()
        embedding_pipeline = EmbeddingPipeline(
            embedder,
            task_index_path=cfg.faiss_tasks_index_path,
            dataset_index_path=cfg.faiss_datasets_index_path,
            task_parquet_path=cfg.task_parquet_path,
            dataset_parquet_path=cfg.dataset_parquet_path,
        )
        embedding_pipeline.embed_all(db_path=cfg.db_path)
        logger.info("--- STEP 4: Vector Embedding and Indexing Finished ---")

        # STEP 5: Build Knowledge Graph
        logger.info("--- STEP 5: Building Basic Knowledge Graph ---")
        graph_builder = GraphBuilder(db_path=cfg.db_path, graph_path=cfg.graph_path)
        graph_builder.build_basic_graph()
        graph_builder.save_graph()
        logger.info(
            f"--- STEP 5: Basic Knowledge Graph Finished. Saved to: {cfg.graph_path} ---"
        )

        # STEP 6: Merge Tasks
        logger.info("--- STEP 6: Starting Task Merging ---")
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
        logger.info(
            f"--- STEP 6: Task Merging Finished. Saved to: {cfg.graph_processed_path} ---"
        )

        # STEP 7: Merge Datasets
        logger.info("--- STEP 7: Starting Dataset Merging ---")
        dataset_merger = DatasetMerger(
            db_path=cfg.db_path,
            graph_path=cfg.graph_processed_path,  # Input is TaskMerger's output
            graph_processed_path=cfg.graph_processed_path,  # Output overwrites
            dataset_faiss_path=cfg.faiss_datasets_index_path,
            dataset_parquet_path=cfg.dataset_parquet_path,
            llm_client=self.llm_client,
            cache_manager=cache,
            similarity_threshold=cfg.dataset_merge_similarity_threshold,
            k_neighbors=cfg.dataset_merge_k_neighbors,
            llm_retries=cfg.llm_retries,
            llm_retry_delay=cfg.llm_retry_delay,
        )
        dataset_merger.merge_datasets()
        dataset_merger.save_graph()  # Save the graph with merged datasets
        logger.info(
            f"--- STEP 7: Dataset Merging Finished. Saved to: {cfg.graph_processed_path} ---"
        )

        # STEP 8: Separate Task Graph
        logger.info("--- STEP 8: Building Task Similarity Graph ---")
        merged_graph_builder = GraphBuilder(
            db_path=cfg.db_path,
            graph_path=cfg.graph_processed_path,
            save_path=cfg.graph_tasks_path,
        )
        merged_graph_builder.build_and_save_task_similarity_graph()
        logger.info(
            f"--- STEP 8: Task Similarity Graph Finished. Saved to: {cfg.graph_tasks_path} ---"
        )

        logger.info("Build process completed successfully.")

    def search(
        self,
        task: str,
        top_k_datasets: int = 5,
        initial_faiss_k: int = 2,
        similarity_threshold: float = 0.85,
        min_seed_similarity: float = 0.6,
        pagerank_alpha: float = 0.85,
    ):
        logger.info(f"Starting search for task: '{task}' with top_k={top_k_datasets}")
        searcher = self._get_searcher()
        results = searcher.search(
            task,
            top_k_datasets=top_k_datasets,
            initial_faiss_k=initial_faiss_k,
            similarity_threshold=similarity_threshold,
            min_seed_similarity=min_seed_similarity,
            pagerank_alpha=pagerank_alpha,
        )
        logger.info(f"Search found {len(results)} results.")
        return results

    def qa(
        self,
        task_description: str,
        top_k_datasets: int = 5,
        initial_faiss_k: int = 2,
        similarity_threshold: float = 0.85,
        min_seed_similarity: float = 0.6,
        pagerank_alpha: float = 0.85,
    ) -> str:
        logger.info(f"Starting QA for task: '{task_description}'")
        qa_engine = self._get_qa_engine()
        answer = qa_engine.answer(
            task_description,
            top_k_datasets=top_k_datasets,
            initial_faiss_k=initial_faiss_k,
            similarity_threshold=similarity_threshold,
            min_seed_similarity=min_seed_similarity,
            pagerank_alpha=pagerank_alpha,
        )
        logger.info("QA process finished.")
        return answer
