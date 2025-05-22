from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

from task_oriented_dataset_search.extraction.client import OpenAIClient
from task_oriented_dataset_search.extraction.extractor import StandardExtractor
from task_oriented_dataset_search.extraction.file_extractor import extract_file
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
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 30.0
    temperature: float = 0.1

    def __post_init__(self):
        if self.db_path is None:
            self.db_path = os.path.join(self.cache_root, "tiny_db.json")


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
