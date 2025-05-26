import logging
from task_oriented_dataset_search.preprocessing.interface import Document
from task_oriented_dataset_search.preprocessing.loader import get_loader
from task_oriented_dataset_search.utils.cache import CacheManager

logger = logging.getLogger(__name__)
cache_manager = CacheManager()


def preprocess(path: str) -> Document:
    fingerprint = cache_manager.fingerprint_file(path)
    logger.debug(f"File fingerprint for {path}: {fingerprint}")

    def text_loader(cache_path: str) -> str:
        logger.debug(f"Loading preprocessed text from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    def text_saver(cache_path: str, text: str) -> None:
        logger.debug(f"Saving preprocessed text to cache: {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)

    def compute_text() -> str:
        logger.debug(f"Cache miss for {path}. Computing text...")
        loader = get_loader(path)
        doc = loader.load(path)
        logger.debug(f"Successfully computed text for {path}.")
        return doc.text

    text = cache_manager.load_or_compute(
        step="preprocessing",
        fingerprint=fingerprint,
        compute_fn=compute_text,
        loader=text_loader,
        saver=text_saver,
        ext=".txt",
    )

    return Document(text=text, metadata={"path": path})
