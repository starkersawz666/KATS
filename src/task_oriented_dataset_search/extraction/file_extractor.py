import json
import logging
import os
from typing import Dict
from task_oriented_dataset_search.extraction.interface import BaseExtractor
from task_oriented_dataset_search.utils.cache import CacheManager

logger = logging.getLogger(__name__)


def extract_file(
    txt_path: str,
    extractor: BaseExtractor,
    cache: CacheManager,
    extraction_step: str = "extraction",
):
    fname = os.path.basename(txt_path)
    fingerprint, _ = os.path.splitext(fname)
    logger.debug(f"Starting extraction for: {fname} (fingerprint: {fingerprint})")

    def loader(pth: str):
        logger.debug(f"Loading extraction result from cache: {pth}")
        with open(pth, "r", encoding="utf-8") as f:
            return json.load(f)

    def saver(pth: str, data: Dict):
        logger.debug(f"Saving extraction result to cache: {pth}")
        with open(pth, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def compute_fn():
        logger.debug(f"Cache miss for {fname}. Computing extraction...")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            logger.debug(f"Text length for {fname}: {len(text)} characters.")
        result = extractor.extract(text)
        logger.debug(f"Successfully computed extraction for {fname}.")
        return result

    try:
        result = cache.load_or_compute(
            step=extraction_step,
            fingerprint=fingerprint,
            compute_fn=compute_fn,
            loader=loader,
            saver=saver,
            ext=".json",
        )
        logger.debug(f"Finished extraction for: {fname}")
        return result
    except Exception as e:
        logger.error(f"Failed during load_or_compute for {fname}: {e}", exc_info=True)
        raise
