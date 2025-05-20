import json
import os
from typing import Dict
from task_oriented_dataset_search.extraction.interface import BaseExtractor
from task_oriented_dataset_search.utils.cache import CacheManager


def extract_file(
    txt_path: str,
    extractor: BaseExtractor,
    cache: CacheManager,
    extraction_step: str = "extraction",
):
    fname = os.path.basename(txt_path)
    fingerprint, _ = os.path.splitext(fname)

    def loader(pth: str):
        with open(pth, "r", encoding="utf-8") as f:
            return json.load(f)

    def saver(pth: str, data: Dict):
        with open(pth, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def compute_fn():
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        return extractor.extract(text)

    result = cache.load_or_compute(
        step=extraction_step,
        fingerprint=fingerprint,
        compute_fn=compute_fn,
        loader=loader,
        saver=saver,
        ext=".json",
    )

    return result
