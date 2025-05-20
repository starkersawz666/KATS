from task_oriented_dataset_search.utils.cache import CacheManager

cache = CacheManager()


def get_extraction_cache(text: str) -> str:
    fingerprint = cache.fingerprint_text(text)
    return fingerprint
