from task_oriented_dataset_search.preprocessing.interface import Document
from task_oriented_dataset_search.preprocessing.loader import get_loader
from task_oriented_dataset_search.utils.cache import CacheManager


cache_manager = CacheManager()

def preprocess(path: str) -> Document:
    fingerprint = cache_manager.fingerprint_file(path)

    def text_loader(cache_path: str) -> str:
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def text_saver(cache_path: str, text: str) -> None:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    def compute_text() -> str:
        loader = get_loader(path)
        doc = loader.load(path)
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