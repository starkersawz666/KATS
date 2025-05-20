import os, json, hashlib
from typing import Callable, Any

class CacheManager:
    def __init__(self, cache_root: str = ".cache"):
        self.cache_root = cache_root
        os.makedirs(cache_root, exist_ok=True)

    def _step_dir(self, step: str) -> str:
        d = os.path.join(self.cache_root, step)
        os.makedirs(d, exist_ok=True)
        return d

    def fingerprint_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def fingerprint_file(self, path: str) -> str:
        with open(path, "rb") as f:
            return self.fingerprint_bytes(f.read())

    def fingerprint_text(self, text: str) -> str:
        return self.fingerprint_bytes(text.encode("utf-8"))

    def get_cache_path(self, step: str, fingerprint: str, ext: str) -> str:
        return os.path.join(self._step_dir(step), f"{fingerprint}{ext}")

    def load_if_exists(self,
                       step: str,
                       fingerprint: str,
                       loader: Callable[[str], Any],
                       ext: str) -> Any | None:
        path = self.get_cache_path(step, fingerprint, ext)
        if os.path.exists(path):
            return loader(path)
        return None

    def save(self,
             step: str,
             fingerprint: str,
             data: Any,
             saver: Callable[[str, Any], None],
             ext: str) -> None:
        path = self.get_cache_path(step, fingerprint, ext)
        saver(path, data)

    def load_or_compute(self,
                        step: str,
                        fingerprint: str,
                        compute_fn: Callable[[], Any],
                        loader: Callable[[str], Any],
                        saver: Callable[[str, Any], None],
                        ext: str) -> Any:
        cached = self.load_if_exists(step, fingerprint, loader, ext)
        if cached is not None:
            return cached
        result = compute_fn()
        self.save(step, fingerprint, result, saver, ext)
        return result
