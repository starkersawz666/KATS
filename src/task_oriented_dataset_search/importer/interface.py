from pathlib import Path


class BaseImporter:
    def import_file(self, json_path: Path) -> None:
        raise NotImplementedError()

    def import_all(self, cache_dir: Path) -> None:
        raise NotImplementedError()
