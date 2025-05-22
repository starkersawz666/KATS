import json
import uuid
from pathlib import Path

from tinydb import TinyDB, Query

from .interface import BaseImporter


class TinyDBImporter(BaseImporter):

    def __init__(self, db_path: Path | str = ".cache/tiny_db.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.db = TinyDB(self.db_path)

        self.docs_tbl = self.db.table("documents")
        self.datasets_tbl = self.db.table("datasets")
        self.tasks_tbl = self.db.table("tasks")

        self.DocQ = Query()
        self.DatasetQ = Query()
        self.TaskQ = Query()

    def import_file(self, json_path: Path) -> None:
        json_path = Path(json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not data:
            return
        if data.get("datasets_used") is False:
            return

        fingerprint = json_path.stem

        if not self.docs_tbl.contains(self.DocQ.id == fingerprint):
            self.docs_tbl.insert({"id": fingerprint})

        datasets = data.get("datasets")
        if not isinstance(datasets, list):
            return

        for ds in datasets:
            title = ds.get("title", "")
            description = ds.get("description", "")
            link = ds.get("link", "")
            # reference = ds.get("reference", "")
            dataset_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{fingerprint}:{title}").hex
            if not self.datasets_tbl.contains(self.DatasetQ.id == dataset_id):
                self.datasets_tbl.insert(
                    {
                        "id": dataset_id,
                        "document_id": fingerprint,
                        "title": title,
                        "description": description,
                        "link": link,
                        # "reference": reference,
                    }
                )

            task_description = ds.get("task_description", "")
            keywords = ds.get("task", "")
            task_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"{dataset_id}:{task_description}"
            ).hex
            if not self.tasks_tbl.contains(self.TaskQ.id == task_id):
                self.tasks_tbl.insert(
                    {
                        "id": task_id,
                        "dataset_id": dataset_id,
                        "task_description": task_description,
                        "keywords": keywords,
                    }
                )

    def import_all(self, cache_dir: Path | str) -> None:
        cache_dir = Path(cache_dir)
        for json_file in cache_dir.glob("*.json"):
            try:
                self.import_file(json_file)
            except Exception:
                pass
