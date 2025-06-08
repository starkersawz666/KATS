import logging
import math
import os

from tinydb import Query, TinyDB
from task_oriented_dataset_search.extraction.client import OpenAIClient
from task_oriented_dataset_search.search.searcher import Searcher


logger = logging.getLogger(__name__)


class QAEngine:
    def __init__(self, qa_client: OpenAIClient, searcher: Searcher, db_path: str):
        logger.debug("Initializing QAEngine")
        self.qa_client = qa_client
        self.searcher = searcher
        self.db_path = db_path
        self._prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")

    def _read_prompt(self, relpath: str) -> str:
        path = os.path.join(self._prompts_dir, os.path.basename(relpath))
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found at {path}")
            raise

    def answer(
        self,
        task_description: str,
        top_k_datasets: int = 5,
        search_rate: float = 2.0,
        initial_faiss_k: int = 2,
        similarity_threshold: float = 0.85,
        min_seed_similarity: float = 0.6,
        pagerank_alpha: float = 0.85,
    ) -> str:
        if search_rate < 1.0:
            logger.warning(
                f"Search rate is set to 1.0 since it is {search_rate}, which is less than 1.0."
            )
            search_rate = 1.0

        search_results = self.searcher.search(
            task_description,
            top_k_datasets=math.ceil(top_k_datasets * search_rate),
            initial_faiss_k=initial_faiss_k,
            similarity_threshold=similarity_threshold,
            min_seed_similarity=min_seed_similarity,
            pagerank_alpha=pagerank_alpha,
        )

        if not search_results:
            return "Sorry, I cannot find any related dataset for your question."

        db = TinyDB(self.db_path)
        tasks_tbl = db.table("tasks")
        TaskQ = Query()

        context_list = []
        for i, ds in enumerate(search_results):
            ds_id = ds.get("id")
            title = ds.get("title", "N/A")
            description = ds.get("description", "N/A")
            link = ds.get("link", "N/A")

            associated_tasks = tasks_tbl.search(TaskQ.dataset_id == ds_id)
            tasks_info = (
                "; ".join([t.get("task_description", "N/A") for t in associated_tasks])
                if associated_tasks
                else "N/A"
            )

            context_list.append(
                f"{i+1}. **Dataset**: {title}\n"
                f"   **Description**: {description}\n"
                f"   **Related Tasks**: {tasks_info}\n"
                f"   **Link**: {link}"
            )

        context_str = "\n\n".join(context_list)

        try:
            prompt_template = self._read_prompt("qa_prompt.txt")
        except FileNotFoundError:
            return (
                "Sorry, an error occurred while preparing the answer (prompt missing)."
            )

        prompt = prompt_template.format(
            top_k=top_k_datasets,
            context_str=context_str,
            task_description=task_description,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.qa_client.chat(messages)
            answer_content = response.choices[0].message.content
            return (
                answer_content
                if answer_content
                else "Sorry, I could not generate an answer."
            )
        except Exception as e:
            logger.error(f"QA LLM call failed: {e}")
            return "Sorry, an error occurred while generating the answer."
