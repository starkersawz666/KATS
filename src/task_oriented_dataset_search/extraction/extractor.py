import logging
import os
from typing import Dict
from task_oriented_dataset_search.extraction.client import BaseLLMClient
from task_oriented_dataset_search.extraction.interface import (
    BaseExtractor,
    ExtractionResult,
)
import re
import json

logger = logging.getLogger(__name__)


class StandardExtractor(BaseExtractor):
    def __init__(self, client: BaseLLMClient):
        self.client = client
        self._prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")

    def _read_prompt(self, relpath: str) -> str:
        path = os.path.join(self._prompts_dir, os.path.basename(relpath))
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _extract_json_from_response_bracket(response: str, del_comments=True):
        start_index = end_index = -1
        if "[" in response:
            start_index = response.find("[")
        elif "{" in response:
            start_index = response.find("{")
        else:
            start_index = response.find("```json") + 8
        if "]" in response:
            end_index = response.rfind("]")
        elif "}" in response:
            end_index = response.rfind("}")
        else:
            end_index = response.rfind("```") - 1
        if start_index != -1 and end_index != -1:
            response = response[start_index : end_index + 1].replace("\_", "_")
        if del_comments:
            return json.loads(
                re.sub(r"\/\/.*$", "", response, flags=re.MULTILINE), strict=False
            )
        else:
            return json.loads(response, strict=False)

    @staticmethod
    def _extract_json_from_response_brace(response: str, del_comments=True):
        start_index = end_index = -1
        if "{" in response:
            start_index = response.find("{")
        elif "[" in response:
            start_index = response.find("[")
        else:
            start_index = response.find("```json") + 8
        if "}" in response:
            end_index = response.rfind("}")
        elif "]" in response:
            end_index = response.rfind("]")
        else:
            end_index = response.rfind("```") - 1
        if start_index != -1 and end_index != -1:
            response = response[start_index : end_index + 1].replace("\_", "_")
        if del_comments:
            return json.loads(
                re.sub(r"\/\/.*$", "", response, flags=re.MULTILINE), strict=False
            )
        else:
            return json.loads(response, strict=False)

    @staticmethod
    def _extract_json_from_response(response: str):
        extract_funs = (
            (StandardExtractor._extract_json_from_response_bracket, False),
            (StandardExtractor._extract_json_from_response_brace, False),
            (StandardExtractor._extract_json_from_response_bracket, True),
            (StandardExtractor._extract_json_from_response_brace, True),
        )
        for fun in extract_funs:
            try:
                return fun[0](response, fun[1])
            except Exception as e:
                pass

    def extract(
        self,
        text: str,
        prompts: Dict[str, str] = {
            "dataset_existence": "dataset_existence_prompt.txt",
            "dataset_extraction": "dataset_extraction_prompt.txt",
            "reference_extraction": "reference_extraction_prompt.txt",
            "keywords_extraction": "keywords_extraction_prompt.txt",
        },
        **tpl_kwargs,
    ) -> ExtractionResult:
        messages = [
            {
                "role": "system",
                "content": self._read_prompt(prompts["dataset_existence"]),
            },
            {"role": "user", "content": text},
        ]
        response_dexi = self.client.chat(messages)
        assistant_msg_dexi = response_dexi.choices[0].message
        messages.append(
            {
                "role": assistant_msg_dexi.role or "assistant",
                "content": assistant_msg_dexi.content,
            }
        )

        json_dexi = self._extract_json_from_response(assistant_msg_dexi.content)
        flag_dataset_used = list(json_dexi.values())[0]
        if not flag_dataset_used:
            return {"datasets_used": False, "datasets": []}

        messages.append(
            {
                "role": "user",
                "content": self._read_prompt(prompts["dataset_extraction"]),
            }
        )
        response_dext = self.client.chat(messages)
        assistant_msg_dext = response_dext.choices[0].message
        messages.append(
            {
                "role": assistant_msg_dext.role or "assistant",
                "content": assistant_msg_dext.content,
            }
        )
        json_dext = self._extract_json_from_response(assistant_msg_dext.content)

        result: ExtractionResult = {"datasets_used": True, "datasets": []}
        for dataset in json_dext:
            result["datasets"].append(
                {
                    "title": dataset.get("title"),
                    "description": dataset.get("description"),
                    "task_description": dataset.get("task"),
                    "link": dataset.get("link"),
                }
            )

        # for dataset in result["datasets"]:
        #     template = self._read_prompt(prompts["reference_extraction"])
        #     prompt = template.replace("__{dataset_name}__", dataset["title"])
        #     messages.append({"role": "user", "content": prompt})
        #     response_ref = self.client.chat(messages)
        #     assistant_msg_ref = response_ref.choices[0].message
        #     messages.append(
        #         {
        #             "role": assistant_msg_ref.role or "assistant",
        #             "content": assistant_msg_ref.content,
        #         }
        #     )
        #     json_ref = self._extract_json_from_response(assistant_msg_ref.content)
        #     dataset["reference"] = json_ref.get("reference")

        for dataset in result["datasets"]:
            template = self._read_prompt(prompts["keywords_extraction"])
            keywords_messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": dataset["task_description"]},
            ]
            response_kext = self.client.chat(keywords_messages)
            assistant_kext = response_kext.choices[0].message.content
            keywords = [w.strip() for w in assistant_kext.split(",") if w.strip()]
            if keywords[-1].endswith("."):
                keywords[-1] = keywords[-1][0:-1]
            dataset["task"] = keywords

        return result
