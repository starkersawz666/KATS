import logging
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseLLMClient:

    def chat(self, messages: List[Dict[str, str]], **options) -> Dict[str, Any]:
        raise NotImplementedError()


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.1,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.temperature = temperature

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        temp = self.temperature if temperature is None else temperature
        logger.debug(
            f"Calling OpenAI chat with model: {self._model}, temperature: {temp}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
            **kwargs,
        )
        usage = response.usage
        if usage:
            logger.debug(f"OpenAI API call successful. Usage: {usage}")
        else:
            logger.debug("OpenAI API call successful.")
        return response
