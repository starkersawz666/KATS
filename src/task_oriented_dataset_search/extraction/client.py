from typing import List, Dict, Any

from openai import OpenAI


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
        **kwargs
    ) -> Dict[str, Any]:
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            **kwargs
        )
