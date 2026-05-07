import os
import json
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class NavigatorLLM:
    """Minimal Navigator HTTP LLM client wrapper.

    This provides a small compatibility surface used by the project:
    - `invoke(prompt)` returning an object with `.content` (string)
    - `with_structured_output(model_cls)` returning an object with
      `invoke(prompt)` that parses JSON into the provided Pydantic model.

        The API key, endpoint, and model are read from environment variables with
        no hardcoded secrets or model defaults:
      - `NAVIGATOR_API_KEY`
      - `NAVIGATOR_API_ENDPOINT`
      - `NAVIGATOR_MODEL`
    """

    def __init__(self, temperature: Optional[float] = None) -> None:
        self.api_key = os.getenv("NAVIGATOR_API_KEY")
        if not self.api_key:
            raise ValueError("NAVIGATOR_API_KEY is required.")

        base_url = os.getenv("NAVIGATOR_API_ENDPOINT")
        if not base_url:
            raise ValueError("NAVIGATOR_API_ENDPOINT is required.")
        # Remove /v1/chat/completions if present to get base URL
        if base_url.endswith("/v1/chat/completions"):
            base_url = base_url.replace("/v1/chat/completions", "")
        self.model = os.getenv("NAVIGATOR_MODEL")
        if not self.model:
            raise ValueError("NAVIGATOR_MODEL is required.")

        self.temperature = 0.0 if temperature is None else temperature
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def invoke(self, prompt: str) -> LLMResponse:
        content = self._call_api(prompt)
        return LLMResponse(content=content)

    def with_structured_output(self, model_cls):
        return StructuredNavigatorLLM(self, model_cls)


class StructuredNavigatorLLM:
    def __init__(self, base_llm: NavigatorLLM, model_cls) -> None:
        self.base = base_llm
        self.model_cls = model_cls

    def invoke(self, prompt: str):
        response = self.base.client.chat.completions.parse(
            model=self.base.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.base.temperature,
            response_format=self.model_cls,
        )
        return response.choices[0].message.parsed


def get_llm() -> NavigatorLLM:
    return NavigatorLLM()
