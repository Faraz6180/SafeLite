"""OpenRouter provider implementation using the official HTTP API."""

from __future__ import annotations

import json
from typing import Any

from planner.models import ActionPlan

from .base import BaseLLMProvider, LLMProviderError
from .config import LLMConfig
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser


class OpenRouterProvider(BaseLLMProvider):
    """Generate plans using the OpenRouter chat completions API."""

    def __init__(self, config: LLMConfig | None = None, api_key: str | None = None, http_client: Any | None = None) -> None:
        cfg = config or LLMConfig(provider="openrouter")
        cfg.api_key = api_key or cfg.api_key
        super().__init__(cfg)
        self.http_client = http_client or self._default_http_client()
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()

    def generate_plan(self, instruction: str) -> ActionPlan:
        if not self.config.api_key:
            raise LLMProviderError("OpenRouter API key is missing.")

        prompt = self.prompt_builder.build(instruction)
        payload = {
            "model": self.config.model_name or "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "developer", "content": prompt.developer_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = self.http_client.post_json(
                "https://openrouter.ai/api/v1/chat/completions",
                {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                payload,
            )
        except TimeoutError as exc:
            raise LLMProviderError("OpenRouter request timed out.") from exc
        except Exception as exc:
            raise LLMProviderError(f"OpenRouter request failed: {exc}") from exc

        if isinstance(response, dict) and "error" in response:
            raise LLMProviderError(f"OpenRouter provider error: {response['error']}")

        try:
            content = response["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LLMProviderError("OpenRouter returned an unexpected response shape.") from exc

        return self.parser.parse(content)

    def health_check(self) -> bool:
        if not self.config.api_key:
            return False
        try:
            self.http_client.post_json(
                "https://openrouter.ai/api/v1/chat/completions",
                {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                {"model": self.config.model_name or "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            )
        except Exception:
            return False
        return True

    def validate_response(self, payload: Any) -> ActionPlan:
        return self.parser.parse(payload)

    class _DefaultHTTPClient:
        def post_json(self, url: str, headers: dict[str, str], payload: dict[str, object]) -> object:
            import urllib.request
            import urllib.error

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                return {"error": {"message": exc.read().decode("utf-8")}}
            except urllib.error.URLError as exc:
                raise TimeoutError(str(exc.reason)) from exc

    def _default_http_client(self) -> Any:
        return self._DefaultHTTPClient()
