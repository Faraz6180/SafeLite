"""Groq provider implementation using the official HTTP API."""

from __future__ import annotations

import json
from typing import Any

from groq import Groq

from planner.models import ActionPlan

from .base import BaseLLMProvider, LLMProviderError
from .config import LLMConfig
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser


class GroqProvider(BaseLLMProvider):
    """Generate plans through the Groq chat completions API."""

    def __init__(self, config: LLMConfig | None = None, api_key: str | None = None, http_client: Any | None = None) -> None:
        cfg = config or LLMConfig(provider="groq")
        cfg.api_key = api_key or cfg.api_key
        super().__init__(cfg)
        self.http_client = http_client or self._default_http_client()
        self.client = http_client if http_client is not None else (Groq(api_key=self.config.api_key) if self.config.api_key else None)
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()

    def generate_plan(self, instruction: str) -> ActionPlan:
        if not self.config.api_key:
            raise LLMProviderError("Groq API key is missing.")

        prompt = self.prompt_builder.build(instruction)
        payload = {
            "model": self.config.model_name or "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "developer", "content": prompt.developer_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

        try:
            if self.client is None:
                raise LLMProviderError("Groq client is not initialized.")
            if hasattr(self.client, "post_json"):
                response = self.client.post_json(
                    "https://api.groq.com/openai/v1/chat/completions",
                    {"Authorization": f"Bearer {self.config.api_key}"},
                    payload,
                )
                if isinstance(response, dict) and "error" in response:
                    error_message = response["error"]
                    if isinstance(error_message, dict):
                        error_message = error_message.get("message", str(error_message))
                    raise LLMProviderError(f"Groq provider error: {error_message}")
                try:
                    content = response["choices"][0]["message"]["content"]
                except Exception as exc:
                    raise LLMProviderError("Groq returned an unexpected response shape.") from exc
                return self.parser.parse(content)

            response = self.client.chat.completions.create(**payload)
            try:
                content = response.choices[0].message.content
            except Exception as exc:
                raise LLMProviderError("Groq returned an unexpected response shape.") from exc
            return self.parser.parse(content)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Groq request failed: {exc}") from exc

    def health_check(self) -> bool:
        if not self.config.api_key:
            raise LLMProviderError("Groq API key is missing.")
        try:
            if self.client is None:
                raise LLMProviderError("Groq client is not initialized.")
            if hasattr(self.client, "post_json"):
                response = self.client.post_json(
                    "https://api.groq.com/openai/v1/chat/completions",
                    {"Authorization": f"Bearer {self.config.api_key}"},
                    {"model": self.config.model_name or "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                )
                if isinstance(response, dict) and "error" in response:
                    error_message = response["error"]
                    if isinstance(error_message, dict):
                        error_message = error_message.get("message", str(error_message))
                    raise LLMProviderError(f"Groq provider error: {error_message}")
            else:
                self.client.chat.completions.create(
                    model=self.config.model_name or "llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5,
                )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Groq health check failed: {exc}") from exc
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
