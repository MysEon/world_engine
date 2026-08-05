from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import LLMSettings, Settings, api_key_for

DEFAULT_SYSTEM = "You are a helpful assistant."


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper over the OpenAI SDK with provider-switchable base_url."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            cfg = self.settings.llm
            kwargs: dict[str, Any] = {"base_url": cfg.base_url}
            key = api_key_for(self.settings)
            if key:
                kwargs["api_key"] = key
            else:
                kwargs["api_key"] = "not-needed"
            self._client = OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        cfg = self.settings.llm
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else cfg.temperature,
            "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"LLM call failed ({cfg.provider}): {exc}") from exc
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        text = self.chat(messages, json_mode=True, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            cleaned = text.strip().strip("`").strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                raise LLMError(f"LLM returned non-JSON in json_mode: {text[:500]}") from exc
