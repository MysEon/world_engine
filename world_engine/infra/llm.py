from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from .llm_settings import EffectiveLLM, LLMSettingsStore
from .logging import get_logger

logger = get_logger("llm")


class LLMError(RuntimeError):
    pass


def _sanitize(text: str, secret: str | None) -> str:
    if secret and secret in text:
        text = text.replace(secret, "***")
    return text


def test_llm_connection(eff: EffectiveLLM) -> tuple[str, float]:
    """Send a minimal chat completion against the given config. Returns (model, latency_ms)."""
    client = OpenAI(base_url=eff.base_url, api_key=eff.api_key or "not-needed")
    start = time.perf_counter()
    client.chat.completions.create(
        model=eff.model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=8,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return eff.model, latency_ms


class LLMClient:
    """Provider-switchable LLM client that re-resolves config on every call.

    Reads the effective config from an LLMSettingsStore, so runtime config
    changes take effect on the next call without a restart (hot reload).
    """

    def __init__(self, store: LLMSettingsStore):
        self.store = store
        self._client: OpenAI | None = None
        self._client_key: tuple[str, str | None] | None = None

    @property
    def effective(self) -> EffectiveLLM:
        return self.store.effective()

    @property
    def client(self) -> OpenAI:
        eff = self.effective
        key = (eff.base_url, eff.api_key)
        if self._client_key != key:
            self._client = OpenAI(base_url=eff.base_url, api_key=eff.api_key or "not-needed")
            self._client_key = key
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        eff = self.effective
        kwargs: dict[str, Any] = {
            "model": eff.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else eff.temperature,
            "max_tokens": max_tokens if max_tokens is not None else eff.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("llm_error", provider=eff.provider, model=eff.model, error=_sanitize(str(exc), eff.api_key))
            raise LLMError(f"LLM call failed ({eff.provider}): {_sanitize(str(exc), eff.api_key)}") from exc
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
