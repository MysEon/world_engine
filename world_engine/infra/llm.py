from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

from .llm_settings import EffectiveLLM, LLMSettingsStore
from .logging import get_logger

logger = get_logger("llm")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.MULTILINE)


class LLMError(RuntimeError):
    pass


def _sanitize(text: str, secret: str | None) -> str:
    if secret and secret in text:
        text = text.replace(secret, "***")
    return text


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Tolerantly parse an LLM JSON reply (empty, fenced, or wrapped in prose)."""
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


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

    Resilience: retries on transient failures, empty replies, and non-JSON
    responses; JSON replies are parsed tolerantly (markdown fences / prose).
    """

    def __init__(self, store: LLMSettingsStore, max_retries: int = 2, retry_delay: float = 0.5):
        self.store = store
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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

        last_error = "空响应"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
            except Exception as exc:  # noqa: BLE001
                last_error = _sanitize(str(exc), eff.api_key)
                logger.error("llm_error", provider=eff.provider, model=eff.model, attempt=attempt + 1, error=last_error)
                if attempt == self.max_retries:
                    raise LLMError(f"LLM 调用失败（{eff.provider}）：{last_error}") from exc
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (attempt + 1))
        raise LLMError(f"LLM 连续 {self.max_retries + 1} 次返回空响应（{eff.provider}）")

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        last_detail = ""
        for attempt in range(self.max_retries + 1):
            call_kwargs = dict(kwargs)
            if attempt > 0 and call_kwargs.get("max_tokens"):
                call_kwargs["max_tokens"] = int(call_kwargs["max_tokens"] * (1 + 0.5 * attempt))
            text = self.chat(messages, json_mode=True, **call_kwargs)
            data = _try_parse_json(text)
            if data is not None:
                return data
            last_detail = text[:200] or "(空响应)"
            logger.warning("llm_json_retry", attempt=attempt + 1, snippet=last_detail)
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (attempt + 1))
        raise LLMError(f"LLM 连续 {self.max_retries + 1} 次未返回有效 JSON：{last_detail}")
