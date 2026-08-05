from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel

from .config import LLMSettings, PROVIDER_DEFAULTS


class RuntimeLLMConfig(BaseModel):
    """Overrides saved at runtime. Fields that stay None fall back to defaults."""

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class EffectiveLLM(BaseModel):
    """The fully-resolved LLM configuration used for a call."""

    provider: str
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.8
    max_tokens: int = 1024


def build_effective(
    cfg: RuntimeLLMConfig,
    defaults: LLMSettings,
    env: Mapping[str, str] | None = None,
) -> EffectiveLLM:
    """Resolve priority: runtime overrides > provider defaults > config defaults > env key."""
    env = env if env is not None else os.environ
    provider = cfg.provider or defaults.provider
    base_url = cfg.base_url or defaults.base_url
    model = cfg.model or defaults.model
    env_name = defaults.api_key_env
    if provider in PROVIDER_DEFAULTS:
        pdef = PROVIDER_DEFAULTS[provider]
        base_url = base_url or pdef["base_url"]
        model = model or pdef["model"]
        env_name = pdef.get("api_key_env", env_name)
    api_key = cfg.api_key or env.get(env_name)
    return EffectiveLLM(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
    )


class LLMSettingsStore:
    """Runtime LLM configuration: persists to disk, survives restarts.

    The api_key is stored on disk for the user's own process but is never
    returned through the API (callers must use public()/effective()).
    """

    def __init__(self, path: Path, defaults: LLMSettings):
        self.path = path
        self.defaults = defaults
        self._lock = threading.RLock()
        self._config = self._load()

    def _load(self) -> RuntimeLLMConfig:
        if not self.path.exists():
            return RuntimeLLMConfig()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return RuntimeLLMConfig.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return RuntimeLLMConfig()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            self._config.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def update(self, patch: RuntimeLLMConfig) -> dict:
        """Merge the patch into the current runtime config and persist.

        None fields keep their current value; an empty api_key clears it.
        Returns the public view (no secret).
        """
        with self._lock:
            new = RuntimeLLMConfig(
                provider=patch.provider if patch.provider is not None else self._config.provider,
                base_url=patch.base_url if patch.base_url is not None else self._config.base_url,
                model=patch.model if patch.model is not None else self._config.model,
                api_key=self._config.api_key,
            )
            if patch.api_key is not None:
                new.api_key = patch.api_key or None
            self._config = new
            self._save()
        return self.public()

    def effective(self) -> EffectiveLLM:
        with self._lock:
            return build_effective(self._config, self.defaults)

    def public(self) -> dict:
        """Safe view for the API: never contains the api_key itself."""
        eff = self.effective()
        return {
            "provider": eff.provider,
            "base_url": eff.base_url,
            "model": eff.model,
            "api_key_set": bool(eff.api_key),
        }
