from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config.yaml")


class LLMSettings(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str | None = None
    temperature: float = 0.8
    max_tokens: int = 1024


class EngineSettings(BaseModel):
    max_turns: int = 50
    db_path: Path = Field(default_factory=lambda: Path("data/world.db"))
    memory_retrieval_top_k: int = 6


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)


PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini", "api_key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b", "api_key_env": ""},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
}


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    llm_cfg = data.get("llm", {}) or {}
    llm = LLMSettings(**llm_cfg)

    if not llm.base_url and llm.provider in PROVIDER_DEFAULTS:
        defaults = PROVIDER_DEFAULTS[llm.provider]
        llm.base_url = defaults["base_url"]
        if llm.model == "deepseek-chat" or llm_cfg.get("model") is None:
            llm.model = defaults["model"]
        if not llm.api_key_env or llm.api_key_env == "DEEPSEEK_API_KEY":
            llm.api_key_env = defaults["api_key_env"]

    if not llm.api_key_env:
        raise ValueError(f"provider '{llm.provider}' requires no API key but config is missing one; set llm.api_key_env")

    engine_cfg = data.get("engine", {}) or {}
    engine = EngineSettings(**engine_cfg)
    if not engine.db_path.is_absolute():
        base = path.resolve().parent
        engine.db_path = base / engine.db_path

    return Settings(llm=llm, engine=engine)


def api_key_for(settings: Settings) -> str | None:
    env = settings.llm.api_key_env
    if not env:
        return None
    return os.environ.get(env)
