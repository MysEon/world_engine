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
    reflect_every: int = 5
    memory_archive_cutoff: int = 4
    planner_max_events: int = 80
    chapter_max_tokens: int = 2500


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    default_preset: str = "presets/tavern.yaml"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    json_output: bool = False


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


#: The only supported providers: DeepSeek and any OpenAI-compatible endpoint
#: (LM Studio / Ollama / other compatible gateways) via a custom base_url.
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
}


def validate_provider(provider: str) -> None:
    """Raise ValueError for any provider outside PROVIDER_DEFAULTS."""
    if provider not in PROVIDER_DEFAULTS:
        supported = "、".join(PROVIDER_DEFAULTS)
        raise ValueError(f"不支持的 provider：'{provider}'；仅支持：{supported}")


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    llm_cfg = data.get("llm", {}) or {}
    llm = LLMSettings(**llm_cfg)
    validate_provider(llm.provider)

    defaults = PROVIDER_DEFAULTS[llm.provider]
    if not llm.base_url:
        llm.base_url = defaults["base_url"]
    if llm_cfg.get("model") is None:
        llm.model = defaults["model"]
    if not llm_cfg.get("api_key_env"):
        llm.api_key_env = defaults["api_key_env"]

    base = path.resolve().parent
    engine_cfg = data.get("engine", {}) or {}
    engine = EngineSettings(**engine_cfg)
    if not engine.db_path.is_absolute():
        engine.db_path = base / engine.db_path

    server_cfg = data.get("server", {}) or {}
    server = ServerSettings(**server_cfg)
    if not Path(server.default_preset).is_absolute():
        server.default_preset = str(base / server.default_preset)

    logging_cfg = data.get("logging", {}) or {}
    logging_settings = LoggingSettings(**logging_cfg)

    return Settings(llm=llm, engine=engine, server=server, logging=logging_settings)


def api_key_for(settings: Settings) -> str | None:
    env = settings.llm.api_key_env
    if not env:
        return None
    return os.environ.get(env)
