"""Regression tests for the bugs fixed in the review round.

Each test pins one previously broken behaviour:
- config with an absolute db_path no longer crashes on server.default_preset
- only deepseek / OpenAI-compatible providers are supported
- switching provider at runtime resets stale base_url/model
- unknown worlds are 404 and are NOT silently created
- a step's event log only contains that turn's events
- export does not silently plan a novel (no LLM side effect on GET)
- re-planning refuses to overwrite written chapters unless forced
- a failed chapter write keeps the chapters finished so far
- CORS preflight from the Vite dev origin is allowed
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from test_api import FakeLLM, make_client

from world_engine.infra.config import PROVIDER_DEFAULTS, LLMSettings, load_settings
from world_engine.infra.llm import LLMError
from world_engine.infra.llm_settings import (
    LLMSettingsStore,
    RuntimeLLMConfig,
    build_effective,
)

DEEPSEEK_LLM = {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "api_key_env": "DEEPSEEK_API_KEY",
}


def _write_config(tmp_path: Path, llm: dict | None = None, **overrides) -> Path:
    data = {
        "llm": dict(llm or DEEPSEEK_LLM),
        "engine": {"db_path": str(tmp_path / "world.db")},
        "server": {"default_preset": "presets/tavern.yaml"},
    }
    data.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class TwoChapterPlannerLLM(FakeLLM):
    """FakeLLM with a two-chapter outline so multi-chapter writes can be tested."""

    def chat_json(self, messages, **kwargs):
        content = messages[-1]["content"]
        if "原始事件流" in content:
            return {
                "title": "荒山茶馆",
                "arc": "测试用两章大纲。",
                "chapters": [
                    {"number": 1, "title": "茶香初会", "summary": "相遇", "start_turn": 1, "end_turn": 1},
                    {"number": 2, "title": "夜谈", "summary": "夜谈", "start_turn": 1, "end_turn": 1},
                ],
            }
        return super().chat_json(messages, **kwargs)


class FailOnSecondChapterLLM(TwoChapterPlannerLLM):
    """Fails whenever a prompt for chapter two shows up (order-independent)."""

    def chat(self, messages, **kwargs):
        text = "\n".join(m.get("content", "") for m in messages)
        if "夜谈" in text:
            raise LLMError("simulated failure while writing chapter 2")
        return super().chat(messages, **kwargs)


# -- config ------------------------------------------------------------------


def test_absolute_db_path_does_not_crash_on_relative_preset(tmp_path: Path):
    config = _write_config(tmp_path)
    settings = load_settings(config)

    assert settings.engine.db_path == tmp_path / "world.db"
    assert settings.server.default_preset == str(tmp_path / "presets" / "tavern.yaml")


def test_load_settings_rejects_unsupported_provider(tmp_path: Path):
    config = _write_config(tmp_path, llm={"provider": "anthropic"})

    with pytest.raises(ValueError, match="deepseek|openai|不支持"):
        load_settings(config)


def test_only_deepseek_and_openai_providers_are_supported():
    assert set(PROVIDER_DEFAULTS) == {"deepseek", "openai"}


# -- llm settings resolution --------------------------------------------------


def test_build_effective_switching_provider_uses_new_provider_defaults():
    defaults = LLMSettings(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    )
    eff = build_effective(RuntimeLLMConfig(provider="openai"), defaults, env={})
    assert eff.base_url == PROVIDER_DEFAULTS["openai"]["base_url"]
    assert eff.model == PROVIDER_DEFAULTS["openai"]["model"]

    custom = build_effective(
        RuntimeLLMConfig(provider="openai", base_url="http://localhost:1234/v1", model="local-model"),
        defaults,
        env={},
    )
    assert custom.base_url == "http://localhost:1234/v1"
    assert custom.model == "local-model"


def test_switching_provider_drops_stale_base_url_and_model(tmp_path: Path):
    defaults = LLMSettings(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    )
    store = LLMSettingsStore(tmp_path / "llm_settings.json", defaults)
    store.update(RuntimeLLMConfig(provider="deepseek", base_url="https://api.deepseek.com", model="deepseek-chat"))

    public = store.update(RuntimeLLMConfig(provider="openai"))

    assert public["base_url"] == PROVIDER_DEFAULTS["openai"]["base_url"]
    assert public["model"] == PROVIDER_DEFAULTS["openai"]["model"]


def test_settings_put_accepts_openai_compatible_endpoint(tmp_path: Path):
    client = make_client(tmp_path)
    resp = client.put(
        "/api/settings/llm",
        json={"provider": "openai", "base_url": "http://localhost:1234/v1", "model": "local-model"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "openai"
    assert body["base_url"] == "http://localhost:1234/v1"
    assert body["model"] == "local-model"
    assert body["api_key_set"] is False


def test_settings_put_rejects_unsupported_provider(tmp_path: Path):
    client = make_client(tmp_path)
    resp = client.put("/api/settings/llm", json={"provider": "anthropic"})
    assert resp.status_code == 400


# -- world lifecycle ----------------------------------------------------------


def test_get_unknown_world_is_404_and_not_created(tmp_path: Path):
    client = make_client(tmp_path)

    resp = client.get("/api/worlds/nope")

    assert resp.status_code == 404
    assert client.get("/api/worlds").json() == []


def test_history_of_unknown_world_is_404(tmp_path: Path):
    client = make_client(tmp_path)

    assert client.get("/api/worlds/nope/history").status_code == 404


def test_step_events_only_contain_current_turn(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})

    first = client.post("/api/worlds/w1/step", json={}).json()
    second = client.post("/api/worlds/w1/step", json={}).json()

    history = client.get("/api/worlds/w1/history", params={"limit": 100}).json()
    current = [e["content"] for e in history if e["turn"] == 2]

    assert first["events"]
    assert second["events"] == current


# -- novel pipeline -----------------------------------------------------------


def test_export_without_plan_is_404_instead_of_planning(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})

    assert client.get("/api/worlds/w1/novel/export").status_code == 404


def test_replan_requires_force_once_chapters_are_written(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})
    client.post("/api/worlds/w1/step", json={})

    assert client.post("/api/worlds/w1/novel/plan").status_code == 200
    assert client.post("/api/worlds/w1/novel/write").status_code == 200

    assert client.post("/api/worlds/w1/novel/plan").status_code == 409
    forced = client.post("/api/worlds/w1/novel/plan", json={"force": True})
    assert forced.status_code == 200


def test_failed_chapter_write_keeps_finished_chapters(tmp_path: Path):
    client = make_client(tmp_path, llm=FailOnSecondChapterLLM())
    client.post("/api/worlds", json={"world_id": "w1"})
    client.post("/api/worlds/w1/step", json={})
    assert client.post("/api/worlds/w1/novel/plan").status_code == 200

    resp = client.post("/api/worlds/w1/novel/write", json={"chapters": [1, 2]})
    assert resp.status_code == 502

    novel = client.get("/api/worlds/w1/novel").json()
    assert novel["chapters"][0]["content"]
    assert novel["chapters"][0]["status"] == "edited"


# -- CORS ---------------------------------------------------------------------


def test_cors_preflight_allows_vite_dev_origin(tmp_path: Path):
    client = make_client(tmp_path)

    resp = client.options(
        "/api/worlds",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


# -- legacy runtime config ----------------------------------------------------


def test_legacy_unsupported_runtime_provider_is_ignored(tmp_path: Path):
    path = tmp_path / "llm_settings.json"
    path.write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com",
                "model": "claude-3-5-haiku",
            }
        ),
        encoding="utf-8",
    )
    defaults = LLMSettings(
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
    )

    store = LLMSettingsStore(path, defaults)

    eff = store.effective()
    assert eff.provider == "deepseek"
    assert eff.base_url == "https://api.deepseek.com"
    assert eff.model == "deepseek-chat"
    assert store.public()["provider"] == "deepseek"
