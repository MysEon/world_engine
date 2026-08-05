from pathlib import Path

from fastapi.testclient import TestClient

from world_engine.api.app import create_app
from world_engine.infra.config import EngineSettings, LLMSettings, LoggingSettings, Settings


class FakeLLM:
    def chat_json(self, messages, **kwargs):
        return {"thought": "t", "speech": "s", "action": "a", "memories": []}

    def chat(self, messages, **kwargs):
        return "n"


def make_client(tmp_path: Path, api_key_env: str = "NO_SUCH_ENV") -> TestClient:
    settings = Settings(
        llm=LLMSettings(
            provider="custom",
            base_url="http://localhost:1/v1",
            model="local-model",
            api_key_env=api_key_env,
        ),
        engine=EngineSettings(db_path=tmp_path / "test.db", reflect_every=99),
        logging=LoggingSettings(level="CRITICAL"),
    )
    app = create_app(settings, llm=FakeLLM())
    return TestClient(app)


def test_get_default_settings(tmp_path: Path):
    client = make_client(tmp_path)
    resp = client.get("/api/settings/llm")
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "custom"
    assert data["api_key_set"] is False
    assert "api_key" not in data


def test_put_save_and_merge(tmp_path: Path):
    client = make_client(tmp_path)
    resp = client.put(
        "/api/settings/llm",
        json={"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key": "sk-secret"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "deepseek"
    assert data["model"] == "deepseek-chat"
    assert data["api_key_set"] is True
    assert "sk-secret" not in resp.text

    partial = client.put("/api/settings/llm", json={"model": "deepseek-reasoner"})
    assert partial.json()["model"] == "deepseek-reasoner"
    assert partial.json()["provider"] == "deepseek"


def test_api_key_never_returns_secret(tmp_path: Path):
    client = make_client(tmp_path)
    client.put("/api/settings/llm", json={"provider": "deepseek", "base_url": "x", "model": "m", "api_key": "sk-topsecret"})
    resp = client.get("/api/settings/llm")
    assert "sk-topsecret" not in resp.text
    assert resp.json()["api_key_set"] is True


def test_settings_persist_across_app_restart(tmp_path: Path):
    make_client(tmp_path).put("/api/settings/llm", json={"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-persist"})
    client2 = make_client(tmp_path)
    assert client2.get("/api/settings/llm").json()["model"] == "gpt-4o-mini"


def test_test_endpoint_success(tmp_path: Path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr("world_engine.api.routes.test_llm_connection", lambda eff: (eff.model, 42.5))
    resp = client.post(
        "/api/settings/llm/test",
        json={"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key": "sk-x"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "latency_ms": 42.5, "model": "deepseek-chat"}


def test_test_endpoint_failure_sanitizes_key(tmp_path: Path, monkeypatch):
    client = make_client(tmp_path)

    def boom(eff):
        raise RuntimeError("connection refused, key sk-secret123 rejected")

    monkeypatch.setattr("world_engine.api.routes.test_llm_connection", boom)
    resp = client.post(
        "/api/settings/llm/test",
        json={"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key": "sk-secret123"},
    )
    assert resp.status_code == 502
    assert "sk-secret123" not in resp.json()["detail"]
    assert "connection refused" in resp.json()["detail"]


def test_hot_reload_uses_runtime_config(tmp_path: Path):
    from world_engine.narrative.engine import WorldEngine

    settings = Settings(
        llm=LLMSettings(provider="custom", base_url="http://old:1/v1", model="old", api_key_env="NO_SUCH_ENV"),
        engine=EngineSettings(db_path=tmp_path / "test.db", reflect_every=99),
    )
    store_path = tmp_path / "llm_settings.json"
    from world_engine.infra.llm_settings import LLMSettingsStore, RuntimeLLMConfig

    store = LLMSettingsStore(store_path, settings.llm)
    engine = WorldEngine(settings, "w1", llm_store=store)

    captured = {}

    class RecordingLLM:
        def __init__(self, eff):
            self.eff = eff

        def chat(self, messages, **kwargs):
            captured["base_url"] = self.eff.base_url
            captured["model"] = self.eff.model
            return "ok"

        def chat_json(self, messages, **kwargs):
            captured["base_url"] = self.eff.base_url
            return {"thought": "t", "speech": "s", "action": "a", "memories": []}

    engine.llm = RecordingLLM(store.effective())
    engine.load_world(
        {
            "name": "w",
            "setting": "s",
            "characters": [{"name": "甲", "persona": {}}],
        }
    )
    engine.step()
    assert captured["base_url"] == "http://old:1/v1"

    store.update(RuntimeLLMConfig(provider="deepseek", base_url="https://api.deepseek.com", model="deepseek-chat", api_key="sk-hot"))
    engine.llm = RecordingLLM(store.effective())
    engine.step()
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["model"] == "deepseek-chat"
