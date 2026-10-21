from pathlib import Path

from fastapi.testclient import TestClient

from world_engine.api.app import create_app
from world_engine.infra.config import (
    EngineSettings,
    LLMSettings,
    LoggingSettings,
    ServerSettings,
    Settings,
)


class FakeLLM:
    def chat_json(self, messages, **kwargs):
        content = messages[-1]["content"]
        if "原始事件流" in content:
            return {
                "title": "荒山茶馆",
                "arc": "三个过客在茶馆相遇，一封信鸽引出一桩旧案。",
                "chapters": [
                    {"number": 1, "title": "茶香初会", "summary": "三个过客各怀心事地相遇", "start_turn": 1, "end_turn": 1},
                ],
            }
        return {
            "thought": "此人来路不明，我须留神。",
            "speech": "这位兄台，面生得很。",
            "action": "端起茶碗，不动声色。",
            "memories": [{"content": "茶馆里来了个生面孔", "importance": 6}],
        }

    def chat(self, messages, **kwargs):
        return "茶馆里雾气氤氲，一个生面孔推门而入。"


def make_client(tmp_path: Path, llm=None) -> TestClient:
    settings = Settings(
        llm=LLMSettings(provider="fake", model="fake", api_key_env="", base_url="http://x"),
        engine=EngineSettings(db_path=tmp_path / "test.db", reflect_every=99),
        server=ServerSettings(default_preset=str(tmp_path / "preset.yaml")),
        logging=LoggingSettings(level="CRITICAL"),
    )
    import yaml

    (tmp_path / "preset.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "荒山茶馆",
                "setting": "一处江湖茶馆。",
                "locations": ["大堂"],
                "characters": [
                    {"name": "沈雁", "role": "剑客", "persona": {"core": "寡言"}},
                    {"name": "柳娘", "role": "老板娘", "persona": {"core": "精明"}},
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    app = create_app(settings, llm=llm if llm is not None else FakeLLM())
    return TestClient(app)


def test_create_and_status(tmp_path: Path):
    client = make_client(tmp_path)
    resp = client.post("/api/worlds", json={"world_id": "w1"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "荒山茶馆"
    assert [c["name"] for c in data["characters"]] == ["沈雁", "柳娘"]

    status = client.get("/api/worlds/w1")
    assert status.json()["turn"] == 0


def test_step_returns_god_view(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})
    resp = client.post("/api/worlds/w1/step", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["turn"] == 1
    assert data["narrative"]
    names = [a["name"] for a in data["agents"]]
    assert "沈雁" in names
    shen = next(a for a in data["agents"] if a["name"] == "沈雁")
    assert shen["thought"]
    assert shen["speech"]
    assert shen["memories"]


def test_step_with_director_event(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})
    resp = client.post(
        "/api/worlds/w1/step",
        json={"event": {"content": "一只血染的信鸽撞进茶馆", "targets": ["沈雁"]}},
    )
    assert resp.status_code == 200
    assert resp.json()["turn"] == 1


def test_history(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})
    client.post("/api/worlds/w1/step", json={})
    resp = client.get("/api/worlds/w1/history")
    assert resp.status_code == 200
    kinds = {e["kind"] for e in resp.json()}
    assert {"thought", "speech", "action", "narration"} <= kinds


def test_novel_pipeline(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})
    client.post("/api/worlds/w1/step", json={})

    plan = client.post("/api/worlds/w1/novel/plan")
    assert plan.status_code == 200
    assert plan.json()["title"] == "荒山茶馆"

    write = client.post("/api/worlds/w1/novel/write")
    assert write.status_code == 200
    assert write.json()["chapters"][0]["content"]

    novel = client.get("/api/worlds/w1/novel")
    assert novel.status_code == 200
    assert novel.json()["chapters"][0]["status"] == "edited"

    export = client.get("/api/worlds/w1/novel/export")
    assert export.status_code == 200
    assert "## 第1章" in export.text


def test_novel_404_before_plan(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})
    resp = client.get("/api/worlds/w1/novel")
    assert resp.status_code == 404


def test_list_worlds(tmp_path: Path):
    client = make_client(tmp_path)
    client.post("/api/worlds", json={"world_id": "w1"})
    client.post("/api/worlds", json={"world_id": "w2"})
    assert sorted(client.get("/api/worlds").json()) == ["w1", "w2"]
