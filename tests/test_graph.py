from pathlib import Path

from world_engine.director import DirectorEvent, apply_director_event, memories_for_director_event
from world_engine.graph import WorldEngine
from world_engine.world_state import Character, WorldState


class FakeLLM:
    def __init__(self, with_memories: bool = True):
        self.with_memories = with_memories

    def chat_json(self, messages, **kwargs):
        return {
            "thought": "我暗中打量着眼前的人。",
            "speech": "这位兄台，面生得很。",
            "action": "端起茶碗，不动声色。",
            "memories": (
                [{"content": "茶馆里来了个生面孔", "importance": 6}]
                if self.with_memories
                else []
            ),
        }

    def chat(self, messages, **kwargs):
        return "茶馆里，一个生面孔走了进来，气氛微微收紧。"


def make_engine(tmp_path: Path, llm: FakeLLM | None = None) -> WorldEngine:
    from world_engine.config import EngineSettings, LLMSettings, Settings

    settings = Settings(
        llm=LLMSettings(provider="fake", model="fake", api_key_env="", base_url="http://x"),
        engine=EngineSettings(db_path=tmp_path / "test.db"),
    )
    engine = WorldEngine(settings, "test-world")
    engine.llm = llm or FakeLLM()  # type: ignore[assignment]
    engine.load_world(
        {
            "name": "测试茶馆",
            "setting": "一处江湖茶馆。",
            "locations": ["大堂"],
            "characters": [
                {"name": "沈雁", "role": "剑客", "persona": {"core": "寡言"}},
                {"name": "阿蛮", "role": "少年", "persona": {"core": "热血"}},
            ],
        }
    )
    return engine


def test_full_step_without_director(tmp_path: Path):
    engine = make_engine(tmp_path)
    result = engine.step()

    assert result.turn == 1
    assert "生面孔" in result.narrative
    assert engine.world.turn == 1
    assert len(engine.world.narrative) == 1
    assert engine.store.memories("test-world", "沈雁")[0].content == "茶馆里来了个生面孔"


def test_step_with_director_event(tmp_path: Path):
    engine = make_engine(tmp_path, FakeLLM(with_memories=False))
    event = DirectorEvent(content="一只信鸽撞进茶馆，翅膀上绑着血染的竹筒。", targets=["沈雁"])
    result = engine.step(director_event=event)

    assert result.turn == 1
    memories = engine.store.memories("test-world", "沈雁")
    assert any("信鸽" in m.content for m in memories)
    assert engine.store.memories("test-world", "阿蛮") == []


def test_director_event_helpers():
    state = WorldState(name="w", characters=[Character(name="甲"), Character(name="乙")])
    event = DirectorEvent(content="地震了", targets=["甲"])
    events = apply_director_event(state, event, turn=5)
    assert events[0].kind == "director"
    memories = memories_for_director_event(state, event, turn=5)
    assert [m.character for m in memories] == ["甲"]
