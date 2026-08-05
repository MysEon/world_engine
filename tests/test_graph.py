from pathlib import Path

from world_engine.core.models import Character, DirectorEvent, WorldState
from world_engine.narrative.director import apply_director_event, memories_for_director_event
from world_engine.narrative.engine import WorldEngine


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
    from world_engine.infra.config import EngineSettings, LLMSettings, Settings

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


def test_write_time_dedup(tmp_path: Path):
    engine = make_engine(tmp_path)
    engine.step()
    engine.step()
    mems = engine.store.memories("test-world", "沈雁")
    assert sum(1 for m in mems if "生面孔" in m.content) == 1


def test_thoughts_persisted_and_returned(tmp_path: Path):
    engine = make_engine(tmp_path)
    result = engine.step()

    assert "沈雁" in result.thoughts
    assert result.thoughts["沈雁"]
    events = engine.store.events("test-world", limit=None)
    thought_events = [e for e in events if e.kind == "thought"]
    assert any(e.actor == "沈雁" for e in thought_events)


def test_no_cross_character_memory_leak(tmp_path: Path):
    from world_engine.core.models import MemoryRecord

    engine = make_engine(tmp_path, FakeLLM(with_memories=False))
    engine.store.add_memories(
        "test-world",
        [
            MemoryRecord(character="沈雁", content="只有沈雁知道：信鸽是冲我来的", importance=9, turn=0),
            MemoryRecord(character="阿蛮", content="只有阿蛮知道：师父的信要送往柳城", importance=9, turn=0),
        ],
    )

    given: dict[str, list[str]] = {}
    for name, agent in engine.agents.items():
        original_act = agent.act

        def act(name=name, agent=agent, original_act=original_act):
            def _inner(world, turn, perception, memories):
                given[name] = [m.content for m in memories]
                return original_act(world, turn, perception, memories)

            return _inner

        agent.act = act()

    engine.step()

    assert any("只有沈雁知道" in c for c in given["沈雁"])
    assert not any("只有阿蛮知道" in c for c in given["沈雁"])
    assert any("只有阿蛮知道" in c for c in given["阿蛮"])
    assert not any("只有沈雁知道" in c for c in given["阿蛮"])


class ReflectFakeLLM(FakeLLM):
    def chat_json(self, messages, **kwargs):
        if "记忆片段" in messages[-1]["content"]:
            return {"reflections": [{"content": "他对陌生人的警惕在加深", "importance": 9}]}
        return super().chat_json(messages, **kwargs)


def make_reflecting_engine(tmp_path: Path):
    from world_engine.infra.config import EngineSettings, LLMSettings, Settings

    settings = Settings(
        llm=LLMSettings(provider="fake", model="fake", api_key_env="", base_url="http://x"),
        engine=EngineSettings(db_path=tmp_path / "test.db", reflect_every=1),
    )
    engine = WorldEngine(settings, "test-world")
    engine.llm = ReflectFakeLLM(with_memories=True)  # type: ignore[assignment]
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


def test_reflection_and_archiving(tmp_path: Path):
    from world_engine.core.models import MemoryRecord

    engine = make_reflecting_engine(tmp_path)
    engine.store.add_memories(
        "test-world",
        [MemoryRecord(character="沈雁", content="屋檐下的一只旧瓦罐", importance=3, turn=0)],
    )
    engine.step()

    all_mems = engine.store.memories("test-world", "沈雁", include_archived=True)
    assert any(m.kind == "reflection" and "警惕" in m.content for m in all_mems)
    assert any(m.archived and m.importance == 3 for m in all_mems)
    active = engine.store.memories("test-world", "沈雁")
    assert not any(m.importance == 3 for m in active)
