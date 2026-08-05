from pathlib import Path

from world_engine.core.models import Character, MemoryRecord, WorldEvent, WorldState
from world_engine.core.storage import WorldStore


def test_store_roundtrip(tmp_path: Path):
    store = WorldStore(tmp_path / "test.db")
    state = WorldState(
        name="测试世界",
        setting="一片荒原",
        locations=["茶棚"],
        characters=[Character(name="张三", role="旅人")],
    )
    store.save("w1", state)

    loaded = store.load("w1")
    assert loaded is not None
    assert loaded.name == "测试世界"
    assert loaded.characters[0].name == "张三"


def test_events_and_memories(tmp_path: Path):
    store = WorldStore(tmp_path / "test.db")
    store.add_event("w1", WorldEvent(turn=1, kind="speech", actor="张三", content="你好"))
    store.add_memories(
        "w1",
        [MemoryRecord(character="张三", content="遇到了李四", importance=6, turn=1)],
    )

    events = store.events("w1")
    assert events[0].actor == "张三"
    assert store.memories("w1", "张三")[0].content == "遇到了李四"
    assert store.memories("w1", "李四") == []
