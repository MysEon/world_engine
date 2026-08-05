from pathlib import Path

from world_engine.infra.config import EngineSettings, LLMSettings, Settings
from world_engine.narrative.engine import WorldEngine


class NovelFakeLLM:
    def chat_json(self, messages, **kwargs):
        content = messages[-1]["content"]
        if "原始事件流" in content:
            return {
                "title": "荒山茶馆",
                "arc": "从平静到暗流涌动的江湖开端",
                "chapters": [
                    {"number": 1, "title": "茶香初会", "summary": "三个过客在茶馆里各怀心思地相遇", "start_turn": 1, "end_turn": 2},
                    {"number": 2, "title": "信鸽夜来", "summary": "血染的信鸽带来了旧案线索", "start_turn": 3, "end_turn": 4},
                ],
            }
        return {"thought": "我暗中打量来人", "speech": "这位兄台面生得很。", "action": "端起茶碗。", "memories": []}

    def chat(self, messages, **kwargs):
        content = messages[-1]["content"]
        if "待润色的初稿" in content:
            return "（润色后）茶馆里雾气氤氲，三个过客各怀心事地落了座。"
        return "茶馆里雾气氤氲，一个生面孔推门而入，远处的山道上隐隐传来马蹄声。"


def make_engine(tmp_path: Path) -> WorldEngine:
    settings = Settings(
        llm=LLMSettings(provider="fake", model="fake", api_key_env="", base_url="http://x"),
        engine=EngineSettings(db_path=tmp_path / "test.db", reflect_every=99),
    )
    engine = WorldEngine(settings, "test-world")
    engine.llm = NovelFakeLLM()  # type: ignore[assignment]
    engine.load_world(
        {
            "name": "荒山茶馆",
            "setting": "一处江湖茶馆。",
            "locations": ["大堂"],
            "characters": [
                {"name": "沈雁", "role": "剑客", "persona": {"core": "寡言"}},
                {"name": "柳娘", "role": "老板娘", "persona": {"core": "精明"}},
            ],
        }
    )
    return engine


def run_four_turns(engine: WorldEngine) -> None:
    for _ in range(4):
        engine.step()


def test_plan_novel(tmp_path: Path):
    engine = make_engine(tmp_path)
    run_four_turns(engine)

    novel = engine.plan_novel()
    assert novel.title == "荒山茶馆"
    assert novel.arc
    assert [c.number for c in novel.chapters] == [1, 2]
    assert novel.chapters[0].start_turn == 1
    assert novel.chapters[-1].end_turn == 4
    assert engine.store.load_novel("test-world").title == "荒山茶馆"


def test_write_novel(tmp_path: Path):
    engine = make_engine(tmp_path)
    run_four_turns(engine)

    novel = engine.write_novel()
    assert all(c.content for c in novel.chapters)
    assert all(c.status == "edited" for c in novel.chapters)
    assert "润色后" in novel.chapters[0].content
    assert engine.store.load_novel("test-world").chapters[0].content


def test_write_specific_chapter(tmp_path: Path):
    engine = make_engine(tmp_path)
    run_four_turns(engine)
    engine.plan_novel()

    novel = engine.write_novel([1])
    written = [c.number for c in novel.chapters if c.content]
    unwritten = [c.number for c in novel.chapters if not c.content]
    assert written == [1]
    assert unwritten == [2]


def test_export_novel(tmp_path: Path):
    engine = make_engine(tmp_path)
    run_four_turns(engine)
    engine.write_novel()

    markdown = engine.export_novel()
    assert "# 荒山茶馆" in markdown
    assert "## 目录" in markdown
    assert "## 第1章 茶香初会" in markdown
    assert "（润色后）" in markdown
