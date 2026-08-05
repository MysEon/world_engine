from __future__ import annotations

from typing import Callable

from ..core.models import Novel, WorldEvent, WorldState
from ..core.storage import WorldStore
from ..infra.llm import LLMClient
from ..infra.logging import get_logger
from .editor import ChapterEditor
from .planner import NovelPlanner
from .writer import ChapterWriter

logger = get_logger("novel")


class NovelPipeline:
    """Orchestrates the novel production pipeline: plan -> write -> edit -> export.

    Reads only the world's structured chronicle (core.models.WorldEvent), never the
    per-turn narration, so fiction quality is owned entirely by this layer.
    """

    def __init__(
        self,
        llm_provider: Callable[[], LLMClient],
        planner_max_events: int = 80,
        chapter_max_tokens: int = 2500,
    ):
        self.planner = NovelPlanner(llm_provider, max_events=planner_max_events)
        self.writer = ChapterWriter(llm_provider, max_tokens=chapter_max_tokens)
        self.editor = ChapterEditor(llm_provider)

    def plan(self, world: WorldState, store: WorldStore) -> Novel:
        events = store.events(world.world_id, limit=None)
        novel = self.planner.plan(world, events)
        store.save_novel(world.world_id, novel)
        logger.info(
            "novel_plan",
            world_id=world.world_id,
            title=novel.title,
            chapters=len(novel.chapters),
        )
        return novel

    def write(self, world: WorldState, store: WorldStore, novel: Novel, numbers: list[int] | None = None) -> Novel:
        events = store.events(world.world_id, limit=None)
        targets = [c for c in novel.chapters if numbers is None or c.number in numbers]
        for chapter in targets:
            draft = self.writer.write(world, chapter, events)
            polished = self.editor.edit(world, chapter, events, draft)
            chapter.content = polished
            chapter.status = "edited"
            logger.info(
                "novel_chapter",
                world_id=world.world_id,
                chapter=chapter.number,
                title=chapter.title,
                chars=len(polished),
            )
        store.save_novel(world.world_id, novel)
        return novel

    def export(self, novel: Novel) -> str:
        parts = [
            f"# {novel.title or '未命名之书'}",
            "",
            f"> {novel.arc}" if novel.arc else "",
            "",
            "## 目录",
            "",
        ]
        for chapter in novel.chapters:
            parts.append(f"{chapter.number}. {chapter.title}")
        parts.append("")
        parts.append("---")
        parts.append("")
        for chapter in novel.chapters:
            parts.append(f"## 第{chapter.number}章 {chapter.title}")
            parts.append("")
            parts.append(chapter.content if chapter.content else "*（本章尚未撰写）*")
            parts.append("")
        return "\n".join(parts)
