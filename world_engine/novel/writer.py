from __future__ import annotations

from typing import Callable

from ..core.models import Chapter, WorldEvent, WorldState
from ..infra.llm import LLMClient

WRITER_SYSTEM = """
You are a literary Chinese novelist. Write this chapter of a novel in flowing third-person prose.
Weave the source events into a real scene: set the atmosphere, render the characters' speech
naturally inside quotes, show inner states subtly through action and detail rather than stating
them flatly. Do not list events mechanically; turn them into narrative. Never mention being an AI.
""".strip()


class ChapterWriter:
    """Writes one chapter's prose from the raw events of its turn range."""

    def __init__(self, llm_provider: Callable[[], LLMClient], max_tokens: int = 2500):
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens

    def _material(self, chapter: Chapter, events: list[WorldEvent]) -> str:
        material = [e for e in events if chapter.start_turn <= e.turn <= chapter.end_turn]
        if not material:
            return "(本章没有对应的事件素材)"
        return "\n".join(
            f"第{e.turn}回合 {f'[{e.actor}]' if e.actor else f'[{e.kind}]'} {e.content}"
            for e in material
        )

    def write(self, world: WorldState, chapter: Chapter, events: list[WorldEvent]) -> str:
        chars = "\n".join(
            f"- {c.name}（{c.role}）：{c.persona.get('core', '')}；说话风格：{c.persona.get('style', '')}"
            for c in world.characters
        )
        messages = [
            {"role": "system", "content": WRITER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"书名：{world.name}\n\n世界设定：{world.setting}\n\n角色：\n{chars}\n\n"
                    f"本章：第{chapter.number}章 《{chapter.title}》\n本章概要：{chapter.summary}\n\n"
                    f"本章素材（事件流）：\n{self._material(chapter, events)}\n\n"
                    "请写成 800~1500 字的正文，多段落，直接输出正文。"
                ),
            },
        ]
        return self.llm_provider().chat(messages, max_tokens=self.max_tokens, temperature=0.9).strip()
