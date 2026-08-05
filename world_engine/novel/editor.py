from __future__ import annotations

from typing import Callable

from ..core.models import Chapter, WorldEvent, WorldState
from ..infra.llm import LLMClient

EDITOR_SYSTEM = """
You are the editor of a literary novel. Polish this chapter: smooth awkward phrasing, keep
character voices consistent, fix continuity slips (names, timing), and make sure the chapter
delivers its planned beat. Stay faithful to the source events — do not add new plot that did
not happen. Output only the polished chapter text. Write in Chinese.
""".strip()


class ChapterEditor:
    """A quality pass over a chapter: continuity, voice consistency, polish."""

    def __init__(self, llm_provider: Callable[[], LLMClient]):
        self.llm_provider = llm_provider

    def edit(self, world: WorldState, chapter: Chapter, events: list[WorldEvent], draft: str) -> str:
        material = [e for e in events if chapter.start_turn <= e.turn <= chapter.end_turn]
        source = "\n".join(
            f"第{e.turn}回合 {f'[{e.actor}]' if e.actor else f'[{e.kind}]'} {e.content}"
            for e in material
        )
        messages = [
            {"role": "system", "content": EDITOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"世界设定：{world.setting}\n\n本章概要：{chapter.summary}\n\n"
                    f"本章素材（事实清单）：\n{source or '(无)'}\n\n"
                    f"待润色的初稿：\n{draft}\n\n"
                    "输出润色后的完整正文。"
                ),
            },
        ]
        return self.llm_provider().chat(messages, max_tokens=3000, temperature=0.7).strip()
