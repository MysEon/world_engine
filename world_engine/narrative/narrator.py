from __future__ import annotations

from typing import Callable

from ..infra.llm import LLMClient

NARRATOR_SYSTEM = """
You are the world's narrator, a Chinese novelist. Given a turn's raw events from the story,
write 2-4 sentences of flowing, literary prose in the third person. Do NOT reveal private
thoughts of the characters in the prose; thoughts are interior. Render speech naturally
inside quotes. Do not say "the narrator". Output only the prose, no headings.
""".strip()


class Narrator:
    """Turns a turn's raw events into flowing third-person prose."""

    def __init__(self, llm_provider: Callable[[], LLMClient]):
        self.llm_provider = llm_provider

    def narrate(self, turn: int, raw_events: list[str]) -> str:
        if not raw_events:
            return ""
        messages = [
            {"role": "system", "content": NARRATOR_SYSTEM},
            {"role": "user", "content": f"第 {turn} 回合的原始事件：\n" + "\n".join(raw_events)},
        ]
        return self.llm_provider().chat(messages, max_tokens=400, temperature=0.9).strip()
