from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from ..core.models import MemoryRecord
from ..core.storage import WorldStore
from ..infra.llm import LLMClient

REFLECT_SYSTEM = """
You are a sharp observer of human behavior. Reflect on a character's recent memories and
surface high-level insights — patterns, goals, tensions, and relationships — that go beyond
any single event. Never mention being an AI or a simulation. Write in Chinese.
""".strip()

REFLECT_CONTRACT = """
You MUST reply with a single JSON object, exactly this shape:
{
  "reflections": [
    {"content": "<one high-level insight about this character's situation or nature>", "importance": 8}
  ]
}
Generate 1 to 3 reflections. Each insight should be a generalization that could guide the
character's future choices (e.g. "他对陌生人越来越警惕，因为信鸽事件让他联想到旧案"),
NOT a restatement of any single event. importance must be between 8 and 10.
""".strip()


class ReflectionResult(BaseModel):
    reflections: list[dict[str, Any]] = Field(default_factory=list)


class Reflector:
    """Periodic memory consolidation: LLM reflections + archiving of old low-value memories."""

    def __init__(self, llm_provider: Callable[[], LLMClient], window: int = 5, archive_cutoff: int = 4):
        self.llm_provider = llm_provider
        self.window = window
        self.archive_cutoff = archive_cutoff

    def reflect(
        self,
        character: str,
        memories: list[MemoryRecord],
        current_turn: int,
    ) -> ReflectionResult:
        recent = [m for m in memories if m.turn > current_turn - self.window and not m.archived]
        if not recent:
            return ReflectionResult()
        lines = "\n".join(f"- (第{m.turn}回合) {m.content}" for m in recent)
        messages = [
            {"role": "system", "content": REFLECT_SYSTEM},
            {
                "role": "user",
                "content": f"角色：{character}\n\n这是角色最近的记忆片段：\n{lines}\n\n"
                + REFLECT_CONTRACT,
            },
        ]
        data = self.llm_provider().chat_json(messages)
        return ReflectionResult.model_validate(data)

    def run(self, store: WorldStore, world_id: str, character: str, current_turn: int) -> dict[str, int]:
        memories = store.memories(world_id, character, include_archived=True)
        result = self.reflect(character, memories, current_turn)

        new_records = [
            MemoryRecord(
                character=character,
                content=r.get("content", ""),
                importance=min(10, max(8, int(r.get("importance", 8)))),
                turn=current_turn,
                kind="reflection",
            )
            for r in result.reflections
        ]
        if new_records:
            store.add_memories(world_id, new_records)

        to_archive = [
            m.id
            for m in memories
            if m.id is not None
            and not m.archived
            and m.kind != "reflection"
            and m.importance <= self.archive_cutoff
            and m.turn <= current_turn - self.window
        ]
        store.archive(world_id, to_archive)

        return {"reflections": len(new_records), "archived": len(to_archive)}
