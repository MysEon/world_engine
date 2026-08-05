from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from ..core.models import Chapter, Novel, WorldEvent, WorldState
from ..infra.llm import LLMClient

PLANNER_SYSTEM = """
You are a master novelist and story editor. Given a story world, its characters, and the raw
event stream of what actually happened, you must structure this material into a novel outline.
Divide the events into chapters, give each chapter a title and a one-line summary, and shape an
overall narrative arc with clear build-up, climax, and resolution. Do not invent events that are
not in the stream — structure what happened, do not add to it. Write in Chinese.
""".strip()

PLANNER_CONTRACT = """
You MUST reply with a single JSON object, exactly this shape:
{
  "title": "<novel title>",
  "arc": "<one sentence describing the overall narrative arc>",
  "chapters": [
    {
      "number": 1,
      "title": "<chapter title>",
      "summary": "<one-line summary of what happens in this chapter>",
      "start_turn": <int, first turn this chapter covers>,
      "end_turn": <int, last turn this chapter covers>
    }
  ]
}
Chapters must be contiguous and cover the whole event stream in order (chapter 1 starts at the
first turn, the last chapter ends at the last turn, and start_turn of the next chapter equals
end_turn + 1 of the previous one). Aim for 3-8 chapters total. "arc" must be present.
""".strip()


class PlanResult(BaseModel):
    title: str = ""
    arc: str = ""
    chapters: list[Chapter] = Field(default_factory=list)


class NovelPlanner:
    """Turns the world chronicle into a structured novel outline."""

    def __init__(self, llm_provider: Callable[[], LLMClient], max_events: int = 80):
        self.llm_provider = llm_provider
        self.max_events = max_events

    def _condense(self, events: list[WorldEvent]) -> str:
        for e in events:
            who = f"[{e.actor}]" if e.actor else f"[{e.kind}]"
            yield f"第{e.turn}回合 {who} {e.content}"

    def plan(self, world: WorldState, events: list[WorldEvent]) -> Novel:
        stream = list(self._condense(events))
        if len(stream) > self.max_events:
            stream = stream[-self.max_events:]
        if not stream:
            return Novel(world_id=world.world_id, title=world.name)

        chars = "\n".join(f"- {c.name}（{c.role}）：{c.persona.get('core', '')}" for c in world.characters)
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"小说题材：{world.setting}\n\n角色：\n{chars}\n\n"
                    f"原始事件流：\n" + "\n".join(stream) + "\n\n" + PLANNER_CONTRACT
                ),
            },
        ]
        data = self.llm_provider().chat_json(messages)
        result = PlanResult.model_validate(data)
        result.chapters.sort(key=lambda c: c.number)
        return Novel(world_id=world.world_id, title=result.title, arc=result.arc, chapters=result.chapters)
