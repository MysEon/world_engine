from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .llm import LLMClient
from .memory import retrieve
from .world_state import Character, MemoryRecord, WorldState

AGENT_OUTPUT_CONTRACT = """
You MUST reply with a single JSON object, exactly this shape:
{
  "thought": "<your inner thought, not said aloud>",
  "speech": "<what you say out loud, or empty string if silent>",
  "action": "<brief description of what you physically do>",
  "memories": [{"content": "<notable thing you learned or felt this turn>", "importance": <int 1-10>}]
}
"thought" and "action" are required. "speech" may be empty. "memories" may be empty, but if you
learn something significant about another character or the world, record it with a higher importance.
""".strip()


class AgentOutput(BaseModel):
    thought: str = ""
    speech: str = ""
    action: str = ""
    memories: list[dict[str, Any]] = Field(default_factory=list)


class CharacterAgent:
    def __init__(self, character: Character, llm: LLMClient):
        self.character = character
        self.llm = llm

    def system_prompt(self, world: WorldState) -> str:
        p = self.character.persona
        return f"""
You are {self.character.name}, {self.character.role} in the story world "{world.name}".

WORLD SETTING: {world.setting}

YOUR CORE SELF: {p.get('core', '')}
YOUR TRAITS: {p.get('traits', '')}
YOUR SPEAKING STYLE: {p.get('style', '')}
YOUR GOALS: {p.get('goals', '')}
YOUR SECRETS: {p.get('secrets', '')}

You are fully immersed. Never break character. Never mention being an AI or a simulation.
You only know what your own memories and the current scene tell you. Respond as this character.
""".strip()

    def perception_block(
        self,
        world: WorldState,
        current_turn: int,
        perception: str,
        memories: list[MemoryRecord],
    ) -> str:
        recalled = retrieve(perception, memories, current_turn, top_k=4)
        mem_lines = "\n".join(f"- {m.content}" for m in recalled) or "- (no strong recollection)"
        return f"""
CURRENT SCENE:
{perception}

WHAT YOU REMEMBER (recalled just now):
{mem_lines}
""".strip()

    def act(
        self,
        world: WorldState,
        current_turn: int,
        perception: str,
        memories: list[MemoryRecord],
    ) -> AgentOutput:
        messages = [
            {"role": "system", "content": self.system_prompt(world)},
            {
                "role": "user",
                "content": self.perception_block(world, current_turn, perception, memories)
                + "\n\n"
                + "Decide what you do and say right now. "
                + AGENT_OUTPUT_CONTRACT,
            },
        ]
        data = self.llm.chat_json(messages)
        return AgentOutput.model_validate(data)
