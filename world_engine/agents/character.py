from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.models import Character, MemoryRecord, WorldState
from ..infra.llm import LLMClient
from ..memory.retrieval import retrieve

AGENT_OUTPUT_CONTRACT = """
你必须只回复一个 JSON 对象，结构严格如下：
{
  "thought": "<你的内心想法，不说出口>",
  "speech": "<你说出口的话；若沉默则为空字符串>",
  "action": "<你身体动作的简短描述>",
  "memories": [{"content": "<本回合你学到的或感受到的重要事情>", "importance": <整数 1-10>}]
}
"thought" 与 "action" 必填；"speech" 可为空；"memories" 可为空。
【重要】所有内容字段（thought / speech / action / memories 的 content）必须一律使用中文。
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
你是 {self.character.name}，{self.character.role}，身处故事世界「{world.name}」。

世界设定：{world.setting}

你的核心性格：{p.get('core', '')}
你的特质：{p.get('traits', '')}
你的说话风格：{p.get('style', '')}
你的目标：{p.get('goals', '')}
你的秘密：{p.get('secrets', '')}

你必须彻底沉浸于角色之中。绝不能跳出角色，绝不能提及自己是 AI 或这是一场模拟。
你只能依据自己的记忆和当前场景知道的信息来行动，以这个角色的身份回应。
【重要】你的所有输出必须一律使用中文。
""".strip()

    def perception_block(
        self,
        current_turn: int,
        perception: str,
        memories: list[MemoryRecord],
    ) -> str:
        recalled = retrieve(perception, memories, current_turn, top_k=4)
        mem_lines = "\n".join(f"- {m.content}" for m in recalled) or "-（暂无强烈的记忆）"
        return f"""
当前场景：
{perception}

你此刻记起的事（刚刚检索到的记忆）：
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
                "content": self.perception_block(current_turn, perception, memories)
                + "\n\n"
                + "Decide what you do and say right now. "
                + AGENT_OUTPUT_CONTRACT,
            },
        ]
        data = self.llm.chat_json(messages)
        return AgentOutput.model_validate(data)
