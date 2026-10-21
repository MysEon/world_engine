from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from ..core.models import WorldEvent, WorldState
from ..infra.llm import LLMClient, LLMError

ADVISOR_SYSTEM = """
你是文字世界模拟器的「导演参谋」。用户是这个世界的主宰，但常常想不出下一步该发生什么。
你的任务：根据这个世界此刻的状态，给出 3 条可以立刻注入的导演事件，把故事推向更紧张、更有戏的方向。

规则：
1. 每条事件是「世界突然发生的一个变化」，一句话说清，20–60 个汉字；要具体、可视、在场角色立刻能感知。
2. targets 里的名字必须来自给定的角色名单（可 1–3 个）；空数组表示全体在场角色都能感知，谨慎使用。
3. importance 取 1–10 的整数：5=日常扰动，7=改变关系或处境，9=生死级大事。
4. reason 用 15–40 个字说明「为什么现在提这个」：它利用了哪条现有矛盾，会逼出谁的什么反应。
5. 三条要拉开方向：一条加压、一条给角色选择、一条引入外来变量；不要三条都在同一个矛盾上打转。
6. 避开最近已经发生的事件方向，不要复述已经发生的事。
7. 只写外部发生的事，不要写角色的内心独白，也不要提「建议」「选项」「AI」这类词。
8. 只输出 JSON，不要解释，不要代码块围栏。

输出格式：
{"suggestions": [{"content": "...", "targets": ["角色名"], "importance": 7, "reason": "..."}]}
""".strip()


class DirectorSuggestion(BaseModel):
    content: str = Field(min_length=1, max_length=300)
    targets: list[str] = Field(default_factory=list)
    importance: int = Field(default=6)
    reason: str = Field(default="", max_length=200)


class AdviceResult(BaseModel):
    suggestions: list[DirectorSuggestion] = Field(default_factory=list)


class DirectorAdvisor:
    """A second opinion: proposes creator events without touching the world."""

    def __init__(self, llm_provider: Callable[[], LLMClient]):
        self.llm_provider = llm_provider

    def suggest(self, world: WorldState, events: list[WorldEvent], count: int = 3) -> list[DirectorSuggestion]:
        roster = "\n".join(
            f"- {c.name}（{c.role}，{c.location or '未知'}）" for c in world.characters
        )
        recent = "\n".join(
            f"[第{e.turn}回合] {e.actor}·{e.kind}：{e.content}" for e in events[-20:]
        ) or "（世界还没开始）"
        messages = [
            {"role": "system", "content": ADVISOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"世界：{world.name}\n设定：{world.setting}\n"
                    f"地点：{'、'.join(world.locations) or '未定'}\n\n"
                    f"角色名单：\n{roster}\n\n"
                    f"当前回合：{world.turn}\n最近事件：\n{recent}\n\n"
                    f"请给 {count} 条导演事件。"
                ),
            },
        ]
        data = self.llm_provider().chat_json(messages)
        try:
            result = AdviceResult.model_validate(data)
        except ValidationError as exc:
            raise LLMError(f"导演建议格式不合法：{exc.errors()[:1]}") from None

        known = {c.name for c in world.characters}
        cleaned: list[DirectorSuggestion] = []
        seen: set[str] = set()
        for raw in result.suggestions:
            content = " ".join(raw.content.split())
            if not content or content in seen:
                continue
            seen.add(content)
            cleaned.append(
                DirectorSuggestion(
                    content=content,
                    targets=[t for t in raw.targets if t in known][:3],
                    importance=max(1, min(10, raw.importance)),
                    reason=" ".join(raw.reason.split()),
                )
            )
            if len(cleaned) >= count:
                break
        if not cleaned:
            raise LLMError("导演建议为空")
        return cleaned
