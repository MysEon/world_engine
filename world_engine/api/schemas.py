from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.models import DirectorEvent


class CreateWorldRequest(BaseModel):
    world_id: str = "default"
    preset_path: str | None = None


class DirectorEventRequest(BaseModel):
    content: str
    targets: list[str] = Field(default_factory=list)
    importance: int = Field(default=8, ge=1, le=10)

    def to_core(self) -> DirectorEvent:
        return DirectorEvent(content=self.content, targets=self.targets, importance=self.importance)


class StepRequest(BaseModel):
    event: DirectorEventRequest | None = None


class AgentView(BaseModel):
    name: str
    thought: str = ""
    speech: str = ""
    action: str = ""
    memories: list[str] = Field(default_factory=list)


class StepResponse(BaseModel):
    turn: int
    narrative: str = ""
    agents: list[AgentView] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)


class CharacterView(BaseModel):
    name: str
    role: str
    location: str
    persona: dict[str, Any] = Field(default_factory=dict)


class WorldSummary(BaseModel):
    world_id: str
    name: str
    turn: int
    setting: str = ""
    locations: list[str] = Field(default_factory=list)
    characters: list[CharacterView] = Field(default_factory=list)


class ChapterView(BaseModel):
    number: int
    title: str = ""
    summary: str = ""
    start_turn: int = 0
    end_turn: int = 0
    content: str = ""
    status: str = "outlined"


class NovelView(BaseModel):
    title: str = ""
    arc: str = ""
    chapters: list[ChapterView] = Field(default_factory=list)


class WriteRequest(BaseModel):
    chapters: list[int] | None = None


class NovelPlanRequest(BaseModel):
    """force=true allows re-planning even when chapters already have content."""

    force: bool = False


class LLMSettingsRequest(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class LLMSettingsResponse(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key_set: bool


class LLMTestRequest(LLMSettingsRequest):
    pass


class LLMTestResponse(BaseModel):
    ok: bool
    latency_ms: float
    model: str


def to_world_summary(world_id: str, world) -> WorldSummary:
    return WorldSummary(
        world_id=world_id,
        name=world.name,
        turn=world.turn,
        setting=world.setting,
        locations=world.locations,
        characters=[CharacterView(name=c.name, role=c.role, location=c.location, persona=c.persona) for c in world.characters],
    )


def to_novel_view(novel) -> NovelView:
    return NovelView(
        title=novel.title,
        arc=novel.arc,
        chapters=[ChapterView(**ch.model_dump()) for ch in novel.chapters],
    )
