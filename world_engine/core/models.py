from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Character(BaseModel):
    name: str
    role: str = "character"
    persona: dict[str, Any] = Field(default_factory=dict)
    location: str = "unknown"
    status: dict[str, Any] = Field(default_factory=dict)


class WorldEvent(BaseModel):
    turn: int
    kind: str
    actor: str | None = None
    content: str
    created_at: str = Field(default_factory=utcnow)


class MemoryRecord(BaseModel):
    id: int | None = None
    character: str
    content: str
    importance: int = Field(default=3, ge=1, le=10)
    turn: int = 0
    kind: str = "event"
    archived: bool = False
    created_at: str = Field(default_factory=utcnow)


class WorldState(BaseModel):
    name: str
    setting: str = ""
    locations: list[str] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    turn: int = 0
    narrative: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utcnow)


class DirectorEvent(BaseModel):
    """A change injected into the world by the creator (you, the user)."""

    content: str = Field(..., description="What the creator does to the world")
    targets: list[str] = Field(default_factory=list, description="Characters who perceive it directly, empty = everyone")
    importance: int = Field(default=8, ge=1, le=10)
