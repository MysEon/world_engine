from __future__ import annotations

from pydantic import BaseModel, Field

from .world_state import WorldEvent, WorldState


class DirectorEvent(BaseModel):
    """A change injected into the world by the creator (you, the user)."""

    content: str = Field(..., description="What the creator does to the world")
    targets: list[str] = Field(default_factory=list, description="Characters who perceive it directly, empty = everyone")
    importance: int = Field(default=8, ge=1, le=10)


def apply_director_event(state: WorldState, event: DirectorEvent, turn: int) -> list[WorldEvent]:
    """Apply a creator event: log it as a world event and plant memories in targets.

    Returns the world events that should be persisted.
    """
    targets = event.targets or [c.name for c in state.characters]
    events: list[WorldEvent] = [
        WorldEvent(turn=turn, kind="director", content=event.content, actor="导演")
    ]
    return events


def memories_for_director_event(state: WorldState, event: DirectorEvent, turn: int):
    """Memories to plant so targeted characters 'witness' the creator's change."""
    targets = event.targets or [c.name for c in state.characters]
    from .world_state import MemoryRecord

    return [
        MemoryRecord(
            character=t,
            content=f"发生了一件大事：{event.content}",
            importance=event.importance,
            turn=turn,
        )
        for t in targets
    ]
