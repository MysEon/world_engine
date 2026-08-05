from __future__ import annotations

from ..core.models import DirectorEvent, MemoryRecord, WorldEvent, WorldState


def apply_director_event(state: WorldState, event: DirectorEvent, turn: int) -> list[WorldEvent]:
    """Apply a creator event: returns the world events that should be persisted.

    The event itself does not mutate the world directly; its consequences flow
    through the memories it plants and how characters react next turn.
    """
    return [WorldEvent(turn=turn, kind="director", content=event.content, actor="导演")]


def memories_for_director_event(state: WorldState, event: DirectorEvent, turn: int) -> list[MemoryRecord]:
    """Memories to plant so targeted characters 'witness' the creator's change."""
    targets = event.targets or [c.name for c in state.characters]
    return [
        MemoryRecord(
            character=t,
            content=f"发生了一件大事：{event.content}",
            importance=event.importance,
            turn=turn,
        )
        for t in targets
    ]
