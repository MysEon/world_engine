from __future__ import annotations

from ..core.models import WorldEvent

KIND_LABELS = {
    "thought": "内心独白",
    "speech": "说话",
    "action": "行动",
    "narration": "叙述",
    "director": "世界异变",
}


def format_events(
    events: list[WorldEvent],
    start_turn: int | None = None,
    end_turn: int | None = None,
) -> str:
    """Render a slice of the chronicle with human-readable kind labels.

    The event stream is the novel's fact list; labels make it obvious which
    material is inner monologue (内心独白), dialogue, action, or world change.
    """
    filtered = [
        e
        for e in events
        if (start_turn is None or e.turn >= start_turn) and (end_turn is None or e.turn <= end_turn)
    ]
    if not filtered:
        return "(本章没有对应的事件素材)"
    lines = []
    for e in filtered:
        label = KIND_LABELS.get(e.kind, e.kind)
        who = e.actor if e.actor else label
        lines.append(f"第{e.turn}回合 [{who}·{label}] {e.content}")
    return "\n".join(lines)
