from __future__ import annotations

from ..core.models import DirectorEvent, WorldState


def build_scene(world: WorldState, turn: int, director_event: DirectorEvent | None = None) -> str:
    """The shared stage: what every character can perceive this turn."""
    last = world.narrative[-1] if world.narrative else "此刻一切如常，平静的一天。"
    scene_lines = [
        f"时间：第 {turn} 回合",
        f"地点：{'、'.join(world.locations) if world.locations else '未知'}",
        "",
        "此刻的状况：",
        last,
    ]
    if director_event is not None:
        scene_lines.append("")
        scene_lines.append("【世界发生了异变】")
        scene_lines.append(director_event.content)
    return "\n".join(scene_lines)


def build_agent_perception(scene: str, live_log: str) -> str:
    """Per-character view: the shared scene plus what just happened in their presence."""
    if not live_log:
        return scene
    return f"{scene}\n\n刚才在场的人：\n{live_log}"
