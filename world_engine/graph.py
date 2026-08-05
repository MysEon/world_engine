from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import AgentOutput, CharacterAgent
from .config import Settings
from .director import DirectorEvent, apply_director_event, memories_for_director_event
from .llm import LLMClient
from .world_state import Character, MemoryRecord, WorldEvent, WorldState, WorldStore

NARRATOR_SYSTEM = """
You are the world's narrator, a Chinese novelist. Given a turn's raw events from the story,
write 2-4 sentences of flowing, literary prose in the third person. Do NOT reveal private
thoughts of the characters in the prose; thoughts are interior. Render speech naturally
inside quotes. Do not say "the narrator". Output only the prose, no headings.
""".strip()


class EngineState(TypedDict, total=False):
    world_id: str
    turn: int
    pending_event: DirectorEvent
    outputs: dict[str, AgentOutput]
    narrative: str
    scene: str


class StepResult:
    def __init__(self, turn: int, narrative: str, event_log: list[str]):
        self.turn = turn
        self.narrative = narrative
        self.event_log = event_log


class WorldEngine:
    def __init__(self, settings: Settings, world_id: str):
        self.settings = settings
        self.world_id = world_id
        self.store = WorldStore(settings.engine.db_path)
        self.llm = LLMClient(settings)
        self.world: WorldState | None = None
        self.agents: dict[str, CharacterAgent] = {}
        self.graph = self._build_graph()

    # -- lifecycle -----------------------------------------------------------

    def load_world(self, preset: dict[str, Any]) -> WorldState:
        state = self.store.load(self.world_id)
        if state is None:
            state = self._init_from_preset(preset)
        self.world = state
        self.agents = {c.name: CharacterAgent(c, self.llm) for c in state.characters}
        return state

    def _init_from_preset(self, preset: dict[str, Any]) -> WorldState:
        chars = [
            Character(
                name=c["name"],
                role=c.get("role", "character"),
                persona=c.get("persona", {}),
                location=c.get("location", preset.get("locations", ["未知"])[0] if preset.get("locations") else "未知"),
            )
            for c in preset.get("characters", [])
        ]
        state = WorldState(
            name=preset["name"],
            setting=preset.get("setting", ""),
            locations=preset.get("locations", []),
            characters=chars,
            turn=0,
            narrative=[],
        )
        self.store.save(self.world_id, state)
        seeds: list[MemoryRecord] = []
        for c in preset.get("characters", []):
            for mem in c.get("initial_memories", []):
                seeds.append(MemoryRecord(character=c["name"], content=mem, importance=5, turn=0))
        if seeds:
            self.store.add_memories(self.world_id, seeds)
        return state

    def current_turn(self) -> int:
        return self.world.turn + 1 if self.world else 1

    # -- graph ---------------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(EngineState)
        g.add_node("apply_director", self._apply_director)
        g.add_node("perceive", self._perceive)
        g.add_node("act", self._act)
        g.add_node("narrate", self._narrate)
        g.add_node("persist", self._persist)
        g.add_edge(START, "apply_director")
        g.add_edge("apply_director", "perceive")
        g.add_edge("perceive", "act")
        g.add_edge("act", "narrate")
        g.add_edge("narrate", "persist")
        g.add_edge("persist", END)
        return g.compile()

    def _apply_director(self, state: EngineState) -> dict[str, Any]:
        turn = self.current_turn()
        event = state.get("pending_event")
        if event is not None:
            for ev in apply_director_event(self.world, event, turn):
                self.store.add_event(self.world_id, ev)
            self.store.add_memories(self.world_id, memories_for_director_event(self.world, event, turn))
        return {}

    def _perceive(self, state: EngineState) -> dict[str, Any]:
        turn = self.current_turn()
        last = self.world.narrative[-1] if self.world.narrative else "此刻一切如常，平静的一天。"
        scene_lines = [
            f"时间：第 {turn} 回合",
            f"地点：{'、'.join(self.world.locations) if self.world.locations else '未知'}",
            "",
            "此刻的状况：",
            last,
        ]
        event = state.get("pending_event")
        if event is not None:
            scene_lines.append("")
            scene_lines.append("【世界发生了异变】")
            scene_lines.append(event.content)
        return {"scene": "\n".join(scene_lines)}

    def _act(self, state: EngineState) -> dict[str, Any]:
        turn = self.current_turn()
        scene = state.get("scene", "")
        outputs: dict[str, AgentOutput] = {}
        turn_log: list[str] = []
        for name, agent in self.agents.items():
            live = "\n".join(turn_log)
            perception = scene + (("\n\n刚才在场的人：\n" + live) if live else "")
            memories = self.store.memories(self.world_id, name)
            out = agent.act(self.world, turn, perception, memories)
            outputs[name] = out
            if out.speech:
                turn_log.append(f"{name}：\"{out.speech}\"")
            turn_log.append(f"{name}：{out.action}")
        return {"outputs": outputs}

    def _narrate(self, state: EngineState) -> dict[str, Any]:
        outputs = state.get("outputs", {})
        raw = []
        for name, out in outputs.items():
            raw.append(f"{name} 内心：{out.thought}")
            if out.speech:
                raw.append(f"{name} 说：{out.speech}")
            raw.append(f"{name} 行动：{out.action}")
        turn = self.current_turn()
        if not raw:
            return {"narrative": ""}
        messages = [
            {"role": "system", "content": NARRATOR_SYSTEM},
            {"role": "user", "content": f"第 {turn} 回合的原始事件：\n" + "\n".join(raw)},
        ]
        prose = self.llm.chat(messages, max_tokens=400, temperature=0.9).strip()
        return {"narrative": prose}

    def _persist(self, state: EngineState) -> dict[str, Any]:
        turn = self.current_turn()
        narrative = state.get("narrative", "")
        outputs = state.get("outputs", {})
        if narrative:
            self.world.narrative.append(narrative)
            self.store.add_event(self.world_id, WorldEvent(turn=turn, kind="narration", content=narrative))
        memories: list[MemoryRecord] = []
        for name, out in outputs.items():
            if out.speech:
                self.store.add_event(self.world_id, WorldEvent(turn=turn, kind="speech", actor=name, content=out.speech))
            self.store.add_event(self.world_id, WorldEvent(turn=turn, kind="action", actor=name, content=out.action))
            for m in out.memories:
                importance = int(m.get("importance", 4))
                memories.append(MemoryRecord(character=name, content=m.get("content", ""), importance=max(1, min(10, importance)), turn=turn))
        if memories:
            self.store.add_memories(self.world_id, memories)
        self.world.turn = turn
        self.store.save(self.world_id, self.world)
        return {}

    # -- public API ----------------------------------------------------------

    def step(self, director_event: DirectorEvent | None = None) -> StepResult:
        turn = self.current_turn()
        self.graph.invoke(
            {
                "world_id": self.world_id,
                "turn": turn,
                "pending_event": director_event,
            }
        )
        event_log = [e.content for e in self.store.events(self.world_id, limit=8)]
        narrative = self.world.narrative[-1] if self.world.narrative else ""
        return StepResult(turn=turn, narrative=narrative, event_log=event_log)
