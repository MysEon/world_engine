from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..agents.character import AgentOutput, CharacterAgent
from ..core.models import Character, DirectorEvent, MemoryRecord, WorldEvent, WorldState
from ..core.storage import WorldStore
from ..infra.config import Settings
from ..infra.llm import LLMClient
from ..memory.reflection import Reflector
from ..memory.retrieval import is_duplicate
from .director import apply_director_event, memories_for_director_event
from .narrator import Narrator
from .perception import build_agent_perception, build_scene


class EngineState(TypedDict, total=False):
    world_id: str
    turn: int
    pending_event: DirectorEvent
    outputs: dict[str, AgentOutput]
    narrative: str
    scene: str
    reflect_due: bool


class StepResult:
    def __init__(self, turn: int, narrative: str, event_log: list[str]):
        self.turn = turn
        self.narrative = narrative
        self.event_log = event_log


class WorldEngine:
    """Composition root: wires every subsystem into a single LangGraph turn pipeline.

    This is the only module allowed to know about all others. Each subsystem
    (memory, agents, narrative) stays decoupled from its siblings.
    """

    def __init__(self, settings: Settings, world_id: str):
        self.settings = settings
        self.world_id = world_id
        self.store = WorldStore(settings.engine.db_path)
        self.llm = LLMClient(settings)
        self.narrator = Narrator(lambda: self.llm)
        self.world: WorldState | None = None
        self.agents: dict[str, CharacterAgent] = {}
        self.reflect_every = max(1, settings.engine.reflect_every)
        self.reflector = Reflector(
            lambda: self.llm,
            window=self.reflect_every,
            archive_cutoff=settings.engine.memory_archive_cutoff,
        )
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
        locations = preset.get("locations", [])
        chars = [
            Character(
                name=c["name"],
                role=c.get("role", "character"),
                persona=c.get("persona", {}),
                location=c.get("location", locations[0] if locations else "未知"),
            )
            for c in preset.get("characters", [])
        ]
        state = WorldState(
            name=preset["name"],
            setting=preset.get("setting", ""),
            locations=locations,
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
        g.add_node("reflect", self._reflect)
        g.add_edge(START, "apply_director")
        g.add_edge("apply_director", "perceive")
        g.add_edge("perceive", "act")
        g.add_edge("act", "narrate")
        g.add_edge("narrate", "persist")
        g.add_conditional_edges(
            "persist",
            self._route_after_persist,
            {"reflect": "reflect", "done": END},
        )
        g.add_edge("reflect", END)
        return g.compile()

    def _route_after_persist(self, state: EngineState) -> str:
        return "reflect" if state.get("reflect_due") else "done"

    def _apply_director(self, state: EngineState) -> dict[str, Any]:
        turn = self.current_turn()
        event = state.get("pending_event")
        if event is not None:
            for ev in apply_director_event(self.world, event, turn):
                self.store.add_event(self.world_id, ev)
            self.store.add_memories(self.world_id, memories_for_director_event(self.world, event, turn))
        return {}

    def _perceive(self, state: EngineState) -> dict[str, Any]:
        return {"scene": build_scene(self.world, self.current_turn(), state.get("pending_event"))}

    def _act(self, state: EngineState) -> dict[str, Any]:
        turn = self.current_turn()
        scene = state.get("scene", "")
        outputs: dict[str, AgentOutput] = {}
        turn_log: list[str] = []
        for name, agent in self.agents.items():
            live = "\n".join(turn_log)
            perception = build_agent_perception(scene, live)
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
        return {"narrative": self.narrator.narrate(self.current_turn(), raw)}

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
            existing = self.store.memories(self.world_id, name)
            for m in out.memories:
                content = m.get("content", "")
                if not content or is_duplicate(content, existing):
                    continue
                importance = int(m.get("importance", 4))
                memories.append(MemoryRecord(character=name, content=content, importance=max(1, min(10, importance)), turn=turn))
        if memories:
            self.store.add_memories(self.world_id, memories)
        self.world.turn = turn
        self.store.save(self.world_id, self.world)
        return {"reflect_due": turn % self.reflect_every == 0}

    def _reflect(self, state: EngineState) -> dict[str, Any]:
        completed_turn = self.world.turn
        for name in self.agents:
            self.reflector.run(self.store, self.world_id, name, completed_turn)
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
