from __future__ import annotations

from ..core.storage import WorldStore
from ..infra.config import Settings
from ..infra.llm import LLMClient
from ..infra.preset import load_preset
from ..narrative.engine import WorldEngine


class WorldManager:
    """Holds live WorldEngine instances per world_id, lazy-created on demand."""

    def __init__(self, settings: Settings, llm: LLMClient | None = None):
        self.settings = settings
        self._llm = llm
        self._engines: dict[str, WorldEngine] = {}
        self._store = WorldStore(settings.engine.db_path)

    def get(self, world_id: str, preset_path: str | None = None) -> WorldEngine:
        engine = self._engines.get(world_id)
        if engine is None:
            engine = WorldEngine(self.settings, world_id)
            if self._llm is not None:
                engine.llm = self._llm
            engine.load_world(load_preset(preset_path or self.settings.server.default_preset))
            self._engines[world_id] = engine
        return engine

    def world_ids(self) -> list[str]:
        return self._store.list_worlds()

    def engines(self) -> dict[str, WorldEngine]:
        return self._engines
