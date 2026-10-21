from __future__ import annotations

import threading

from ..core.storage import WorldStore
from ..infra.config import Settings
from ..infra.llm import LLMClient
from ..infra.llm_settings import LLMSettingsStore
from ..infra.preset import load_preset
from ..narrative.engine import WorldEngine


class WorldManager:
    """Holds live WorldEngine instances per world_id, lazy-created on demand."""

    def __init__(self, settings: Settings, llm: LLMClient | None = None):
        self.settings = settings
        self._llm = llm
        self._engines: dict[str, WorldEngine] = {}
        self._store = WorldStore(settings.engine.db_path)
        self.llm_store = LLMSettingsStore(
            settings.engine.db_path.parent / "llm_settings.json",
            settings.llm,
        )
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def get(self, world_id: str, preset_path: str | None = None) -> WorldEngine:
        """Create the world if needed, otherwise continue the persisted one."""
        engine = self._engines.get(world_id)
        if engine is None:
            engine = WorldEngine(self.settings, world_id, llm_store=self.llm_store)
            if self._llm is not None:
                engine.llm = self._llm
            engine.load_world(load_preset(preset_path or self.settings.server.default_preset))
            self._engines[world_id] = engine
        return engine

    def get_existing(self, world_id: str) -> WorldEngine:
        """Return an already-persisted world; unknown ids raise KeyError (never create)."""
        if self._store.load(world_id) is None:
            raise KeyError(world_id)
        return self.get(world_id)

    def lock(self, world_id: str) -> threading.Lock:
        """Per-world mutex so turns of one world are serialised."""
        with self._locks_guard:
            lock = self._locks.get(world_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[world_id] = lock
            return lock

    def world_ids(self) -> list[str]:
        return self._store.list_worlds()

    def engines(self) -> dict[str, WorldEngine]:
        return self._engines
