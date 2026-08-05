from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
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
    character: str
    content: str
    importance: int = Field(default=3, ge=1, le=10)
    turn: int = 0
    created_at: str = Field(default_factory=utcnow)


class WorldState(BaseModel):
    name: str
    setting: str = ""
    locations: list[str] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    turn: int = 0
    narrative: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utcnow)


class WorldStore:
    """SQLite persistence for world state, event log, and memory streams."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS worlds (
                id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                kind TEXT NOT NULL,
                actor TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                world_id TEXT NOT NULL,
                character TEXT NOT NULL,
                content TEXT NOT NULL,
                importance INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.conn.commit()

    def load(self, world_id: str) -> WorldState | None:
        row = self.conn.execute(
            "SELECT state_json FROM worlds WHERE id = ?", (world_id,)
        ).fetchone()
        if row is None:
            return None
        return WorldState.model_validate_json(row["state_json"])

    def save(self, world_id: str, state: WorldState) -> None:
        state.updated_at = utcnow()
        self.conn.execute(
            """INSERT INTO worlds (id, state_json, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET state_json = excluded.state_json,
                                             updated_at = excluded.updated_at""",
            (world_id, state.model_dump_json(), state.updated_at),
        )
        self.conn.commit()

    def add_event(self, world_id: str, event: WorldEvent) -> None:
        self.conn.execute(
            """INSERT INTO events (world_id, turn, kind, actor, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (world_id, event.turn, event.kind, event.actor, event.content, event.created_at),
        )
        self.conn.commit()

    def events(self, world_id: str, limit: int = 20) -> list[WorldEvent]:
        rows = self.conn.execute(
            """SELECT turn, kind, actor, content, created_at FROM events
               WHERE world_id = ? ORDER BY id DESC LIMIT ?""",
            (world_id, limit),
        ).fetchall()
        return [WorldEvent(**dict(r)) for r in reversed(rows)]

    def add_memories(self, world_id: str, records: list[MemoryRecord]) -> None:
        self.conn.executemany(
            """INSERT INTO memories (world_id, character, content, importance, turn, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (world_id, r.character, r.content, r.importance, r.turn, r.created_at)
                for r in records
            ],
        )
        self.conn.commit()

    def memories(self, world_id: str, character: str) -> list[MemoryRecord]:
        rows = self.conn.execute(
            """SELECT character, content, importance, turn, created_at FROM memories
               WHERE world_id = ? AND character = ? ORDER BY id""",
            (world_id, character),
        ).fetchall()
        return [MemoryRecord(**dict(r)) for r in rows]

    def close(self) -> None:
        self.conn.close()
