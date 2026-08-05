from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import MemoryRecord, WorldEvent, WorldState, utcnow


class WorldStore:
    """SQLite persistence for world state, event log, and memory streams.

    Depends only on core.models; swap out for another backend behind the same
    method surface if the project outgrows SQLite.
    """

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
                kind TEXT DEFAULT 'event',
                archived INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )"""
        )
        self._migrate_memories(cur)
        self.conn.commit()

    def _migrate_memories(self, cur: sqlite3.Cursor) -> None:
        cols = {r["name"] for r in cur.execute("PRAGMA table_info(memories)").fetchall()}
        if "kind" not in cols:
            cur.execute("ALTER TABLE memories ADD COLUMN kind TEXT DEFAULT 'event'")
        if "archived" not in cols:
            cur.execute("ALTER TABLE memories ADD COLUMN archived INTEGER DEFAULT 0")

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
            """INSERT INTO memories (world_id, character, content, importance, turn, kind, archived, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    world_id,
                    r.character,
                    r.content,
                    r.importance,
                    r.turn,
                    r.kind,
                    1 if r.archived else 0,
                    r.created_at,
                )
                for r in records
            ],
        )
        self.conn.commit()

    def memories(self, world_id: str, character: str, include_archived: bool = False) -> list[MemoryRecord]:
        query = (
            """SELECT id, character, content, importance, turn, kind, archived, created_at
               FROM memories WHERE world_id = ? AND character = ?"""
            + ("" if include_archived else " AND archived = 0")
            + " ORDER BY id"
        )
        rows = self.conn.execute(query, (world_id, character)).fetchall()
        return [MemoryRecord(**dict(r)) for r in rows]

    def archive(self, world_id: str, ids: list[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(
            f"UPDATE memories SET archived = 1 WHERE world_id = ? AND id IN ({placeholders})",
            [world_id, *ids],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
