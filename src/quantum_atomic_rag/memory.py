"""SQLite-backed atomic memory for ingestion, retrieval, and graph rendering."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AtomicNode(BaseModel):
    node_id: str
    content: str
    source: str = "manual"
    orbital_shell: int = Field(ge=1, le=7)
    is_nucleus: bool = False
    nucleus_id: str | None = None
    potential_energy: float
    utility: float = Field(ge=0.0, le=1.0)
    momentum: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime


class MemoryQueryResult(BaseModel):
    query: str
    retrieved_nodes: list[AtomicNode]
    retrieval_trace: list[dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: float
    participating_agents: int = 4


class MemoryMetrics(BaseModel):
    nuclei: int
    atoms: int
    shells_populated: list[int]
    links: int
    active_atoms: int
    average_utility: float
    average_momentum: float


class KnowledgeStore:
    """Small transactional SQLite store shared by the browser API and UI."""

    def __init__(self, database_path: str = "data/atomic_memory.sqlite3") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS atoms (
                    node_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    orbital_shell INTEGER NOT NULL CHECK (orbital_shell BETWEEN 1 AND 7),
                    is_nucleus INTEGER NOT NULL DEFAULT 0,
                    nucleus_id TEXT,
                    potential_energy REAL NOT NULL,
                    utility REAL NOT NULL CHECK (utility BETWEEN 0 AND 1),
                    momentum REAL NOT NULL CHECK (momentum BETWEEN 0 AND 1),
                    metadata TEXT NOT NULL,
                    embedding TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS atom_links (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    strength REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id)
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(atoms)")}
            migrations = {
                "is_nucleus": "ALTER TABLE atoms ADD COLUMN is_nucleus INTEGER NOT NULL DEFAULT 0",
                "nucleus_id": "ALTER TABLE atoms ADD COLUMN nucleus_id TEXT",
                "embedding": "ALTER TABLE atoms ADD COLUMN embedding TEXT",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            self._backfill_structure(connection)

    def _backfill_structure(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("SELECT * FROM atoms ORDER BY updated_at, node_id").fetchall()
        if not rows:
            return
        nucleus = next((row for row in rows if row["is_nucleus"]), rows[0])
        connection.execute(
            "UPDATE atoms SET is_nucleus = CASE WHEN node_id = ? THEN 1 ELSE 0 END, nucleus_id = ?",
            (nucleus["node_id"], nucleus["node_id"]),
        )
        for row in rows:
            if row["node_id"] == nucleus["node_id"]:
                continue
            strength = self._token_overlap(row["content"], nucleus["content"])
            connection.execute(
                "UPDATE atoms SET orbital_shell = ? WHERE node_id = ?",
                (2 if strength >= 0.2 else 3, row["node_id"]),
            )
            if strength > 0:
                connection.execute(
                    "INSERT OR REPLACE INTO atom_links(source_id, target_id, strength) VALUES (?, ?, ?)",
                    (nucleus["node_id"], row["node_id"], strength),
                )

    def ingest(
        self,
        content: str,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> AtomicNode:
        normalized = content.strip()
        if not normalized:
            raise ValueError("content cannot be empty")
        node_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        now = datetime.now(UTC)
        with self._connect() as connection:
            existing = connection.execute("SELECT * FROM atoms WHERE node_id = ?", (node_id,)).fetchone()
            if existing:
                return self._row_to_node(existing)
            existing_rows = connection.execute("SELECT * FROM atoms ORDER BY updated_at, node_id").fetchall()
            is_nucleus = not existing_rows
            nucleus_id = node_id if is_nucleus else next(
                (row["node_id"] for row in existing_rows if row["is_nucleus"]), existing_rows[0]["node_id"]
            )
            best_overlap = max(
                (self._token_overlap(normalized, row["content"]) for row in existing_rows), default=0.0
            )
            orbital_shell = 1 if is_nucleus else (2 if best_overlap >= 0.2 else 3)
            node = AtomicNode(
                node_id=node_id,
                content=normalized,
                source=source[:256],
                orbital_shell=orbital_shell,
                is_nucleus=is_nucleus,
                nucleus_id=nucleus_id,
                potential_energy=1.0 / orbital_shell,
                utility=0.5,
                momentum=0.0,
                metadata=metadata or {},
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO atoms (node_id, content, source, orbital_shell, is_nucleus, nucleus_id, potential_energy, utility, momentum, metadata, embedding, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node.node_id,
                    node.content,
                    node.source,
                    node.orbital_shell,
                    int(node.is_nucleus),
                    node.nucleus_id,
                    node.potential_energy,
                    node.utility,
                    node.momentum,
                    json.dumps(node.metadata),
                    json.dumps(embedding) if embedding else None,
                    node.updated_at.isoformat(),
                ),
            )
            if not is_nucleus:
                for row in existing_rows:
                    strength = self._token_overlap(normalized, row["content"])
                    if strength > 0:
                        connection.execute(
                            "INSERT OR REPLACE INTO atom_links(source_id, target_id, strength) VALUES (?, ?, ?)",
                            (node.node_id, row["node_id"], strength),
                        )
            return node

    def search(self, query: str, limit: int = 12, embedding: list[float] | None = None) -> MemoryQueryResult:
        started = time.perf_counter()
        tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM atoms ORDER BY utility DESC, updated_at DESC").fetchall()
            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                content_tokens = set(re.findall(r"[a-z0-9]{3,}", row["content"].lower()))
                lexical = len(tokens & content_tokens) / max(len(tokens), 1)
                semantic = 0.0
                if embedding and row["embedding"]:
                    semantic = self._cosine_similarity(embedding, json.loads(row["embedding"]))
                score = semantic * 0.8 + lexical * 0.15 + row["utility"] * 0.05
                scored.append((score, row))
            scored.sort(key=lambda item: item[0], reverse=True)
            selected_scores = [(score, row) for score, row in scored if score > 0][:limit]
            selected = [row for score, row in selected_scores]
            if selected:
                connection.executemany(
                    "UPDATE atoms SET utility = MIN(1.0, utility + 0.05), momentum = MIN(1.0, momentum + 0.1), updated_at = ? WHERE node_id = ?",
                    [(datetime.now(UTC).isoformat(), row["node_id"]) for row in selected],
                )
                selected = [
                    connection.execute("SELECT * FROM atoms WHERE node_id = ?", (row["node_id"],)).fetchone()
                    for row in selected
                ]
            nodes = [self._row_to_node(row) for row in selected]
        return MemoryQueryResult(
            query=query,
            retrieved_nodes=nodes,
            retrieval_trace=[
                {
                    "node_id": row["node_id"],
                    "match_score": round(score, 4),
                    "orbital_shell": row["orbital_shell"],
                    "utility_before_update": round(row["utility"], 4),
                    "momentum_before_update": round(row["momentum"], 4),
                }
                for score, row in selected_scores
            ],
            execution_time_ms=(time.perf_counter() - started) * 1000,
        )

    def graph(self) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as connection:
            nodes = [
                self._row_to_node(row).model_dump(mode="json")
                for row in connection.execute("SELECT * FROM atoms ORDER BY utility DESC").fetchall()
            ]
            links = [dict(row) for row in connection.execute("SELECT * FROM atom_links").fetchall()]
        return {"nodes": nodes, "links": links}

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0])

    def metrics(self) -> MemoryMetrics:
        with self._connect() as connection:
            summary = connection.execute(
                "SELECT COUNT(*) AS atoms, COUNT(DISTINCT CASE WHEN is_nucleus = 1 THEN node_id END) AS nuclei, "
                "AVG(utility) AS average_utility, AVG(momentum) AS average_momentum, "
                "COUNT(CASE WHEN utility >= 0.6 OR momentum >= 0.4 THEN 1 END) AS active_atoms FROM atoms"
            ).fetchone()
            shells = [row[0] for row in connection.execute("SELECT DISTINCT orbital_shell FROM atoms ORDER BY orbital_shell")]
            links = connection.execute("SELECT COUNT(*) FROM atom_links").fetchone()[0]
        return MemoryMetrics(
            nuclei=summary["nuclei"] or 0,
            atoms=summary["atoms"] or 0,
            shells_populated=shells,
            links=links,
            active_atoms=summary["active_atoms"] or 0,
            average_utility=round(summary["average_utility"] or 0.0, 4),
            average_momentum=round(summary["average_momentum"] or 0.0, 4),
        )

    @staticmethod
    def _token_overlap(left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-z0-9]{3,}", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9]{3,}", right.lower()))
        return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
        return numerator / denominator if denominator else 0.0

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> AtomicNode:
        return AtomicNode(
            node_id=row["node_id"],
            content=row["content"],
            source=row["source"],
            orbital_shell=row["orbital_shell"],
            potential_energy=row["potential_energy"],
            is_nucleus=bool(row["is_nucleus"]),
            nucleus_id=row["nucleus_id"],
            utility=row["utility"],
            momentum=row["momentum"],
            metadata=json.loads(row["metadata"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
