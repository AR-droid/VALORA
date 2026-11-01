"""SQLite persistence for decisions, agent runs, benchmark results and ledger blocks.

One file, no server: fine for a single-instance deployment (a Hugging Face Space
or one container). Every write is wrapped so a storage failure never takes the
simulation down.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    text TEXT,
    levers TEXT,
    status TEXT,
    tick INTEGER,
    predicted TEXT,
    realized TEXT,
    record TEXT
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    kind TEXT,
    goal TEXT,
    result TEXT
);
CREATE TABLE IF NOT EXISTS bench_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    scenario TEXT,
    agent TEXT,
    score REAL,
    metrics TEXT
);
CREATE TABLE IF NOT EXISTS blocks (
    idx INTEGER PRIMARY KEY,
    hash TEXT,
    block TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower())}


class Store:
    def __init__(self, path: str = "valora.db"):
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _exec(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self.lock:
            try:
                cur = self.conn.execute(sql, args)
                rows = cur.fetchall()
                self.conn.commit()
                return rows
            except sqlite3.Error as exc:
                print(f"[store] {exc}")
                return []

    # ------------------------------------------------------------- decisions
    def save_decision(self, record: dict[str, Any]) -> None:
        self._exec(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
            (record["id"], record.get("created_at", _now()), record.get("text", ""),
             json.dumps(record.get("levers", {})), record.get("status", ""), int(record.get("tick", 0)),
             json.dumps(record.get("predicted", {})), json.dumps(record.get("realized")),
             json.dumps(record, default=str)),
        )

    def set_realized(self, decision_id: str, realized: dict) -> None:
        rows = self._exec("SELECT record FROM decisions WHERE id=?", (decision_id,))
        if not rows:
            return
        rec = json.loads(rows[0][0])
        rec["realized"] = realized
        self._exec("UPDATE decisions SET realized=?, record=? WHERE id=?",
                   (json.dumps(realized), json.dumps(rec, default=str), decision_id))

    def decisions(self, limit: int = 50) -> list[dict]:
        rows = self._exec("SELECT record FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,))
        return [json.loads(r[0]) for r in rows]

    def get_decision(self, decision_id: str) -> dict | None:
        rows = self._exec("SELECT record FROM decisions WHERE id=?", (decision_id,))
        return json.loads(rows[0][0]) if rows else None

    def search_decisions(self, query: str, limit: int = 5) -> list[dict]:
        """Rank past decisions by word overlap with the query (simple, dependency-free recall)."""
        q = _tokens(query)
        scored = []
        for rec in self.decisions(limit=500):
            lever_words = " ".join(rec.get("levers", {}).keys()).replace("_", " ")
            overlap = len(q & _tokens(rec.get("text", "") + " " + lever_words))
            if overlap:
                scored.append((overlap, rec))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:limit]]

    # ------------------------------------------------------------ agent runs
    def save_run(self, run_id: str, kind: str, goal: str, result: dict) -> None:
        self._exec("INSERT OR REPLACE INTO agent_runs VALUES (?,?,?,?,?)",
                   (run_id, _now(), kind, goal, json.dumps(result, default=str)))

    def runs(self, kind: str | None = None, limit: int = 20) -> list[dict]:
        if kind:
            rows = self._exec("SELECT id, created_at, kind, goal, result FROM agent_runs WHERE kind=? "
                              "ORDER BY created_at DESC LIMIT ?", (kind, limit))
        else:
            rows = self._exec("SELECT id, created_at, kind, goal, result FROM agent_runs "
                              "ORDER BY created_at DESC LIMIT ?", (limit,))
        return [{"id": r[0], "created_at": r[1], "kind": r[2], "goal": r[3], "result": json.loads(r[4])}
                for r in rows]

    # ------------------------------------------------------------- benchmark
    def save_bench(self, scenario: str, agent: str, score: float, metrics: dict) -> None:
        self._exec("INSERT INTO bench_results (created_at, scenario, agent, score, metrics) VALUES (?,?,?,?,?)",
                   (_now(), scenario, agent, float(score), json.dumps(metrics, default=str)))

    def leaderboard(self) -> list[dict]:
        rows = self._exec(
            "SELECT scenario, agent, MAX(score), COUNT(*), MAX(created_at) FROM bench_results "
            "GROUP BY scenario, agent ORDER BY scenario, MAX(score) DESC"
        )
        return [{"scenario": r[0], "agent": r[1], "best_score": round(r[2], 3), "runs": r[3], "last_run": r[4]}
                for r in rows]

    # ---------------------------------------------------------------- ledger
    def save_block(self, block: dict) -> None:
        self._exec("INSERT OR REPLACE INTO blocks VALUES (?,?,?)",
                   (int(block["index"]), block["hash"], json.dumps(block, default=str)))

    def load_blocks(self) -> list[dict]:
        return [json.loads(r[0]) for r in self._exec("SELECT block FROM blocks ORDER BY idx")]
