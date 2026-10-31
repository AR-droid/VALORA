"""Tamper-evident ledger: SHA-256 hash chain of blocks with Merkle roots.

Every economy tick, policy decision and agent run is recorded here, so any run can
be audited after the fact: change one record and every later block hash breaks.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

GENESIS = "0" * 64


def hash_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def merkle_root(items: list[dict]) -> str:
    if not items:
        return hash_json([])
    layer = [hash_json(x) for x in items]
    while len(layer) > 1:
        layer = [
            hash_json({"left": layer[i], "right": layer[i + 1] if i + 1 < len(layer) else layer[i]})
            for i in range(0, len(layer), 2)
        ]
    return layer[0]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    BLOCK_FIELDS = ("index", "timestamp", "previous_hash", "merkle_root", "transactions")

    def __init__(self, on_block: Callable[[dict], None] | None = None, max_audit: int = 500):
        self.blocks: list[dict] = []
        self.pending: list[dict] = []
        self.audit: list[dict] = []
        self.lock = threading.RLock()
        self.on_block = on_block
        self.max_audit = max_audit

    # ------------------------------------------------------------------ writes
    def record(self, kind: str, payload: dict, *, actor: str = "valora", memo: str = "") -> dict:
        """Queue a structured record (tick snapshot, policy decision, agent run...)."""
        tx = {
            "id": str(uuid.uuid4()),
            "kind": kind,
            "from": actor,
            "to": "ledger",
            "amount": float(payload.get("amount", 0.0)) if isinstance(payload, dict) else 0.0,
            "memo": memo or kind,
            "payload_hash": hash_json(payload),
            "payload": payload,
            "timestamp": _now(),
            "status": "pending",
        }
        tx["hash"] = hash_json({k: v for k, v in tx.items() if k != "status"})
        with self.lock:
            self.pending.append(tx)
            self._audit({"action": f"recorded:{kind}", "from": actor, "to": "ledger", "memo": tx["memo"],
                         "tx_id": tx["id"], "hash": tx["hash"]})
        return tx

    def add_transaction(self, tx: dict) -> dict:
        """Backwards-compatible value transfer (from, to, amount, memo)."""
        if not isinstance(tx, dict):
            raise ValueError("Transaction must be a JSON object")
        missing = [f for f in ("from", "to", "amount") if f not in tx]
        if missing:
            raise ValueError(f"Missing transaction fields: {', '.join(missing)}")
        amount = float(tx["amount"])
        if amount <= 0:
            raise ValueError("Transaction amount must be positive")
        rec = self.record("transfer", {"amount": amount, "from": str(tx["from"]), "to": str(tx["to"])},
                          actor=str(tx["from"]), memo=str(tx.get("memo", "")))
        rec["to"] = str(tx["to"])
        return rec

    def mine(self) -> dict | None:
        with self.lock:
            if not self.pending:
                return None
            txs = [{**tx, "status": "confirmed"} for tx in self.pending]
            core = {
                "index": len(self.blocks) + 1,
                "timestamp": _now(),
                "previous_hash": self.blocks[-1]["hash"] if self.blocks else GENESIS,
                "merkle_root": merkle_root(txs),
                "transactions": txs,
            }
            block = {**core, "hash": hash_json(core)}
            self.blocks.append(block)
            self.pending = []
            self._audit({"action": "mined", "block_index": block["index"], "hash": block["hash"],
                         "tx_count": len(txs)})
        if self.on_block:
            try:
                self.on_block(block)
            except Exception as exc:  # persistence must never break the chain
                print(f"[ledger] on_block failed: {exc}")
        return block

    def load(self, blocks: list[dict]) -> None:
        with self.lock:
            self.blocks = list(blocks)

    def _audit(self, entry: dict) -> None:
        entry.setdefault("timestamp", _now())
        self.audit.append(entry)
        if len(self.audit) > self.max_audit:
            del self.audit[: len(self.audit) - self.max_audit]

    # ------------------------------------------------------------------- reads
    def verify(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        with self.lock:
            for i, block in enumerate(self.blocks):
                expected_prev = GENESIS if i == 0 else self.blocks[i - 1]["hash"]
                if block.get("previous_hash") != expected_prev:
                    errors.append(f"Block {block.get('index')} previous_hash mismatch")
                if block.get("hash") != hash_json({k: block[k] for k in self.BLOCK_FIELDS}):
                    errors.append(f"Block {block.get('index')} hash invalid")
                if block.get("merkle_root") != merkle_root(block.get("transactions", [])):
                    errors.append(f"Block {block.get('index')} merkle root invalid")
        return not errors, errors

    def status(self) -> dict:
        with self.lock:
            return {
                "ledger_height": len(self.blocks),
                "pending_count": len(self.pending),
                "tip_hash": self.blocks[-1]["hash"] if self.blocks else None,
            }

    def get_block(self, index: int) -> dict | None:
        with self.lock:
            return self.blocks[index - 1] if 1 <= index <= len(self.blocks) else None

    def get_blocks(self, limit: int | None = None) -> list[dict]:
        with self.lock:
            data = list(self.blocks)
        return data[-limit:] if limit else data

    def get_pending(self) -> list[dict]:
        with self.lock:
            return list(self.pending)

    def get_audit(self, limit: int | None = None) -> list[dict]:
        with self.lock:
            data = list(self.audit)
        return data[-limit:] if limit else data
