"""The running platform: one live economy, its ledger, memory, and the agents.

Agents never touch the live economy directly. They work on a consistent copy
(``snapshot_ctx``); only ``submit_policy`` changes the live economy, and only after
the council's vote and the lever guardrails.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from .agentic.copilot import PolicyCopilot
from .agentic.council import EconomicCouncil
from .agentic.llm import LLMClient
from .agentic.policy import clamp_levers
from .agentic.tools import ToolContext
from .config import Settings, get_settings
from .engine import Economy, EconomyConfig, Policy
from .harness.bench import run_benchmark
from .ledger import Ledger
from .store import Store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Runtime:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.lock = threading.RLock()
        self.store = Store(self.s.db_path)
        self.ledger = Ledger(on_block=self.store.save_block)
        self.ledger.load(self.store.load_blocks())
        self.economy = Economy(EconomyConfig(seed=self.s.seed))
        self.llm = LLMClient(self.s)
        self.council_runs: dict[str, dict] = {}
        self.copilot_runs: dict[str, dict] = {}
        self.bench_job: dict[str, Any] = {"state": "idle"}
        self._ticker: threading.Thread | None = None
        self._stop = threading.Event()
        self.ledger.record("genesis_economy", {"seed": self.s.seed, **self.economy.indicators()},
                           memo="economy started")

    # --------------------------------------------------------------- economy
    def snapshot_ctx(self) -> ToolContext:
        with self.lock:
            copy = self.economy.fork()
        return ToolContext(economy=copy, store=self.store, ledger=self.ledger)

    def step(self, n: int = 1) -> list[dict]:
        snaps = []
        with self.lock:
            for _ in range(max(1, min(120, n))):
                snap = self.economy.step()
                snaps.append(snap)
                self.ledger.record("tick", snap, memo=f"tick {snap['tick']} ({snap['phase']})")
                if snap["tick"] % 3 == 0:
                    self.ledger.mine()
            self._evaluate_due_decisions()
        return snaps

    def apply_shock(self, shock_type: str, magnitude: float, duration: int) -> dict:
        with self.lock:
            s = self.economy.apply_shock(shock_type, magnitude, duration)
        self.ledger.record("shock", {"type": s.type, "magnitude": s.magnitude, "duration": s.duration},
                           memo=f"shock {s.type} x{s.magnitude}")
        return self.economy.status()

    def start(self) -> None:
        if self.s.tick_seconds <= 0 or self._ticker:
            return

        def loop():
            while not self._stop.wait(self.s.tick_seconds):
                try:
                    self.step(1)
                except Exception:
                    traceback.print_exc()

        self._ticker = threading.Thread(target=loop, daemon=True, name="valora-ticker")
        self._ticker.start()

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- council
    def deliberate(self, text: str, *, apply: bool = False, author: str = "api",
                   levers: dict | None = None, duration: int | None = None) -> dict:
        ctx = self.snapshot_ctx()
        council = EconomicCouncil(ctx, llm=self.llm)
        if levers:
            orig_parse = council.parse
            council.parse = lambda t: {"levers": clamp_levers(levers), "duration": int(duration or 12),  # type: ignore[method-assign]
                                       "parser": "given", "summary": ""} if clamp_levers(levers) else orig_parse(t)
        result = council.deliberate(text)
        result["author"] = author
        status = "rejected"
        if result["consensus"]["consensus_reached"]:
            status = "approved"
            if apply:
                pol = Policy(label=text[:80], levers=result["final_levers"], duration=result["final_duration"],
                             id=result["id"], source=f"council:{author}")
                with self.lock:
                    applied = self.economy.apply_policy(pol)
                    result["applied_at_tick"] = self.economy.tick
                result["applied_levers"] = applied.levers
                status = "applied"
        result["status"] = status
        self.council_runs[result["id"]] = result
        if len(self.council_runs) > 50:
            self.council_runs.pop(next(iter(self.council_runs)))
        self.store.save_decision({
            "id": result["id"], "created_at": result["created_at"], "text": text,
            "levers": result["final_levers"], "status": status, "tick": self.economy.tick,
            "predicted": result["predicted"], "realized": None, "verdict": result["verdict"],
            "consensus": result["consensus"], "mode": result["mode"], "author": author,
            "start_indicators": self.economy.indicators(),
        })
        self.ledger.record("policy_decision", {
            "decision_id": result["id"], "text": text, "levers": result["final_levers"], "status": status,
            "votes": result["consensus"]["votes"], "verdict": result["verdict"]["decision"],
            "predicted": result["predicted"]["summary"],
        }, actor="council", memo=f"{status}: {text[:60]}")
        return result

    def _evaluate_due_decisions(self) -> None:
        """Compare what the council predicted with what actually happened, 12 months on."""
        for rec in self.store.decisions(limit=100):
            if rec.get("status") != "applied" or rec.get("realized"):
                continue
            if self.economy.tick - int(rec.get("tick", 0)) < 12:
                continue
            start = rec.get("start_indicators") or {}
            now = self.economy.indicators()
            realized = {
                "evaluated_at_tick": self.economy.tick,
                "change": {k: round(now[k] - start.get(k, now[k]), 3) for k in
                           ["gdp", "inflation", "unemployment", "wealth_gini"]},
                "predicted_policy_end": (rec.get("predicted") or {}).get("policy_path_end"),
            }
            self.store.set_realized(rec["id"], realized)
            self.ledger.record("outcome_review", {"decision_id": rec["id"], **realized}, actor="harness",
                               memo=f"outcome review: {rec.get('text', '')[:50]}")

    def revoke(self, policy_id: str) -> bool:
        with self.lock:
            ok = self.economy.revoke_policy(policy_id)
        if ok:
            self.ledger.record("policy_revoked", {"policy_id": policy_id}, memo="policy revoked")
        return ok

    # -------------------------------------------------------------- copilot
    def copilot(self, goal: str, horizon: int = 24) -> dict:
        ctx = self.snapshot_ctx()
        res = PolicyCopilot(ctx, llm=self.llm).run(goal, horizon=horizon)
        self.copilot_runs[res["id"]] = res
        if len(self.copilot_runs) > 50:
            self.copilot_runs.pop(next(iter(self.copilot_runs)))
        self.store.save_run(res["id"], "copilot", res["goal"], {k: v for k, v in res.items() if k != "series"})
        prop = res.get("proposal") or {}
        self.ledger.record("copilot_run", {"run_id": res["id"], "goal": res["goal"], "proposal": prop.get("label"),
                                           "levers": prop.get("levers"), "planner": res.get("planner")},
                           actor="copilot", memo=f"copilot: {res['goal'][:60]}")
        return res

    # ------------------------------------------------------------ benchmark
    def start_bench(self, agents: list[str] | None = None, scenarios: list[str] | None = None) -> dict:
        if self.bench_job.get("state") == "running":
            return self.bench_job
        job = {"id": str(uuid.uuid4()), "state": "running", "started_at": _now(), "progress": [], "result": None}
        self.bench_job = job

        def work():
            try:
                res = run_benchmark(agents, scenarios, store=self.store,
                                    progress=lambda m: job["progress"].append(m))
                for r in res["results"]:
                    r.pop("series", None)
                job["result"] = res
                job["state"] = "done"
                self.ledger.record("benchmark", {"ranking": res["ranking"]}, actor="harness", memo="benchmark run")
            except Exception as exc:
                job["state"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            job["finished_at"] = _now()

        threading.Thread(target=work, daemon=True, name="valora-bench").start()
        return job

    def meta(self) -> dict:
        return {"llm_enabled": self.llm.enabled, "model": self.llm.model, "tick": self.economy.tick,
                "tick_seconds": self.s.tick_seconds, "usage": self.llm.usage.as_dict(),
                "regulator": type(self.economy.regulator).__name__}


_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> Runtime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = Runtime()
            _runtime.start()
        return _runtime


def reset_runtime(rt: Runtime | None) -> None:
    """For tests: install a specific runtime (or clear it)."""
    global _runtime
    with _runtime_lock:
        if _runtime and _runtime is not rt:
            _runtime.stop()
        _runtime = rt


__all__ = ["Runtime", "get_runtime", "reset_runtime"]
