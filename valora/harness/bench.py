"""Benchmark harness: run policy-making agents through fixed, seeded scenarios.

A *policy agent* is anything with ``decide(economy) -> dict | None`` that returns
levers (or ``None`` to keep the current stance). Every ``decide_every`` months the
harness asks the agent for a decision, applies it through the same guardrails the
live app uses, and at the end scores the whole run with ``welfare_score``.

Same scenario + same seed = same shocks and same agent luck, so scores are
comparable across agents and reproducible across machines. Each run's tick
snapshots are hashed into a chain; the final hash is the run's fingerprint.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Protocol

import numpy as np
import yaml

from ..agentic.copilot import PolicyCopilot
from ..agentic.llm import LLMClient
from ..agentic.policy import clamp_levers, welfare_score
from ..agentic.tools import ToolContext
from ..engine import Economy, EconomyConfig, Policy
from ..ledger.chain import GENESIS, hash_json


@dataclass
class Scenario:
    name: str
    description: str
    seed: int
    ticks: int
    shocks: list[dict] = field(default_factory=list)


def load_scenarios() -> dict[str, Scenario]:
    out = {}
    for f in sorted(resources.files("valora.harness").joinpath("scenarios").iterdir(), key=lambda p: p.name):
        if f.name.endswith((".yaml", ".yml")):
            d = yaml.safe_load(f.read_text())
            out[d["name"]] = Scenario(d["name"], d.get("description", ""), int(d.get("seed", 0)),
                                      int(d.get("ticks", 36)), list(d.get("shocks") or []))
    return out


class PolicyAgent(Protocol):
    name: str

    def decide(self, economy: Economy) -> dict | None: ...


class NoPolicy:
    name = "no-policy"

    def decide(self, economy: Economy) -> dict | None:
        return None


class FixedStimulus:
    name = "fixed-stimulus"

    def decide(self, economy: Economy) -> dict | None:
        return {"spending_change": 0.02}


class TaylorRule:
    """Classic rule: raise rates when inflation is above 2%, cut when unemployment is high."""

    name = "taylor-rule"

    def decide(self, economy: Economy) -> dict | None:
        rate = 0.5 * (economy.inflation - 2.0) / 100 - 0.5 * (economy.unemployment - 5.0) / 100
        spend = 0.03 if economy.unemployment > 8 else 0.0
        return {"rate_change": rate, "spending_change": spend}


class CopilotAgent:
    """The Policy Copilot used as a closed-loop policy-maker."""

    def __init__(self, use_llm: bool = False, goal: str = "Maximise overall welfare", horizon: int = 12):
        self.use_llm = use_llm
        self.goal = goal
        self.horizon = horizon
        self.name = "copilot-llm" if use_llm else "copilot-offline"
        self.llm = LLMClient() if use_llm else _OfflineLLM()

    def decide(self, economy: Economy) -> dict | None:
        res = PolicyCopilot(ToolContext(economy), llm=self.llm).run(self.goal, horizon=self.horizon)
        prop = res.get("proposal") or {}
        if prop.get("welfare_gain", 0) <= 0:
            return None
        return prop.get("levers")


class _OfflineLLM(LLMClient):
    @property
    def enabled(self) -> bool:
        return False


AGENTS: dict[str, Callable[[], Any]] = {
    "no-policy": NoPolicy,
    "fixed-stimulus": FixedStimulus,
    "taylor-rule": TaylorRule,
    "copilot-offline": lambda: CopilotAgent(use_llm=False),
    "copilot-llm": lambda: CopilotAgent(use_llm=True),
}


def run_scenario(scenario: Scenario, agent: PolicyAgent, decide_every: int = 6) -> dict[str, Any]:
    t0 = time.time()
    econ = Economy(EconomyConfig(seed=scenario.seed))
    start_gdp = econ.gdp
    shocks = {}
    for s in scenario.shocks:
        shocks.setdefault(int(s["tick"]), []).append(s)
    history: list[dict] = []
    decisions: list[dict] = []
    chain = GENESIS
    for t in range(scenario.ticks):
        for s in shocks.get(t, []):
            econ.apply_shock(s["type"], float(s.get("magnitude", 1.0)), int(s.get("duration", 6)))
        if t % decide_every == 0:
            levers = clamp_levers(agent.decide(econ) or {})
            econ.policies = [p for p in econ.policies if p.source != "bench-agent"]
            if levers:
                econ.apply_policy(Policy(label=f"{agent.name}@{t}", levers=levers, duration=decide_every,
                                         source="bench-agent"))
            decisions.append({"tick": t, "levers": levers})
        snap = econ.step()
        chain = hash_json({"prev": chain, "snap": snap})
        history.append(snap)
    infl = [h["inflation"] for h in history]
    unemp = [h["unemployment"] for h in history]
    return {
        "scenario": scenario.name,
        "agent": agent.name,
        "score": welfare_score(history, start_gdp),
        "metrics": {
            "mean_inflation": round(float(np.mean(infl)), 3),
            "max_inflation": round(float(np.max(infl)), 3),
            "mean_unemployment": round(float(np.mean(unemp)), 3),
            "max_unemployment": round(float(np.max(unemp)), 3),
            "gdp_end": round(history[-1]["gdp"], 2),
            "wealth_gini_end": history[-1]["wealth_gini"],
            "public_debt": round(econ.public_debt, 1),
        },
        "decisions": decisions,
        "run_hash": chain,
        "seconds": round(time.time() - t0, 2),
        "series": {k: [h[k] for h in history] for k in ["gdp", "inflation", "unemployment"]},
    }


def run_benchmark(agent_names: list[str] | None = None, scenario_names: list[str] | None = None,
                  store=None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    scenarios = load_scenarios()
    names = scenario_names or list(scenarios)
    agents = agent_names or ["no-policy", "fixed-stimulus", "taylor-rule", "copilot-offline"]
    results = []
    for sname in names:
        if sname not in scenarios:
            continue
        for aname in agents:
            if aname not in AGENTS:
                continue
            if progress:
                progress(f"{sname} / {aname}")
            r = run_scenario(scenarios[sname], AGENTS[aname]())
            results.append(r)
            if store is not None:
                store.save_bench(sname, aname, r["score"], {**r["metrics"], "run_hash": r["run_hash"]})
    table: dict[str, dict[str, float]] = {}
    for r in results:
        table.setdefault(r["agent"], {})[r["scenario"]] = r["score"]
    ranking = sorted(({"agent": a, "mean_score": round(float(np.mean(list(v.values()))), 3), "scores": v}
                      for a, v in table.items()), key=lambda x: -x["mean_score"])
    return {"ranking": ranking, "results": results,
            "scenarios": {n: scenarios[n].description for n in names if n in scenarios}}
