"""Tools the agents can call. Each tool has a JSON Schema so an LLM can call it.

Tools only *read* the live economy or simulate on a fork. Nothing here changes the
live economy: applying a policy goes through the council's vote and guardrails.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..engine import LEVER_BOUNDS, SHOCK_PROFILES, Economy
from ..engine.actors import simulate_economy
from ..ledger import Ledger
from ..store import Store
from .policy import LEVER_HELP, clamp_levers, describe_effect, parse_policy_text, what_if


@dataclass
class ToolContext:
    economy: Economy
    store: Store | None = None
    ledger: Ledger | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


LEVER_PROPS = {k: {"type": "number", "description": f"{LEVER_HELP[k]}; allowed {lo} to {hi}"}
               for k, (lo, hi) in LEVER_BOUNDS.items()}


def _compact_whatif(r: dict) -> dict:
    return {"levers": r["levers"], "duration": r["duration"], "horizon": r["horizon"],
            "delta": {k: v for k, v in r["delta"].items() if k.startswith("mean_") or k == "end_wealth_gini"},
            "welfare_gain": r["welfare"]["gain"], "public_debt_delta": r["public_debt_delta"],
            "summary": describe_effect(r)}


def build_tools(ctx: ToolContext) -> dict[str, Tool]:
    def get_economy_status() -> dict:
        st = ctx.economy.status()
        st["recent"] = ctx.economy.history[-6:]
        st["agents"] = ctx.economy.agents_status()
        return st

    def simulate_policy(duration: int = 12, horizon: int = 24, **levers) -> dict:
        return _compact_whatif(what_if(ctx.economy, levers, int(duration), int(horizon)))

    def compare_policies(options: list[dict], horizon: int = 24) -> dict:
        results = []
        for opt in (options or [])[:6]:
            lv = clamp_levers(opt.get("levers", opt))
            r = what_if(ctx.economy, lv, int(opt.get("duration", 12)), int(horizon))
            results.append({"label": opt.get("label", json.dumps(lv)), **_compact_whatif(r)})
        results.sort(key=lambda x: -x["welfare_gain"])
        return {"ranked": results}

    def stress_test(shock_type: str, magnitude: float = 1.0, duration: int = 12, horizon: int = 24, **levers) -> dict:
        """Run the policy under a shock, versus the same shock with no policy."""
        shock = [{"type": shock_type, "magnitude": magnitude, "duration": duration}]
        return _compact_whatif(what_if(ctx.economy, levers, 12, int(horizon), shocks=shock))

    def sector_impact(policy_text: str) -> dict:
        res = simulate_economy(policy_text, time_steps=6)
        return {"signals": res["policy_signals"], "most_affected": res["selected_actors"],
                "composite_end": {k: v[-1] for k, v in res["actor_series"].items()}}

    def parse_policy(policy_text: str) -> dict:
        return parse_policy_text(policy_text)

    def recall_past_decisions(query: str) -> dict:
        if not ctx.store:
            return {"matches": []}
        out = []
        for rec in ctx.store.search_decisions(query, limit=4):
            out.append({"text": rec.get("text"), "levers": rec.get("levers"), "status": rec.get("status"),
                        "predicted": (rec.get("predicted") or {}).get("summary"),
                        "realized": rec.get("realized")})
        return {"matches": out}

    def ledger_audit() -> dict:
        if not ctx.ledger:
            return {"available": False}
        ok, errors = ctx.ledger.verify()
        return {"valid": ok, "errors": errors[:5], **ctx.ledger.status()}

    tools = [
        Tool("get_economy_status", "Current indicators, business-cycle phase, active policies and shocks, "
             "the last 6 months, and what the household, firm, bank and regulator agents are doing.",
             {"type": "object", "properties": {}}, get_economy_status),
        Tool("simulate_policy", "Simulate a policy on a copy of the economy and report its effect versus no "
             "policy over the horizon (months). Use this before recommending anything.",
             {"type": "object", "properties": {**LEVER_PROPS,
                                               "duration": {"type": "integer", "description": "months active, 1-60"},
                                               "horizon": {"type": "integer", "description": "months to simulate, 3-72"}}},
             simulate_policy),
        Tool("compare_policies", "Simulate up to 6 candidate policies and rank them by welfare gain.",
             {"type": "object", "properties": {
                 "options": {"type": "array", "items": {"type": "object", "properties": {
                     "label": {"type": "string"}, "duration": {"type": "integer"},
                     "levers": {"type": "object", "properties": LEVER_PROPS}}}},
                 "horizon": {"type": "integer"}}, "required": ["options"]},
             compare_policies),
        Tool("stress_test", "Check how a policy holds up under a shock (" + ", ".join(SHOCK_PROFILES) + ").",
             {"type": "object", "properties": {"shock_type": {"type": "string", "enum": list(SHOCK_PROFILES)},
                                               "magnitude": {"type": "number"}, **LEVER_PROPS},
              "required": ["shock_type"]}, stress_test),
        Tool("sector_impact", "Which sectors (households, firms, banks, exporters, energy...) a policy text "
             "affects most, from the sector model.",
             {"type": "object", "properties": {"policy_text": {"type": "string"}}, "required": ["policy_text"]},
             sector_impact),
        Tool("parse_policy", "Convert policy text into numeric levers with the keyword parser.",
             {"type": "object", "properties": {"policy_text": {"type": "string"}}, "required": ["policy_text"]},
             parse_policy),
        Tool("recall_past_decisions", "Search memory for similar past policy decisions and how they turned out.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
             recall_past_decisions),
        Tool("ledger_audit", "Verify the hash-chained audit ledger.", {"type": "object", "properties": {}},
             ledger_audit),
    ]
    return {t.name: t for t in tools}


def call_tool(tools: dict[str, Tool], name: str, args: dict | str | None) -> dict:
    if name not in tools:
        return {"error": f"unknown tool {name}"}
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return {"error": "arguments were not valid JSON"}
    try:
        return tools[name].fn(**(args or {}))
    except TypeError as exc:
        return {"error": f"bad arguments: {exc}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
