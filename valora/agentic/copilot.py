"""Policy Copilot: a goal-driven, tool-using agent.

Give it a goal ("bring unemployment under 5% without pushing inflation over 4%").
It inspects the economy, recalls past decisions, simulates candidate policies,
stress-tests the best one and returns a proposal with the evidence behind it.

Two planners share one trace format:

* LLM planner: native tool calling against any OpenAI-compatible model, capped at
  ``AGENT_MAX_STEPS`` model calls.
* Offline planner: a deterministic search over candidate policies, used when no
  API key is set or the model fails, so the agent always returns a real answer.

Either way, the final proposal is re-simulated by the harness before it is shown,
so the numbers in the answer are checked, not just claimed by the model.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from .llm import LLMClient, parse_json
from .policy import clamp_levers, describe_effect, what_if
from .tools import ToolContext, build_tools, call_tool

SYSTEM = """You are VALORA Policy Copilot, an economic-policy agent working inside a simulated economy.
One simulation tick is one month. You can call tools to inspect the economy and simulate policies on a copy.
Method:
1. Call get_economy_status. 2. Call recall_past_decisions for similar goals.
3. Use compare_policies on 3-6 diverse candidates that serve the user's goal.
4. stress_test the best candidate under a plausible shock.
5. Answer.
Rules: never invent numbers, only quote tool results. Keep levers inside the documented ranges.
Final answer: a short explanation (under 180 words) of what you recommend and why, with the simulated effects,
then a JSON block in ```json fences with keys: label (string), levers (object), duration (months),
rationale (one sentence)."""

GOAL_TERMS = {
    "unemployment": ("mean_unemployment", -1.5),
    "jobs": ("mean_unemployment", -1.5),
    "employment": ("mean_unemployment", -1.5),
    "inflation": ("mean_inflation", -1.5),
    "prices": ("mean_inflation", -1.5),
    "growth": ("mean_gdp", 0.05),
    "gdp": ("mean_gdp", 0.05),
    "inequality": ("end_wealth_gini", -200.0),
    "poverty": ("end_wealth_gini", -200.0),
    "confidence": ("mean_consumer_confidence", 0.1),
}


def goal_weights(goal: str) -> dict[str, float]:
    """Weights per metric. Terms after 'without', 'while' or 'but' are constraints and count less."""
    g = goal.lower()
    m = re.search(r"\b(without|while|but|keeping|as long as)\b", g)
    head, tail = (g[: m.start()], g[m.start():]) if m else (g, "")
    w: dict[str, float] = {}
    for term, (metric, weight) in GOAL_TERMS.items():
        if term in head:
            w[metric] = weight
        elif term in tail and metric not in w:
            w[metric] = weight * 0.3
    return w


def goal_score(result: dict, weights: dict[str, float]) -> float:
    d = result["delta"]
    score = result["welfare_gain"] if "welfare_gain" in result else result["welfare"]["gain"]
    for metric, weight in weights.items():
        score += weight * d.get(metric, 0.0)
    debt = result.get("public_debt_delta", 0.0)
    return round(score - 0.0002 * max(0.0, debt), 4)


def candidate_policies(goal: str) -> list[dict]:
    """A diverse menu of single and combined levers, biased toward the goal."""
    g = goal.lower()
    menu = [
        {"label": "Infrastructure stimulus 3%", "levers": {"spending_change": 0.03}, "duration": 12},
        {"label": "Tax cut 3 points", "levers": {"tax_change": -0.03}, "duration": 12},
        {"label": "Targeted transfers 6%", "levers": {"transfer_change": 0.06}, "duration": 12},
        {"label": "Rate cut 1 point", "levers": {"rate_change": -0.01}, "duration": 12},
        {"label": "Rate hike 1 point", "levers": {"rate_change": 0.01}, "duration": 12},
        {"label": "Deregulation", "levers": {"regulation_change": -0.5}, "duration": 18},
        {"label": "Transfers 5% + rate hike 0.5", "levers": {"transfer_change": 0.05, "rate_change": 0.005},
         "duration": 12},
        {"label": "Fiscal consolidation", "levers": {"spending_change": -0.03, "tax_change": 0.02}, "duration": 12},
    ]
    if any(k in g for k in ["inflation", "prices", "overheat"]):
        menu += [{"label": "Rate hike 2 points", "levers": {"rate_change": 0.02}, "duration": 12},
                 {"label": "Spending restraint + rate hike", "levers": {"spending_change": -0.02, "rate_change": 0.01},
                  "duration": 12}]
    if any(k in g for k in ["unemployment", "jobs", "recession", "growth"]):
        menu += [{"label": "Big stimulus 6%", "levers": {"spending_change": 0.06}, "duration": 12},
                 {"label": "Stimulus 4% + rate cut 1", "levers": {"spending_change": 0.04, "rate_change": -0.01},
                  "duration": 12}]
    if any(k in g for k in ["inequality", "poverty", "low-income", "welfare"]):
        menu += [{"label": "Transfers 10% funded by tax +2", "levers": {"transfer_change": 0.10, "tax_change": 0.02},
                  "duration": 18}]
    return menu


class PolicyCopilot:
    def __init__(self, ctx: ToolContext, llm: LLMClient | None = None, max_steps: int | None = None):
        self.ctx = ctx
        self.llm = llm or LLMClient()
        self.tools = build_tools(ctx)
        self.max_steps = max_steps or self.llm.s.agent_max_steps

    # --------------------------------------------------------------- public
    def run(self, goal: str, horizon: int = 24) -> dict[str, Any]:
        goal = (goal or "").strip() or "Maximise overall welfare: steady growth, low unemployment, ~2% inflation."
        started = datetime.now(timezone.utc).isoformat()
        result = None
        if self.llm.enabled:
            result = self._run_llm(goal, horizon)
        if result is None:
            result = self._run_offline(goal, horizon)
        proposal = result.get("proposal")
        if proposal:
            verified = what_if(self.ctx.economy, proposal["levers"], proposal.get("duration", 12), horizon,
                               label=proposal.get("label", "copilot"))
            proposal["verified_effect"] = describe_effect(verified)
            proposal["welfare_gain"] = verified["welfare"]["gain"]
            proposal["goal_score"] = goal_score(verified, goal_weights(goal))
            result["series"] = verified["series"]
            result["delta"] = verified["delta"]
        result.update({"id": str(uuid.uuid4()), "goal": goal, "started_at": started,
                       "model": self.llm.model, "usage": self.llm.usage.as_dict()})
        return result

    # --------------------------------------------------------- LLM planner
    def _run_llm(self, goal: str, horizon: int) -> dict | None:
        messages: list[dict] = [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": f"Goal: {goal}\nUse a horizon of {horizon} months."}]
        schemas = [t.schema() for t in self.tools.values()]
        trace: list[dict] = []
        for step in range(self.max_steps):
            msg = self.llm.chat(messages, tools=schemas, max_tokens=900)
            if msg is None:
                return None if not trace else self._finish_from_trace(goal, trace, horizon, note="model stopped early")
            calls = msg.get("tool_calls") or []
            if msg.get("content"):
                trace.append({"step": step + 1, "type": "thought", "text": msg["content"][:1200]})
            if not calls:
                return self._final_from_text(goal, msg.get("content") or "", trace, horizon)
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls[:4]:
                fn = c.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments", "{}")
                out = call_tool(self.tools, name, args)
                trace.append({"step": step + 1, "type": "tool", "tool": name, "args": _safe_json(args),
                              "result": _shorten(out)})
                messages.append({"role": "tool", "tool_call_id": c.get("id", name), "name": name,
                                 "content": json.dumps(out, default=str)[:6000]})
        messages.append({"role": "user", "content": "Step budget reached. Give your final answer now."})
        msg = self.llm.chat(messages, max_tokens=700)
        if msg and msg.get("content"):
            return self._final_from_text(goal, msg["content"], trace, horizon)
        return self._finish_from_trace(goal, trace, horizon, note="step budget reached")

    def _final_from_text(self, goal: str, text: str, trace: list[dict], horizon: int) -> dict:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
        data = parse_json(m.group(1)) if m else None
        answer = re.sub(r"```json.*?```", "", text, flags=re.S).strip()
        if data and clamp_levers(data.get("levers", {})):
            proposal = {"label": str(data.get("label", "Copilot proposal"))[:120],
                        "levers": clamp_levers(data["levers"]),
                        "duration": int(min(60, max(1, int(data.get("duration", 12) or 12)))),
                        "rationale": str(data.get("rationale", ""))[:400]}
            trace.append({"step": len(trace) + 1, "type": "answer", "text": answer[:2000]})
            return {"answer": answer, "proposal": proposal, "trace": trace, "planner": "llm"}
        return self._finish_from_trace(goal, trace, horizon, note="model gave no valid proposal", answer=answer)

    def _finish_from_trace(self, goal: str, trace: list[dict], horizon: int, note: str, answer: str = "") -> dict:
        offline = self._run_offline(goal, horizon)
        offline["trace"] = trace + [{"step": len(trace) + 1, "type": "thought",
                                     "text": f"{note}; falling back to the deterministic planner."}] + offline["trace"]
        if answer:
            offline["answer"] = answer + "\n\n" + offline["answer"]
        offline["planner"] = "llm+offline"
        return offline

    # ----------------------------------------------------- offline planner
    def _run_offline(self, goal: str, horizon: int) -> dict:
        trace: list[dict] = []
        weights = goal_weights(goal)
        status = call_tool(self.tools, "get_economy_status", {})
        ind = status.get("indicators", {})
        trace.append({"step": 1, "type": "tool", "tool": "get_economy_status", "args": {},
                      "result": {"phase": status.get("current_cycle"), "indicators": ind}})
        memory = call_tool(self.tools, "recall_past_decisions", {"query": goal})
        trace.append({"step": 2, "type": "tool", "tool": "recall_past_decisions", "args": {"query": goal},
                      "result": {"matches": len(memory.get("matches", []))}})
        trace.append({"step": 3, "type": "thought", "text":
                      f"Goal weights {({k: round(v, 2) for k, v in weights.items()}) or 'balanced welfare'}. Economy is in {status.get('current_cycle')} with "
                      f"inflation {ind.get('inflation')}% and unemployment {ind.get('unemployment')}%. "
                      "Simulating a menu of candidate policies."})
        menu = candidate_policies(goal)
        ranked = []
        for opt in menu:
            r = what_if(self.ctx.economy, opt["levers"], opt["duration"], horizon, label=opt["label"])
            ranked.append((goal_score(r, weights), opt, r))
        ranked.sort(key=lambda x: -x[0])
        trace.append({"step": 4, "type": "tool", "tool": "compare_policies",
                      "args": {"options": [o["label"] for o in menu], "horizon": horizon},
                      "result": [{"label": o["label"], "goal_score": s, "welfare_gain": r["welfare"]["gain"]}
                                 for s, o, r in ranked[:5]]})
        # Refine: scale the best candidate up and down, and blend it with the runner-up.
        best_score, best, best_r = ranked[0]
        refinements = []
        for scale in (0.5, 1.5):
            refinements.append({"label": f"{best['label']} x{scale}",
                                "levers": {k: v * scale for k, v in best["levers"].items()},
                                "duration": best["duration"]})
        second = ranked[1][1]
        blend = dict(best["levers"])
        for k, v in second["levers"].items():
            blend[k] = blend.get(k, 0.0) + 0.5 * v
        refinements.append({"label": f"{best['label']} + half of {second['label']}", "levers": blend,
                            "duration": best["duration"]})
        for opt in refinements:
            lv = clamp_levers(opt["levers"])
            r = what_if(self.ctx.economy, lv, opt["duration"], horizon, label=opt["label"])
            s = goal_score(r, weights)
            if s > best_score:
                best_score, best, best_r = s, {**opt, "levers": lv}, r
        trace.append({"step": 5, "type": "tool", "tool": "compare_policies",
                      "args": {"options": [o["label"] for o in refinements]},
                      "result": {"chosen": best["label"], "goal_score": best_score}})
        stress = call_tool(self.tools, "stress_test", {"shock_type": "energy_price_spike", "magnitude": 1.0,
                                                      **best["levers"]})
        trace.append({"step": 6, "type": "tool", "tool": "stress_test",
                      "args": {"shock_type": "energy_price_spike", **best["levers"]},
                      "result": {"welfare_gain_under_shock": stress.get("welfare_gain")}})
        d = best_r["delta"]
        answer = (
            f"**Recommendation: {best['label']}** for {best['duration']} months.\n\n"
            f"Simulated against doing nothing over {horizon} months: unemployment {d['mean_unemployment']:+.2f} pts, "
            f"inflation {d['mean_inflation']:+.2f} pts, GDP {d['mean_gdp']:+.1f}, "
            f"wealth Gini {d['end_wealth_gini']:+.3f}, public debt {best_r['public_debt_delta']:+.0f}. "
            f"It ranked first of {len(menu) + len(refinements)} candidates on your goal "
            f"and still gains {stress.get('welfare_gain', 0):+.2f} welfare points under an energy price shock."
        )
        if memory.get("matches"):
            answer += f"\n\nMemory: {len(memory['matches'])} similar past decisions were considered."
        trace.append({"step": 7, "type": "answer", "text": answer})
        proposal = {"label": best["label"], "levers": clamp_levers(best["levers"]), "duration": best["duration"],
                    "rationale": f"Best goal score ({best_score:+.2f}) among simulated candidates."}
        return {"answer": answer, "proposal": proposal, "trace": trace, "planner": "offline-search",
                "candidates": [{"label": o["label"], "goal_score": s} for s, o, _ in ranked]}


def _safe_json(args: Any) -> Any:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args[:500]
    return args


def _shorten(out: Any, limit: int = 1500) -> Any:
    text = json.dumps(out, default=str)
    if len(text) <= limit:
        return out
    return {"truncated": text[:limit] + "..."}
