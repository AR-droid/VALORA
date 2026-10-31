"""Economic Council: a multi-agent deliberation that ends in a vote.

Pipeline for one policy proposal:

1. Parser agent turns the text into bounded levers (LLM JSON, checked against the
   keyword parser; guardrails clamp every value).
2. The harness simulates the policy against a no-policy baseline and under a
   stress shock. This evidence is shared with every agent.
3. Economic Analyst and Tax Advisor write their assessments.
4. Policy Supporter and Policy Opposer debate for two rounds, each seeing the
   other's last argument.
5. Policy Critic gives a verdict: Proceed, Modify (with adjusted levers) or Pause.
6. Every agent votes; votes are weighted and the policy passes on a majority.

With an LLM key each agent is a model call with a role prompt and JSON output.
Without one, each agent is a rule-based reasoner over the same evidence, so the
council still produces grounded, numeric arguments.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from ..engine.actors import simulate_economy
from .llm import LLMClient
from .policy import LEVER_HELP, clamp_levers, describe_effect, parse_policy_text, what_if
from .tools import ToolContext

ROLES = {
    "Economic Analyst": {"weight": 1.0, "brief": "Assess macro effects: growth, jobs, inflation, and the cycle."},
    "Tax Advisor": {"weight": 1.0, "brief": "Assess fiscal cost, public debt, tax incidence and distribution."},
    "Policy Supporter": {"weight": 0.5, "brief": "Argue for the policy using the evidence."},
    "Policy Opposer": {"weight": 0.5, "brief": "Argue against the policy: risks, trade-offs, failure modes."},
    "Policy Critic": {"weight": 1.5, "brief": "Neutral judge. Decide Proceed, Modify or Pause."},
}

ROLE_SYSTEM = (
    "You are the {role} on VALORA's economic council, deliberating about a policy for a simulated economy "
    "(one tick = one month). {brief} Use only the numbers in the evidence; do not invent data. "
    "Reply in JSON."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EconomicCouncil:
    def __init__(self, ctx: ToolContext, llm: LLMClient | None = None):
        self.ctx = ctx
        self.llm = llm or LLMClient()

    # ----------------------------------------------------------------- parse
    def parse(self, text: str) -> dict[str, Any]:
        keyword = parse_policy_text(text)
        if self.llm.enabled:
            prompt = (
                "Convert this policy into levers. Levers (omit unused ones):\n"
                + "\n".join(f"- {k}: {v}" for k, v in LEVER_HELP.items())
                + '\nReturn JSON: {"levers": {...}, "duration": months, "summary": "one sentence"}\n\n'
                f"Policy: {text}"
            )
            data = self.llm.complete_json("You translate economic policy text into numeric levers.", prompt)
            if data and clamp_levers(data.get("levers", {})):
                return {"levers": clamp_levers(data["levers"]),
                        "duration": int(min(60, max(1, int(data.get("duration") or keyword["duration"])))),
                        "summary": str(data.get("summary", ""))[:300], "parser": "llm",
                        "keyword_check": keyword["levers"]}
        if not keyword["levers"]:
            keyword["levers"] = {"spending_change": 0.01}
            keyword["note"] = "No lever recognised; testing a small 1% spending change."
        return {**keyword, "parser": "keyword", "summary": ""}

    # -------------------------------------------------------------- evidence
    def evidence(self, text: str, parsed: dict, horizon: int = 24) -> dict[str, Any]:
        econ = self.ctx.economy
        main = what_if(econ, parsed["levers"], parsed["duration"], horizon, label=text[:60])
        stress = what_if(econ, parsed["levers"], parsed["duration"], horizon, label="stress",
                         shocks=[{"type": "energy_price_spike", "magnitude": 1.0, "duration": 9}])
        sectors = simulate_economy(text, time_steps=6)
        return {
            "status": econ.status(),
            "effect": main,
            "effect_text": describe_effect(main),
            "stress": {"welfare_gain": stress["welfare"]["gain"], "delta": stress["delta"],
                       "text": describe_effect(stress)},
            "sectors": {"most_affected": sectors["selected_actors"], "signals": sectors["policy_signals"]},
        }

    def _evidence_brief(self, text: str, parsed: dict, ev: dict) -> str:
        st = ev["status"]
        return json.dumps({
            "policy_text": text,
            "levers": parsed["levers"],
            "duration_months": parsed["duration"],
            "economy_now": {"phase": st["current_cycle"], **st["indicators"]},
            "simulated_effect_vs_no_policy": ev["effect_text"],
            "welfare_gain": ev["effect"]["welfare"]["gain"],
            "under_energy_shock": ev["stress"]["text"],
            "most_affected_sectors": ev["sectors"]["most_affected"],
        }, default=str)

    # ------------------------------------------------------------ deliberate
    def deliberate(self, text: str, *, horizon: int = 24) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Policy text is required")
        parsed = self.parse(text)
        ev = self.evidence(text, parsed, horizon)
        brief = self._evidence_brief(text, parsed, ev)
        use_llm = self.llm.enabled

        agents: list[dict] = []
        for role in ["Economic Analyst", "Tax Advisor"]:
            out = self._llm_assess(role, brief) if use_llm else None
            agents.append(out or self._rule_assess(role, parsed, ev))

        debate = self._debate(brief, parsed, ev, use_llm)
        for role, stance in [("Policy Supporter", "Supporter"), ("Policy Opposer", "Opposer")]:
            lines = [d["message"] for d in debate if d["speaker"] == role]
            vote = "approve" if stance == "Supporter" else "reject"
            agents.append({"agent": role, "task": ROLES[role]["brief"], "output": "\n\n".join(lines),
                           "vote": vote, "confidence": 0.6})

        verdict = (self._llm_verdict(brief, debate) if use_llm else None) or self._rule_verdict(parsed, ev)
        critic_vote = {"Proceed": "approve", "Modify": "approve", "Pause": "reject"}[verdict["decision"]]
        agents.append({"agent": "Policy Critic", "task": ROLES["Policy Critic"]["brief"],
                       "output": f"**Verdict: {verdict['decision']}**\n\n" + "\n".join(f"- {r}" for r in verdict["reasons"]),
                       "vote": critic_vote, "confidence": verdict.get("confidence", 0.7)})

        final_levers, final_duration = parsed["levers"], parsed["duration"]
        if verdict["decision"] == "Modify" and verdict.get("modified_levers"):
            final_levers = clamp_levers(verdict["modified_levers"]) or final_levers
        final_effect = ev["effect"] if final_levers == parsed["levers"] else what_if(
            self.ctx.economy, final_levers, final_duration, horizon, label="modified")

        consensus = self._consensus(agents)
        final_output = self._summary(text, parsed, ev, verdict, consensus, final_levers, final_effect)
        return {
            "id": str(uuid.uuid4()),
            "created_at": _now(),
            "policy": text,
            "parsed": parsed,
            "evidence": {"effect_text": ev["effect_text"], "stress_text": ev["stress"]["text"],
                         "welfare_gain": ev["effect"]["welfare"]["gain"], "delta": ev["effect"]["delta"],
                         "sectors": ev["sectors"], "series": final_effect["series"]},
            "agents": agents,
            "debate": debate,
            "verdict": verdict,
            "consensus": consensus,
            "final_levers": final_levers,
            "final_duration": final_duration,
            "predicted": {"summary": describe_effect(final_effect), "delta": final_effect["delta"],
                          "policy_path_end": {k: v for k, v in final_effect["policy"].items() if k.startswith("end_")},
                          "horizon": horizon},
            "final_output": final_output,
            "mode": "llm" if use_llm else "offline",
            "model": self.llm.model,
            "usage": self.llm.usage.as_dict(),
        }

    # ------------------------------------------------------------ LLM roles
    def _llm_assess(self, role: str, brief: str) -> dict | None:
        sys = ROLE_SYSTEM.format(role=role, brief=ROLES[role]["brief"])
        user = (f"Evidence:\n{brief}\n\nWrite your assessment in at most 120 words, as 3 bullet points and a "
                'one-line recommendation. Then vote. JSON: {"report": "...", "vote": "approve|reject|abstain", '
                '"confidence": 0.0-1.0}')
        data = self.llm.complete_json(sys, user, max_tokens=500)
        if not data or not data.get("report"):
            return None
        vote = str(data.get("vote", "abstain")).lower()
        return {"agent": role, "task": ROLES[role]["brief"], "output": str(data["report"])[:1500],
                "vote": vote if vote in {"approve", "reject", "abstain"} else "abstain",
                "confidence": float(min(1.0, max(0.0, float(data.get("confidence", 0.5) or 0.5))))}

    def _llm_verdict(self, brief: str, debate: list[dict]) -> dict | None:
        sys = ROLE_SYSTEM.format(role="Policy Critic", brief=ROLES["Policy Critic"]["brief"])
        transcript = "\n".join(f"{d['speaker']}: {d['message']}" for d in debate)
        user = (f"Evidence:\n{brief}\n\nDebate:\n{transcript}\n\nDecide. If Modify, give adjusted levers "
                "(same lever names, within ranges). JSON: {\"decision\": \"Proceed|Modify|Pause\", "
                "\"reasons\": [3 short strings], \"modified_levers\": {}, \"confidence\": 0.0-1.0}")
        data = self.llm.complete_json(sys, user, max_tokens=450)
        if not data or data.get("decision") not in {"Proceed", "Modify", "Pause"}:
            return None
        reasons = [str(r)[:240] for r in (data.get("reasons") or [])][:3] or ["No reasons given."]
        return {"decision": data["decision"], "reasons": reasons,
                "modified_levers": clamp_levers(data.get("modified_levers") or {}),
                "confidence": float(data.get("confidence", 0.7) or 0.7), "source": "llm"}

    def _debate(self, brief: str, parsed: dict, ev: dict, use_llm: bool) -> list[dict]:
        debate: list[dict] = []
        if use_llm:
            for _round in range(2):
                for role, stance in [("Policy Supporter", "Supporter"), ("Policy Opposer", "Opposer")]:
                    last = debate[-1]["message"] if debate else "(opening statement)"
                    sys = ROLE_SYSTEM.format(role=role, brief=ROLES[role]["brief"])
                    user = (f"Evidence:\n{brief}\n\nLast argument from the other side: {last}\n\n"
                            'Reply in at most 60 words, citing numbers from the evidence. JSON: {"message": "..."}')
                    data = self.llm.complete_json(sys, user, max_tokens=250)
                    if not data or not data.get("message"):
                        return self._rule_debate(parsed, ev)
                    debate.append({"speaker": role, "stance": stance, "message": str(data["message"])[:600]})
            return debate
        return self._rule_debate(parsed, ev)

    # ---------------------------------------------------------- rule roles
    def _rule_assess(self, role: str, parsed: dict, ev: dict) -> dict:
        d = ev["effect"]["delta"]
        st = ev["status"]["indicators"]
        gain = ev["effect"]["welfare"]["gain"]
        debt = ev["effect"]["public_debt_delta"]
        if role == "Economic Analyst":
            lines = [
                f"- The economy is in {ev['status']['current_cycle']}: inflation {st['inflation']:.1f}%, "
                f"unemployment {st['unemployment']:.1f}%, confidence {st['consumer_confidence']:.0f}.",
                f"- Simulated over {ev['effect']['horizon']} months, the policy moves unemployment "
                f"{d['mean_unemployment']:+.2f} pts and inflation {d['mean_inflation']:+.2f} pts.",
                f"- Average GDP changes by {d['mean_gdp']:+.1f}; under an energy shock the welfare gain is "
                f"{ev['stress']['welfare_gain']:+.2f}.",
                f"- Recommendation: {'support' if gain > 0 else 'do not support'} (welfare {gain:+.2f}).",
            ]
            vote = "approve" if gain > 0 else "reject"
        else:
            levers = parsed["levers"]
            lines = [
                f"- Levers: {', '.join(f'{k} {v:+.3f}' for k, v in levers.items())} for {parsed['duration']} months.",
                f"- Public debt changes by {debt:+.0f} versus no policy; the tax rate is now "
                f"{st['tax_rate'] * 100:.1f}%.",
                f"- Wealth Gini moves {d['end_wealth_gini']:+.3f}, "
                f"so the policy is {'progressive' if d['end_wealth_gini'] < 0 else 'regressive or neutral'}.",
            ]
            vote = "approve" if (gain > 0.3 or (gain > 0 and debt < 1500)) else "reject"
            lines.append("- Recommendation: " + ("fiscally acceptable for what it delivers." if vote == "approve"
                                              else "costly for what it delivers; fund it or shrink it."))
        return {"agent": role, "task": ROLES[role]["brief"], "output": "\n".join(lines), "vote": vote,
                "confidence": round(min(0.95, 0.5 + abs(gain) / 4), 2)}

    def _rule_debate(self, parsed: dict, ev: dict) -> list[dict]:
        d = ev["effect"]["delta"]
        debt = ev["effect"]["public_debt_delta"]
        sectors = ", ".join(ev["sectors"]["most_affected"][:3])
        good, bad = [], []
        if d["mean_unemployment"] < 0:
            good.append(f"it cuts unemployment by {abs(d['mean_unemployment']):.2f} points on average")
        else:
            bad.append(f"unemployment rises {d['mean_unemployment']:.2f} points")
        if d["mean_inflation"] < 0:
            good.append(f"it lowers inflation by {abs(d['mean_inflation']):.2f} points")
        else:
            bad.append(f"inflation goes up {d['mean_inflation']:.2f} points")
        if d["mean_gdp"] > 0:
            good.append(f"average GDP is {d['mean_gdp']:.1f} higher")
        else:
            bad.append(f"average GDP is {abs(d['mean_gdp']):.1f} lower")
        if d["end_wealth_gini"] < 0:
            good.append("wealth inequality narrows")
        else:
            bad.append(f"wealth inequality widens by {d['end_wealth_gini']:.3f}")
        if debt > 0:
            bad.append(f"public debt grows by {debt:.0f}")
        else:
            good.append(f"public debt falls by {abs(debt):.0f}")
        stress = ev["stress"]["welfare_gain"]
        return [
            {"speaker": "Policy Supporter", "stance": "Supporter",
             "message": "In simulation " + (", ".join(good[:2]) or "the downside is limited") + "."},
            {"speaker": "Policy Opposer", "stance": "Opposer",
             "message": "The cost is real: " + (", ".join(bad[:2]) or "the gains are small for the risk taken") + "."},
            {"speaker": "Policy Supporter", "stance": "Supporter",
             "message": f"It holds up under an energy price shock (welfare {stress:+.2f}) and reaches {sectors}."},
            {"speaker": "Policy Opposer", "stance": "Opposer",
             "message": "Simulated gains depend on the model; " + (bad[-1] if bad else "we should review it after a year")
             + ", so set a review date and an exit rule."},
        ]

    def _rule_verdict(self, parsed: dict, ev: dict) -> dict:
        gain = ev["effect"]["welfare"]["gain"]
        stress = ev["stress"]["welfare_gain"]
        d = ev["effect"]["delta"]
        reasons = [f"Welfare gain {gain:+.2f} versus no policy over {ev['effect']['horizon']} months.",
                   f"Under an energy shock the gain is {stress:+.2f}.",
                   f"Unemployment {d['mean_unemployment']:+.2f} pts, inflation {d['mean_inflation']:+.2f} pts."]
        if gain > 0.3 and stress > -0.5:
            return {"decision": "Proceed", "reasons": reasons, "modified_levers": {}, "confidence": 0.75,
                    "source": "rules"}
        if gain > -0.5:
            half = {k: v * 0.5 for k, v in parsed["levers"].items()}
            return {"decision": "Modify", "reasons": reasons + ["Halving the policy limits the downside."],
                    "modified_levers": half, "confidence": 0.6, "source": "rules"}
        return {"decision": "Pause", "reasons": reasons, "modified_levers": {}, "confidence": 0.7, "source": "rules"}

    # ---------------------------------------------------------- consensus
    def _consensus(self, agents: list[dict]) -> dict:
        approve = reject = 0.0
        votes = []
        for a in agents:
            w = ROLES[a["agent"]]["weight"] * float(a.get("confidence", 0.5))
            if a["vote"] == "approve":
                approve += w
            elif a["vote"] == "reject":
                reject += w
            votes.append({"agent": a["agent"], "vote": a["vote"], "weight": round(w, 3)})
        return {"approve_weight": round(approve, 3), "reject_weight": round(reject, 3),
                "consensus_reached": approve > reject, "votes": votes}

    def _summary(self, text, parsed, ev, verdict, consensus, levers, effect) -> str:
        outcome = "APPROVED" if consensus["consensus_reached"] else "REJECTED"
        lv = ", ".join(f"{k} {v:+.3f}" for k, v in levers.items())
        return (
            f"## Council decision: {outcome}\n\n"
            f"Policy: {text}\n\n"
            f"Levers: {lv} for {parsed['duration']} months (parsed by {parsed['parser']}).\n\n"
            f"Critic verdict: {verdict['decision']}. Votes: {consensus['approve_weight']:.2f} approve vs "
            f"{consensus['reject_weight']:.2f} reject.\n\n"
            f"Simulated effect: {describe_effect(effect)}\n\n"
            f"Under stress: {ev['stress']['text']}"
        )
