"""Exercise the LLM code paths with a scripted fake model (no network)."""

import json

from valora.agentic.copilot import PolicyCopilot
from valora.agentic.council import EconomicCouncil
from valora.agentic.llm import LLMClient
from valora.agentic.tools import ToolContext
from valora.engine import Economy


class ScriptedLLM(LLMClient):
    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)
        self.seen = []

    @property
    def enabled(self):
        return True

    @property
    def model(self):
        return "scripted"

    def chat(self, messages, **kw):
        self.seen.append((messages, kw))
        return self.replies.pop(0) if self.replies else None


def _call(name, args, i=0):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_copilot_llm_tool_loop_and_verified_proposal():
    final = ("Use a 3% stimulus.\n```json\n" + json.dumps(
        {"label": "Stimulus 3%", "levers": {"spending_change": 0.03, "junk": 1}, "duration": 12,
         "rationale": "cuts unemployment"}) + "\n```")
    llm = ScriptedLLM([
        {"content": "Checking the economy.", "tool_calls": [_call("get_economy_status", {})]},
        {"content": "", "tool_calls": [_call("simulate_policy", {"spending_change": 0.03, "horizon": 6}, 1)]},
        {"content": final},
    ])
    r = PolicyCopilot(ToolContext(Economy()), llm=llm).run("cut unemployment", horizon=12)
    assert r["planner"] == "llm"
    assert r["proposal"]["levers"] == {"spending_change": 0.03}
    assert "verified_effect" in r["proposal"]
    tools_used = [s["tool"] for s in r["trace"] if s["type"] == "tool"]
    assert tools_used == ["get_economy_status", "simulate_policy"]
    # tool results were fed back to the model
    assert any(m.get("role") == "tool" for m in llm.seen[2][0])


def test_copilot_falls_back_when_model_gives_no_proposal():
    llm = ScriptedLLM([{"content": "I think taxes should change."}])
    r = PolicyCopilot(ToolContext(Economy()), llm=llm).run("growth", horizon=12)
    assert r["planner"] == "llm+offline" and r["proposal"]["levers"]


def test_council_llm_roles():
    j = lambda d: {"content": json.dumps(d)}  # noqa: E731
    replies = [j({"levers": {"tax_change": -0.03}, "duration": 18, "summary": "tax cut"}),
               j({"report": "- growth up", "vote": "approve", "confidence": 0.8}),
               j({"report": "- debt up", "vote": "reject", "confidence": 0.6})]
    replies += [j({"message": f"point {i}"}) for i in range(4)]
    replies += [j({"decision": "Modify", "reasons": ["a", "b", "c"], "modified_levers": {"tax_change": -0.015},
                   "confidence": 0.9})]
    r = EconomicCouncil(ToolContext(Economy()), llm=ScriptedLLM(replies)).deliberate("Cut taxes")
    assert r["parsed"]["parser"] == "llm" and r["parsed"]["duration"] == 18
    assert [d["message"] for d in r["debate"]] == ["point 0", "point 1", "point 2", "point 3"]
    assert r["verdict"]["decision"] == "Modify"
    assert r["final_levers"] == {"tax_change": -0.015}
    assert r["mode"] == "llm"
