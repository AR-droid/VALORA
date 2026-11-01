from valora.agentic.copilot import PolicyCopilot, goal_weights
from valora.agentic.council import EconomicCouncil
from valora.agentic.policy import clamp_levers, parse_policy_text, what_if
from valora.agentic.tools import ToolContext, build_tools, call_tool
from valora.engine import Economy


def test_parser_reads_direction_and_size():
    assert parse_policy_text("Cut income taxes by 3% for 2 years") == {"levers": {"tax_change": -0.03}, "duration": 24}
    lv = parse_policy_text("Increase transfer payments, ease small-business credit, and tighten emissions standards")
    assert lv["levers"]["transfer_change"] > 0
    assert lv["levers"]["rate_change"] < 0
    assert lv["levers"]["regulation_change"] > 0
    assert parse_policy_text("Raise interest rates by 50 basis points")["levers"] == {"rate_change": 0.005}


def test_guardrails_clamp_and_drop_junk():
    assert clamp_levers({"tax_change": 9, "nope": 1, "rate_change": "x", "spending_change": float("nan")}) == {
        "tax_change": 0.10}


def test_what_if_does_not_touch_live_economy():
    e = Economy()
    before = e.indicators()
    r = what_if(e, {"spending_change": 0.04}, 12, 12)
    assert e.indicators() == before and e.tick == 0
    assert r["delta"]["mean_unemployment"] < 0


def test_tools_have_schemas_and_handle_bad_calls():
    tools = build_tools(ToolContext(Economy()))
    for t in tools.values():
        assert t.schema()["function"]["parameters"]["type"] == "object"
    assert "error" in call_tool(tools, "nope", {})
    assert "error" in call_tool(tools, "simulate_policy", "{not json")
    assert "summary" in call_tool(tools, "simulate_policy", {"tax_change": -0.02, "horizon": 6})


def test_council_offline_produces_full_deliberation():
    r = EconomicCouncil(ToolContext(Economy())).deliberate("Launch a 4% infrastructure stimulus")
    assert [a["agent"] for a in r["agents"]] == [
        "Economic Analyst", "Tax Advisor", "Policy Supporter", "Policy Opposer", "Policy Critic"]
    assert len(r["debate"]) == 4
    assert r["verdict"]["decision"] in {"Proceed", "Modify", "Pause"}
    assert r["final_levers"]
    assert r["mode"] == "offline"


def test_goal_weights_treat_without_clause_as_constraint():
    w = goal_weights("Bring unemployment down without pushing inflation above 4%")
    assert w["mean_unemployment"] < w["mean_inflation"] < 0


def test_copilot_offline_returns_verified_proposal():
    r = PolicyCopilot(ToolContext(Economy())).run("Bring unemployment down", horizon=12)
    assert r["planner"] == "offline-search"
    assert r["proposal"]["levers"]
    assert "verified_effect" in r["proposal"]
    assert any(s["type"] == "tool" for s in r["trace"])
