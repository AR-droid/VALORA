from valora.harness.bench import AGENTS, load_scenarios, run_scenario


def test_scenarios_load():
    sc = load_scenarios()
    assert {"baseline", "financial_crisis", "pandemic"} <= set(sc)


def test_run_is_reproducible():
    sc = load_scenarios()["energy_shock"]
    a = run_scenario(sc, AGENTS["taylor-rule"]())
    b = run_scenario(sc, AGENTS["taylor-rule"]())
    assert a["run_hash"] == b["run_hash"] and a["score"] == b["score"]


def test_policy_beats_no_policy_in_a_crisis():
    sc = load_scenarios()["financial_crisis"]
    assert run_scenario(sc, AGENTS["fixed-stimulus"]())["score"] > run_scenario(sc, AGENTS["no-policy"]())["score"]
