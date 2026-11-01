import numpy as np

from valora.engine import LEVER_BOUNDS, Economy, EconomyConfig, Policy


def test_same_seed_same_history():
    a = Economy(EconomyConfig(seed=3)).run(24)
    b = Economy(EconomyConfig(seed=3)).run(24)
    assert a == b


def test_indicators_stay_in_sane_ranges():
    e = Economy()
    for snap in e.run(120):
        assert 0 < snap["gdp"] < 10_000
        assert -5 <= snap["inflation"] <= 30
        assert 0 < snap["unemployment"] <= 40
        assert 0 <= snap["consumer_confidence"] <= 100
        assert 0.05 <= snap["tax_rate"] <= 0.5
        assert 0 <= snap["wealth_gini"] <= 1


def test_fork_is_independent():
    e = Economy()
    f = e.fork()
    f.run(12)
    assert e.tick == 0 and f.tick == 12


def test_policy_is_clamped_and_expires():
    e = Economy()
    p = e.apply_policy(Policy("huge", {"spending_change": 5.0, "bogus": 1.0}, duration=3))
    assert p.levers == {"spending_change": LEVER_BOUNDS["spending_change"][1]}
    e.run(3)
    assert e.policies == []


def test_stimulus_lowers_unemployment_and_recession_shock_raises_it():
    base = Economy()
    calm = base.fork()
    stim = base.fork()
    stim.apply_policy(Policy("stim", {"spending_change": 0.05}, 24))
    crisis = base.fork()
    crisis.apply_shock("financial_crisis", 1.5, 12)
    u = lambda h: np.mean([s["unemployment"] for s in h])  # noqa: E731
    hc, hs, hx = calm.run(24), stim.run(24), crisis.run(24)
    assert u(hs) < u(hc) < u(hx)


def test_rate_hike_lowers_inflation():
    base = Economy()
    calm, hike = base.fork(), base.fork()
    hike.apply_policy(Policy("hike", {"rate_change": 0.02}, 24))
    infl = lambda h: np.mean([s["inflation"] for s in h])  # noqa: E731
    assert infl(hike.run(24)) < infl(calm.run(24))
