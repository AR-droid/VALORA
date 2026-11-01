"""Turning policy text into bounded levers, and measuring a policy by simulating it.

``what_if`` is the evidence every agent relies on: it forks the live economy,
runs it with and without the policy for the same number of months, and reports
the difference. Because the fork copies the random state too, both runs see the
same luck, so the difference is the policy's effect.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..engine import LEVER_BOUNDS, Economy, Policy

METRICS = ["gdp", "inflation", "unemployment", "consumer_confidence", "wealth_gini", "tax_rate"]

LEVER_HELP = {
    "tax_change": "change in the tax rate as a fraction (-0.03 = cut by 3 points)",
    "spending_change": "extra government spending as a fraction of potential output (0.04 = +4%)",
    "rate_change": "change in the policy interest rate as a fraction (0.01 = +1 point)",
    "transfer_change": "cash transfers per household as a fraction of the average wage (0.05 = 5%)",
    "regulation_change": "regulatory burden from -1 (deregulate) to +1 (much tighter)",
}

_UP = r"\b(increas|rais|hik|boost|expand|more|higher|tighten|add|grow|lift|introduc|launch)\w*"
_DOWN = r"\b(cut|reduc|lower|decreas|slash|eas|less|fewer|loosen|trim|remov|scrap|abolish|end)\w*"


def _clause(text: str, start: int, end: int) -> tuple[int, int]:
    """Bounds of the clause around a match: split on , ; . and ' and '."""
    seps = [",", ";", ".", " and "]
    lo = max([text.rfind(sep, 0, start) + len(sep) for sep in seps if text.rfind(sep, 0, start) >= 0] or [0])
    his = [text.find(sep, end) for sep in seps if text.find(sep, end) >= 0]
    return lo, min(his) if his else len(text)


def _pct_near(text: str, start: int, end: int) -> float | None:
    lo, hi = _clause(text, start, end)
    window = text[lo:hi]
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|pp|points?|bps|basis points)", window)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit in {"bps", "basis points"}:
        return val / 10000.0
    return val / 100.0


def _direction(text: str, start: int, end: int) -> float:
    """Sign from the nearest verb before the keyword (same clause), else just after it."""
    clause_start = max(text.rfind(",", 0, start), text.rfind(";", 0, start), text.rfind(".", 0, start),
                       text.rfind(" and ", 0, start) + 4 if text.rfind(" and ", 0, start) >= 0 else -1)
    before = text[max(0, clause_start + 1, start - 45): start]
    verbs = [(m.start(), 1.0) for m in re.finditer(_UP, before)] + [(m.start(), -1.0) for m in re.finditer(_DOWN, before)]
    if verbs:
        return max(verbs)[1]
    after = text[end: end + 25]
    up, down = re.search(_UP, after), re.search(_DOWN, after)
    if down and (not up or down.start() < up.start()):
        return -1.0
    return 1.0


def parse_policy_text(text: str) -> dict[str, Any]:
    """Keyword parser used offline and as a cross-check on the LLM's parse."""
    t = text.lower()
    levers: dict[str, float] = {}
    rules = [
        ("tax_change", r"\b(tax|taxes|vat|gst|levy|income tax|corporate tax)\b", 0.02),
        ("spending_change", r"\b(spending|stimulus|infrastructure|public investment|budget|grant|subsid\w*)\b", 0.03),
        ("rate_change", r"\b(interest rate|policy rate|rates|repo|monetary|borrowing cost|credit)\b", 0.01),
        ("transfer_change", r"\b(transfer\w*|cash|benefit\w*|welfare|ubi|basic income|pension\w*|voucher\w*)\b", 0.05),
        ("regulation_change", r"\b(regulat\w*|compliance|standard\w*|emission\w*|red tape|inspection\w*)\b", 0.5),
    ]
    for lever, pattern, default in rules:
        m = re.search(pattern, t)
        if not m:
            continue
        sign = _direction(t, m.start(), m.end())
        if lever == "regulation_change":
            levers[lever] = sign * default
            continue
        pct = _pct_near(t, m.start(), m.end())
        mag = pct if pct is not None else default
        if lever == "spending_change" and re.search(r"\bsubsid", t) and sign < 0:
            sign = -1.0
        levers[lever] = sign * mag
    if "rate_change" not in levers and re.search(r"\b(tighten|hawkish)\b.*\b(money|monetary)\b", t):
        levers["rate_change"] = 0.01
    duration = 12
    m = re.search(r"(\d+)\s*(month|year)", t)
    if m:
        duration = int(m.group(1)) * (12 if m.group(2).startswith("year") else 1)
    return {"levers": clamp_levers(levers), "duration": int(min(60, max(1, duration)))}


def clamp_levers(levers: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in (levers or {}).items():
        if k not in LEVER_BOUNDS:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(f) or f == 0.0:
            continue
        lo, hi = LEVER_BOUNDS[k]
        out[k] = round(float(min(hi, max(lo, f))), 4)
    return out


def _summary(history: list[dict]) -> dict[str, float]:
    if not history:
        return {}
    out = {f"mean_{k}": float(np.mean([h[k] for h in history])) for k in METRICS}
    out.update({f"end_{k}": float(history[-1][k]) for k in METRICS})
    return out


def welfare_score(history: list[dict], start_gdp: float | None = None, inflation_target: float = 2.0,
                  unemployment_target: float = 4.5) -> float:
    """Higher is better: output gained, minus misses on inflation and jobs, minus inequality."""
    if not history:
        return 0.0
    infl = np.array([h["inflation"] for h in history])
    unemp = np.array([h["unemployment"] for h in history])
    ref = start_gdp or history[0]["gdp"]
    growth = (float(np.mean([h["gdp"] for h in history])) / ref - 1.0) * 100.0 if ref else 0.0
    gini = history[-1]["wealth_gini"]
    score = (
        0.6 * growth
        - 1.0 * float(np.mean(np.abs(infl - inflation_target)))
        - 1.2 * float(np.mean(np.maximum(unemp - unemployment_target, 0)))
        - 10.0 * max(0.0, gini - 0.25)
    )
    return round(float(score), 3)


def what_if(economy: Economy, levers: dict[str, float], duration: int = 12, horizon: int = 24,
            label: str = "what-if", shocks: list[dict] | None = None) -> dict[str, Any]:
    """Simulate the policy against a no-policy baseline from the same starting point."""
    levers = clamp_levers(levers)
    horizon = int(min(72, max(3, horizon)))
    base = economy.fork()
    test = economy.fork()
    for s in shocks or []:
        for e in (base, test):
            e.apply_shock(s.get("type", "demand_collapse"), float(s.get("magnitude", 1.0)), int(s.get("duration", 6)))
    if levers:
        test.apply_policy(Policy(label=label, levers=levers, duration=duration, source="what-if"))
    start_gdp = economy.gdp
    hb = base.run(horizon)
    ht = test.run(horizon)
    sb, st = _summary(hb), _summary(ht)
    delta = {k: round(st[k] - sb[k], 4) for k in sb}
    return {
        "levers": levers,
        "duration": duration,
        "horizon": horizon,
        "baseline": {k: round(v, 3) for k, v in sb.items()},
        "policy": {k: round(v, 3) for k, v in st.items()},
        "delta": delta,
        "welfare": {"baseline": welfare_score(hb, start_gdp), "policy": welfare_score(ht, start_gdp),
                    "gain": round(welfare_score(ht, start_gdp) - welfare_score(hb, start_gdp), 3)},
        "public_debt_delta": round(test.public_debt - base.public_debt, 1),
        "series": {
            "tick": [h["tick"] for h in ht],
            **{f"baseline_{k}": [h[k] for h in hb] for k in ["gdp", "inflation", "unemployment"]},
            **{f"policy_{k}": [h[k] for h in ht] for k in ["gdp", "inflation", "unemployment"]},
        },
    }


def describe_effect(result: dict[str, Any]) -> str:
    d = result["delta"]
    return (
        f"Over {result['horizon']} months versus no policy: GDP {d['mean_gdp']:+.1f} (avg), "
        f"inflation {d['mean_inflation']:+.2f} pts, unemployment {d['mean_unemployment']:+.2f} pts, "
        f"confidence {d['mean_consumer_confidence']:+.1f}, wealth Gini {d['end_wealth_gini']:+.3f}; "
        f"welfare score {result['welfare']['gain']:+.2f}; public debt {result['public_debt_delta']:+.0f}."
    )
