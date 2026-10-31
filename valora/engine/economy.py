"""Tick-based agent economy.

One tick is one month. Each tick:

1. the regulator agent adjusts the tax rate toward its targets,
2. firms hire from the labour force based on last month's demand,
3. households earn wages (or benefits), pay tax, receive transfers and dividends,
4. each household's Q-learning brain picks how much to spend; the bank's credit
   model approves or denies loans for households short of cash,
5. firms set prices with their learning brains, the goods market clears,
6. GDP, inflation, unemployment and confidence are computed from what happened,
7. every brain receives its reward and learns,
8. a snapshot of the tick is returned (the caller hashes it into the ledger).

Business-cycle phases and shocks act on productivity, costs and confidence, so
they move the economy through the agents rather than by editing indicators.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..agents import BankRiskBrain, ConsumerQBrain, FirmAdaptiveBrain, make_regulator

PHASES = ["expansion", "peak", "recession", "trough"]
PHASE_TICKS = {"expansion": 18, "peak": 4, "recession": 8, "trough": 6}
# Annualised productivity growth and confidence drift for each phase.
PHASE_TRENDS = {
    "expansion": {"growth": 0.03, "confidence": 4.0, "cost": 0.0},
    "peak": {"growth": 0.01, "confidence": -2.0, "cost": 0.01},
    "recession": {"growth": -0.03, "confidence": -8.0, "cost": -0.01},
    "trough": {"growth": 0.005, "confidence": 3.0, "cost": 0.0},
}
# Shock profiles per unit magnitude: annualised productivity hit, cost pressure,
# confidence points, and a direct demand multiplier.
SHOCK_PROFILES = {
    "financial_crisis": {"growth": -0.08, "cost": 0.0, "confidence": -18.0, "demand": -0.10, "credit": -0.5},
    "pandemic": {"growth": -0.10, "cost": 0.02, "confidence": -20.0, "demand": -0.15, "credit": -0.1},
    "energy_price_spike": {"growth": -0.02, "cost": 0.12, "confidence": -6.0, "demand": -0.03, "credit": 0.0},
    "supply_cost": {"growth": -0.01, "cost": 0.10, "confidence": -4.0, "demand": 0.0, "credit": 0.0},
    "demand_collapse": {"growth": 0.0, "cost": 0.0, "confidence": -15.0, "demand": -0.20, "credit": 0.0},
    "tech_boom": {"growth": 0.06, "cost": -0.02, "confidence": 12.0, "demand": 0.05, "credit": 0.1},
}

# Policy levers and the range each one is allowed to take. Guardrails clamp to these.
INVEST = 0.05  # autonomous investment, share of potential output

LEVER_BOUNDS = {
    "tax_change": (-0.10, 0.10),  # change in tax rate, fraction (0.02 = +2 points)
    "spending_change": (-0.10, 0.15),  # extra government demand, fraction of potential output
    "rate_change": (-0.04, 0.04),  # change in policy interest rate, fraction
    "transfer_change": (-0.10, 0.20),  # transfers to households, fraction of average wage
    "regulation_change": (-1.0, 1.0),  # tighter (+) or looser (-) regulation
}


@dataclass
class Policy:
    """A concrete, bounded policy the engine can apply."""

    label: str
    levers: dict[str, float]
    duration: int = 12
    id: str = ""
    source: str = "user"
    started_tick: int = 0

    def clamped(self) -> Policy:
        levers = {}
        for k, v in self.levers.items():
            if k in LEVER_BOUNDS and v is not None:
                lo, hi = LEVER_BOUNDS[k]
                levers[k] = float(min(hi, max(lo, float(v))))
        return Policy(self.label, levers, int(min(60, max(1, self.duration))), self.id, self.source, self.started_tick)


@dataclass
class Shock:
    type: str
    magnitude: float
    duration: int
    started_tick: int = 0
    label: str = ""


@dataclass
class EconomyConfig:
    households: int = 60
    firms: int = 5
    seed: int = 7
    wage: float = 100.0
    units_per_worker: float = 7.0
    base_tax: float = 0.20
    base_rate: float = 0.04
    benefit: float = 40.0


@dataclass
class Household:
    skill: float
    brain: ConsumerQBrain
    savings: float = 300.0
    debt: float = 0.0
    employed: bool = True
    last_spending: float = 0.0


class Economy:
    def __init__(self, config: EconomyConfig | None = None, burn_in: int = 24):
        self.cfg = config or EconomyConfig()
        self.rng = np.random.RandomState(self.cfg.seed)
        n, m = self.cfg.households, self.cfg.firms
        skills = np.clip(self.rng.lognormal(0.0, 0.35, n), 0.4, 3.0)
        self.households = [
            Household(skill=float(s), brain=ConsumerQBrain(actions=[0.55, 0.7, 0.85, 0.95], seed=self.cfg.seed * 1000 + i), savings=200 + 200 * float(s))
            for i, s in enumerate(skills)
        ]
        self.bank = BankRiskBrain(seed=self.cfg.seed)
        self.regulator = make_regulator()
        unit_cost = self.cfg.wage / self.cfg.units_per_worker
        cap = int(1.3 * n * self.cfg.units_per_worker / m)
        self.firms = [FirmAdaptiveBrain(cost=unit_cost, capacity=cap, seed=self.cfg.seed * 10 + j) for j in range(m)]
        self.firm_demand = [n * self.cfg.units_per_worker * 0.9 / m] * m

        self.tick = 0
        self.phase = "expansion"
        self.phase_tick = 0
        self.productivity = 1.0
        self.wage = self.cfg.wage
        self.tax_rate = self.cfg.base_tax
        self.regulator_adjust = 0.0
        self.policy_rate = self.cfg.base_rate
        self.price_index: float | None = None
        self.inflation = 2.0  # percent, annualised
        self.unemployment = 5.0  # percent
        self.confidence = 70.0
        self.employment_rate = 0.95
        self.gdp = 1000.0
        self._gdp_scale: float | None = None
        self.last_tax_revenue = 0.0
        self.public_debt = 0.0
        self.policies: list[Policy] = []
        self.shocks: list[Shock] = []
        self.history: list[dict] = []
        self.loans_issued = 0
        self.defaults = 0
        if burn_in:
            self._burn_in(burn_in)

    def _burn_in(self, ticks: int) -> None:
        """Let learners settle, then rebase GDP to 1000 and restart the clock."""
        for _ in range(ticks):
            self.step()
        self._gdp_scale = (self._gdp_scale or 1.0) * 1000.0 / max(self.gdp, 1.0)
        self.gdp = 1000.0
        self.tick = 0
        self.phase, self.phase_tick = "expansion", 0
        self.history = []
        self.loans_issued = self.defaults = 0

    # ------------------------------------------------------------ interventions
    def apply_policy(self, policy: Policy) -> Policy:
        p = policy.clamped()
        p.started_tick = self.tick
        self.policies.append(p)
        return p

    def revoke_policy(self, policy_id: str) -> bool:
        before = len(self.policies)
        self.policies = [p for p in self.policies if p.id != policy_id]
        return len(self.policies) < before

    def apply_shock(self, shock_type: str, magnitude: float = 1.0, duration: int = 6, label: str = "") -> Shock:
        s = Shock(shock_type, float(max(0.0, min(3.0, magnitude))), int(max(1, min(60, duration))), self.tick, label)
        self.shocks.append(s)
        self.confidence = float(np.clip(self.confidence + SHOCK_PROFILES.get(shock_type, {}).get("confidence", -3.0)
                                        * s.magnitude * 0.5, 0, 100))
        return s

    def fork(self) -> Economy:
        """Deep copy for what-if runs; the original is untouched."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ helpers
    def _levers(self) -> dict[str, float]:
        total = {k: 0.0 for k in LEVER_BOUNDS}
        for p in self.policies:
            for k, v in p.levers.items():
                total[k] += v
        return total

    def _shock_sum(self) -> dict[str, float]:
        total = {"growth": 0.0, "cost": 0.0, "demand": 0.0, "credit": 0.0, "confidence": 0.0}
        for s in self.shocks:
            prof = SHOCK_PROFILES.get(s.type, {"growth": -0.01, "confidence": -3.0})
            for k in total:
                total[k] += prof.get(k, 0.0) * s.magnitude
        return total

    def _expire(self) -> None:
        self.policies = [p for p in self.policies if self.tick - p.started_tick < p.duration]
        self.shocks = [s for s in self.shocks if self.tick - s.started_tick < s.duration]

    def _advance_phase(self) -> None:
        self.phase_tick += 1
        if self.phase_tick >= PHASE_TICKS[self.phase]:
            self.phase = PHASES[(PHASES.index(self.phase) + 1) % len(PHASES)]
            self.phase_tick = 0

    # --------------------------------------------------------------------- step
    def step(self) -> dict[str, Any]:
        cfg = self.cfg
        levers = self._levers()
        shocks = self._shock_sum()
        trend = PHASE_TRENDS[self.phase]

        # 1. Regulator agent nudges taxes toward its targets.
        reg_delta = float(self.regulator.act({
            "gdp_norm": self.gdp / 1000.0,
            "inflation": self.inflation / 100.0,
            "unemployment": self.unemployment / 100.0,
        }))
        self.regulator_adjust = float(np.clip(self.regulator_adjust + reg_delta * 0.25, -0.08, 0.08))
        self.tax_rate = float(np.clip(cfg.base_tax + self.regulator_adjust + levers["tax_change"], 0.05, 0.5))
        self.policy_rate = float(np.clip(cfg.base_rate + levers["rate_change"], 0.0, 0.15))

        # Productivity follows the business cycle and shocks.
        self.productivity *= 1.0 + (trend["growth"] + shocks["growth"]) / 12.0
        self.productivity = float(np.clip(self.productivity, 0.5, 2.5))
        cost_pressure = 1.0 + 0.02 * levers["regulation_change"] + shocks["cost"] + trend["cost"]

        # 2. Hiring: firms employ enough workers for last month's demand.
        upw = cfg.units_per_worker * self.productivity
        labour_demand = sum(min(f.capacity * self.productivity, d) for f, d in zip(self.firms, self.firm_demand)) / upw
        n = len(self.households)
        self.employment_rate = float(np.clip(labour_demand / n, 0.5, 0.97))
        employed_count = int(round(self.employment_rate * n))
        noise = self.rng.normal(0, 0.3, n)
        order = np.argsort(-(np.array([h.skill for h in self.households]) + noise))
        employed_idx = set(order[:employed_count].tolist())

        # 3. Incomes.
        transfers_per_h = levers["transfer_change"] * self.wage
        wages_paid = 0.0
        dividends_pool = getattr(self, "_dividends", 0.0)
        skill_total = sum(h.skill for h in self.households)
        tax_revenue = 0.0
        disposable = []
        for i, h in enumerate(self.households):
            h.employed = i in employed_idx
            gross = self.wage * h.skill if h.employed else cfg.benefit
            if h.employed:
                wages_paid += gross
            div = dividends_pool * h.skill / skill_total
            tax = (gross + div) * self.tax_rate if h.employed else div * self.tax_rate
            tax_revenue += tax
            disposable.append(gross + div - tax + transfers_per_h)

        # 4. Households spend; the bank lends to those short of cash.
        conf_mult = 0.75 + 0.5 * self.confidence / 100.0
        rate_mult = 1.0 - 2.0 * (self.policy_rate - cfg.base_rate)
        demand_mult = max(0.3, conf_mult * rate_mult * (1.0 + shocks["demand"]))
        credit_tight = shocks["credit"] < -0.2
        total_spending = 0.0
        loans_now = defaults_now = 0
        for h, inc in zip(self.households, disposable):
            frac = h.brain.act({"inflation": self.inflation, "savings": h.savings})
            spend = max(0.0, (frac * inc + 0.03 * max(h.savings, 0.0)) * demand_mult)
            debt_cap = 600.0 * (1.0 - 0.5 * levers["regulation_change"])
            if h.savings + inc - spend < 20.0 and h.debt < debt_cap:
                decision = self.bank.act({"savings": max(h.savings, 0.0), "income": inc * h.skill / 1.5,
                                          "inflation": self.inflation / 100.0})
                if decision == "approve" and not (credit_tight and self.rng.rand() < 0.5):
                    h.debt += 150.0
                    h.savings += 150.0
                    loans_now += 1
            h.savings += inc - spend
            # Debt service: interest plus 5% principal, paid from savings.
            if h.debt > 0:
                service = h.debt * (0.05 + (self.policy_rate + 0.03) / 12.0)
                if h.savings >= service:
                    h.savings -= service
                    h.debt = max(0.0, h.debt - h.debt * 0.05)
                    if h.debt < 1.0:
                        h.debt = 0.0
                        self.bank.learn(+1.0, {})
                elif self.rng.rand() < (0.15 if not h.employed else 0.02) * (1.0 - 0.6 * levers["regulation_change"]):
                    h.debt = 0.0  # default: the bank writes the loan off
                    defaults_now += 1
                    self.bank.learn(-1.0, {})
            h.last_spending = spend
            total_spending += spend

        # Government spends what it collected last month plus any stimulus.
        potential_value = n * upw * (self.wage / upw) * 1.3
        gov_spending = max(0.0, (0.12 + levers["spending_change"]) * potential_value)
        self.public_debt += gov_spending + transfers_per_h * n - tax_revenue
        self.last_tax_revenue = tax_revenue
        investment = INVEST * potential_value * conf_mult * (1.0 + 4.0 * trend["growth"] + shocks["demand"])
        nominal_demand = total_spending + gov_spending + investment

        # 5. Firms price and produce; demand flows toward cheaper firms.
        actions = [f.act({"demand_signal": d / max(1.0, f.capacity * self.productivity), "cost_pressure":
                          cost_pressure * self.wage / cfg.wage / self.productivity}) for f, d in
                   zip(self.firms, self.firm_demand)]
        if not hasattr(self, "_firm_prices"):
            self._firm_prices = [a["price"] for a in actions]
        for j, a in enumerate(actions):
            # Sticky prices: each month a firm closes a quarter of the gap to its target price.
            self._firm_prices[j] += 0.25 * (a["price"] - self._firm_prices[j])
            a["price"] = self._firm_prices[j]
        prices = np.array([a["price"] for a in actions])
        weights = np.exp(-2.0 * (prices / prices.mean() - 1.0))
        weights /= weights.sum()
        sales_units, revenue, profit_total = [], 0.0, 0.0
        for j, (f, a) in enumerate(zip(self.firms, actions)):
            units_wanted = nominal_demand * weights[j] / a["price"]
            supply = min(f.capacity * self.productivity, max(units_wanted * 1.05, a["production"] * self.productivity))
            sold = min(supply, units_wanted)
            rev = sold * a["price"]
            cost = supply * a["unit_cost"]
            profit = rev - cost
            f.learn(profit / 100.0, {})
            self.firm_demand[j] = units_wanted
            sales_units.append(sold)
            revenue += rev
            profit_total += profit
        self._dividends = max(0.0, 0.8 * profit_total)

        # 6. Macro aggregates.
        sold_total = sum(sales_units) or 1.0
        avg_price = revenue / sold_total
        new_index = 100.0 * avg_price / (cfg.wage / cfg.units_per_worker * 1.3)
        self.price_index = new_index if self.price_index is None else 0.8 * self.price_index + 0.2 * new_index
        self._price_hist = (getattr(self, "_price_hist", []) + [self.price_index])[-13:]
        base = self._price_hist[0]
        months = len(self._price_hist) - 1
        yoy = ((self.price_index / base) ** (12.0 / months) - 1.0) * 100 if months else 2.0
        utilisation = nominal_demand / max(potential_value, 1.0)
        demand_pull = (utilisation - 0.85) * 6.0
        # Expectations channel: a policy rate above neutral pulls inflation down.
        rate_gap = (self.policy_rate - cfg.base_rate) * 100.0
        target = 0.6 * yoy + 0.4 * (2.0 + demand_pull) - 0.6 * rate_gap
        self.inflation = float(np.clip(0.8 * self.inflation + 0.2 * target, -5.0, 30.0))
        # Wages follow prices and labour-market tightness.
        tightness = (self.employment_rate - 0.95) * 0.02
        self.wage *= 1.0 + 0.8 * self.inflation / 1200.0 + tightness / 12.0

        real_output = sold_total * (cfg.wage / cfg.units_per_worker * 1.3)
        if self._gdp_scale is None:
            self._gdp_scale = 1000.0 / max(real_output, 1.0)
        gdp_prev = self.gdp
        self.gdp = float(0.6 * self.gdp + 0.4 * real_output * self._gdp_scale)
        self.unemployment = float(np.clip(0.6 * self.unemployment + 0.4 * (1 - self.employment_rate) * 100, 0.5, 40))
        growth = (self.gdp / gdp_prev - 1.0) * 100 if gdp_prev else 0.0
        self.confidence = float(np.clip(
            self.confidence + (trend["confidence"] + shocks["confidence"] * 0.3) / 12.0
            + growth * 0.5 - max(0.0, self.inflation - 4.0) * 0.15 - max(0.0, self.unemployment - 6.0) * 0.08
            + (70.0 - self.confidence) * 0.04 - 0.4 * defaults_now,
            0.0, 100.0))

        # 7. Household learning: utility from consumption, pain from inflation and joblessness.
        for h in self.households:
            r = math.log1p(h.last_spending) / 5.0 - max(self.inflation, 0.0) / 50.0 - (0.0 if h.employed else 0.2)
            r -= 0.2 if h.savings < 0 else 0.0
            h.brain.learn(r, {"inflation": self.inflation, "savings": h.savings})

        self.loans_issued += loans_now
        self.defaults += defaults_now
        self.tick += 1
        self._advance_phase()
        self._expire()
        snap = self.snapshot(extra={
            "gdp_growth_pct": round(growth, 3),
            "loans": loans_now,
            "defaults": defaults_now,
            "avg_price": round(avg_price, 3),
            "gov_spending": round(gov_spending, 2),
            "tax_revenue": round(tax_revenue, 2),
        })
        self.history.append(snap)
        if len(self.history) > 2000:
            del self.history[:500]
        return snap

    def run(self, ticks: int) -> list[dict]:
        return [self.step() for _ in range(ticks)]

    # -------------------------------------------------------------------- views
    def gini(self) -> float:
        w = np.sort(np.array([max(h.savings - h.debt, 0.0) for h in self.households]))
        if w.sum() <= 0:
            return 0.0
        n = len(w)
        return float((2 * np.arange(1, n + 1) - n - 1).dot(w) / (n * w.sum()))

    def indicators(self) -> dict[str, float]:
        return {
            "gdp": round(self.gdp, 3),
            "inflation": round(self.inflation, 3),
            "unemployment": round(self.unemployment, 3),
            "consumer_confidence": round(self.confidence, 3),
            "tax_rate": round(self.tax_rate, 4),
            "policy_rate": round(self.policy_rate, 4),
            "wealth_gini": round(self.gini(), 4),
        }

    def snapshot(self, extra: dict | None = None) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "phase": self.phase,
            **self.indicators(),
            "employment_rate": round(self.employment_rate, 4),
            "productivity": round(self.productivity, 4),
            "active_policies": [p.label for p in self.policies],
            "active_shocks": [s.type for s in self.shocks],
            **(extra or {}),
        }

    def status(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "current_cycle": self.phase,
            "days_in_cycle": self.phase_tick * 30,
            "indicators": self.indicators(),
            "active_shocks": [
                {"type": s.type, "magnitude": s.magnitude, "ends_in_ticks": s.duration - (self.tick - s.started_tick)}
                for s in self.shocks
            ],
            "active_policies": [
                {"id": p.id, "label": p.label, "levers": p.levers, "source": p.source,
                 "ends_in_ticks": p.duration - (self.tick - p.started_tick)}
                for p in self.policies
            ],
        }

    def agents_status(self) -> dict[str, Any]:
        hs = self.households
        return {
            "households": {
                "count": len(hs),
                "employed": sum(h.employed for h in hs),
                "avg_savings": round(float(np.mean([h.savings for h in hs])), 2),
                "avg_debt": round(float(np.mean([h.debt for h in hs])), 2),
                "avg_spending": round(float(np.mean([h.last_spending for h in hs])), 2),
            },
            "firms": [{"price": f.price, "capacity": f.capacity,
                       "preferred_markup": float(f.markups[int(np.argmax(f.values))])} for f in self.firms],
            "bank": {"loans_issued": self.loans_issued, "defaults": self.defaults,
                     "training_samples": int(len(self.bank.y))},
            "regulator": {"type": type(self.regulator).__name__, "tax_rate": round(self.tax_rate, 4),
                          "adjustment": round(self.regulator_adjust, 4)},
        }
