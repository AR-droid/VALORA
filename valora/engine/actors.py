from __future__ import annotations

from dataclasses import dataclass

PolicySignals = dict[str, float]
StateVector = dict[str, float]


@dataclass
class ActorBlueprint:
    """Static description of a macro actor in the economy."""

    name: str
    sector: str
    keywords: list[str]
    base_state: StateVector
    behavioural_rules: dict[str, float]
    reward_penalty: str


@dataclass
class SimulationAgent:
    """Runtime representation of an actor tailored to a specific policy."""

    name: str
    sector: str
    state: StateVector
    behavioural_rules: dict[str, float]
    reward_penalty: str

    def step(self, policy_signals: PolicySignals, channels: dict[str, float], time_index: int) -> StateVector:
        """Advance the agent state by one time step based on policy signals and targeted channels."""

        fiscal_effect = policy_signals.get("fiscal", 0.0)
        monetary_effect = policy_signals.get("monetary", 0.0)
        regulatory_effect = policy_signals.get("regulation", 0.0)

        net_pressure = (
            fiscal_effect * self.behavioural_rules.get("fiscal_sensitivity", 0.0)
            + monetary_effect * self.behavioural_rules.get("monetary_sensitivity", 0.0)
            + regulatory_effect * self.behavioural_rules.get("regulatory_sensitivity", 0.0)
        )

        decay = max(0.35, 1.0 - 0.06 * time_index)
        directional_weight = self.behavioural_rules.get("policy_alignment", 1.0)
        adjustment_rate = self.behavioural_rules.get("adjustment_rate", 0.25)
        delta = net_pressure * adjustment_rate * decay * directional_weight

        channel_magnifier = {
            "consumption_support": 1.15,
            "credit_easing": 1.1,
            "green_transition": 1.05,
            "trade_push": 1.0,
            "price_control": 0.9,
            "labor_support": 1.1,
        }

        for param, value in list(self.state.items()):
            sensitivity = self.behavioural_rules.get(f"{param}_sensitivity", 1.0)
            param_bias = self.behavioural_rules.get(f"{param}_bias", 0.0)

            targeted = 0.0
            name = param.lower()
            for channel, weight in channel_magnifier.items():
                if channel not in channels:
                    continue
                if channel == "consumption_support" and any(k in name for k in ["income", "consumption", "poverty", "price"]):
                    targeted += channels[channel] * weight
                if channel == "credit_easing" and any(k in name for k in ["loan", "credit", "default", "liquidity"]):
                    targeted += channels[channel] * weight
                if channel == "green_transition" and any(k in name for k in ["carbon", "energy", "fuel", "kwh", "emission"]):
                    targeted += channels[channel] * weight
                if channel == "trade_push" and any(k in name for k in ["export", "import", "shipping", "trade", "exchange"]):
                    targeted += channels[channel] * weight
                if channel == "price_control" and "cost" in name:
                    targeted += channels[channel] * weight
                if channel == "labor_support" and any(k in name for k in ["labor", "employment", "unemployment", "job"]):
                    targeted += channels[channel] * weight

            drift = 0.01 * param_bias
            updated = value + delta * sensitivity + targeted * 0.4 + drift
            self.state[param] = max(-5.0, min(5.0, round(updated, 3)))

        return dict(self.state)


def _actor_blueprints() -> list[ActorBlueprint]:
    return [
        ActorBlueprint(
            name="Households (General)",
            sector="households_general",
            keywords=["household", "consumer", "family", "income", "spending"],
            base_state={
                "income_distribution": 0.0,
                "avg_household_size": 3.2,
                "consumption_elasticity": 0.5,
                "unemployment_susceptibility": 0.4,
                "liquidity_buffer_pct": 0.2,
                "essential_spending_share": 0.65,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.8,
                "monetary_sensitivity": 0.25,
                "regulatory_sensitivity": 0.2,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.35,
                "income_distribution_sensitivity": 1.2,
                "unemployment_susceptibility_sensitivity": 1.1,
            },
            reward_penalty="Rewarded when disposable income and job stability rise; penalized by cost shocks.",
        ),
        ActorBlueprint(
            name="Low-Income Households",
            sector="households_low_income",
            keywords=["low income", "poverty", "welfare", "benefit", "subsidy"],
            base_state={
                "poverty_risk": 0.6,
                "benefit_dependency_rate": 0.45,
                "price_sensitivity": 0.8,
                "informal_sector_participation": 0.3,
                "default_probability": 0.25,
            },
            behavioural_rules={
                "fiscal_sensitivity": 1.0,
                "monetary_sensitivity": 0.2,
                "regulatory_sensitivity": 0.15,
                "policy_alignment": 1.1,
                "adjustment_rate": 0.45,
                "poverty_risk_sensitivity": 1.3,
                "price_sensitivity_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by transfers and subsidies; penalized by inflation or benefit cuts.",
        ),
        ActorBlueprint(
            name="High-Income Households",
            sector="households_high_income",
            keywords=["wealth", "luxury", "investment", "asset", "offshore"],
            base_state={
                "asset_ownership_ratio": 0.85,
                "luxury_spending_share": 0.25,
                "investment_propensity": 0.7,
                "offshore_saving_tendency": 0.35,
                "wealth_tax_exposure": 0.2,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.6,
                "monetary_sensitivity": 0.4,
                "regulatory_sensitivity": 0.3,
                "policy_alignment": 0.9,
                "adjustment_rate": 0.3,
                "investment_propensity_sensitivity": 1.2,
                "wealth_tax_exposure_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by investment incentives; penalized by wealth and capital taxes.",
        ),
        ActorBlueprint(
            name="Small Firms",
            sector="firms_small",
            keywords=["small business", "sme", "credit", "startup", "compliance"],
            base_state={
                "unit_cost": 1.0,
                "fixed_cost": 0.4,
                "variable_cost": 0.6,
                "production_capacity": 0.5,
                "market_power": 0.2,
                "loan_dependency": 0.55,
                "time_to_scale_production": 0.4,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.7,
                "monetary_sensitivity": 0.5,
                "regulatory_sensitivity": 0.6,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.4,
                "production_capacity_sensitivity": 1.2,
                "loan_dependency_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by tax breaks and credit access; penalized by compliance costs.",
        ),
        ActorBlueprint(
            name="Medium Firms",
            sector="firms_medium",
            keywords=["mid-size", "factory", "labor", "inventory", "supply"],
            base_state={
                "capacity_utilization_rate": 0.6,
                "labor_intensity": 0.55,
                "cost_pass_through_rate": 0.4,
                "credit_demand": 0.5,
                "inventory_buffer": 0.45,
                "supply_chain_risk": 0.35,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.65,
                "monetary_sensitivity": 0.35,
                "regulatory_sensitivity": 0.4,
                "policy_alignment": 0.95,
                "adjustment_rate": 0.32,
                "capacity_utilization_rate_sensitivity": 1.1,
                "supply_chain_risk_sensitivity": 1.05,
            },
            reward_penalty="Rewarded by demand growth and stable credit; penalized by supply disruptions.",
        ),
        ActorBlueprint(
            name="Large Corporations",
            sector="corporate_large",
            keywords=["corporate", "multinational", "shareholder", "debt", "investment"],
            base_state={
                "capital_intensity": 0.75,
                "global_exposure": 0.55,
                "supply_chain_depth": 0.5,
                "pricing_strategy": 0.45,
                "shareholder_pressure": 0.6,
                "debt_maturity_profile": 0.5,
                "investment_rate": 0.55,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.55,
                "monetary_sensitivity": 0.45,
                "regulatory_sensitivity": 0.45,
                "policy_alignment": 0.9,
                "adjustment_rate": 0.28,
                "investment_rate_sensitivity": 1.15,
                "global_exposure_sensitivity": 1.05,
            },
            reward_penalty="Rewarded by predictable policy and incentives; penalized by regulatory drag.",
        ),
        ActorBlueprint(
            name="Banks (Commercial)",
            sector="banks_commercial",
            keywords=["bank", "loan", "credit", "capital", "npl"],
            base_state={
                "capital_adequacy_ratio": 0.12,
                "loan_portfolio_risk": 0.35,
                "liquidity_coverage_ratio": 1.05,
                "interest_margin": 0.03,
                "nonperforming_loans_ratio": 0.04,
                "lending_policy": 0.5,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.25,
                "monetary_sensitivity": 0.85,
                "regulatory_sensitivity": 0.5,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.3,
                "loan_portfolio_risk_sensitivity": 1.2,
                "lending_policy_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by lower funding costs and healthy credit demand; penalized by rising defaults.",
        ),
        ActorBlueprint(
            name="Microfinance Institutions",
            sector="microfinance",
            keywords=["microfinance", "rural", "women", "small loan", "cap"],
            base_state={
                "avg_loan_size": 0.002,
                "rural_penetration": 0.6,
                "default_rate_mfi": 0.08,
                "women_borrower_share": 0.7,
                "interest_rate_cap_policy_exposure": 0.5,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.45,
                "monetary_sensitivity": 0.55,
                "regulatory_sensitivity": 0.5,
                "policy_alignment": 1.05,
                "adjustment_rate": 0.33,
                "default_rate_mfi_sensitivity": 1.2,
                "rural_penetration_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by concessional funding and targeted subsidies; penalized by tighter caps or rising defaults.",
        ),
        ActorBlueprint(
            name="Central Bank",
            sector="central_bank",
            keywords=["central bank", "policy rate", "reserve", "inflation", "liquidity"],
            base_state={
                "inflation_target": 0.05,
                "policy_rate": 0.04,
                "reserve_requirement": 0.09,
                "open_market_operations_capacity": 0.7,
                "communication_bias": 0.2,
                "currency_stability_priority": 0.6,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.2,
                "monetary_sensitivity": 1.0,
                "regulatory_sensitivity": 0.25,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.27,
                "policy_rate_sensitivity": 1.25,
                "currency_stability_priority_sensitivity": 1.1,
            },
            reward_penalty="Rewarded for meeting inflation and stability goals; penalized by volatility and credibility loss.",
        ),
        ActorBlueprint(
            name="Government (Fiscal Authority)",
            sector="fiscal_authority",
            keywords=["budget", "spending", "tax", "fiscal", "transfer"],
            base_state={
                "fiscal_space": 0.35,
                "spending_priorities": 0.5,
                "tax_bases": 0.55,
                "budget_rigidity": 0.45,
                "automatic_stabilizers": 0.6,
                "transfer_payments_ratio": 0.4,
            },
            behavioural_rules={
                "fiscal_sensitivity": 1.1,
                "monetary_sensitivity": 0.2,
                "regulatory_sensitivity": 0.25,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.3,
                "spending_priorities_sensitivity": 1.1,
                "transfer_payments_ratio_sensitivity": 1.15,
            },
            reward_penalty="Rewarded by revenue buoyancy and growth; penalized by fiscal stress and inefficiency.",
        ),
        ActorBlueprint(
            name="Tax Department / Revenue Authority",
            sector="tax_department",
            keywords=["tax authority", "compliance", "gst", "enforcement", "revenue"],
            base_state={
                "income_tax_enforcement_strength": 0.55,
                "corporate_tax_enforcement_strength": 0.5,
                "gst_compliance_rate": 0.65,
                "evasion_detection_prob": 0.4,
                "admin_efficiency": 0.5,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.95,
                "monetary_sensitivity": 0.1,
                "regulatory_sensitivity": 0.35,
                "policy_alignment": 1.0,
                "adjustment_rate": 0.34,
                "gst_compliance_rate_sensitivity": 1.1,
                "admin_efficiency_sensitivity": 1.05,
            },
            reward_penalty="Rewarded by higher compliance and efficiency; penalized by evasion and enforcement gaps.",
        ),
        ActorBlueprint(
            name="Regulatory Authority (Non-Monetary)",
            sector="regulator_non_monetary",
            keywords=["regulator", "compliance", "inspection", "penalty", "standard"],
            base_state={
                "regulatory_power_score": 0.5,
                "compliance_cost_factor": 0.45,
                "inspection_frequency": 0.5,
                "penalty_rate": 0.35,
                "sector_focus": 0.4,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.2,
                "monetary_sensitivity": 0.15,
                "regulatory_sensitivity": 1.0,
                "policy_alignment": 1.05,
                "adjustment_rate": 0.31,
                "inspection_frequency_sensitivity": 1.15,
                "compliance_cost_factor_sensitivity": 1.2,
            },
            reward_penalty="Rewarded by effective oversight; penalized when burdens hamper growth.",
        ),
        ActorBlueprint(
            name="Investors / Financial Markets",
            sector="financial_markets",
            keywords=["market", "investor", "portfolio", "liquidity", "risk"],
            base_state={
                "portfolio_allocation": 0.5,
                "risk_tolerance": 0.55,
                "reaction_speed": 0.6,
                "market_liquidity": 0.5,
                "foreign_flow_sensitivity": 0.45,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.45,
                "monetary_sensitivity": 0.55,
                "regulatory_sensitivity": 0.35,
                "policy_alignment": 1.05,
                "adjustment_rate": 0.36,
                "reaction_speed_sensitivity": 1.2,
                "market_liquidity_sensitivity": 1.1,
            },
            reward_penalty="Rewarded by clarity and liquidity; penalized by uncertainty and capital controls.",
        ),
        ActorBlueprint(
            name="Exporters / Trade Sector",
            sector="exporters",
            keywords=["export", "trade", "currency", "tariff", "shipping"],
            base_state={
                "export_volume": 0.5,
                "import_dependency": 0.45,
                "exchange_rate_exposure": 0.55,
                "global_demand_sensitivity": 0.6,
                "shipping_cost_index": 0.4,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.55,
                "monetary_sensitivity": 0.65,
                "regulatory_sensitivity": 0.35,
                "policy_alignment": 1.1,
                "adjustment_rate": 0.38,
                "exchange_rate_exposure_sensitivity": 1.2,
                "shipping_cost_index_sensitivity": 1.05,
            },
            reward_penalty="Rewarded by competitive exchange rates and logistics support; penalized by tariffs and high shipping costs.",
        ),
        ActorBlueprint(
            name="Energy Producers / Utility Sector",
            sector="energy_utilities",
            keywords=["energy", "utility", "fuel", "grid", "carbon", "kwh"],
            base_state={
                "generation_capacity": 0.55,
                "fuel_dependency_mix": 0.5,
                "cost_per_kwh": 0.12,
                "carbon_intensity": 0.4,
                "grid_stability_factor": 0.6,
                "energy_price_pass_through_rate": 0.5,
            },
            behavioural_rules={
                "fiscal_sensitivity": 0.4,
                "monetary_sensitivity": 0.25,
                "regulatory_sensitivity": 0.85,
                "policy_alignment": 1.05,
                "adjustment_rate": 0.37,
                "grid_stability_factor_sensitivity": 1.15,
                "cost_per_kwh_sensitivity": 1.2,
            },
            reward_penalty="Rewarded by investment support and stable regulation; penalized by emissions costs and input volatility.",
        ),
    ]


def extract_policy_signals(policy_text: str) -> tuple[PolicySignals, dict[str, float]]:
    """Heuristic extraction of the policy's directional signals and targeted channels."""

    lowered = policy_text.lower()

    def _score(words: list[str]) -> int:
        return sum(lowered.count(w) for w in words)

    fiscal_words = ["tax", "spend", "stimulus", "subsidy", "grant", "budget", "rebate", "transfer"]
    monetary_words = ["rate", "interest", "liquidity", "reserve", "monetary", "credit", "loan"]
    regulatory_words = ["regulation", "standard", "compliance", "emission", "safety", "rule", "tariff", "inspection"]

    fiscal_score = _score(fiscal_words)
    monetary_score = _score(monetary_words)
    regulatory_score = _score(regulatory_words)

    direction = -1.0 if any(k in lowered for k in ["cut", "reduce", "ease"]) else 1.0

    fiscal_intensity = direction * min(1.0, 0.1 + 0.05 * fiscal_score)
    monetary_intensity = direction * min(1.0, 0.1 + 0.05 * monetary_score)
    regulatory_intensity = min(1.0, 0.1 + 0.05 * regulatory_score)

    signals = {
        "fiscal": round(fiscal_intensity, 3),
        "monetary": round(monetary_intensity, 3),
        "regulation": round(regulatory_intensity, 3),
    }

    channels = {
        "consumption_support": round(0.2 + 0.05 * _score(["transfer", "benefit", "income", "cash", "voucher"]), 3),
        "credit_easing": round(0.15 + 0.05 * _score(["credit", "loan", "rate", "bank", "liquidity", "refinance"]), 3),
        "green_transition": round(0.1 + 0.05 * _score(["emission", "carbon", "renewable", "energy", "grid", "fuel"]), 3),
        "trade_push": round(0.1 + 0.05 * _score(["export", "tariff", "trade", "shipping", "fx", "exchange"]), 3),
        "price_control": round(0.1 + 0.05 * _score(["price cap", "ceiling", "control", "limit", "cap" ]), 3),
        "labor_support": round(0.1 + 0.05 * _score(["job", "employment", "hiring", "wage", "labor"]), 3),
    }

    return signals, channels


def select_affected_actors(policy_text: str, blueprints: list[ActorBlueprint], top_n: int = 5) -> list[ActorBlueprint]:
    lowered = policy_text.lower()

    def _score(bp: ActorBlueprint) -> tuple[int, float]:
        keyword_score = sum(lowered.count(k) for k in bp.keywords)
        base_priority = bp.behavioural_rules.get("fiscal_sensitivity", 0) + bp.behavioural_rules.get(
            "monetary_sensitivity", 0
        ) + bp.behavioural_rules.get("regulatory_sensitivity", 0)
        return keyword_score, base_priority

    sorted_bps = sorted(blueprints, key=lambda bp: _score(bp), reverse=True)
    return sorted_bps[:top_n]


def build_agents_for_policy(policy_text: str) -> tuple[list[SimulationAgent], PolicySignals, dict[str, float]]:
    blueprints = _actor_blueprints()
    policy_signals, channels = extract_policy_signals(policy_text)
    selected_bps = select_affected_actors(policy_text, blueprints, top_n=5)

    agents = [
        SimulationAgent(
            name=bp.name,
            sector=bp.sector,
            state=dict(bp.base_state),
            behavioural_rules=dict(bp.behavioural_rules),
            reward_penalty=bp.reward_penalty,
        )
        for bp in selected_bps
    ]
    return agents, policy_signals, channels


def simulate_economy(policy_text: str, time_steps: int = 8) -> dict[str, object]:
    """Simulate the evolution of the economy over time for the top impacted actors."""

    agents, policy_signals, channels = build_agents_for_policy(policy_text)
    timeline: list[dict[str, StateVector]] = []
    actor_series: dict[str, list[float]] = {agent.name: [] for agent in agents}
    actor_timelines: dict[str, dict[str, list[float]]] = {agent.name: {} for agent in agents}
    global_series: list[dict[str, float]] = []

    def _composite_index(state: StateVector) -> float:
        if not state:
            return 0.0
        return round(sum(state.values()) / len(state), 3)

    def _track_timelines(snapshot: dict[str, StateVector]):
        for actor, params in snapshot.items():
            for param, value in params.items():
                actor_timelines.setdefault(actor, {}).setdefault(param, []).append(value)

    def _macro(snapshot: dict[str, StateVector], last: dict[str, float]) -> dict[str, float]:
        base = last or {"gdp": 100.0, "inflation": 2.5, "unemployment": 5.2}

        def _avg(keys: list[str]) -> float:
            vals: list[float] = []
            for state in snapshot.values():
                for k in keys:
                    if k in state:
                        vals.append(state[k])
            return sum(vals) / len(vals) if vals else 0.0

        consumption = _avg(["income_distribution", "consumption_elasticity", "essential_spending_share", "poverty_risk"])
        production = _avg([
            "production_capacity",
            "capacity_utilization_rate",
            "investment_rate",
            "generation_capacity",
        ])
        credit = _avg(["loan_dependency", "credit_demand", "lending_policy", "loan_portfolio_risk", "default_probability"])
        trade = _avg(["export_volume", "global_demand_sensitivity", "exchange_rate_exposure", "shipping_cost_index"])
        energy_cost = _avg(["cost_per_kwh", "carbon_intensity", "energy_price_pass_through_rate", "fuel_dependency_mix"])

        gdp_change = (
            consumption * 2.2
            + production * 2.8
            + trade * 1.7
            - energy_cost * 0.6
            - credit * 1.0
            + policy_signals.get("fiscal", 0.0) * 3.2
            + policy_signals.get("monetary", 0.0) * 2.4
            - policy_signals.get("regulation", 0.0) * 1.2
            + channels.get("trade_push", 0.0) * 1.3
        )

        inflation_change = (
            consumption * 0.9
            + energy_cost * 1.6
            - production * 0.4
            - policy_signals.get("monetary", 0.0) * 1.1
            + policy_signals.get("fiscal", 0.0) * 0.7
            + channels.get("price_control", 0.0) * -0.6
        )

        unemployment_change = (
            -production * 1.3
            - consumption * 0.7
            + credit * 0.4
            + policy_signals.get("regulation", 0.0) * 0.5
            - channels.get("labor_support", 0.0) * 0.8
        )

        noise = 0.3
        return {
            "gdp": round(max(50.0, min(150.0, base["gdp"] + gdp_change * 0.35 + noise)), 3),
            "inflation": round(max(0.1, min(15.0, base["inflation"] + inflation_change * 0.25 + 0.1)), 3),
            "unemployment": round(max(0.5, min(20.0, base["unemployment"] + unemployment_change * 0.3 - 0.05)), 3),
        }

    for t in range(time_steps):
        snapshot: dict[str, StateVector] = {}
        for agent in agents:
            snapshot[agent.name] = agent.step(policy_signals, channels, time_index=t)
        timeline.append(snapshot)
        _track_timelines(snapshot)

        actor_indices = {_name: _composite_index(state) for _name, state in snapshot.items()}
        for name, score in actor_indices.items():
            actor_series[name].append(score)

        last_global = global_series[-1] if global_series else {}
        global_series.append(_macro(snapshot, last_global))

    return {
        "policy_signals": {**policy_signals, **channels},
        "selected_actors": [agent.name for agent in agents],
        "timeline": timeline,
        "actor_series": actor_series,
        "actor_timelines": actor_timelines,
        "global_series": global_series,
    }


if __name__ == "__main__":
    sample_policy = (
        "Increase transfer payments, ease small-business credit, and tighten emissions standards for utilities."
    )
    result = simulate_economy(sample_policy, time_steps=6)

    print("Selected actors:")
    for name in result["selected_actors"]:
        print(f"- {name}")

    print("\nGlobal trajectory (GDP, inflation, unemployment):")
    for idx, metrics in enumerate(result["global_series"], start=1):
        print(f"t={idx}: {metrics}")

    print("\nActor composite indices:")
    for actor, series in result["actor_series"].items():
        print(f"{actor}: {series}")
