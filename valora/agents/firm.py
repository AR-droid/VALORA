"""Firm that adapts its price markup with a simple bandit over markup levels."""

from __future__ import annotations

import numpy as np

from .base import BaseBrain


class FirmAdaptiveBrain(BaseBrain):
    """Chooses a markup over unit cost with epsilon-greedy value estimates.

    Production follows the demand signal, capped at capacity. Reward is profit,
    so the firm learns which markup earns the most in the current environment.
    """

    role = "firm"

    def __init__(self, cost: float = 6.0, capacity: int = 100, epsilon: float = 0.1, seed: int = 0):
        self.cost = cost
        self.capacity = capacity
        self.markups = np.array([1.05, 1.15, 1.3, 1.5, 1.75])
        self.values = np.zeros(len(self.markups))
        self.counts = np.zeros(len(self.markups))
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        self.last_idx: int | None = None
        self.price = float(self.cost * self.markups[2])

    def act(self, state):
        demand = float(state["demand_signal"])
        cost_pressure = float(state.get("cost_pressure", 1.0))
        if self.rng.rand() < self.epsilon:
            idx = int(self.rng.randint(len(self.markups)))
        else:
            idx = int(np.argmax(self.values))
        self.last_idx = idx
        unit_cost = self.cost * cost_pressure
        self.price = round(unit_cost * float(self.markups[idx]), 2)
        production = int(self.capacity * min(1.0, max(0.1, demand)))
        return {"price": self.price, "production": production, "unit_cost": unit_cost}

    def learn(self, reward, next_state):
        if self.last_idx is None:
            return
        i = self.last_idx
        self.counts[i] += 1
        self.values[i] += (reward - self.values[i]) / self.counts[i]
