"""Tabular Q-learning consumer."""

from __future__ import annotations

import numpy as np

from .base import BaseBrain


class ConsumerQBrain(BaseBrain):
    """State: (inflation bucket, savings bucket). Actions: share of income spent."""

    role = "consumer"

    def __init__(
        self,
        inflation_bins: int = 5,
        savings_bins: int = 5,
        actions=None,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int = 0,
    ):
        self.rng = np.random.RandomState(seed)
        self.inflation_bins = inflation_bins
        self.savings_bins = savings_bins
        self.actions = np.array(actions if actions is not None else [0.3, 0.5, 0.7, 0.9])
        self.q = np.zeros((inflation_bins, savings_bins, len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state: tuple[int, int] | None = None
        self.last_action_idx: int | None = None

    def _discretize(self, inflation_pct: float, savings: float) -> tuple[int, int]:
        # Inflation arrives in percent (2.0 means 2%); buckets are 2 points wide.
        inf_idx = int(np.clip(inflation_pct / 2.0, 0, self.inflation_bins - 1))
        sav_idx = int(np.clip(savings / 200.0, 0, self.savings_bins - 1))
        return inf_idx, sav_idx

    def act(self, state):
        s = self._discretize(state["inflation"], state["savings"])
        if self.rng.rand() < self.epsilon:
            a_idx = int(self.rng.randint(len(self.actions)))
        else:
            a_idx = int(np.argmax(self.q[s[0], s[1]]))
        self.last_state, self.last_action_idx = s, a_idx
        return float(self.actions[a_idx])

    def learn(self, reward, next_state):
        if self.last_state is None or self.last_action_idx is None:
            return
        nxt = self._discretize(next_state["inflation"], next_state["savings"])
        best_next = float(np.max(self.q[nxt[0], nxt[1]]))
        s0, s1 = self.last_state
        a = self.last_action_idx
        self.q[s0, s1, a] += self.alpha * (reward + self.gamma * best_next - self.q[s0, s1, a])
