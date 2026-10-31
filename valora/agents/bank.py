"""Bank that approves or denies loans with a RandomForest credit model."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseBrain


class BankRiskBrain(BaseBrain):
    """Features: savings, income, inflation (fraction). Action: approve or deny.

    The model is pretrained on synthetic data and refit online from observed
    repayments, so it learns from defaults that actually happen in the simulation.
    """

    role = "bank"

    def __init__(self, seed: int = 0, n: int = 1500, refit_every: int = 50):
        self.rng = np.random.RandomState(seed)
        self.model = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
        self.X, self.y = self._synthetic(n)
        self.model.fit(self.X, self.y)
        self.refit_every = refit_every
        self.new_samples = 0
        self.last_features: np.ndarray | None = None

    def _synthetic(self, n: int):
        savings = self.rng.uniform(0, 1000, n)
        income = self.rng.uniform(0, 200, n)
        inflation = self.rng.uniform(0, 0.15, n)
        risk = 0.3 * (1 - savings / 1000) + 0.5 * (1 - income / 200) + 0.2 * inflation / 0.15
        X = np.column_stack([savings, income, inflation])
        y = (risk < 0.5).astype(int)
        return X, y

    def act(self, state):
        feats = np.array([[state["savings"], state["income"], state["inflation"]]], dtype=float)
        self.last_features = feats
        return "approve" if int(self.model.predict(feats)[0]) == 1 else "deny"

    def learn(self, reward, next_state):
        """``reward`` > 0 means the last approved loan was repaid; < 0 means it defaulted."""
        if self.last_features is None or reward == 0:
            return
        self.X = np.vstack([self.X, self.last_features])
        self.y = np.append(self.y, 1 if reward > 0 else 0)
        self.new_samples += 1
        if self.new_samples >= self.refit_every:
            self.model.fit(self.X[-3000:], self.y[-3000:])
            self.new_samples = 0
