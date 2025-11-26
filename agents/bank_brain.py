# bank_brain.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from agents.shared.base_agent import BaseBrain

class BankRiskBrain(BaseBrain):
    """
    Bank loan approval brain using RandomForest.
    Input features:
      - savings
      - income
      - inflation
    Action:
      - approve (1) / deny (0)
    """

    def __init__(self, seed=0):
        self.model = RandomForestClassifier(random_state=seed)

        # ---- PRETRAIN on synthetic data ----
        X, y = self._generate_fake_data()
        self.model.fit(X, y)

    def _generate_fake_data(self, n=2000):
        X = np.zeros((n, 3))
        y = np.zeros(n)

        for i in range(n):
            savings = np.random.uniform(0, 1000)
            income = np.random.uniform(0, 200)
            inflation = np.random.uniform(0, 0.15)

            # Simple risk rule:
            risk = (0.3 * (1 - savings/1000)
                   + 0.5 * (1 - income/200)
                   + 0.2 * inflation)
            X[i] = [savings, income, inflation]
            y[i] = 1 if risk < 0.5 else 0  # approve if risk low

        return X, y

    def act(self, state):
        features = np.array([
            state["savings"],
            state["income"],
            state["inflation"]
        ]).reshape(1, -1)

        decision = self.model.predict(features)[0]
        return "approve" if decision == 1 else "deny"

    def learn(self, reward, next_state):
        # Banks don't learn in this version
        pass
