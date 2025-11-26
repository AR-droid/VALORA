# firm_brain.py
from agents.shared.base_agent import BaseBrain

class FirmHeuristicBrain(BaseBrain):
    """
    Firm with heuristic economic behavior (upgrade to DQN later).
    Controls: price, production.
    Reward: profit.
    """

    def __init__(self, base_price=10.0, cost=6.0, capacity=100, seed=0):
        self.price = base_price
        self.cost = cost
        self.capacity = capacity

    def act(self, state):
        """
        state includes:
            - demand_signal (0.5 low, 1.0 normal, 1.5 high)
        """
        demand = state["demand_signal"]


        if demand > 1.05:
            self.price *= 1.02
        elif demand < 0.95:
            self.price *= 0.98

  
        self.price = max(self.cost * 1.05, self.price)


        production = int(self.capacity * min(1.0, demand))
        return {
            "price": round(self.price, 2),
            "production": production
        }

    def learn(self, reward, next_state):
        """
        For now, heuristic = no learning.
        Later we replace with DQN.
        """
        pass
