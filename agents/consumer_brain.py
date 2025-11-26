
import numpy as np
from agents.shared.base_agent import BaseBrain

class ConsumerQBrain(BaseBrain):
    """
    Q-learning consumer agent.
    State: (inflation_bucket, savings_bucket)
    Actions: spend fractions [0.3, 0.5, 0.7, 0.9]
    """
    def __init__(self,
                 inflation_bins=5,
                 savings_bins=5,
                 actions=None,
                 alpha=0.1,
                 gamma=0.9,
                 epsilon=0.1,
                 seed=0):

        self.rng = np.random.RandomState(seed)
        self.inflation_bins = inflation_bins
        self.savings_bins = savings_bins

        if actions is None:
            self.actions = np.array([0.3, 0.5, 0.7, 0.9])
        else:
            self.actions = np.array(actions)

        # Q-table
        self.q = np.zeros((inflation_bins, savings_bins, len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.last_state = None
        self.last_action_idx = None

    # ----- State Discretization -----
    def _discretize_state(self, inflation, savings):
        inf_idx = int(np.clip(inflation * 100, 0, self.inflation_bins - 1))
        sav_idx = int(np.clip(savings / 200, 0, self.savings_bins - 1))
        return inf_idx, sav_idx

    # ----- Action Selection -----
    def act(self, state):
        inflation = state["inflation"]
        savings = state["savings"]

        s = self._discretize_state(inflation, savings)

        # epsilon-greedy
        if self.rng.rand() < self.epsilon:
            a_idx = self.rng.randint(len(self.actions))
        else:
            a_idx = int(np.argmax(self.q[s[0], s[1]]))

        self.last_state = s
        self.last_action_idx = a_idx
        return self.actions[a_idx]

    # ----- Learning -----
    def learn(self, reward, next_state):
        if self.last_state is None:
            return

        next_s = self._discretize_state(
            next_state["inflation"],
            next_state["savings"]
        )
        best_next = np.max(self.q[next_s[0], next_s[1]])

        s0, s1 = self.last_state
        a = self.last_action_idx

        td = reward + self.gamma * best_next - self.q[s0, s1, a]
        self.q[s0, s1, a] += self.alpha * td
