
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gym import Env
from gym import spaces
from agents.shared.base_agent import BaseBrain


class RegulatorEnv(Env):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-3, high=3, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(1,), dtype=np.float32)

    def reset(self):
     
        self.state = np.array([1.0, 0.02, 0.05], dtype=np.float32)
        return self.state

    def step(self, action):
        delta_tax = float(np.clip(action[0], -0.02, 0.02))

        gdp, inf, unemp = self.state

   
        new_inf = inf + 0.005 - 0.5 * delta_tax
        new_unemp = min(0.5, unemp + 0.01 + 0.4 * delta_tax)
        new_gdp = gdp * (1.0 + 0.005 - 0.4 * delta_tax)

        self.state = np.array([new_gdp, new_inf, new_unemp], dtype=np.float32)

        
        reward = - (abs(new_inf - 0.02) + abs(new_unemp - 0.05))

        done = False
        return self.state, reward, done, {}


class RegulatorPPOBrain(BaseBrain):
    def __init__(self, model_path=None):
        env = DummyVecEnv([lambda: RegulatorEnv()])
        if model_path:
            self.model = PPO.load(model_path)
        else:
            self.model = PPO("MlpPolicy", env, verbose=0)
            self.model.learn(10_000)   # quick train

    def act(self, state):
        obs = np.array([
            state["gdp_norm"],
            state["inflation"],
            state["unemployment"]
        ])
        action, _ = self.model.predict(obs, deterministic=True)
        return float(action[0])

    def learn(self, reward, next_state):
        pass
