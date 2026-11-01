"""Regulator that moves the tax rate toward inflation and unemployment targets.

Two implementations share one interface:

* ``RuleRegulator``: a Taylor-style rule with no heavy dependencies (default).
* ``PPORegulator``: a PPO policy from stable-baselines3, used when the optional
  ``rl`` extra is installed (``pip install valora[rl]``) and ``VALORA_RL=1``.
"""

from __future__ import annotations

import os

import numpy as np

from .base import BaseBrain

INFLATION_TARGET = 0.02
UNEMPLOYMENT_TARGET = 0.05
MAX_STEP = 0.02


class RuleRegulator(BaseBrain):
    role = "regulator"

    def __init__(self, k_inflation: float = 0.25, k_unemployment: float = 0.2):
        self.k_inf = k_inflation
        self.k_unemp = k_unemployment

    def act(self, state):
        inf_gap = state["inflation"] - INFLATION_TARGET
        unemp_gap = state["unemployment"] - UNEMPLOYMENT_TARGET
        # Raise taxes to cool inflation; cut them when unemployment is high.
        delta = self.k_inf * inf_gap - self.k_unemp * unemp_gap
        return float(np.clip(delta, -MAX_STEP, MAX_STEP))


def _build_ppo(timesteps: int = 10_000, model_path: str | None = None):
    import gymnasium as gym  # noqa: PLC0415
    from gymnasium import spaces  # noqa: PLC0415
    from stable_baselines3 import PPO  # noqa: PLC0415

    class RegulatorEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-3, high=3, shape=(3,), dtype=np.float32)
            self.action_space = spaces.Box(low=-MAX_STEP, high=MAX_STEP, shape=(1,), dtype=np.float32)
            self.t = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            rng = self.np_random
            self.state = np.array(
                [1.0, rng.uniform(0.0, 0.08), rng.uniform(0.02, 0.12)], dtype=np.float32
            )
            self.t = 0
            return self.state, {}

        def step(self, action):
            d = float(np.clip(action[0], -MAX_STEP, MAX_STEP))
            gdp, inf, unemp = self.state
            inf = inf + 0.003 - 0.5 * d
            unemp = float(np.clip(unemp + 0.002 + 0.4 * d, 0.0, 0.5))
            gdp = gdp * (1.0 + 0.005 - 0.4 * d)
            self.state = np.array([gdp, inf, unemp], dtype=np.float32)
            self.t += 1
            reward = -(abs(inf - INFLATION_TARGET) + abs(unemp - UNEMPLOYMENT_TARGET))
            return self.state, reward, False, self.t >= 120, {}

    if model_path and os.path.exists(model_path):
        return PPO.load(model_path)
    model = PPO("MlpPolicy", RegulatorEnv(), verbose=0, seed=0)
    model.learn(timesteps)
    if model_path:
        model.save(model_path)
    return model


class PPORegulator(BaseBrain):
    role = "regulator"

    def __init__(self, model_path: str | None = "regulator_ppo.zip"):
        self.model = _build_ppo(model_path=model_path)

    def act(self, state):
        obs = np.array([state["gdp_norm"], state["inflation"], state["unemployment"]], dtype=np.float32)
        action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
        return float(np.asarray(action).ravel()[0])


def make_regulator() -> BaseBrain:
    if os.getenv("VALORA_RL", "") in {"1", "true", "yes"}:
        try:
            return PPORegulator()
        except Exception as exc:  # optional dependency missing or training failed
            print(f"[valora] PPO regulator unavailable ({exc}); using rule-based regulator")
    return RuleRegulator()
