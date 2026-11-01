"""The VALORA agent protocol: every brain observes a dict and returns an action."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class BaseBrain:
    """Minimal agent interface.

    ``act`` maps an observation to an action. ``learn`` receives the reward for the
    previous action and the next observation. Brains that do not learn may keep the
    default no-op ``learn``.
    """

    role: str = "agent"

    def act(self, state: Mapping[str, Any]) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def learn(self, reward: float, next_state: Mapping[str, Any]) -> None:
        return None
