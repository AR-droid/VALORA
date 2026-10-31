"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM: any OpenAI-compatible chat completions endpoint (Groq by default).
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    llm_timeout_s: float = field(default_factory=lambda: _float("LLM_TIMEOUT_S", 45.0))
    llm_max_tokens: int = field(default_factory=lambda: _int("MAX_AGENT_COMPLETION_TOKENS", 450))
    # Hard ceiling on LLM calls per agent run, so a loop can never run away.
    agent_max_steps: int = field(default_factory=lambda: _int("AGENT_MAX_STEPS", 6))

    secret_key: str = field(default_factory=lambda: os.getenv("FLASK_SECRET_KEY", ""))
    db_path: str = field(default_factory=lambda: os.getenv("VALORA_DB", "valora.db"))
    seed: int = field(default_factory=lambda: _int("VALORA_SEED", 7))
    # Seconds between background economy ticks (0 disables the loop).
    tick_seconds: float = field(default_factory=lambda: _float("VALORA_TICK_SECONDS", 30.0))
    port: int = field(default_factory=lambda: _int("PORT", 5004))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key) and os.getenv("VALORA_OFFLINE", "") not in {"1", "true", "yes"}


def get_settings() -> Settings:
    return Settings()
