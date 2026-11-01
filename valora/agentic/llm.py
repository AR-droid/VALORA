"""Small client for any OpenAI-compatible chat completions API (Groq by default).

It supports tool calling and JSON output. When no API key is configured, or the
API fails, callers get ``None`` back and fall back to their deterministic path, so
the platform keeps working offline and in tests.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings, get_settings


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "errors": self.errors[-5:]}


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.usage = LLMUsage()

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    @property
    def model(self) -> str:
        return self.s.llm_model if self.enabled else "offline-deterministic"

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False,
             max_tokens: int | None = None, temperature: float = 0.3) -> dict | None:
        """Return the assistant message dict, or ``None`` when the LLM is unavailable."""
        if not self.enabled:
            return None
        body: dict[str, Any] = {
            "model": self.s.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.s.llm_max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}"}
        url = self.s.llm_base_url.rstrip("/") + "/chat/completions"
        for attempt in range(3):
            try:
                r = httpx.post(url, json=body, headers=headers, timeout=self.s.llm_timeout_s)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                self.usage.calls += 1
                u = data.get("usage") or {}
                self.usage.prompt_tokens += int(u.get("prompt_tokens", 0))
                self.usage.completion_tokens += int(u.get("completion_tokens", 0))
                return data["choices"][0]["message"]
            except Exception as exc:  # network, auth, schema
                self.usage.errors.append(f"{type(exc).__name__}: {exc}"[:300])
                if attempt == 2:
                    return None
                time.sleep(1.0)
        self.usage.errors.append("rate limited or server error after retries")
        return None

    def complete_json(self, system: str, user: str, *, max_tokens: int | None = None) -> dict | None:
        msg = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                        json_mode=True, max_tokens=max_tokens)
        if not msg:
            return None
        return parse_json(msg.get("content") or "")

    def complete_text(self, system: str, user: str, *, max_tokens: int | None = None) -> str | None:
        msg = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_tokens=max_tokens)
        return (msg or {}).get("content") or None


def parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None
