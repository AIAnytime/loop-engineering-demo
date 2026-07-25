"""Thin OpenAI client with token accounting.

Token accounting is not a nice-to-have in a loop. A naive loop that resends the
whole transcript every step grows cost ~N(N+1)/2. The budget meter below is what
lets the loop stop itself before your card does.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import OPENAI_API_KEY, OPENAI_URL

# Reasoning models (o-series) reject `temperature` and meter output as
# `max_completion_tokens`, not `max_tokens`. We special-case them below.
_REASONING = ("o1", "o3", "o4")


@dataclass
class Budget:
    """A running meter shared by every agent in the loop."""

    max_tokens: int
    used_tokens: int = 0
    calls: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def charge(self, model: str, tokens: int) -> None:
        self.used_tokens += tokens
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0) + tokens

    @property
    def pct(self) -> float:
        return 100.0 * self.used_tokens / self.max_tokens if self.max_tokens else 0.0

    def exhausted(self) -> bool:
        return self.used_tokens >= self.max_tokens

    def over(self, pct: int) -> bool:
        return self.pct >= pct


class LLMError(RuntimeError):
    pass


def chat(
    model: str,
    messages: list[dict[str, str]],
    budget: Budget | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1600,
    retries: int = 3,
) -> str:
    """One completion. Retries transport errors only -- never silently loops."""
    if not OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not set (check .env)")
    if budget is not None and budget.exhausted():
        raise LLMError(f"token budget exhausted: {budget.used_tokens}/{budget.max_tokens}")

    payload: dict[str, Any] = {"model": model, "messages": messages}
    if model.startswith(_REASONING):
        # o-series: no temperature; give reasoning room via max_completion_tokens.
        payload["max_completion_tokens"] = max(max_tokens, 4000)
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise LLMError(f"{resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise LLMError(str(data["error"])[:300])
            text = data["choices"][0]["message"]["content"] or ""
            if budget is not None:
                usage = data.get("usage") or {}
                budget.charge(model, int(usage.get("total_tokens", 0)))
            return text.strip()
        except Exception as exc:  # noqa: BLE001 - transport-level backoff
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"LLM call failed after {retries} attempts: {last_err}")


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_BARE_OBJ = re.compile(r"\{.*\}", re.S)


def chat_json(model: str, messages: list[dict[str, str]], **kw: Any) -> dict:
    """Completion that must return a JSON object.

    Structured output is what turns 'the model said it looks fine' into a gate
    condition you can branch on. Free-text verdicts are not verification.
    """
    raw = chat(model, messages, **kw)
    for pattern in (_JSON_BLOCK, _BARE_OBJ):
        match = pattern.search(raw)
        if match:
            try:
                return json.loads(match.group(1) if pattern is _JSON_BLOCK else match.group(0))
            except json.JSONDecodeError:
                continue
    raise LLMError(f"expected JSON, got: {raw[:300]}")
