"""THE LOOP.

Read this file top to bottom

    goal
      -> read state          (external memory, survives the process)
      -> gather evidence     (tools = ground truth)
      -> triage              (cheap model: is the expensive path worth it?)
      -> [ implement -> verify ]  bounded retries, verifier feedback in the loop
      -> human gate          (risk policy decides: publish or escalate)
      -> write state + log   (so tomorrow's run starts where today ended)

Everything that distinguishes this from `while True: agent.run()` is marked
with a `# LOOP:` comment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import agents, tools
from .config import LoopConfig
from .llm import Budget, LLMError
from .state import LoopState, append_run_log


@dataclass
class Attempt:
    n: int
    memo: str
    verdict: str
    score: int
    violations: list[str] = field(default_factory=list)


@dataclass
class LoopResult:
    ticker: str
    outcome: str                    # completed | escalated | skipped | aborted
    attempts: list[Attempt] = field(default_factory=list)
    memo: str | None = None
    gate_reason: str | None = None
    tokens: int = 0
    duration_s: float = 0.0
    escalations: int = 0

    @property
    def final_score(self) -> int:
        return self.attempts[-1].score if self.attempts else 0


Reporter = Callable[[str, str], None]


def _noop(_kind: str, _msg: str) -> None:
    pass


def run_loop(
    ticker: str,
    cfg: LoopConfig,
    state: LoopState | None = None,
    report: Reporter = _noop,
) -> LoopResult:
    """Run one full cycle of the equity-research loop for one ticker."""
    started = time.time()
    ticker = ticker.upper()
    state = state or LoopState.load(cfg.state_path)
    budget = Budget(max_tokens=cfg.max_tokens_per_run)
    result = LoopResult(ticker=ticker, outcome="aborted")

    # LOOP: denylist is checked before a single token is spent. Guardrails are
    # cheapest at the edge of the system, not inside the model's judgement.
    if ticker in cfg.denylist_tickers:
        report("gate", f"{ticker} is on the denylist — refusing to run")
        state.noise.append(f"{ticker}: denylisted, skipped")
        result.outcome = "skipped"
        result.gate_reason = "denylist"
        return _finish(result, state, cfg, budget, started)

    # LOOP: collision detection — two loops must never act on the same item.
    if state.acting_on and state.acting_on != ticker:
        report("gate", f"another loop is acting on {state.acting_on} — skipping")
        result.outcome = "skipped"
        result.gate_reason = "peer collision"
        return _finish(result, state, cfg, budget, started)

    state.start_run(ticker)

    # ---- 1. TOOLS: ground truth first, opinions later --------------------
    report("step", f"fetching evidence for {ticker}")
    evidence = tools.collect_evidence(ticker)
    errors = [k for k, v in evidence.items() if isinstance(v, dict) and "error" in v]
    if errors:
        report("warn", f"degraded evidence: {', '.join(errors)}")

    # ---- 2. TRIAGE: the cheapest possible exit ---------------------------
    report("step", f"triage ({cfg.triage_model})")
    try:
        decision = agents.triage(cfg.triage_model, evidence, _summarize(state), budget)
    except LLMError as exc:
        report("error", f"triage failed: {exc}")
        result.outcome = "aborted"
        return _finish(result, state, cfg, budget, started)

    if not decision.get("worth_running", False):
        # LOOP: empty watchlist -> immediate exit. This single branch is the
        # difference between a $2/day loop and a $200/day loop.
        report("skip", f"triage says stop: {decision.get('reason')}")
        state.noise.append(f"{ticker}: {decision.get('reason')}")
        result.outcome = "skipped"
        result.gate_reason = "triage: not worth running"
        return _finish(result, state, cfg, budget, started)

    focus = decision.get("focus", [])
    report("info", f"focus: {focus}")

    # ---- 3. THE MAKER-CHECKER LOOP ---------------------------------------
    rejection: dict[str, Any] | None = None
    memo = ""
    for attempt in range(1, cfg.max_attempts + 1):
        # LOOP: budget brake. At 80% we stop starting new expensive work.
        if budget.over(cfg.pause_at_budget_pct):
            report("warn", f"budget at {budget.pct:.0f}% — pausing before attempt {attempt}")
            result.gate_reason = "budget brake"
            break

        report("step", f"attempt {attempt}/{cfg.max_attempts}: implementer ({cfg.implementer_model})")
        memo = agents.implement(
            cfg.implementer_model, ticker, evidence, focus, budget,
            rejection=rejection, attempt=attempt,
        )

        report("step", f"attempt {attempt}/{cfg.max_attempts}: verifier ({cfg.verifier_model})")
        review = agents.verify(cfg.verifier_model, evidence, memo, budget)
        result.attempts.append(
            Attempt(n=attempt, memo=memo, verdict=review["verdict"],
                    score=review["score"], violations=review["violations"])
        )
        report(
            "verdict",
            f"{review['verdict']} · score {review['score']}/100"
            + (f" · {len(review['violations'])} violation(s)" if review["violations"] else ""),
        )
        for v in review["violations"][:5]:
            report("violation", str(v))

        if review["verdict"] == "APPROVE":
            rejection = None
            break

        # LOOP: the rejection is *fed forward*. A retry that doesn't carry the
        # reason it failed is just rolling the dice again.
        rejection = review
    else:
        # LOOP: attempt cap reached -> escalate. Never retry forever.
        report("escalate", f"{cfg.max_attempts} attempts exhausted — escalating to human")
        state.inbox.append(
            f"{ticker}: verifier rejected {cfg.max_attempts}x "
            f"(last score {result.final_score}/100) — needs a human read"
        )
        result.outcome = "escalated"
        result.escalations = 1
        result.memo = memo
        return _finish(result, state, cfg, budget, started)

    if not result.attempts or result.attempts[-1].verdict != "APPROVE":
        report("escalate", "stopped without approval — escalating")
        state.inbox.append(f"{ticker}: stopped without approval ({result.gate_reason or 'unknown'})")
        result.outcome = "escalated"
        result.escalations = 1
        result.memo = memo
        return _finish(result, state, cfg, budget, started)

    # ---- 4. HUMAN GATE: verifier approval is necessary, not sufficient ----
    last_review = result.attempts[-1]
    gate_reason = _gate(memo, cfg)
    result.memo = memo
    if gate_reason:
        report("gate", f"human gate triggered: {gate_reason}")
        state.inbox.append(f"{ticker}: APPROVED by verifier (score {last_review.score}) but gated — {gate_reason}")
        result.outcome = "escalated"
        result.escalations = 1
        result.gate_reason = gate_reason
    else:
        path = _publish(memo, ticker, cfg)
        report("done", f"published → {path}")
        state.done.append(f"{ticker}: memo approved (score {last_review.score}/100) → {path.name}")
        result.outcome = "completed"

    return _finish(result, state, cfg, budget, started)


# --------------------------------------------------------------------------
# gate policy, publishing, bookkeeping
# --------------------------------------------------------------------------

def _gate(memo: str, cfg: LoopConfig) -> str | None:
    """Risk policy. Deterministic code, not a model call -- gates must be auditable.

    Parse the *section*, never grep the whole document. A naive
    `"STRONG BUY" in memo` fires on the sentence "13 strong buy ratings" sitting
    in the thesis, and now every run escalates for no reason. Gates that cry
    wolf get switched off, and a gate that is off is not a gate.
    """
    recommendation = _section(memo, "Recommendation")
    if cfg.gate_on_high_conviction and recommendation:
        label = recommendation.upper()
        if label.startswith("STRONG BUY") or label.startswith("STRONG SELL"):
            return f"high-conviction call ({recommendation}) requires human sign-off"
    confidence = _extract_confidence(memo)
    if confidence is not None and confidence < cfg.gate_on_low_confidence:
        return f"low confidence ({confidence:.2f}) — human should decide"
    return None


def _section(memo: str, name: str) -> str | None:
    """First non-empty line of a `## <name>` section, stripped of markdown noise."""
    import re
    match = re.search(rf"^#{{1,4}}\s*{name}\s*$(.*?)(?=^#{{1,4}}\s|\Z)", memo,
                      re.I | re.M | re.S)
    if not match:
        return None
    for line in match.group(1).splitlines():
        cleaned = line.strip().strip("*_`-• ").strip()
        if cleaned:
            return cleaned
    return None


def _extract_confidence(memo: str) -> float | None:
    import re
    raw = _section(memo, "Confidence")
    if raw:
        match = re.search(r"([01]?\.\d+|[01])\b", raw)
        if match:
            return float(match.group(1))
    match = re.search(r"confidence[^0-9]{0,20}([01]?\.\d+)", memo, re.I)
    return float(match.group(1)) if match else None


def _publish(memo: str, ticker: str, cfg: LoopConfig) -> Path:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    import datetime as dt
    path = cfg.artifacts_dir / f"{ticker}_{dt.date.today()}.md"
    path.write_text(memo, encoding="utf-8")
    return path


def _summarize(state: LoopState) -> str:
    return (
        f"last_run={state.last_run}; runs={state.runs}; "
        f"open_human_items={len(state.inbox)}; recent_done={state.done[-3:]}"
    )


def _finish(result: LoopResult, state: LoopState, cfg: LoopConfig,
            budget: Budget, started: float) -> LoopResult:
    result.tokens = budget.used_tokens
    result.duration_s = round(time.time() - started, 1)
    state.finish_run()
    state.save(cfg.state_path)
    append_run_log(
        cfg.run_log_path,
        {
            "pattern": cfg.name,
            "target": result.ticker,
            "outcome": result.outcome,
            "attempts": len(result.attempts),
            "final_score": result.final_score,
            "escalations": result.escalations,
            "gate_reason": result.gate_reason,
            "tokens": result.tokens,
            "duration_s": result.duration_s,
            "models": {
                "triage": cfg.triage_model,
                "implementer": cfg.implementer_model,
                "verifier": cfg.verifier_model,
            },
        },
    )
    return result
