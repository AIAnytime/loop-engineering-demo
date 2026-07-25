"""Loop configuration.

In loop engineering, the *loop* is the artifact you design -- not the prompt.
Everything a human decides up front lives here: cadence, models, budget,
stop conditions, the denylist, and the gate policy.

Rule of thumb from the workshop: if you cannot point at the line of code that
stops the loop, you do not have a loop -- you have a `while True`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# NOTE: the key in .env is spelled FINHUB (Finnhub's own docs use FINNHUB);
# we accept both so nobody loses 10 minutes to a typo during the workshop.
FINNHUB_API_KEY = os.getenv("FINHUB_API_KEY") or os.getenv("FINNHUB_API_KEY", "")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
FINNHUB_BASE = "https://finnhub.io/api/v1"


@dataclass(frozen=True)
class LoopConfig:
    """Every knob a loop needs before it is allowed to run unattended."""

    # ---- identity -------------------------------------------------------
    name: str = "equity-research-loop"
    goal: str = (
        "Produce a one-page equity research memo whose every numeric claim is "
        "traceable to data fetched this run, with an explicit recommendation, "
        "confidence, and risk section."
    )

    # ---- models: cheap for triage/build, stronger & independent for verify --
    # Maker-checker only works if the checker is not the maker. Ideal is a
    # different family; here (OpenAI-only) we use a stronger, independent
    # reasoning model to verify -- it reads the evidence line by line and does
    # not share the cheap implementer's shortcuts. Never let one model grade itself.
    triage_model: str = os.getenv("TRIAGE_MODEL", "gpt-4o-mini")
    implementer_model: str = os.getenv("IMPLEMENTER_MODEL", "gpt-4o-mini")
    verifier_model: str = os.getenv("VERIFIER_MODEL", "o4-mini")

    # ---- stop conditions (designed BEFORE the loop runs) ----------------
    max_attempts: int = 3                 # bounded retries, per item
    max_tokens_per_run: int = 120_000     # hard budget cap
    pause_at_budget_pct: int = 80         # soft brake -> triage only
    min_verifier_score: int = 80          # gate threshold, 0-100

    # ---- human gate policy ----------------------------------------------
    # Anything matching these escalates instead of auto-publishing.
    gate_on_high_conviction: bool = True   # STRONG BUY / STRONG SELL -> human
    gate_on_low_confidence: float = 0.5    # confidence < this -> human
    denylist_tickers: tuple[str, ...] = ("GME", "AMC", "DJT")  # meme//restricted

    # ---- external memory -------------------------------------------------
    state_path: Path = PROJECT_ROOT / "STATE.md"
    run_log_path: Path = PROJECT_ROOT / "loop-run-log.jsonl"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # ---- cadence (informational here; the scheduler owns it) -------------
    cadence: str = "1d"
    autonomy_tier: str = "L2"  # L1 report-only | L2 assisted | L3 unattended

    watchlist: tuple[str, ...] = field(default=("AAPL", "MSFT", "NVDA"))


DEFAULT = LoopConfig()


def missing_keys() -> list[str]:
    """Fail loudly and early rather than three LLM calls deep."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not FINNHUB_API_KEY:
        missing.append("FINHUB_API_KEY")
    return missing
