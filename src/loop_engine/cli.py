"""Workshop CLI.

    python -m loop_engine.cli --ticker AAPL          # one loop cycle
    python -m loop_engine.cli --watchlist            # the whole watch list
    python -m loop_engine.cli --ticker AAPL --naive  # the one-shot contrast
    python -m loop_engine.cli --state                # show external memory
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import DEFAULT, LoopConfig, missing_keys
from .llm import Budget, chat
from .loop import LoopResult, run_loop
from .state import LoopState
from . import tools

console = Console()

_STYLES = {
    "step": ("bold cyan", "▸"),
    "info": ("dim", " "),
    "warn": ("bold yellow", "!"),
    "error": ("bold red", "✗"),
    "skip": ("yellow", "↷"),
    "verdict": ("bold magenta", "⚖"),
    "violation": ("red", "  ·"),
    "gate": ("bold yellow", "🚦"),
    "escalate": ("bold yellow", "↑"),
    "done": ("bold green", "✓"),
}


def report(kind: str, msg: str) -> None:
    style, glyph = _STYLES.get(kind, ("white", "-"))
    console.print(f"[{style}]{glyph} {msg}[/]")


def naive_baseline(ticker: str, cfg: LoopConfig) -> None:
    """The 'before' picture: one prompt, one answer, zero verification.

    Run this next to the loop during the workshop. Same model, same question --
    the difference is entirely in the system around it.
    """
    budget = Budget(max_tokens=cfg.max_tokens_per_run)
    console.rule("[bold]NAIVE: single prompt, no tools, no verification")
    answer = chat(
        cfg.implementer_model,
        [{"role": "user", "content": f"Write a one-page equity research memo on {ticker.upper()} "
                                      "with a recommendation and confidence."}],
        budget=budget,
        max_tokens=1200,
    )
    console.print(Markdown(answer))
    console.print(
        Panel(
            "Every number above came out of the model's weights, not a data source.\n"
            "Nothing checked it. Nothing could have stopped it. It looks exactly as\n"
            "confident as a memo that is correct.",
            title="[bold red]What just happened",
            border_style="red",
        )
    )


def show_result(result: LoopResult) -> None:
    table = Table(title=f"Loop run · {result.ticker}", show_header=True, header_style="bold")
    table.add_column("attempt")
    table.add_column("verdict")
    table.add_column("score")
    table.add_column("violations")
    for a in result.attempts:
        verdict_style = "green" if a.verdict == "APPROVE" else "red"
        table.add_row(str(a.n), f"[{verdict_style}]{a.verdict}[/]", f"{a.score}/100",
                      str(len(a.violations)))
    console.print(table)
    console.print(
        f"[bold]outcome:[/] {result.outcome}   "
        f"[bold]tokens:[/] {result.tokens:,}   "
        f"[bold]duration:[/] {result.duration_s}s"
        + (f"   [bold]gate:[/] {result.gate_reason}" if result.gate_reason else "")
    )
    if result.memo:
        console.print(Panel(Markdown(result.memo), title="artifact", border_style="green"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loop engineering workshop demo")
    parser.add_argument("--ticker", help="ticker to research, e.g. AAPL")
    parser.add_argument("--watchlist", action="store_true", help="run the whole watch list")
    parser.add_argument("--naive", action="store_true", help="run the no-loop baseline instead")
    parser.add_argument("--state", action="store_true", help="print STATE.md and exit")
    parser.add_argument("--evidence", action="store_true", help="print raw tool output and exit")
    parser.add_argument("--max-attempts", type=int, help="override the retry cap")
    parser.add_argument("--verifier", help="override the verifier model")
    args = parser.parse_args(argv)

    cfg = DEFAULT
    overrides = {}
    if args.max_attempts:
        overrides["max_attempts"] = args.max_attempts
    if args.verifier:
        overrides["verifier_model"] = args.verifier
    if overrides:
        cfg = LoopConfig(**{**cfg.__dict__, **overrides})

    if args.state:
        if cfg.state_path.exists():
            console.print(Markdown(cfg.state_path.read_text(encoding="utf-8")))
        else:
            console.print("[yellow]no STATE.md yet — run the loop once[/]")
        return 0

    missing = missing_keys()
    if missing:
        console.print(f"[bold red]missing env vars:[/] {', '.join(missing)}  (add them to .env)")
        return 1

    if args.evidence:
        ticker = args.ticker or "AAPL"
        console.print_json(json.dumps(tools.collect_evidence(ticker)))
        return 0

    if args.naive:
        naive_baseline(args.ticker or "AAPL", cfg)
        return 0

    targets = list(cfg.watchlist) if args.watchlist else [args.ticker or "AAPL"]
    state = LoopState.load(cfg.state_path)
    if not state.watch:
        state.watch = list(cfg.watchlist)

    for ticker in targets:
        console.rule(f"[bold]LOOP · {ticker} · tier {cfg.autonomy_tier} · cadence {cfg.cadence}")
        result = run_loop(ticker, cfg, state=state, report=report)
        show_result(result)

    console.rule("[bold]external memory (STATE.md)")
    console.print(Markdown(cfg.state_path.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
