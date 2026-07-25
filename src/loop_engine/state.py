"""External memory: STATE.md on disk.

    "The memory must be on disk, not in the context.
     The agent forgets, but the repository doesn't."  -- Addy Osmani

The agent's context window dies with the process. STATE.md survives it, and is
the handoff point between today's run and tomorrow's. It is also human-readable
on purpose: a state file you cannot skim is a state file nobody reviews.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

HEADER = "# Loop State — equity-research-loop"

_TEMPLATE = """{header}

Last run: {last_run}
Runs completed: {runs}

## Done
{done}

## Human Inbox (needs a decision from you)
{inbox}

## Watch List
{watch}

## Recent Noise (ignored this run)
{noise}
"""


@dataclass
class LoopState:
    last_run: str = "never"
    runs: int = 0
    done: list[str] = field(default_factory=list)
    inbox: list[str] = field(default_factory=list)
    watch: list[str] = field(default_factory=list)
    noise: list[str] = field(default_factory=list)
    acting_on: str | None = None  # collision detection between parallel loops

    # -- rendering ---------------------------------------------------------
    @staticmethod
    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {i}" for i in items) if items else "_(empty)_"

    def render(self) -> str:
        return _TEMPLATE.format(
            header=HEADER,
            last_run=self.last_run,
            runs=self.runs,
            done=self._bullets(self.done[-10:]),
            inbox=self._bullets(self.inbox[-10:]),
            watch=self._bullets(self.watch),
            noise=self._bullets(self.noise[-10:]),
        )

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(self.render(), encoding="utf-8")
        # A machine-readable sidecar so the next run doesn't have to parse prose.
        path.with_suffix(".json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LoopState":
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            return cls(**json.loads(sidecar.read_text(encoding="utf-8")))
        return cls()

    # -- hygiene -----------------------------------------------------------
    def prune(self, keep: int = 20) -> None:
        """State rot is real: closed items pile up until the loop drowns in them."""
        self.done = self.done[-keep:]
        self.noise = self.noise[-keep:]

    def start_run(self, target: str) -> None:
        self.acting_on = target
        self.last_run = dt.datetime.now().isoformat(timespec="seconds")
        self.runs += 1

    def finish_run(self) -> None:
        self.acting_on = None
        self.prune()


def append_run_log(path: Path, record: dict) -> None:
    """One JSON line per run. This is what your weekly dashboard reads."""
    record = {"logged_at": dt.datetime.now().isoformat(timespec="seconds"), **record}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
