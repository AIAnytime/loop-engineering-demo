"""Convert _nb_source.py (cell-marked python) into a Colab-ready .ipynb.

The source is a plain .py file so it can be linted, diffed and — most importantly —
actually executed end to end before a room full of people watches it run.

    python notebooks/build_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "_nb_source.py"
OUT = HERE / "loop_engineering_crewai_demo.ipynb"


def build() -> dict:
    text = SRC.read_text(encoding="utf-8")
    markers = list(re.finditer(r"^# %%(?: \[(markdown)\])?\s*$", text, re.M))
    cells = []

    for i, marker in enumerate(markers):
        start = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        body = text[start:end].strip("\n")
        is_md = marker.group(1) == "markdown"

        if is_md:
            # strip the leading "# " comment prefix from markdown cells
            lines = [re.sub(r"^# ?", "", line) for line in body.split("\n")]
            source = "\n".join(lines).strip("\n")
            cell = {"cell_type": "markdown", "metadata": {}, "source": _split(source)}
        else:
            # un-comment the pip install so it becomes a real shell cell in Colab
            source = re.sub(r"^# (!pip .*)$", r"\1", body, flags=re.M).strip("\n")
            cell = {"cell_type": "code", "execution_count": None, "metadata": {},
                    "outputs": [], "source": _split(source)}
        if source.strip():
            cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def _split(source: str) -> list[str]:
    lines = source.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


if __name__ == "__main__":
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    print(f"✓ {OUT.name} — {code} code cells, {md} markdown cells")
