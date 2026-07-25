# Loop Engineering — workshop kit

A 60-minute technical workshop on building AI agents that keep working until the job is
**verified** done: slides, a runnable loop, and a Colab notebook.

| Artifact | What it is |
|---|---|
| `Loop_Engineering_Workshop.pptx` | 68 slides, light theme, diagram-led, full speaker notes |
| `explainer.md` | The long-form written version — read this before you present |
| `src/loop_engine/` | The loop from first principles, no framework, ~400 lines |
| `notebooks/loop_engineering_crewai_demo.ipynb` | The same loop with CrewAI, Colab-ready |
| `deck/make_diagrams.py` | Every diagram in the deck, generated from Graphviz |

---

## Setup

Requires Python 3.13 and Graphviz (`brew install graphviz`, only needed to rebuild diagrams).

```bash
uv venv --python 3.13 .venv
uv pip install -e .
```

`.env` in the project root:

```
OPENAI_API_KEY=sk-...
FINHUB_API_KEY=...
```

Get keys at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) and
[finnhub.io/register](https://finnhub.io/register) (free tier is enough).

---

## Run the demo

```bash
# the baseline: one prompt, no tools, no verification — watch it invent numbers
.venv/bin/python -m loop_engine.cli --ticker AAPL --naive

# the loop: evidence → triage → implement → verify → gate → state
.venv/bin/python -m loop_engine.cli --ticker AAPL

# the whole watch list
.venv/bin/python -m loop_engine.cli --watchlist

# what the loop remembers
.venv/bin/python -m loop_engine.cli --state

# raw tool output, no LLM calls
.venv/bin/python -m loop_engine.cli --evidence --ticker NVDA
```

A typical run:

```
▸ fetching evidence for MSFT
▸ triage (gpt-4o-mini)
▸ attempt 1/3: implementer (gpt-4o-mini)
▸ attempt 1/3: verifier (o4-mini)
⚖ REJECT · score 25/100 · 2 violation(s)
  · fabricated: '63%' is not grounded in the evidence JSON
  · overconfident: confidence 0.9 exceeds 0.8 when material errors are present
▸ attempt 2/3: implementer (gpt-4o-mini)
▸ attempt 2/3: verifier (o4-mini)
⚖ REJECT · score 30/100 · 1 violation(s)
  · labeled the intraday high $389.03 as the 52-week high; evidence shows $555.45
▸ attempt 3/3: implementer (gpt-4o-mini)
▸ attempt 3/3: verifier (o4-mini)
⚖ APPROVE · score 100/100
✓ published → MSFT_2026-07-25.md
```

~30 seconds, ~11k tokens, about one cent.

---

## Read the code in this order

| File | The idea |
|---|---|
| `config.py` | Every knob a human decides up front: models, budget, stop conditions, gate policy |
| `tools.py` | Finnhub calls → the **evidence table**. A number not in here is a hallucination |
| `agents.py` | Triage / implementer / **adversarial verifier** prompts. Separate calls, fresh contexts |
| `loop.py` | **The loop.** Every line separating it from `while True` is marked `# LOOP:` |
| `state.py` | `STATE.md` — memory that survives the process |
| `cli.py` | The workshop harness |

Start with `loop.py`. It reads top to bottom in one sitting.

---

## The notebook

`notebooks/loop_engineering_crewai_demo.ipynb` — upload to Colab, add your two keys as Colab
secrets (`OPENAI_API_KEY`, `FINHUB_API_KEY`) or paste them when prompted.

It builds the same loop from CrewAI primitives and ends by showing CrewAI's own
`Task(guardrail=..., max_retries=3)` — the framework's built-in verification loop — plus what it
still doesn't give you (cadence, durable state, gates, budgets, run log).

The notebook is generated from `notebooks/_nb_source.py`, which is a plain Python file so it can be
executed end to end before a workshop:

```bash
python notebooks/build_notebook.py
```

---

## Rebuild the deck

```bash
uv pip install -e ".[deck]"
python deck/make_diagrams.py   # 31 PNGs → deck/diagrams/
python deck/make_deck.py       # → Loop_Engineering_Workshop.pptx
```

Diagrams are Graphviz DOT + two matplotlib charts. Edit `DIAGRAMS` in `make_diagrams.py`,
edit slide content at the bottom of `make_deck.py`. Text auto-shrinks to fit its box, so
editing a slide won't silently push text off the edge.

---

## Loop config

Everything is in `src/loop_engine/config.py`:

```python
max_attempts        = 3          # bounded retries, then escalate
max_tokens_per_run  = 120_000    # hard budget cap
pause_at_budget_pct = 80         # soft brake
min_verifier_score  = 80         # gate threshold
denylist_tickers    = ("GME", "AMC", "DJT")
autonomy_tier       = "L2"       # L1 report | L2 assisted | L3 unattended
```

Override models by env var: `TRIAGE_MODEL`, `IMPLEMENTER_MODEL`, `VERIFIER_MODEL`.

**Try this:** set `VERIFIER_MODEL` to the same model as the implementer, run it five times, and
count how many fabrications get through. It's the most convincing experiment in the kit.

---

## Generated at runtime

`STATE.md` · `STATE.json` · `loop-run-log.jsonl` · `artifacts/` — the loop's memory and its receipts.
Safe to delete; they'll be recreated on the next run.
