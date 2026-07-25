# %% [markdown]
# # Loop Engineering with CrewAI — an agent that keeps working until the job is *verified* done
#
# **Workshop notebook.** By the end of this you will have built, and watched fail, a multi-agent
# loop that researches a stock and refuses to publish a number it cannot prove.
#
# We build it in the order the ideas arrived:
#
# | Part | What we add | The idea it teaches |
# |---|---|---|
# | 0 | A single prompt | Why one-shot answers are confidently wrong |
# | 1 | Finnhub tools | Ground truth — the evidence table |
# | 2 | CrewAI agents | Roles: triage, implementer, verifier |
# | 3 | One crew pass | Still not a loop |
# | 4 | An adversarial verifier | Maker–checker: the model that wrote it doesn't grade it |
# | 5 | **The outer loop** | Bounded retries that carry the reason they failed |
# | 6 | STATE.md on disk | Memory that survives the process |
# | 7 | A human gate | Deterministic risk policy, not a model call |
# | 8 | `guardrail=` + `max_retries=` | The same loop, as a CrewAI primitive |
#
# **You need:** an [OpenAI](https://platform.openai.com/api-keys) key and a free
# [Finnhub](https://finnhub.io/register) key.

# %% [markdown]
# ## Setup
#
# Pinned versions — a workshop is a bad place to discover an API change.

# %%
# !pip install -q crewai==1.15.5 requests

# %% [markdown]
# ### Keys — from Colab Secrets
#
# Open the **🔑 Secrets** panel in the Colab sidebar and add two secrets, with notebook
# access toggled **on**:
#
# | Name | Value |
# |---|---|
# | `OPENAI_API_KEY` | your key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
# | `FINHUB_API_KEY` | your free key from [finnhub.io/register](https://finnhub.io/register) |
#
# We read them with `userdata.get(...)` — nothing is ever printed or hard-coded in a cell.

# %%
import os
import getpass

try:
    from google.colab import userdata          # present in Colab
except ImportError:
    userdata = None                            # running elsewhere → use env vars


def _load_key(name: str, label: str) -> str:
    """Colab Secret via userdata.get → env var → last-resort prompt."""
    if userdata is not None:
        try:
            value = userdata.get(name)         # the Colab way: read from Secrets
            if value:
                return value
        except Exception:
            pass                               # secret missing → fall through
    if os.getenv(name):
        return os.environ[name]
    return getpass.getpass(f"{label}: ")


OPENAI_API_KEY = _load_key("OPENAI_API_KEY", "OpenAI API key")
FINNHUB_API_KEY = _load_key("FINHUB_API_KEY", "Finnhub API key")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Keep CrewAI non-interactive: the tracing prompt will block a notebook cell.
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

print("keys loaded ✓")

# %% [markdown]
# ### The three models
#
# One deliberate choice here, and it is the most important idea in the notebook:
# **the verifier is not the model that wrote the memo.** It runs as a separate call, with a
# fresh context and an adversarial brief — it never grades its own homework.
#
# * **Implementer** — `gpt-4o-mini`. Cheap and fast. It does the writing.
# * **Verifier** — `o4-mini`, a **reasoning model**. You want your checker to *think*: methodical,
#   line-by-line grounding checks are exactly what reasoning models are good at. We tested five
#   OpenAI models on the same memos, and this was the only one that both approved fully-grounded
#   work *and* caught invented numbers without inventing objections of its own.
#
# In production the strongest form of this rule is a *different model family* entirely (e.g. Claude
# checking GPT), so the reviewer shares none of the writer's blind spots. A stronger, independent
# reasoning model from the same vendor is the next best thing — and it's what we use here.

# %%
from crewai import LLM

# Reasoning models (o-series) reject both `temperature` and `max_tokens` (they meter output as
# `max_completion_tokens`). Rather than special-case the param name, we simply omit the cap for
# them and let the model default — the verifier returns a small JSON object either way.
_REASONING = ("o1", "o3", "o4")


def openai_llm(model: str, temperature: float = 0.2, max_tokens: int = 2000) -> LLM:
    kwargs = dict(model=f"openai/{model}", api_key=OPENAI_API_KEY)
    if not model.startswith(_REASONING):
        kwargs["temperature"] = temperature    # o-series: fixed at 1, don't send
        kwargs["max_tokens"] = max_tokens       # o-series: uses max_completion_tokens, omit
    return LLM(**kwargs)


TRIAGE_LLM = openai_llm("gpt-4o-mini")             # cheap: is this worth doing?
IMPLEMENTER_LLM = openai_llm("gpt-4o-mini", 0.3)   # cheap: does the work
VERIFIER_LLM = openai_llm("o4-mini")               # reasoning model: judges it, independently

print("models wired ✓")

# %% [markdown]
# ### One Colab-specific helper: run the crew off the event loop
#
# Colab's kernel already has a **running asyncio event loop**. Recent CrewAI refuses a
# synchronous `crew.kickoff()` from inside one and raises:
#
# > *RuntimeError: Agent execution was invoked synchronously from within a running event loop.*
#
# `nest_asyncio` does **not** fix it — CrewAI checks for a running loop explicitly. The clean fix
# is to run `kickoff()` on a worker thread, which has no loop of its own. We wrap it once here and
# route every `kickoff()` in this notebook through `run_crew(...)`, so the code below stays plain
# and synchronous.

# %%
import concurrent.futures


def run_crew(crew, inputs: dict | None = None):
    """kickoff() a crew from a worker thread, so Colab's running event loop doesn't reject it."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: crew.kickoff(inputs=inputs or {})).result()


print("run_crew ready ✓")

# %% [markdown]
# ---
# ## Part 0 — The baseline: one prompt, no tools, no verification
#
# Before we build anything, let's see the failure mode we're solving. Same model,
# same question — just no system around it.

# %%
import requests

TICKER = "AAPL"


def raw_completion(model: str, prompt: str, max_tokens: int = 700) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


naive = raw_completion(
    "gpt-4o-mini",
    f"Write a short equity research note on {TICKER}. Include the current price, "
    f"the P/E ratio, and a recommendation with a confidence score.",
)
print(naive)

# %% [markdown]
# 👉 **Look at the numbers it just gave you.** Every one of them came out of the model's
# weights, not a data source. Some may even be close — that's what makes it dangerous.
#
# Nothing checked them. Nothing *could* have. And it reads exactly as confident as a note
# that happens to be right.

# %% [markdown]
# ---
# ## Part 1 — Tools are ground truth
#
# The fix starts here. Every number the agent is allowed to write must come from a tool call,
# and we keep the tool output around as an **evidence table** so the verifier can check
# the memo against it line by line.

# %%
import datetime as dt
import json

from crewai.tools import tool

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _finnhub(path: str, **params):
    params["token"] = FINNHUB_API_KEY
    resp = requests.get(f"{FINNHUB_BASE}/{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


@tool("Get stock quote")
def get_quote(ticker: str) -> str:
    """Latest price, day range and previous close for a ticker. Returns JSON."""
    q = _finnhub("quote", symbol=ticker)
    return json.dumps({"current": q.get("c"), "change": q.get("d"),
                       "percent_change": q.get("dp"), "high": q.get("h"),
                       "low": q.get("l"), "prev_close": q.get("pc")})


@tool("Get company profile")
def get_profile(ticker: str) -> str:
    """Company name, exchange, industry and market cap. Returns JSON."""
    p = _finnhub("stock/profile2", symbol=ticker)
    return json.dumps({"name": p.get("name"), "industry": p.get("finnhubIndustry"),
                       "market_cap_musd": p.get("marketCapitalization"),
                       "exchange": p.get("exchange")})


_WANTED = {
    "peTTM": "pe_ttm", "psTTM": "ps_ttm", "pbAnnual": "pb_annual", "roeTTM": "roe_ttm",
    "grossMarginTTM": "gross_margin_ttm", "netProfitMarginTTM": "net_margin_ttm",
    "totalDebt/totalEquityQuarterly": "debt_to_equity", "52WeekHigh": "week52_high",
    "52WeekLow": "week52_low", "beta": "beta", "revenueGrowthTTMYoy": "revenue_growth_yoy",
    "epsGrowthTTMYoy": "eps_growth_yoy",
}


@tool("Get financial metrics")
def get_metrics(ticker: str) -> str:
    """Valuation, margin, leverage and growth metrics. Returns JSON."""
    raw = (_finnhub("stock/metric", symbol=ticker, metric="all") or {}).get("metric", {})
    return json.dumps({v: raw.get(k) for k, v in _WANTED.items() if raw.get(k) is not None})


@tool("Get analyst consensus")
def get_consensus(ticker: str) -> str:
    """Most recent analyst recommendation counts. Returns JSON."""
    rows = _finnhub("stock/recommendation", symbol=ticker) or []
    if not rows:
        return json.dumps({})
    r = rows[0]
    return json.dumps({"period": r.get("period"), "strong_buy": r.get("strongBuy"),
                       "buy": r.get("buy"), "hold": r.get("hold"),
                       "sell": r.get("sell"), "strong_sell": r.get("strongSell")})


@tool("Get recent company news")
def get_news(ticker: str) -> str:
    """Headlines from the last 14 days. Returns a JSON list."""
    today = dt.date.today()
    rows = _finnhub("company-news", symbol=ticker,
                    **{"from": str(today - dt.timedelta(days=14)), "to": str(today)}) or []
    return json.dumps([{"headline": r.get("headline"), "source": r.get("source")}
                       for r in rows[:5]])


TOOLS = [get_quote, get_profile, get_metrics, get_consensus, get_news]
print(f"{len(TOOLS)} tools registered ✓")

# %% [markdown]
# The agents will call those tools themselves. But the *verifier* needs the raw truth
# independently — so we also fetch it directly. This dict is the contract:
# **a number that is not in here is, by definition, a hallucination.**

# %%
def collect_evidence(ticker: str) -> dict:
    evidence = {"ticker": ticker.upper(),
                "fetched_at": dt.datetime.now().isoformat(timespec="seconds")}
    for key, fn in (("quote", get_quote), ("profile", get_profile),
                    ("metrics", get_metrics), ("analyst_consensus", get_consensus),
                    ("news", get_news)):
        try:
            # .run() calls the underlying function directly, bypassing the agent
            evidence[key] = json.loads(fn.run(ticker=ticker))
        except Exception as exc:
            # degrade loudly: the verifier must distinguish "no data" from "never fetched"
            evidence[key] = {"error": str(exc)[:160]}
    return evidence


EVIDENCE = collect_evidence(TICKER)
print(json.dumps(EVIDENCE, indent=2)[:1200])

# %% [markdown]
# ---
# ## Part 2 — Three agents, three jobs
#
# * **Triage** — cheap. Decides whether the expensive path runs at all. The single biggest
#   cost lever in loop engineering is the branch that says *nothing to do, stop now*.
# * **Implementer** — writes the memo, using the tools.
# * **Verifier** — adversarial. Fresh context, different model, default stance **reject**.

# %%
from crewai import Agent, Crew, Process, Task

triage_analyst = Agent(
    role="Triage Analyst",
    goal="Decide whether writing a full research memo on {ticker} is worth the tokens.",
    backstory=(
        "You are the cheap first pass in an automated research loop. You never write the "
        "memo yourself. You decide whether it is worth writing, and what it must focus on."
    ),
    llm=TRIAGE_LLM,
    verbose=False,
    allow_delegation=False,
)

research_analyst = Agent(
    role="Equity Research Analyst",
    goal="Write a one-page memo on {ticker} in which every number is traceable to a tool result.",
    backstory=(
        "You are a disciplined analyst. You would rather write 'not available' than estimate. "
        "You know an adversarial reviewer will check every figure you write against the raw data."
    ),
    llm=IMPLEMENTER_LLM,
    tools=TOOLS,
    verbose=False,
    allow_delegation=False,
    max_iter=8,
)

verifier = Agent(
    role="Adversarial Verifier",
    goal="Find every reason to reject the memo before it reaches a human.",
    backstory=(
        "You are a compliance reviewer with a default stance of REJECT. You did not write "
        "this memo and you owe its author nothing. A fabricated number is a fatal violation. "
        "You approve only what you can verify line by line against the evidence."
    ),
    llm=VERIFIER_LLM,
    verbose=False,
    allow_delegation=False,
)

print("agents ready ✓")

# %% [markdown]
# ---
# ## Part 3 — One pass through the crew
#
# This is where most tutorials stop: agents, tasks, `kickoff()`, done. Watch what we get.

# %%
MEMO_SPEC = """
Write a one-page equity research memo on {ticker} in Markdown with EXACTLY these sections:
## Snapshot        (price, market cap, industry)
## Thesis          (3 bullets, each citing a number or headline you fetched)
## Risks           (3 bullets, at least one from valuation or leverage data)
## Recommendation  (one of: STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
## Confidence      (a single decimal 0.0-1.0 on its own line)
## Evidence Used   (bullet list: field name -> value, for every number you cited)

HARD RULES:
- Use the tools. Every numeric claim must come from a tool result. Invent nothing.
- If a field is missing, write "not available". Never estimate.
- End with: _Automated research artifact. Not investment advice._
"""


def write_memo(ticker: str, feedback: str | None = None) -> str:
    description = MEMO_SPEC.format(ticker=ticker)
    if feedback:
        description += (
            "\n\nYour previous attempt was REJECTED by the verifier.\n"
            f"{feedback}\n\nRewrite the memo in full, fixing every violation."
        )
    task = Task(
        description=description,
        expected_output="A six-section markdown memo, every number traceable to a tool result.",
        agent=research_analyst,
    )
    crew = Crew(agents=[research_analyst], tasks=[task],
                process=Process.sequential, verbose=False)
    return str(run_crew(crew, {"ticker": ticker}))


memo_v1 = write_memo(TICKER)
print(memo_v1)

# %% [markdown]
# It looks professional. That is precisely the problem — **you cannot tell by reading it**
# whether those numbers are real. Neither can your users.
#
# So we stop reading, and we make a machine check it.

# %% [markdown]
# ---
# ## Part 4 — The verifier: maker ≠ checker
#
# Three details do all the work here:
#
# 1. **Default stance is REJECT.** The framing measurably changes behaviour.
# 2. **Structured JSON out.** "Looks good to me" is not a gate condition. `score >= 80` is.
# 3. **The code re-derives the verdict.** A model will happily say `APPROVE` while listing a
#    fabrication in the same breath. Trust its observations, not its conclusion.

# %%
import re

VERIFIER_SPEC = """
You are reviewing a research memo written by a DIFFERENT model. It does not get to grade itself.

EVIDENCE JSON (the only facts that exist):
{evidence}

MEMO UNDER REVIEW:
{memo}

THE ONE RULE THAT OVERRIDES YOUR INSTINCTS:
The EVIDENCE JSON is the sole source of truth. Judge every number ONLY against it. Do NOT use
your own world knowledge of what a value "should" be. If the evidence says the market cap is
4,891,183 million, then 4,891,183 million is CORRECT here, even if you believe the real figure is
different. A number that matches the evidence is GROUNDED, full stop -- flagging it because it
"seems too high/low" is the single most common verifier error, and it is itself a failure.

Check in this order:
1. GROUNDING: every number in the memo must trace to a value in the evidence JSON (the
   "Evidence Used" section maps them for you -- check there first). A number is grounded if it
   equals an evidence value after rounding, dropping/adding thousands separators, or restating
   units (4891183.499 -> "$4,891,183.50 million" -> "$4.89 trillion" are ALL the same grounded
   value). Only a number with NO corresponding evidence value is "fabricated: <value>". Fatal.
2. STRUCTURE: all six sections present.
3. COMPLETENESS: >=3 thesis bullets, >=3 risks, a valid recommendation, a confidence in [0,1],
   and the disclaimer line.
4. HONESTY: missing data reported as "not available", never estimated. A price target must show
   the arithmetic that derives it from an evidence value, or it is fabricated.
5. CALIBRATION: confidence must not exceed 0.8 when key data is missing or erroring.

NOT violations -- never report these:
- A number that appears in the evidence, in ANY rounding or unit form (this includes large market
  caps, prices, and ratios that may look implausible to you -- trust the evidence, not your gut).
- Restating a headline that IS in the evidence, when attributed as a headline.
- Wording, tone, or ordering preferences.
Report at most 5 violations, most severe first, and only material ones. A verifier that rejects
grounded work is exactly as useless as one that approves fabrications.

Return ONLY a JSON object, no prose, no code fence:
{{"verdict": "APPROVE" or "REJECT", "score": <0-100>,
  "violations": ["<specific and quotable>"],
  "required_fixes": ["<imperative instruction to the author>"],
  "recommendation": "<the memo's recommendation label>",
  "confidence": <the memo's confidence number>}}
"""


def _parse_json(text: str) -> dict:
    for pattern in (r"```(?:json)?\s*(\{.*?\})\s*```", r"\{.*\}"):
        match = re.search(pattern, text, re.S)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except json.JSONDecodeError:
                continue
    return {"verdict": "REJECT", "score": 0,
            "violations": ["verifier did not return valid JSON"], "required_fixes": []}


def verify_memo(evidence: dict, memo: str) -> dict:
    task = Task(
        description=VERIFIER_SPEC.format(evidence=json.dumps(evidence, indent=2)[:9000],
                                         memo=memo),
        expected_output="A single JSON object with verdict, score, violations, required_fixes.",
        agent=verifier,
    )
    crew = Crew(agents=[verifier], tasks=[task], process=Process.sequential, verbose=False)
    result = _parse_json(str(run_crew(crew)))

    # Never trust the verifier's own label — re-derive it from what it found.
    score = int(result.get("score", 0) or 0)
    violations = result.get("violations") or []
    fatal = [v for v in violations if "fabricat" in str(v).lower()]
    result["verdict"] = "APPROVE" if (score >= 80 and not fatal) else "REJECT"
    result["score"] = score
    result["violations"] = violations
    result.setdefault("required_fixes", [])
    return result


review_v1 = verify_memo(EVIDENCE, memo_v1)
print(f"verdict: {review_v1['verdict']}   score: {review_v1['score']}/100\n")
for v in review_v1["violations"]:
    print(f"  ✗ {v}")

# %% [markdown]
# ---
# ## Part 5 — The loop
#
# Everything so far has been components. This cell is the actual subject of the workshop.
#
# Look for four things — they are what separate a loop from `while True`:
#
# * `for attempt in range(1, max_attempts + 1)` — **bounded** retries
# * `feedback=...` — the retry carries **the reason it failed**
# * the `else:` on the for loop — the **attempt cap**, i.e. escalation
# * the budget check — a **brake** before the wall

# %%
from dataclasses import dataclass, field


@dataclass
class LoopResult:
    ticker: str
    outcome: str
    memo: str | None = None
    attempts: list[dict] = field(default_factory=list)
    gate_reason: str | None = None


def research_loop(ticker: str, max_attempts: int = 3, verbose: bool = True) -> LoopResult:
    result = LoopResult(ticker=ticker, outcome="aborted")

    # --- tools: ground truth before opinions ---------------------------------
    evidence = collect_evidence(ticker)
    if verbose:
        print(f"▸ evidence fetched for {ticker}")

    # --- triage: the cheapest possible exit -----------------------------------
    triage_task = Task(
        description=(
            "Given this evidence, decide whether a full memo is worth writing.\n"
            f"{json.dumps(evidence, indent=2)[:4000]}\n\n"
            'Return ONLY JSON: {"worth_running": true/false, "reason": "...", '
            '"focus": ["2-4 angles the memo must cover"]}'
        ),
        expected_output="A single JSON object.",
        agent=triage_analyst,
    )
    triage_crew = Crew(agents=[triage_analyst], tasks=[triage_task],
                       process=Process.sequential, verbose=False)
    decision = _parse_json(str(run_crew(triage_crew)))
    if not decision.get("worth_running", True):
        if verbose:
            print(f"↷ triage says stop: {decision.get('reason')}")
        result.outcome = "skipped"
        return result
    if verbose:
        print(f"▸ triage: proceed — focus {decision.get('focus')}")

    # --- the maker-checker loop ----------------------------------------------
    feedback, memo = None, ""
    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"▸ attempt {attempt}/{max_attempts}: implementer writing…")
        memo = write_memo(ticker, feedback=feedback)

        if verbose:
            print(f"▸ attempt {attempt}/{max_attempts}: verifier reviewing…")
        review = verify_memo(evidence, memo)
        result.attempts.append({"attempt": attempt, "verdict": review["verdict"],
                                "score": review["score"],
                                "violations": review["violations"]})
        if verbose:
            print(f"⚖ {review['verdict']} · {review['score']}/100")
            for v in review["violations"][:4]:
                print(f"    ✗ {v}")

        if review["verdict"] == "APPROVE":
            break

        # the retry carries the reason — otherwise it is just a re-roll
        feedback = ("Score: {}/100\nViolations:\n- {}\n\nRequired fixes:\n- {}".format(
            review["score"], "\n- ".join(review["violations"]),
            "\n- ".join(review["required_fixes"])))
    else:
        if verbose:
            print(f"↑ {max_attempts} attempts exhausted — escalating to a human")
        result.outcome = "escalated"
        result.memo = memo
        result.gate_reason = "attempt cap reached"
        return result

    result.memo = memo
    result.outcome = "verified"
    return result


run = research_loop(TICKER)
print(f"\noutcome: {run.outcome} after {len(run.attempts)} attempt(s)")

# %% [markdown]
# 👆 **This is the whole point of the workshop.**
#
# One machine wrote a memo; a *different* machine checked every number against the evidence and
# either signed it off or sent it back with a specific, checkable reason — and no human read a
# single line to make that happen.
#
# How many attempts you see is not fixed, and that is the honest part:
#
# * If this run **approved on the first attempt**, that is the system working — the implementer
#   stayed grounded on a heavily-covered name like AAPL and the verifier confirmed it. A clean
#   pass is a *success*, not a boring demo.
# * You are about to see the other case. In **Part 7** we run the same loop on a different ticker,
#   and watch the verifier reject it **twice** — once for an invented "63%", once for labelling
#   the intraday high as the 52-week high — before the third attempt passes. Same loop, same code,
#   no human in the middle.
#
# The value was never "the model fails." It is that **failure is caught and corrected without you.**

# %%
print(run.memo)

# %% [markdown]
# ---
# ## Part 6 — Memory on disk
#
# > *"The memory must be on disk, not in the context. The agent forgets, but the repository doesn't."*
# > — Addy Osmani
#
# The context window dies when this cell finishes. `STATE.md` is how tomorrow's run
# knows what today's run did.

# %%
from pathlib import Path

STATE_PATH = Path("STATE.md")
RUN_LOG = Path("loop-run-log.jsonl")

STATE_TEMPLATE = """# Loop State — equity-research-loop

Last run: {last_run}
Runs completed: {runs}

## Done
{done}

## Human Inbox (needs a decision from you)
{inbox}

## Watch List
{watch}
"""


def load_state() -> dict:
    sidecar = STATE_PATH.with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return {"runs": 0, "done": [], "inbox": [], "watch": ["AAPL", "MSFT", "NVDA"]}


def save_state(state: dict) -> None:
    def bullets(items):
        return "\n".join(f"- {i}" for i in items[-8:]) if items else "_(empty)_"

    STATE_PATH.write_text(STATE_TEMPLATE.format(
        last_run=dt.datetime.now().isoformat(timespec="seconds"),
        runs=state["runs"], done=bullets(state["done"]),
        inbox=bullets(state["inbox"]), watch=bullets(state["watch"])))
    STATE_PATH.with_suffix(".json").write_text(json.dumps(state, indent=2))


def log_run(record: dict) -> None:
    with RUN_LOG.open("a") as fh:
        fh.write(json.dumps({"logged_at": dt.datetime.now().isoformat(timespec="seconds"),
                             **record}) + "\n")


print("state helpers ready ✓")

# %% [markdown]
# ---
# ## Part 7 — The human gate
#
# Verifier approval is **necessary, not sufficient**. The gate is the last thing between the
# loop and the outside world, and it is deliberately *deterministic code* — not another model
# call. A gate you cannot audit is a rumour.
#
# One war story from building this: the first version of `_gate` ran
# `if "STRONG BUY" in memo.upper()`. The thesis section contained the phrase
# *"13 strong buy ratings"*, so **every** run escalated. Nobody disables a gate on purpose —
# they disable it because it cries wolf. Parse structure, don't grep text.

# %%
DENYLIST = {"GME", "AMC", "DJT"}
LOW_CONFIDENCE = 0.5


def _section(memo: str, name: str) -> str | None:
    match = re.search(rf"^#{{1,4}}\s*{name}\s*$(.*?)(?=^#{{1,4}}\s|\Z)", memo, re.I | re.M | re.S)
    if not match:
        return None
    for line in match.group(1).splitlines():
        cleaned = line.strip().strip("*_`-• ").strip()
        if cleaned:
            return cleaned
    return None


def human_gate(memo: str) -> str | None:
    """Returns a reason to escalate, or None to publish."""
    recommendation = _section(memo, "Recommendation")
    if recommendation and recommendation.upper().startswith(("STRONG BUY", "STRONG SELL")):
        return f"high-conviction call ({recommendation}) requires human sign-off"

    raw = _section(memo, "Confidence") or ""
    match = re.search(r"([01]?\.\d+|[01])\b", raw)
    if match and float(match.group(1)) < LOW_CONFIDENCE:
        return f"low confidence ({match.group(1)}) — a human should decide"
    return None


def run_and_record(ticker: str) -> LoopResult:
    state = load_state()
    if ticker.upper() in DENYLIST:
        print(f"🚦 {ticker} is on the denylist — not running")
        return LoopResult(ticker=ticker, outcome="skipped", gate_reason="denylist")

    state["runs"] += 1
    result = research_loop(ticker)

    if result.outcome == "verified":
        reason = human_gate(result.memo or "")
        if reason:
            print(f"🚦 human gate: {reason}")
            state["inbox"].append(f"{ticker}: verified but gated — {reason}")
            result.outcome, result.gate_reason = "escalated", reason
        else:
            Path(f"{ticker}_memo.md").write_text(result.memo or "")
            print(f"✓ published → {ticker}_memo.md")
            state["done"].append(f"{ticker}: memo approved after "
                                 f"{len(result.attempts)} attempt(s)")
    elif result.outcome == "escalated":
        state["inbox"].append(f"{ticker}: {result.gate_reason}")

    save_state(state)
    log_run({"target": ticker, "outcome": result.outcome,
             "attempts": len(result.attempts),
             "final_score": result.attempts[-1]["score"] if result.attempts else 0,
             "gate_reason": result.gate_reason})
    return result


final = run_and_record("MSFT")
print("\n--- STATE.md ---")
print(STATE_PATH.read_text())

# %% [markdown]
# ---
# ## Part 8 — The same loop, as a CrewAI primitive
#
# CrewAI ships a verification loop for a single task: `guardrail=` plus `max_retries=`.
# The guardrail returns `(True, payload)` to pass, or `(False, feedback)` to send the agent
# back to work with your feedback attached.
#
# That is maker–checker with an attempt cap, in two arguments. Use it — it's good.
#
# **But notice what it does *not* give you**, which is most of this notebook:
# cadence, durable state, the human gate, the denylist, the budget cap, the run log.
# The framework gives you the inner loop. The system around it is still your job.

# %%
def grounding_guardrail(output) -> tuple[bool, str]:
    """Runs after the task. False → CrewAI retries, with this feedback attached."""
    memo = getattr(output, "raw", str(output))
    review = verify_memo(EVIDENCE, memo)
    print(f"   [guardrail] {review['verdict']} · {review['score']}/100")
    if review["verdict"] == "APPROVE":
        return (True, memo)
    return (False, "REJECTED by the verifier. Fix every one of these:\n- "
                   + "\n- ".join(review["violations"]))


guarded_task = Task(
    description=MEMO_SPEC.format(ticker=TICKER),
    expected_output="A six-section markdown memo, every number traceable to a tool result.",
    agent=research_analyst,
    guardrail=grounding_guardrail,   # verification, not vibes
    max_retries=3,                   # the attempt cap
)

guarded_crew = Crew(agents=[research_analyst], tasks=[guarded_task],
                    process=Process.sequential, verbose=False)

# When the guardrail keeps failing, CrewAI RAISES after max_retries -- that raise *is* the
# attempt cap firing, the same escalation branch we wrote by hand in Part 5. Catch it so the
# notebook shows the outcome instead of crashing.
try:
    guarded_result = run_crew(guarded_crew)
    print("✓ guardrail PASSED — the verifier approved within the retry budget\n")
    print(str(guarded_result)[:600])
except Exception as exc:
    print("↑ guardrail EXHAUSTED its retries — CrewAI raised, i.e. escalate to a human:\n")
    print(str(exc)[:400])

print("\ntokens used:", guarded_crew.usage_metrics)

# %% [markdown]
# ---
# ## Recap — what each piece bought us
#
# | Component | Without it |
# |---|---|
# | **Tools** | Numbers come from the model's weights |
# | **Evidence table** | "Is this real?" stays a judgement call, not a lookup |
# | **Separate verifier** | The author grades its own homework, leniently |
# | **Structured verdict** | You cannot branch on "looks good to me" |
# | **Feedback in the retry** | You re-roll the dice with the same prompt |
# | **Attempt cap** | An infinite loop with a credit card |
# | **State on disk** | Every run starts from zero, forever |
# | **Human gate** | The loop's worst call reaches production unescorted |
# | **Run log** | You cannot answer "is this loop earning its keep?" |
#
# ## Exercises
#
# 1. **Break the verifier.** Set `VERIFIER_LLM` to the *same* model as the implementer.
#    Run it 5 times. Count how many fabrications slip through. This is the single most
#    convincing experiment in the notebook.
# 2. **Make it stricter.** Require the memo to compute a fair-value estimate from `pe_ttm`
#    and `eps_growth_yoy`, and have the verifier check the arithmetic.
# 3. **Add a budget brake.** Track `crew.usage_metrics.total_tokens` across attempts and
#    stop starting new attempts past 80% of a cap you choose.
# 4. **Add a second loop** for a different watch list, with its own state file. Then implement
#    `acting_on` collision detection so the two never work the same ticker.
# 5. **Change the cadence.** Wrap `run_and_record` in a scheduler and run it every morning —
#    then read `loop-run-log.jsonl` after a week and decide whether it earned its tokens.
#
# ## Where to go next
#
# * `src/loop_engine/` in the workshop repo — the same loop without a framework, ~400 lines
# * Addy Osmani, *Loop Engineering* (June 2026)
# * `cobusgreyling/loop-engineering` — patterns, readiness scoring, checklists
# * Claude Code `/loop` and `/goal`; OpenAI Codex Automations
#
# **Build the loop. But build it like someone who intends to stay the engineer.**
