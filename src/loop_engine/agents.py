"""The three roles inside the loop: Triage, Implementer, Verifier.

Each one is a *separate call with a fresh context*. That is the whole trick of
maker-checker: the verifier must not inherit the implementer's reasoning, its
optimism, or its blind spots. Shared context means shared mistakes.
"""

from __future__ import annotations

import json
from typing import Any

from .llm import Budget, chat, chat_json

# --------------------------------------------------------------------------
# 1. TRIAGE  -- cheap model, decides whether the expensive path runs at all.
#    The single biggest cost lever in loop engineering: no items -> exit now.
# --------------------------------------------------------------------------

TRIAGE_SYSTEM = """You are the triage step of an automated equity-research loop.
You are cheap and fast. You do NOT write the memo. You decide whether writing one
is worth it, and what the memo must focus on.

Return ONLY JSON:
{
  "worth_running": true|false,
  "reason": "<one sentence>",
  "focus": ["<2-4 specific angles the memo must cover, grounded in the evidence>"],
  "data_gaps": ["<fields that are missing or erroring, or []>"]
}
Set worth_running=false only if the evidence is too broken to write anything honest."""


def triage(model: str, evidence: dict, state_summary: str, budget: Budget) -> dict:
    return chat_json(
        model,
        [
            {"role": "system", "content": TRIAGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Prior loop state:\n{state_summary}\n\n"
                    f"Evidence fetched this run:\n{json.dumps(evidence, indent=2)[:6000]}"
                ),
            },
        ],
        budget=budget,
        max_tokens=600,
    )


# --------------------------------------------------------------------------
# 2. IMPLEMENTER -- writes the artifact. Sees the verifier's last rejection.
# --------------------------------------------------------------------------

IMPLEMENTER_SYSTEM = """You are the implementer sub-agent of an equity-research loop.

Write a one-page research memo in Markdown with EXACTLY these sections:
## Snapshot        (price, market cap, industry -- numbers only from the evidence)
## Thesis          (3 bullets, each citing a number or headline from the evidence)
## Risks           (3 bullets, at least one drawn from valuation or leverage data)
## Recommendation  (one of: STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
## Confidence      (a single decimal 0.0-1.0 on its own line)
## Evidence Used   (bullet list: field name -> value, for every number you cited)

HARD RULES -- the verifier will reject on any violation:
- Every numeric claim must appear in the evidence JSON. Invent nothing.
- Never state a price target unless you derive it from a multiple present in the evidence, and show that arithmetic.
- If a field is missing, say "not available" instead of estimating.
- No investment advice framing beyond the recommendation label. End with the one-line disclaimer:
  "_Automated research artifact. Not investment advice._"

Output the memo only. No preamble."""


def implement(
    model: str,
    ticker: str,
    evidence: dict,
    focus: list[str],
    budget: Budget,
    rejection: dict[str, Any] | None = None,
    attempt: int = 1,
) -> str:
    user = (
        f"Ticker: {ticker}\n"
        f"Focus areas from triage: {focus}\n\n"
        f"EVIDENCE (the only facts you may use):\n{json.dumps(evidence, indent=2)[:9000]}"
    )
    if rejection:
        # Feeding the rejection back in is what makes this a loop rather than a retry.
        user += (
            f"\n\nYour previous attempt (#{attempt - 1}) was REJECTED by the verifier.\n"
            f"Score: {rejection.get('score')}/100\n"
            f"Violations you must fix:\n- " + "\n- ".join(rejection.get("violations", [])) +
            f"\n\nRequired fixes:\n- " + "\n- ".join(rejection.get("required_fixes", [])) +
            "\n\nRewrite the memo in full, fixing every violation. Do not repeat the same mistakes."
        )
    return chat(
        model,
        [{"role": "system", "content": IMPLEMENTER_SYSTEM}, {"role": "user", "content": user}],
        budget=budget,
        max_tokens=1800,
        temperature=0.3,
    )


# --------------------------------------------------------------------------
# 3. VERIFIER -- different model family, fresh context, adversarial by default.
#    Default stance is REJECT: it looks for reasons to fail, not to pass.
# --------------------------------------------------------------------------

VERIFIER_SYSTEM = """You are an ADVERSARIAL verifier in an automated research loop.
Your default stance is REJECT. Approve only what you can verify line by line.

You receive: (a) the evidence JSON that was actually fetched, (b) a memo written
by a different model. The memo's author does not get to grade itself -- you do.

THE ONE RULE THAT OVERRIDES YOUR INSTINCTS:
The evidence JSON is the sole source of truth. Judge every number ONLY against it.
Do NOT use your own world knowledge of what a value "should" be. If the evidence
says the market cap is 4,891,183 million, then that is CORRECT here even if you
believe the real figure differs. Flagging an evidence-backed number because it
"seems too high/low" is the single most common verifier error -- and a failure.

Check, in order:
1. GROUNDING: every number in the memo must trace to a value in the evidence JSON
   (the "Evidence Used" section maps them -- check there first). A number is
   grounded if it equals an evidence value after rounding, dropping/adding
   thousands separators, or restating units (4891183.499 -> "$4,891,183.50
   million" -> "$4.89 trillion" are the same grounded value). Only a number with
   NO corresponding evidence value is "fabricated: <value>". Fatal.
2. STRUCTURE: all six required sections present (Snapshot, Thesis, Risks,
   Recommendation, Confidence, Evidence Used).
3. COMPLETENESS: >=3 thesis bullets, >=3 risk bullets, a recommendation from the
   allowed set, a confidence in [0,1], and the disclaimer line.
4. HONESTY: missing data is reported as "not available", not estimated. A price
   target must show the arithmetic deriving it from an evidence value, or it is
   fabricated.
5. CALIBRATION: confidence is not >0.8 when key data is missing or erroring.

NOT violations -- never report these:
- A number that appears in the evidence, in ANY rounding or unit form (this
  includes large market caps, prices, and ratios that may look implausible to
  you -- trust the evidence, not your gut).
- Restating an evidence headline, when attributed as a headline.
- Wording, tone, or ordering preferences.
Report at most 5 violations, most severe first, and only material ones. A verifier
that rejects grounded work is exactly as useless as one that approves fabrications.

Return ONLY JSON:
{
  "verdict": "APPROVE" | "REJECT",
  "score": <0-100>,
  "violations": ["<specific, quotable>"],
  "required_fixes": ["<imperative instruction to the implementer>"],
  "recommendation": "<the memo's recommendation label, verbatim>",
  "confidence": <the memo's confidence number>
}
Score >= 80 AND zero fatal grounding violations is the only path to APPROVE."""


def verify(model: str, evidence: dict, memo: str, budget: Budget) -> dict:
    result = chat_json(
        model,
        [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"EVIDENCE JSON:\n{json.dumps(evidence, indent=2)[:9000]}\n\n"
                    f"MEMO UNDER REVIEW:\n{memo}"
                ),
            },
        ],
        budget=budget,
        max_tokens=1200,
        temperature=0.0,
    )
    # Never trust a verifier's own summary of itself: re-derive the verdict.
    score = int(result.get("score", 0))
    violations = result.get("violations") or []
    fatal = [v for v in violations if "fabricat" in str(v).lower()]
    result["verdict"] = "APPROVE" if (score >= 80 and not fatal) else "REJECT"
    result["score"] = score
    result["violations"] = violations
    result.setdefault("required_fixes", [])
    return result
