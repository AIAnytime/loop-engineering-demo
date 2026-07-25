# LOOP.md — equity-research-loop

The design document for the loop. Written **before** the loop ran, which is the point.
Copy this file as the template for your own first loop.

---

## Goal

Produce a one-page equity research memo whose every numeric claim is traceable to data fetched
during that run, with an explicit recommendation, a calibrated confidence, and a risk section.

**Not the goal:** giving investment advice, predicting prices, or replacing an analyst's judgement.

## Success criteria (what "done" means)

A run is successful when **all** of these hold:

- The adversarial verifier returns `APPROVE` with a score ≥ 80.
- Zero grounding violations — every number appears in the evidence fetched this run.
- All six sections present, ≥ 3 thesis bullets, ≥ 3 risk bullets, confidence in [0, 1].
- The human gate either cleared it, or escalated it to a named inbox.

"The implementer said it was done" is not a success criterion.

## Cadence

`1d` — once each weekday morning, per ticker on the watch list.

Never sub-minute. If you find yourself wanting a 30-second cadence, you want an event trigger.

## Autonomy tier

**L2 — Assisted.** The loop writes artifacts and updates state; it does not act on the outside
world without a gate. Promotion to L3 is not planned: the value here is at L2.

| Tier | Status |
|---|---|
| L1 Report | Passed — 2 weeks, false positive rate measured |
| L2 Assisted | **Current** |
| L3 Unattended | Not planned. No outside-world writes to automate. |

## Roles

| Role | Model | Context |
|---|---|---|
| Triage | `gpt-4o-mini` | Fresh. Cheap. Decides if the expensive path runs at all. |
| Implementer | `gpt-4o-mini` | Fresh. Sees evidence + the previous rejection. |
| Verifier | `o4-mini` (reasoning) | **Fresh, stronger, independent.** Reads evidence line by line. Default stance REJECT. |

The implementer may not approve its own work. Ever. This is not configurable.
(Ideal is a different *family* — e.g. Claude checking GPT — so the reviewer shares none of the
writer's blind spots. On an OpenAI-only key, a stronger reasoning model is the next best thing.)

## Stop conditions

```yaml
succeed:      verifier APPROVE (score >= 80, zero fatal violations) AND gate clear
escalate:     attempt cap reached (3) OR gate triggered
slow_down:    token budget > 80% of max_tokens_per_run  -> no new attempts started
kill:         two consecutive runs aborting on transport errors
              OR cost per approved memo exceeds value for 2 consecutive weeks
```

**Rule: no stop condition, no autonomy.** If you add a new capability to this loop, add its
stop condition in the same commit.

## Budgets

| Limit | Value |
|---|---|
| Tokens per run | 120,000 (hard cap) |
| Soft brake | 80% — stop starting new attempts |
| Attempts per item | 3, then escalate |
| Observed cost | ~11k tokens / ~$0.01 per completed memo |

## Denylist

Not researched under any circumstances, checked **before** any token is spent:

```
GME, AMC, DJT          # meme / restricted
```

Escalated to a human rather than published:

```
recommendation == STRONG BUY | STRONG SELL     # high conviction
confidence < 0.5                                # low confidence
```

The gate is deterministic code in `loop.py::_gate`, not a model call. It parses the memo's
`## Recommendation` section — it does **not** grep the document. (It did once. The thesis text
"13 strong buy ratings" made every run escalate.)

## State

| File | Purpose |
|---|---|
| `STATE.md` | Human-readable memory: Done / Human Inbox / Watch List / Recent Noise |
| `STATE.json` | Machine-readable sidecar, so the next run doesn't parse prose |
| `loop-run-log.jsonl` | One record per run: outcome, attempts, score, tokens, gate reason |
| `artifacts/` | Published memos |

Pruned every run (`state.prune()`): Done and Recent Noise keep the last 20 entries.
The Human Inbox is **never** auto-pruned — only a human clears it.

## Escalation

Items land in `## Human Inbox` in `STATE.md`. In a real deployment this also pings a channel.

**Named owner:** _(fill this in — an escalation nobody reads is an outage nobody noticed)_

If nobody responds, the loop skips that item on the next cycle and moves on. It does not
block, and it does not retry indefinitely.

## Multi-loop coordination

`state.acting_on` is set for the duration of a run. Any peer loop must skip a ticker already
being acted on:

```bash
grep -h "acting_on" state-*.json | sort | uniq -d
```

If you add a second loop, it gets its own state file and shares this denylist.

## Weekly review checklist

- [ ] Read `loop-run-log.jsonl`. Are attempts trending 2 → 3? That means the verifier or the
      implementer is degrading.
- [ ] False positive rate under 30%?
- [ ] Token per approved memo within 2× of baseline?
- [ ] Human Inbox cleared, or explicitly deferred?
- [ ] Can you explain each published memo's recommendation **in your own words**?
- [ ] Prune `STATE.md`. Delete anything closed.
