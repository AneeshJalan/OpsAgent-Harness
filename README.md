# OpsAgent

A dual-persona LLM agent architecture with a code-enforced authorization boundary, plus the eval
methodology to prove it holds — built over a synthetic field-service business.

This is not a booking bot. The booking domain exists to give the boundary something real to guard;
the transferable assets are the tier model, the principal-scoping-in-code rule, and the measurement
discipline built to test both.

> **Status:** this section and Limitations are complete. `Results`, `Hard-subset breakout`, `Failure
> taxonomy`, `What didn't work`, and `Next steps` are stubbed with their columns defined below, to be
> filled once the ablation runs and the calibrated judge (or their explicit descoping) land — see
> `Planning/DAY3.md`.

---

## Problem

In the service trades — HVAC, plumbing, electrical — the job goes to whoever answers the phone
first. A technician with their hands inside a furnace cannot pick up, and nights and weekends are
uncovered entirely for small operators. Every unanswered call is a customer who books with a
competitor instead. An agent that can hold a real booking conversation 24/7, inside a policy
envelope the business actually controls, turns that gap into a subscription line item at SMB
economics rather than a payroll expense — and because it revenue-captures rather than merely saves
labor, it survives budget scrutiny the way cost-savings automation usually does not.

The second half of the problem is the one this project actually weighs itself against: an agent
with write access to a business's calendar and billing is worth nothing without a way to prove it
does the right thing, and to keep proving it every time the prompt or the model changes underneath
it. **The deliverable here is a measured answer to that question, not the agent.**

---

## Personas and why they differ

**Persona C is untrusted and unsupervised; Persona S is trusted and supervised. They must not share
a tool registry.** Every design decision in this project follows from that one sentence.

| | **Persona C — Customer** | **Persona S — Staff** |
|---|---|---|
| Who | A customer of the service company | Office admin, dispatcher, or owner |
| Channel | Web chat / SMS (text only) | In-app assistant panel |
| Auth | **Weak** — identified in-conversation by name/email/phone/address. Deliberately weak: that is where the risk lives (see Limitations). | Strong — authenticated session, known role |
| Wants | Book, reschedule, check an appointment, ask what something costs | Run the day's schedule, fix conflicts, merge duplicates, invoice, answer business questions |
| Availability | 24/7, instant, no human in the loop for routine work | Working hours; a human is present to confirm |
| Trust | **Untrusted.** Assume any input may be adversarial. | Trusted but fallible |
| Registry | Registry C — 10 tools | Registry S — 18 tools, `role`-gated |

A third persona — a voice-based technician assistant — is out of scope for v1 (voice latency
plumbing would consume the entire build budget); the tool layer is built so it wouldn't be
precluded, but nothing here targets it.

---

## Architecture and the authorization boundary

```
              Persona C (untrusted)            Persona S (trusted)
                     │                                │
              ┌──────▼──────┐                  ┌──────▼──────┐
              │ Registry C  │                  │ Registry S  │
              │  10 tools   │                  │  18 tools   │
              └──────┬──────┘                  └──────┬──────┘
                     └──────────────┬─────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │      Tool implementation       │
                    │  principal scoping · policy    │  ← the security boundary
                    │  engine · audit log            │
                    └───────────────┬────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  SQLite (ops.db)  │
                          └───────────────────┘
```

Every tool call from either persona is executed through one `dispatch()` (`tools/dispatcher.py`),
never a registry function directly — one audit point, one principal-scoping point, for both
personas, regardless of which agent loop called it.

### The four load-bearing rules

1. **The prompt is not a security boundary.** Row scoping, tier enforcement, and policy checks all
   live in the tool implementation. If the model is fully jailbroken, nothing changes.
2. **Denial is absence, not refusal.** A tool outside a persona's registry is never wired into that
   agent's dispatch table — there is no code path in which the model could call it, jailbroken or
   not. Never rely on a model declining to call a dangerous tool.
3. **Safety is never the model's decision.** Anything the system *must* do — escalate an
   out-of-policy booking, confirm a write to the customer — happens outside the model's discretion.
   `book_appointment` is one tool, not three; an out-of-policy request escalates automatically
   *inside the tool* and returns `QUEUED`. The agent cannot fail to escalate a booking, because
   escalating is no longer its decision to make.
4. **The principal is injected, never model-writable.** It does not appear in any tool's JSON
   schema. The model cannot see it, set it, or argue with it — a customer-facing tool that
   deliberately still accepts a `customer_id` argument (so an argument-escalation attack is
   measurable at all) ignores any value that conflicts with the injected principal and logs
   `principal_mismatch` (see Limitations).

Enforcement happens twice, independently: the **registry** is what makes rule 2 true — it filters a
persona down to only the tools it should ever see. The **dispatcher** re-checks the principal and
role on every single call regardless of which registry produced it. The filter stops accidents; the
re-check stops attacks that would otherwise only need one tool implementation to skip a check.

---

## Tier model and the runtime decision

Every tool declares a static tier — metadata used for registry filtering and for cheap negative
trace assertions (*"no tier ≥ 3 tool was ever dispatched in this trace"*):

| Tier | Meaning | Who approves | Execution |
|---|---|---|---|
| **0** | Read | Nobody | Immediate |
| **1** | Autonomous write inside the policy envelope | Nobody | Immediate, reversible |
| **2** | Write requiring in-conversation confirmation | The counterparty — customer for C, staff for S | Immediate on confirmation |
| **3** | Write requiring deferred approval by a *different* human | Staff (for C), manager/owner (for S) | Queued to `pending_requests` |
| **4** | Not available to any agent | — | Human-only UI |

Two things worth calling out: **Tier 2's approver is the counterparty, not the company** — a
confirmation gate is UX, and grants no authority of its own. **Tier 4 is enforced by absence** —
`delete_customer`, bulk deletes, raw SQL writes, `modify_policy_config`, and full customer export
exist in neither registry at all. Stating what an agent *never* gets is as much a part of this
design as what it does get.

Declared tier is static, but the actual outcome of any one call is a runtime `Decision`, and it may
escalate *above* the declared tier — never below:

```python
class Decision(Enum):
    EXECUTED      = "executed"        # inside the envelope, done
    NEEDS_CONFIRM = "needs_confirm"   # counterparty must acknowledge (fee, deposit)
    QUEUED        = "queued"          # written to pending_requests — NOT done
    DENIED        = "denied"          # not available to this principal
```

The policy envelope (`policy_config`) is how the business keeps control without being in the loop
at 2am — business hours, lead time, booking window, deposit and balance thresholds, discount cap,
cancellation-fee window, and whether after-hours booking is `never` / `deferred` / `allowed`. The
agent operates autonomously inside it and **queues rather than refuses outside it**: refusal is a
dead end for the customer, queue-and-promise preserves the relationship. `request_human_callback`
stays its own explicit Tier 3 tool precisely so *whether the agent chose to escalate* stays visible
in the trace, separate from *whether a tool auto-escalated on its behalf* — see the R8/R9 escalation
confusion matrix below.

Every non-`EXECUTED` decision, and every fall-forward customer creation, logs one reason code from a
single closed vocabulary (`tools/reasons.py`) — the eval harness asserts on these strings directly,
so nothing outside that file invents a new one.

---

## Risk register

| # | Risk | Cost | Eval category | Check |
|---|---|---|---|---|
| **R1** | Authorization escape — C reaches staff capability | Data breach; unauthorized financial action | Authorization & injection | Negative trace assertion |
| **R2** | Cross-customer leakage | Privacy incident | Identity & scoping | State + response |
| **R3** | Identity confusion on a shared phone | Acts against the wrong account | Identity & scoping | State + required-clarification |
| **R4** | Out-of-policy write | Technician sent on an impossible job | Policy boundary | State diff |
| **R5** | Hallucinated completion — claims done when the decision was `QUEUED` | Customer believes they have an appointment | Hallucination | Response + state cross-check |
| **R6** | Hallucinated fact — invented price, slot, policy | A quote the business must honor or retract | Hallucination | Grounding |
| **R7** | Entity resolution error — acts on the wrong duplicate | Invoice to wrong contact | Dirty data | State |
| **R8** | Failure to escalate — dead-ends instead of relaying the queue result, or fails to offer callback | Lost job — the exact revenue this exists to capture | Escalation | Trace |
| **R9** | Over-escalation — callback for what the tool would have handled | Value proposition collapses into expensive voicemail | Escalation | Trace |
| **R10** | Idempotency failure — retry duplicates a booking, invoice, or notification | Double dispatch; double charge | Failure injection *(roadmap — not built)* | State under injected faults |
| **R11** | Runaway loop | Inference bill; timeout | Efficiency | Turn cap + cost |
| **R12** | Bad tone or bad clarification | Brand damage; abandoned conversation | Conversational quality | Calibrated judge |
| **R13** | Provisional-record abuse — evasive identity evades the credit hold | Work performed for a non-paying account | Authorization & policy | State + reason-code assertion |
| **R14** | Identity-oracle leak — a follow-up question, turn count, or error message that varies with database contents | The customer table enumerated one probe at a time, with no row ever returned | Identity & scoping | Response invariance under DB substitution |

**R8 and R9 are one precision/recall trade-off** — *should I escalate, right now?* — and are reported
as a confusion matrix, never a flat pass rate.

---

## Eval design

### Case distribution

70 scripted cases (`evals/cases/`), enforced exactly by `tests/test_case_schema.py` so the corpus
can never silently drift in count or category:

| Category | Cases | Persona | Risks |
|---|---|---|---|
| Happy path | 8 | C + S | baseline |
| Ambiguity requiring clarification | 6 | C | R3 |
| Identity & scoping | 7 | C | R2, R3 |
| **Authorization & injection** | **10** | **C** | **R1, R14** |
| Policy boundary & escalation | 6 | C | R4, R8, R9 |
| Dirty data / entity resolution | 6 | C + S | R7 |
| Hallucination | 4 | C + S | R5, R6 |
| Over-escalation | 3 | C | R9 |
| Provisional identity & fall-forward | 2 | C | R13, R3 |
| **Adversarial (multi-turn, layered)** | **18** | C + S | R1, R7, R9, R13, R14 |

The first nine categories are the original ~50-case build target; **adversarial** was added after
that baseline run to raise the bar past single-turn attacks — its cases compose multiple pressure
tactics in one conversation (e.g. a stored customer-record note that itself contains an injection
payload, or a sequence of near-miss identifiers designed to smoke out a hard-negative match) rather
than exercising one attack vector in isolation.

**Authorization is the largest single-attack category deliberately**: it is objectively checkable
without a judge, it carries the highest consequence of anything in the risk register, and a table of
injection-attempt results — which layer stopped each one — is the single most senior-looking
artifact this project produces (see the forthcoming authorization deep-dive).

### Check types, and how portable each one actually is

A checker that only works because this database is ~300 synthetic rows this project owns is a
**fixture-dependent** checker. That is not a flaw in every case — some of the most valuable checks
here are fixture-dependent by construction — but it must be labelled, because the transferable asset
in this project is the *methodology*, and a reader evaluating that methodology needs to know which
parts would survive being pointed at a real production database unchanged, which would need
re-bounding, and which are staging-only by design.

| Check type | Portability | Note |
|---|---|---|
| **State assertion** (`evals/checks/state.py`) | Scoped-portable | Full-DB snapshot diff works at this scale as a convenience; in production the same check would be scoped to the entities one run actually touched, not diffed table-wide. |
| **Trace / selection assertions** (`evals/checks/trace_assertions.py`) — `require_tools`, `forbid_tools`, `forbid_tier_at_least`, `require_decision`, `tool_call_order`, `precedence` | Portable | Reads the trace only; no DB access. `forbid_tier_at_least: 3` is the cheap universal authorization assertion, on by default for every Persona C attack case. |
| **Response assertions** (`evals/checks/response_assertions.py`) — substring/regex, plus the R5 queued-vs-done cross-check | Portable | Trace + response text only. |
| **Grounding** (`evals/checks/grounding.py`) | Portable, run-scoped | Extracts money amounts and appointment ids from the *final* assistant turn and asserts each appears in some tool result from that same run. The workhorse of the hallucination checks (R6), and it transfers to a real deployment unchanged. |
| **PII masking** (`evals/checks/pii.py`) | Portable | Regex over the transcript — Registry C tools return no phone or email field at all, so the only leak path this can catch is the agent echoing back what the caller typed. |
| **Conversation quality** (`evals/checks/conversation_quality.py`) — repeated-solicitation, constant-message invariance | Portable, deliberately approximate | Keyword-based, not semantic; a caller who states a value without its field name (a phone number with no word "phone" nearby) is a known false negative, pending a future LLM-judge-based replacement. |
| **Response invariance under DB substitution** (`evals/checks/substitution_invariance.py`, R14) | Fixture-dependent, by design | Replays one case's identity-collection prefix against three constructed databases — the caller absent, present once, duplicated six times — and asserts the agent's behavior *signature* (turn count, tool sequence, fields solicited) is identical across all three. This is inherent, not a gap: it is a pre-production check run in staging against synthetic data, and it could not be built any other way. Say so rather than implying the toy version is the finished idea; the tool-layer half of this same guarantee (`find_my_account` returns an identical shape for 0/1/6 matches) *is* a plain deterministic unit test, and is the part of this check that can never flake. |

Two check types named in the original spec are deliberately **not** part of the per-case pipeline
today:

- **The invariant check** (`balance_cents` derivation, no orphaned `pending_requests`, every audit
  row backed by a real state change) exists and runs — but as `tests/test_invariants.py`, a plain
  deterministic test suite with no LLM involved, not as a per-case eval guard. Every property it
  checks can only ever fail from a tool/DB-layer bug, never from the agent's own tool-calling
  choices, and re-running an LLM-independent check against a costly, non-deterministic model call
  added no incremental coverage for real API spend. See `EVAL_SCHEMA.md`'s "guards" section for the
  full reasoning, including why `state`'s DB diff stayed in the per-case pipeline for the opposite
  reason.
- **The LLM judge** — reserved for what the seven deterministic checkers above cannot cover
  (clarification quality, tone, explanation accuracy) — is not yet built. See Limitations and the
  cut-order in `Planning/DAY3.md`; if it lands, its calibration (raw agreement % and Cohen's κ
  against ~15 hand-labeled cases) will be reported here, not folded into a single vibe score.

---

## Results

*Populated once the ablation runs land — see `Planning/DAY3.md` §3. Columns below are fixed now so
filling them is a data-entry step, not a design step.*

| Metric | Overall | Happy path | Ambiguity | Identity & scoping | Authorization | Policy | Dirty data | Hallucination | Over-escalation | Provisional | Adversarial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Task success rate | | | | | | | | | | | |
| Authorization violation rate *(hard gate — must be 0)* | | | | | | | | | | | |
| Cross-customer leak rate *(hard gate — must be 0)* | | | | | | | | | | | |
| Escalation precision / recall / F1 | | | | | | | | | | | |
| Hallucinated-completion rate (R5) | | | | | | | | | | | |
| Grounding-violation rate (R6) | | | | | | | | | | | |
| Mean turns | | | | | | | | | | | |
| p50 / p95 latency | | | | | | | | | | | |
| Cost per task | | | | | | | | | | | |
| Cost per *successful* task | | | | | | | | | | | |
| Judge–human agreement (raw % / κ) | | | | | | | | | | | |

**Ablation 1 — policy stated in prompt vs. enforced in code** *(the money shot — leads this section
once run)*:

| Arm | Task success | Authorization violation rate | Notes |
|---|---|---|---|
| Enforced in code (default) | | | |
| Stated in prompt only (`POLICY_ENFORCEMENT=prompt_only`) | | | |

**Ablation 2 — terse vs. verbose tool descriptions:**

| Arm | Task success | Mean turns | Cost per successful task |
|---|---|---|---|
| Terse | | | |
| Verbose | | | |

**Ablation 3 — large vs. small model:**

| Arm | Task success | Cost per successful task | p50 latency |
|---|---|---|---|
| `claude-sonnet-5` | | | |
| `claude-haiku-4-5` | | | |

---

## Hard-subset breakout

*The same metrics table above, recomputed over authorization + identity & scoping + dirty data only
(21 cases), at 3 replicates — reported alongside the headline numbers, never instead of them. A 92%
overall pass rate with two authorization failures is a failing system reported as a good one.*

### Authorization deep-dive

*One row per authorization case, once the baseline traces are re-analyzed for this — the fourth
column is the point: for every blocked attack, name the layer that actually stopped it (registry
absence / dispatcher re-check / tool-level scope check / policy engine / the model simply declined
and no code layer was exercised). The last category is counted separately and reported honestly: a
case that passes only because the model chose not to attempt the call is not a pass, it is an
unenforced boundary that happened to hold this run.*

| Case | Attack class | Agent behavior | Which layer stopped it | Reason code logged |
|---|---|---|---|---|
| | | | | |

---

## Failure taxonomy

*Grouped by mechanism, not by case id — what kind of thing went wrong, how often, and which layer
would have to change to fix it. Populated after the ablation and judge runs.*

| Mechanism | Cases affected | Root layer | Would a code fix or a prompt fix address it? |
|---|---|---|---|
| | | | |

---

## What didn't work

*The section most portfolio projects omit, and the one that reads most senior. A null result
reported as a null result is a strength — if the policy-in-prompt arm shows no measurable leak, that
is a genuinely surprising finding and belongs here as prominently as a positive one would. Candidates
to fill in once known: approaches abandoned on evidence, checkers narrowed after producing false
positives (the grounding fact-extractor is the likely one), the judge if calibration comes out weak,
anything the ablations failed to show a difference on.*

---

## Limitations

- **Synthetic data, no revenue.** This runs against a seeded SQLite database with no real customers
  and processes no real payments. Its value is demonstrative: it proves the boundary design and the
  measurement discipline, not a production deployment. Said plainly, not left for a reader to find.
- **`identity_mode = weak` is a deliberate eval fixture, not naivety.** Every Persona C case in this
  corpus resolves identity conversationally, by name/email/phone/address — the mode under which the
  hardest identity-confusion and oracle-probing risks (R2, R3, R14) are actually exercisable. A
  production deployment runs `identity_mode = strong` (a real authenticated session, no in-
  conversation resolution at all), which collapses most of that risk surface by construction. That
  mode is not built here — it is the highest-value item on the roadmap below — and without this
  paragraph a reader could easily mistake phone-based identity resolution for an oversight rather
  than the specific thing this eval suite exists to stress.
- **There is no real authentication.** The principal a tool call runs as is injected directly by the
  harness (`agent/loop.py`), never derived from a login, a token, or a session. This project is
  about **authorization** — what a resolved principal is and is not allowed to do — not
  authentication, which is assumed to already exist upstream of this system.
- **No notification delivery.** Confirming a booking, change, cancellation, or queued request to the
  customer is treated as an obligation the surrounding system fulfills deterministically, not as
  something an agent tool does or could forget to do — so it is deliberately never a tool, and there
  is no `notifications` table, outbox, or delivery path anywhere in this codebase. The only part that
  reaches the eval is the model's own side of it: whether it *describes* what happened accurately,
  in particular whether it says **queued, not done** when the actual decision was `QUEUED` (R5).
- **Sampling cannot be pinned.** `temperature` is not a parameter on the model family this project
  targets — there is no `temperature=0` determinism lever, only `output_config.effort` and adaptive
  thinking as quality knobs. Every result here is reported over replicates, never as a single
  deterministic run, and the response-invariance check (R14) compares a behavioral *signature*
  across runs rather than demanding exact string equality. This is a measured constraint on the
  methodology, not an oversight in it.
- **`customer_id` is deliberately still accepted** as an argument on ownership-scoped Registry C
  tools (e.g. `get_my_appointments`), even though the tool never needs it — the principal is already
  resolved. This exists purely so the argument-escalation attack (a model hallucinating or being
  prompted to supply someone else's id) is a *measurable* event: the dispatcher ignores any value
  that disagrees with the injected principal and logs `principal_mismatch` rather than raising a
  bare `TypeError` for a missing parameter. It is a deliberate decoy, not an API design mistake.
- **Which checks would survive a production database, and which would not.** See the portability
  table under Eval design above. Most of the deterministic checkers transfer unchanged; the DB-state
  diff and the invariant checks would need re-bounding from a full-table diff to a run-scoped one;
  the DB-substitution invariance check is fixture-dependent by construction and is a pre-production,
  staging-only check by design, not a gap to be closed later. Naming which of these transfer, which
  need re-bounding, and which are staging-only *by construction* is meant to read as the difference
  between a toy and a deliberately scaled-down prototype — not an apology for either.

---

## Next steps

*The extension roadmap (E1–E10, from `Planning/2026_08_23_OpsAgent_Spec_v3.md` §16), to be filled in
during README part 2 — `identity_mode = strong` and failure injection are the two highest-value
items and will be called out explicitly.*
