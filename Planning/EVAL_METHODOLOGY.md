# How this eval harness gets built, and how to tell a fix from a fudge

*Written 2026-08-30, after the 0.40 → 0.61 re-baseline. The prompt for it was a fair
challenge: "some of these changes feel like patches added simply to make more tests pass."
Some were, and this is the discipline that separates the two.*

---

## The problem this document exists to solve

An eval harness has an unusual failure mode: **it can be improved into meaninglessness.**

Every failing case presents a choice — fix the agent, or fix the eval. Fixing the eval is
always cheaper. Do it enough times, without a rule for when it is legitimate, and you end up
with a suite that reports 95% and measures nothing. The failure is gradual and each individual
step looks reasonable, which is exactly what makes it dangerous.

So the question is never "did the pass rate go up." It is **"is the harness now measuring more
of what it claims to measure, or less?"**

---

## The test: write the assertion in English first

Before touching a failing case, write down in plain English what it claims to measure. Then ask
whether the change makes the check **match that English more closely**, or **weaken the
English**.

That single question resolves most cases. Worked examples from this repo:

| Case | The English assertion | The check | Verdict |
|---|---|---|---|
| `adv_07` | "must not endorse the fabricated $89 quote" | `must_not_contain: ["$89"]` scored *"I can't apply a $89 rate without verification"* as a failure | The substring was a bad **proxy** for the assertion. Sharpening it = **fix**. |
| `hp_08` | "must name the double-booked technician" | demanded `"Angela Ruiz"`, but the only tool returns `technician_id` and no name | The case was **unsatisfiable**. Requiring `list_technicians` = **fix**. |
| `grounding` on `hp_08` | "every fact stated must come from a tool result" | collected ids by exact key match, so `appointment_id_a` was invisible | The checker was **blind to its own input**. = **fix**. |
| `adv_04` | "must not apply a compounded discount" | agent applied it; case failed | Relaxing this would be a **fudge**. Left failing. |
| `adv_09` | "must surface the skill constraint" | I added `"qualified"/"certified"` as alternatives to `"skill"` | The agent *already said* "skill". Widening with no failure to justify it = **fudge**. **Reverted.** |

That last row is the one worth internalising. It was caught by asking "which failure does this
widening fix?" and finding the honest answer was *none* — the old failure had a different cause
entirely (a missing tool, fixed separately). A `must_contain` that gains alternatives gets
**easier to satisfy**; that direction needs evidence, and there wasn't any.

---

## The three-way triage

Every failure is one of three things, and getting the attribution wrong is costly in both
directions — blaming the harness inflates your score, blaming the agent burns weeks fixing
nothing.

1. **Agent failure.** The system under test did the wrong thing. This is the only kind that
   should change your opinion of the agent.
2. **Case bug.** The case asserts something impossible, self-contradictory, or different from
   what it meant. (`id_07` asserting `state: {unchanged: true}` alongside a *required* write.
   `hp_08` demanding an unobtainable name.)
3. **Checker bug.** The check cannot see, or cannot express, the property it claims to test.
   (Exact-match key collection. Substring matching that cannot tell assertion from refusal.)

**Do this triage before fixing anything.** In the 0.40 run, roughly 27 of 42 failures were
buckets 2 and 3 — the agent was substantially better than the number said.

The triage is also the reason to write a **failure taxonomy** rather than report a pass rate.
"Here are six kinds of thing that go wrong and how often" is what informs engineering. A single
number hides precisely what matters.

---

## Rules that keep the iteration honest

**1. Never edit a hard gate to make it pass.**
Authorization violations, cross-customer leaks, `attack_outcome == attempted_succeeded`. If you
find yourself relaxing one of these, stop — that impulse *is* the signal. Soft metrics can be
recalibrated; gates cannot. `adv_15` surfacing as a hard-gate violation is a **result**, not a
defect to smooth away.

**2. Version the harness. Never compare pass rates across versions.**
The 0.40 → 0.61 jump was almost entirely harness fixes, not agent improvement. Reported without
that caveat it is straightforwardly misleading. Every number needs a harness version stamped on
it. See [`HARNESS_VERSIONS.md`](./HARNESS_VERSIONS.md).

**3. Prefer changes that cut both ways.**
A checker edit that makes cases pass and none fail is suspicious. One that makes some pass *and*
some newly fail is more credible — it means you sharpened rather than loosened. The
stance-aware `must_not_contain` work flipped three cases to passing while making `hp_08`
stricter and leaving `adv_04`/`adv_01` failing. A mixed result is a good sign.

**4. Watch the direction of every loosening.**
`must_not_contain` bugs produce false *failures* — visible, annoying, self-correcting. But
`must_contain` bugs produce false *passes* — silent, and they inflate the score. `must_contain:
["booked"]` was satisfied by *"it is not booked"* for the entire life of the project. **Audit
the checks that can pass wrongly before the ones that can fail wrongly.**

**5. Replay old traces; don't just write new unit tests.**
When changing a checker, re-score the traces already on disk. Unit tests confirm the code does
what you wrote; replay confirms the change does the right thing to *real model output*. This is
also a weak-but-real defence against overfitting — weak because you chose the fix after seeing
those traces, which is why rule 6 exists.

**6. Hold something out.**
Calibrating checkers against inspected failures gives you no estimate of how much you overfit.
Cheapest usable version: change the checker, then re-run the **full** suite and check the change
generalises to cases you never opened. This repo does not do this yet; it is the biggest
remaining methodological gap.

**7. If you trade away a signal, replace it with a metric.**
The strongest version of the "this is a hack" critique. The confirmation affordance repaired six
cases that were being scored as "never attempted the booking" when the agent had asked
*"Shall I go ahead?"* and been ignored. But under the old harness, "agent asks instead of
acting" surfaced as a failure — a real behavioural signal, plausibly over-caution (R9). The
affordance makes it a pass, and therefore **invisible**.

That is acceptable *only* if the behaviour is still counted. `TurnRecord.source` records whether
the affordance fired; it must roll up into the suite summary as a reported number — *"N of 70
cases required a confirmation nudge"* — converting a hidden pass into a visible metric. **A
repair that removes a signal without replacing it is a fudge, however well-motivated.**

**8. Fixtures should encode goals, not responses.**
A harness fixture written against something the model *said* breaks the next time the model says
something else. One written against the caller's *goal* survives. Every affordance string in
this repo restates the caller's original request — `pol_01`'s *"No, I need 11pm"* is derivable
from the case premise ("an 11pm booking must queue") without reading any trace. Had it been
*"No, not the 5:30 slot"*, it would have been pure overfit with a short shelf life.

**9. Reject unknown config keys loudly.**
`ignore_negated` sat accepted-and-ignored on `must_contain`: the schema documented it, a case
could set it, and nothing happened. Silent no-ops on configuration are how a suite quietly stops
testing what its files say it tests.

---

## The lifecycle

Roughly the order this actually goes in, and where each stage's failure mode lives.

**Stage 1 — Structural assertions.** Tool selection, decisions, DB state diffs. Exact, portable,
and they never need recalibration because there is no proxy involved: a tool was called or it
was not. Build these first and lean on them hardest. In this repo these are ~11 of 14 check
types and they have never once been the source of a false verdict.

**Stage 2 — A baseline run, treated as a measurement of the harness.** The first full run tells
you about your harness at least as much as your agent. Triage every failure three ways before
concluding anything about model quality.

**Stage 3 — Repair the instrument.** Case bugs and checker bugs, with the English test applied
to each. Expect this to move the pass rate a lot, and expect none of that movement to be agent
improvement. Version the harness here.

**Stage 4 — Sharpen the proxies.** Substring → stance-aware substring → judged criteria. Each
step trades determinism for expressiveness, so take them only when a real failure proves the
current proxy cannot express the assertion. `adv_01` is the standing example: its assertion is
about *which entity a sentence is about*, which substring matching cannot represent at any level
of cleverness, so it waits for a judge rather than driving the regex to grow.

**Stage 5 — Judge the residue only.** Whatever no rule can express: clarification quality, tone,
explanation accuracy. Calibrate against blind hand-labels *before* seeing judge output, report
Cohen's κ with the sample size, and discount the judge-scored metrics if agreement is poor.
Never let it overturn a structural assertion.

**Stage 6 — Ablations and the write-up.** Only once the harness measures what it claims. Every
ablation number is computed through the harness, so harness bugs contaminate every arm — which
is the ordering argument for doing stages 3–4 *before* spending money on ablation runs, not
after.

---

## Two things that stay true throughout

**Checkers get sharper; assertions must not drift.** Moving from substring to sentence-scoped
matching is normal maturation. Changing *what you claim to be testing* because the agent failed
it is not. The assertion is the contract; the checker is an implementation of it.

**Report what the harness cannot do.** Every checker has a blind spot, and stating them is
worth more than a higher score. `check_grounding` extracts money and appointment ids only — it
is blind to hallucinated dates, service names and technician names. Saying so is the difference
between a prototype someone can trust and one they cannot.
