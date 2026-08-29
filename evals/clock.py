"""Freezes db.seed_common.now_utc() for the duration of one eval run, harness-only -- production
code and its behavior at real deployment time are completely untouched by this module.

Why freezing is needed: db.seed_common.now_utc() returns real wall-clock time, but the golden
DB's fixtures (db/seed_edge_cases.py) seed appointments at hardcoded absolute dates authored
assuming "today" was near ANCHOR_DATE (2026-08-23) -- real time has already moved past several of
them. Telling the model the real date (so it can resolve "next Tuesday" etc.) would then be
consistent with what tools/policy.py's checks actually enforce, but inconsistent with what the
seed data's "current/upcoming" appointments represent -- and that gap only grows the longer this
checked-in fixture DB goes un-regenerated. Freezing "now" to a fixed reference for the whole run,
and telling the model that same reference (see case_runner.py's _build_context_note), keeps the
model, the seed data, and every policy check in agreement, for every run, forever, regardless of
when the suite is actually executed.

Why patch these specific names, not db.seed_common.now_utc alone: `from db.seed_common import
now_utc` copies a reference to the function object into each importing module's own namespace at
import time -- it is not a live link back to db.seed_common. Patching db.seed_common.now_utc
after those imports have already happened (which they have, by the time any function here runs --
evals/case_runner.py's own top-level imports transitively import every one of these first) only
changes what that name means inside db.seed_common itself; none of the modules below would ever
see it. (agent/loop.py's `dispatch` binding is the identical situation, documented there.)
db.seed_common is still included below -- not because patching it alone would work, but so any
*future* call site written the module-attribute way (`import db.seed_common as seed_common;
seed_common.now_utc()`) is covered without anyone having to remember to add it here; see
tests/test_clock.py for the drift check that keeps this list honest against the "from ... import"
style call sites."""

from __future__ import annotations

import contextlib
from datetime import datetime
from unittest import mock

# A weekday, deliberately placed ahead of the seeded appointments db/seed_edge_cases.py dates at
# 2026-08-25/26/27 so those still read as upcoming, not past, against this reference.
FROZEN_NOW = datetime(2026, 8, 24, 10, 0)

# Every module that reaches now_utc() via `from db.seed_common import now_utc` (or a combined
# import naming it) -- kept as an explicit list, not derived by introspection, so a change here
# is a deliberate, reviewable diff. tests/test_clock.py asserts this stays in sync with the repo.
PATCH_TARGETS = (
    "db.seed_common",
    "approve",
    "tools.audit",
    "tools.registry_c",
    "tools.registry_s",
    "tools.tools_common",
)


@contextlib.contextmanager
def frozen_clock(frozen_at: datetime = FROZEN_NOW):
    """Patches now_utc() to return `frozen_at` in every module in PATCH_TARGETS for the duration
    of the `with` block, then reverts -- exactly the shape of a pytest monkeypatch fixture, but
    usable outside pytest (evals/run_suite.py's real-API runs are not pytest tests)."""
    with contextlib.ExitStack() as stack:
        for target in PATCH_TARGETS:
            stack.enter_context(mock.patch(f"{target}.now_utc", return_value=frozen_at))
        yield frozen_at
