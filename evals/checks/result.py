"""The one return type every checker in this package uses. Deliberately just two fields --
`passed` for the pass/fail gate everything else rolls up into, `detail` as a human-readable
explanation for triage. Nothing here is scored on its own; a checker's caller decides whether a
given result is a guard (invalidates the case on failure) or a scored dimension.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckResult:
    passed: bool
    detail: str = ""
