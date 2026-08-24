"""The principal — who a tool call is running as. Constructed by the harness/dispatcher from
session state and passed to tool functions out-of-band; it never appears in a tool's argument
list, so a model can see it, but never set it or argue with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PrincipalType = Literal["customer", "staff", "system"]
StaffRole = Literal["dispatcher", "manager", "owner"]

_ROLE_RANK: dict[str, int] = {"dispatcher": 0, "manager": 1, "owner": 2}


@dataclass(frozen=True)
class Principal:
    type: PrincipalType
    id: int | None  # None = unresolved (customer) or not applicable (system)
    role: StaffRole | None = None

    def has_role_at_least(self, minimum: StaffRole) -> bool:
        """True if this principal's role outranks or matches `minimum`. Staff-only; a
        non-staff principal never has a role and never passes a role check."""
        if self.type != "staff" or self.role is None:
            return False
        return _ROLE_RANK[self.role] >= _ROLE_RANK[minimum]


# The one non-human principal: used by seed scripts and by approve.py when it executes an
# approved request on a human's behalf. Never constructed from user input.
SYSTEM = Principal(type="system", id=None)

# The starting point of every Persona C session before find_my_account resolves anything.
# Fail-closed: every row-scoped read against this principal must return nothing.
UNRESOLVED_CUSTOMER = Principal(type="customer", id=None)
