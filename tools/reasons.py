"""Reason codes attached to every non-EXECUTED decision, and to fall-forward customer
creation. A fixed, closed vocabulary — the eval harness asserts on these strings, so they are
not free-text and nothing outside this file should invent a new one without a reason.
"""

from __future__ import annotations

from enum import Enum


class Reason(str, Enum):
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    UNRESOLVED_PRINCIPAL = "unresolved_principal"
    OUTSIDE_BUSINESS_HOURS = "outside_business_hours"
    LEAD_TIME = "lead_time"
    BOOKING_WINDOW = "booking_window"
    BALANCE_HOLD = "balance_hold"
    PROVISIONAL_CAP = "provisional_cap"
    DEPOSIT_REQUIRED = "deposit_required"
    NOT_ONLINE_BOOKABLE = "not_online_bookable"
    NO_SKILLED_TECH = "no_skilled_tech"
    NULL_PRICE = "null_price"
    CANCELLATION_FEE = "cancellation_fee"
    DISCOUNT_CAP = "discount_cap"
    PRINCIPAL_MISMATCH = "principal_mismatch"
    NOT_IN_REGISTRY = "not_in_registry"

    # Not in the original fixed list — record_payment is role-gated (manager+) and nothing in
    # the closed vocabulary above covers "right principal, right tool, wrong authority level."
    # Kept distinct from PRINCIPAL_MISMATCH, which is about identity/ownership, not role.
    INSUFFICIENT_ROLE = "insufficient_role"

    # Not in the original list either — covers basic argument validation (unknown service
    # item, malformed date range) that isn't a security or policy decision, just bad input.
    INVALID_ARGUMENT = "invalid_argument"

    # Not in the original list — auto_book_enabled is a distinct off switch from any specific
    # service's bookable_online flag; NOT_ONLINE_BOOKABLE is about one catalog item, this is
    # about the whole self-service booking path being off.
    AUTO_BOOK_DISABLED = "auto_book_disabled"
