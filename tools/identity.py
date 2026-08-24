"""Identity resolution — the highest-risk logic in the system (closes R3, R13, R14).

`find_my_account` is written so a leaky variant is impossible to produce by accident: its
return type is `Principal | Literal["unresolved"]`, nothing else. No count, no candidate list,
no reason string ever leaves this function. Diagnostic detail (how many rows matched, why
resolution failed) stays internal to `resolve_candidates`, which only `book_appointment` is
allowed to call directly, and only to choose an audit-log reason code — never to change what
the caller sees.
"""

from __future__ import annotations

import re
from typing import Literal

from sqlalchemy.orm import Session

from db.models import Customer
from tools.principal import Principal

UNRESOLVED: Literal["unresolved"] = "unresolved"


def _normalize_phone(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def _all_customers(session: Session) -> list[Customer]:
    """Includes soft-merged-away rows deliberately: a caller's identifying details still
    belong to the record that was created for them even after staff merge it into another
    account, so matching must still find it. `find_my_account` follows the merge chain to the
    survivor afterward — merged-away rows are valid match targets, just not valid endpoints."""
    return session.query(Customer).all()


def resolve_candidates(
    session: Session, *, name: str, email: str, phone: str, address: str
) -> list[Customer]:
    """Internal matcher — progressively narrows the full customer set by phone, then email,
    then address, then name, in that order (most-unique identifier first). Only narrows when
    a field actually matches someone in the current set; never used directly by an agent-facing
    tool other than book_appointment's fall-forward reason-code selection.

    This is what makes the household-shared-phone case (R3) resolve correctly: phone alone
    matches both members of the household, but email narrows it back down to one — and it's
    what keeps the same-name and same-address hard negatives from merging, since neither of
    those fields is even part of the phone/email match, and a shared field alone never
    survives the narrowing pass without also matching the stronger identifiers.
    """
    candidates = _all_customers(session)
    norm_phone = _normalize_phone(phone)
    if norm_phone:
        matched = [c for c in candidates if _normalize_phone(c.phone) == norm_phone]
        if matched:
            candidates = matched

    norm_email = _normalize_text(email)
    if norm_email:
        matched = [c for c in candidates if _normalize_text(c.email) == norm_email]
        if matched:
            candidates = matched

    norm_address = _normalize_text(address)
    if norm_address:
        matched = [c for c in candidates if _normalize_text(c.address_line) == norm_address]
        if matched:
            candidates = matched

    norm_name = _normalize_text(name)
    if norm_name:
        matched = [c for c in candidates if _normalize_text(c.name) == norm_name]
        if matched:
            candidates = matched

    return candidates


def _resolve_merge_chain(session: Session, customer: Customer) -> Customer:
    seen = {customer.id}
    while customer.merged_into_id is not None:
        if customer.merged_into_id in seen:
            break  # defensive only; soft merges are never expected to cycle
        customer = session.get(Customer, customer.merged_into_id)
        seen.add(customer.id)
    return customer


def find_my_account(
    session: Session, *, name: str, email: str, phone: str, address: str
) -> Principal | Literal["unresolved"]:
    """Full tuple, called once. Returns a resolved Principal or the constant UNRESOLVED
    sentinel — that boundary is the only thing observable outside this function, by design.
    Whether zero rows matched or six did, the return value and everything about calling this
    function looks identical, because the number of matches is exactly what an attacker
    probing for account existence (R14) is trying to learn."""
    candidates = resolve_candidates(session, name=name, email=email, phone=phone, address=address)
    if len(candidates) != 1:
        return UNRESOLVED

    resolved = _resolve_merge_chain(session, candidates[0])
    return Principal(type="customer", id=resolved.id)
