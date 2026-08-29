"""Identity resolution — the highest-risk logic in the system (closes R3, R13, R14).

`find_my_account` is written so a leaky variant is impossible to produce by accident: its
return type is `Principal | Literal["unresolved"]`, nothing else. No count, no candidate list,
no reason string ever leaves this function. Diagnostic detail (how many rows matched, why
resolution failed) stays internal to `resolve_candidates` — nothing outside this module reads
it, since every fall-forward consumer (`is_provisional`, the eval harness, book_appointment's
own control flow) only ever needs one outcome ("identity unresolved"), never how it failed to
resolve.
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
    session: Session, *, name: str, email: str, phone: str,
    address_line: str, city: str | None = None, zip: str | None = None,
) -> list[Customer]:
    """Internal matcher — progressively narrows the full customer set by phone, then email,
    then street, then city, then zip, then name, in that order (most-unique identifier first).
    Only narrows when a field actually matches someone in the current set; only
    `find_my_account` calls this directly.

    `address_line`/`city`/`zip` are separate parameters, not one flattened "address" string
    the caller states in one breath and we'd have to parse back apart -- `Customer` already
    stores them as separate columns, and matching each directly against its own column is more
    reliable than any string-splitting heuristic over a combined value (a caller might say
    "482 Ocean View Dr, San Diego" or "482 Ocean View Dr in San Diego" or omit the city
    entirely -- the model reliably extracts these into separate fields once the tool schema
    asks for them that way, which a fixed split never generalizes to). `city` and `zip` are
    both optional and only narrow when supplied, matching the existing pattern for every other
    field here; `address_line` is the one part of the address that's always expected, since a
    caller can plausibly omit city or zip in conversation but rarely omits the street.

    This is what makes the household-shared-phone case (R3) resolve correctly: phone alone
    matches both members of the household, but email narrows it back down to one. It's also
    what correctly resolves a caller against a near-duplicate seeded pair sharing phone, email,
    and name (customers 1/2, "Jonathan Reyes") -- street narrows the two apart even though
    every other field is identical between them. And it's what keeps the same-name and
    same-address hard negatives from merging, since neither of those fields is even part of the
    phone/email match, and a shared field alone never survives the narrowing pass without also
    matching the stronger identifiers.
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

    norm_street = _normalize_text(address_line)
    if norm_street:
        matched = [c for c in candidates if _normalize_text(c.address_line) == norm_street]
        if matched:
            candidates = matched

    norm_city = _normalize_text(city)
    if norm_city:
        matched = [c for c in candidates if _normalize_text(c.city) == norm_city]
        if matched:
            candidates = matched

    norm_zip = _normalize_text(zip)
    if norm_zip:
        matched = [c for c in candidates if _normalize_text(c.zip) == norm_zip]
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
    session: Session, *, name: str, email: str, phone: str,
    address_line: str, city: str | None = None, zip: str | None = None,
) -> Principal | Literal["unresolved"]:
    """Full tuple, called once. Returns a resolved Principal or the constant UNRESOLVED
    sentinel — that boundary is the only thing observable outside this function, by design.
    Whether zero rows matched or six did, the return value and everything about calling this
    function looks identical, because the number of matches is exactly what an attacker
    probing for account existence (R14) is trying to learn."""
    candidates = resolve_candidates(
        session, name=name, email=email, phone=phone, address_line=address_line, city=city, zip=zip,
    )
    if len(candidates) != 1:
        return UNRESOLVED

    resolved = _resolve_merge_chain(session, candidates[0])
    return Principal(type="customer", id=resolved.id)
