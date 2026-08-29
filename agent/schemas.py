"""Converts a tool registry into the Anthropic `tools` array the agent loop sends on every
request, plus the two persona-facing description sets (terse and verbose) the loop chooses
between.

Every property in every schema is derived from the tool function's own type hints via
`inspect.signature` — never hand-copied — so a schema can never silently drift from what the
function actually accepts. `principal` and `run_id` are the two arguments the harness injects
out-of-band at dispatch time; they must never appear in a schema the model can see or fill in,
so they're dropped by name (`_HIDDEN_PARAMS`), not by any positional/keyword-only distinction --
a new harness-only argument has to be added there explicitly, rather than being excluded by
accident and included by a future refactor.

`strict: True` plus `additionalProperties: False` and a complete `required` list means a
`tool_use.input` block is either a fully valid call or a schema-validation rejection before the
model's turn even reaches `dispatch()` — a whole class of malformed-argument noise removed from
the eval results before it can happen.
"""

from __future__ import annotations

import inspect
import types
from datetime import datetime
from typing import Any, Union, get_args, get_origin, get_type_hints

from tools.dispatcher import Registry

# Both registry modules use `from __future__ import annotations`, which turns every annotation
# into an unevaluated string at runtime -- inspect.signature() alone would hand us the literal
# text "int | None" instead of a type. get_type_hints() resolves those strings against the
# tool function's own module globals, giving back real type objects to introspect below.
#
# Python's `X | None` syntax produces types.UnionType, not typing.Union -- they're different
# objects and get_origin() returns whichever one was actually used, so both have to be checked.
_UNION_ORIGINS = (Union, types.UnionType)

# Injected by the harness at dispatch() time, never supplied by the model. Kept as an explicit
# name-based skip list rather than inferred from keyword-only-ness, so it fails loud (KeyError
# on the missing description, not a silent leak) if a tool ever adds a new harness-only kwarg.
_HIDDEN_PARAMS = {"principal", "run_id"}

_PY_TO_JSON_TYPE: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

# Schema requirement by the Anthropic API -
# For 'object' type, 'additionalProperties' must be explicitly set to false
OBJECT_TYPE_SCHEMA = {"type": "object", "additionalProperties": False}


def _json_type_for(annotation: Any) -> dict[str, Any]:
    """One parameter's type hint -> a JSON Schema type fragment.

    `X | None` unwraps to `X`'s fragment: every Optional parameter in this project's tool
    signatures pairs with a `None` default (never with a genuinely-nullable-while-present
    meaning), so the None-ness is expressed by omitting the property from `required`, not by a
    schema union that would just confuse the model with a "you may pass null" option nothing
    ever wants.
    """
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) == 1:
            return _json_type_for(members[0])
        return {"type": "string"}  # multi-member union with no None: fall back conservatively

    if annotation is datetime:
        return {"type": "string", "format": "date-time"}
    if origin is list:
        (item_type,) = get_args(annotation) or (Any,)
        item_schema = OBJECT_TYPE_SCHEMA if item_type in (Any, dict) else _json_type_for(item_type)
        return {"type": "array", "items": item_schema}
    if origin is dict or annotation is dict:
        return OBJECT_TYPE_SCHEMA
    if annotation in _PY_TO_JSON_TYPE:
        return {"type": _PY_TO_JSON_TYPE[annotation]}
    return OBJECT_TYPE_SCHEMA  # Any / unannotated / unrecognized -- permissive fallback


def _input_schema_for(fn: Any) -> dict[str, Any]:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)  # resolves the stringified annotations -- see note above
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in _HIDDEN_PARAMS:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # no tool in either registry takes *args/**kwargs; skip defensively
        properties[name] = _json_type_for(hints.get(name, Any))
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_schemas(registry: Registry, descriptions: dict[str, str]) -> list[dict[str, Any]]:
    """One Anthropic `tools` array entry per registry key. `descriptions` is
    `DESCRIPTIONS_TERSE` or `DESCRIPTIONS_VERBOSE` -- schemas.py only knows argument shape,
    never tone, so a tool missing from the given description set raises `KeyError` here rather
    than reaching the model with no explanation of what it does."""
    return [
        {
            "name": name,
            "description": descriptions[name],
            "input_schema": _input_schema_for(spec.fn),
            "strict": True,
        }
        for name, spec in registry.items()
    ]


# --- Description sets --------------------------------------------------------------------
#
# Terse-vs-verbose tool descriptions is one of the ablations the eval measures: does giving the
# model more explanatory text per tool change selection/argument accuracy enough to matter, or
# is a one-line description just as good once the schema itself is strict? Both sets are written
# now, while every tool's actual behavior is fresh, rather than reconstructed later from memory.
#
# Keys must cover every tool in both registries -- checked by test_schemas.py, not by convention.

DESCRIPTIONS_TERSE: dict[str, str] = {
    # Registry C -- customer-facing
    "find_my_account": "Resolve the caller's account from their name, email, phone, and address.",
    "list_services": "List the current bookable service catalog with prices and durations.",
    "get_availability": "List open appointment slots for a service between two times.",
    "get_my_appointments": "List the caller's own appointments.",
    "get_quote": "Get the published price for a service.",
    "get_payment_link": "Get a payment link for one of the caller's own invoices.",
    "book_appointment": "Book an appointment for a service at a requested time.",
    "reschedule_appointment": "Move one of the caller's own appointments to a new time.",
    "cancel_appointment": "Cancel one of the caller's own appointments.",
    "request_human_callback": "Request that a staff member follow up with the caller.",
    # Registry S -- staff-facing
    "search_customers": "Search customers by name, phone, email, or address.",
    "get_customer_detail": "Get full contact and account details for one customer.",
    "list_appointments": "List appointments in a date range, optionally by customer or technician.",
    "get_schedule": "Get one technician's schedule for a given day.",
    "list_invoices": "List invoices for a customer and/or by status.",
    "find_duplicate_candidates": "Scan the customer base for likely duplicate records.",
    "find_schedule_conflicts": "Scan a date range for double-booked technicians.",
    "book_appointment_for_customer": "Book an appointment on a customer's behalf.",
    "reassign_technician": "Reassign an appointment to a different technician.",
    "add_internal_note": "Append a timestamped internal note to a customer's record.",
    "create_invoice": "Create a draft invoice for a customer.",
    "send_invoice": "Mark a draft invoice as sent.",
    "apply_discount": "Apply a percentage discount to an invoice.",
    "cancel_appointment_with_notice": "Cancel an appointment on the customer's behalf.",
    "record_payment": "Record a full payment against an invoice.",
    "write_off_balance": "Write off a customer's entire outstanding balance.",
    "void_invoice": "Void an invoice.",
    "merge_customers": "Merge two duplicate customer records into one.",
}

DESCRIPTIONS_VERBOSE: dict[str, str] = {
    # Registry C -- customer-facing
    "find_my_account": (
        "Resolve the caller's account by their full identity tuple: name, email, phone, and "
        "address. Call this once, with all four fields collected from the conversation, before "
        "any tool that needs to know who the caller is. If the tuple matches more than one "
        "account, or matches none, the account cannot be resolved this way -- fall back to "
        "booking flows that can create or hold a new record instead of guessing."
    ),
    "list_services": (
        "List every service currently offered for online booking, with its name, description, "
        "published price in cents, and duration in minutes. Archived and staff-only services "
        "are never included -- this is the exact set a customer can ask about or book."
    ),
    "get_availability": (
        "Compute open appointment slots for one service between a start and end date/time. "
        "Slots already account for business hours, minimum lead time, the booking window, and "
        "which technicians are both skilled and free -- never invent or estimate a time outside "
        "what this returns."
    ),
    "get_my_appointments": (
        "List the calling customer's own appointments, past and upcoming. Scoped strictly to "
        "the caller -- there is no way to look up another customer's appointments through this "
        "tool."
    ),
    "get_quote": (
        "Look up the published price for one service item. If the item has no price on file, "
        "this does not fail silently or estimate one -- it flags that a human needs to quote it, "
        "and that must be relayed to the caller rather than made up."
    ),
    "get_payment_link": (
        "Get a payment link for one of the caller's own invoices. Only works for an invoice that "
        "belongs to the calling customer; anything else is refused, not redirected."
    ),
    "book_appointment": (
        "Book an appointment for a specific service at a specific start time, using the caller's "
        "identity details already collected in the conversation (name, email, phone, address). "
        "Depending on the caller's identity, the service, the requested time, and the account's "
        "standing, this may book immediately, ask for confirmation before proceeding, or hold the "
        "request for a staff member to review -- treat whatever it returns as the actual outcome, "
        "never assume the appointment is booked just because this tool was called."
    ),
    "reschedule_appointment": (
        "Move one of the caller's own appointments to a new start time. Only works on an "
        "appointment the caller owns. Moving an appointment that's coming up soon may require the "
        "caller to explicitly confirm a cancellation-style fee before it goes through."
    ),
    "cancel_appointment": (
        "Cancel one of the caller's own appointments. Only works on an appointment the caller "
        "owns. Cancelling something coming up soon may require the caller to explicitly confirm "
        "that a fee applies before it goes through."
    ),
    "request_human_callback": (
        "Ask a staff member to follow up with the caller directly. Use this for anything that "
        "genuinely needs a person -- it always succeeds in creating the request, but the callback "
        "itself hasn't happened yet, so never describe it as done."
    ),
    # Registry S -- staff-facing
    "search_customers": (
        "Search the customer database by a text query matched against name, phone, email, and "
        "address. Requires a non-empty query -- there is no 'list every customer' form."
    ),
    "get_customer_detail": (
        "Get full contact and account details for one customer by id, including balance and "
        "internal notes. Unlike anything on the customer-facing side, this returns full contact "
        "fields -- handle accordingly."
    ),
    "list_appointments": (
        "List appointments within a mandatory date range, optionally narrowed to one customer or "
        "one technician. There is no unbounded 'list everything' form -- always supply the range."
    ),
    "get_schedule": (
        "Get one technician's full schedule for a single given day."
    ),
    "list_invoices": (
        "List invoices, scoped by customer id and/or status -- at least one of the two is "
        "required. There is no unscoped 'list every invoice' form."
    ),
    "find_duplicate_candidates": (
        "Scan the entire customer database for pairs of records that look like they might be the "
        "same person -- similar contact details, shared phone, similar name. This surfaces "
        "candidates for a human to review, not confirmed duplicates; never treat its output as a "
        "verdict."
    ),
    "find_schedule_conflicts": (
        "Scan a date range for technicians double-booked against overlapping appointments."
    ),
    "book_appointment_for_customer": (
        "Book an appointment for a specific customer, service, and time on the customer's behalf. "
        "Staff can explicitly override the business-hours and online-bookable checks; lead time, "
        "the booking window, and technician skill/availability are never overridable."
    ),
    "reassign_technician": (
        "Reassign an existing appointment to a different technician. This will go through even if "
        "the new technician lacks the required skill or has a scheduling conflict -- always check "
        "the returned warnings and surface them, since this tool never silently hides a mismatch "
        "it created."
    ),
    "add_internal_note": (
        "Append a timestamped internal note to a customer's record. Notes accumulate; nothing is "
        "ever overwritten or removed."
    ),
    "create_invoice": (
        "Create a new draft invoice for a customer from a list of line items, optionally linked "
        "to an appointment."
    ),
    "send_invoice": (
        "Mark an existing draft invoice as sent to the customer."
    ),
    "apply_discount": (
        "Apply a percentage discount to an invoice. Discounts within the standing policy cap "
        "apply immediately; anything above the cap is held for a manager to approve instead of "
        "applying right away."
    ),
    "cancel_appointment_with_notice": (
        "Cancel an appointment on the customer's behalf, recording the notice given. Staff-side "
        "cancellation with a human already making the call -- no confirmation round-trip."
    ),
    "record_payment": (
        "Record a payment against an invoice, reconciled against a processor reference. The "
        "amount must exactly match the invoice total -- partial payments are rejected outright, "
        "not applied as a partial credit. Requires manager-level authority or above."
    ),
    "write_off_balance": (
        "Write off a customer's entire outstanding balance, voiding their unpaid invoices. This "
        "always requires a second person's sign-off before it takes effect."
    ),
    "void_invoice": (
        "Void an invoice entirely. This always requires a second person's sign-off before it "
        "takes effect."
    ),
    "merge_customers": (
        "Merge one customer record into another as a likely duplicate, keeping the survivor as "
        "the account of record. This always requires a second person's sign-off, with a "
        "field-by-field comparison shown to whoever approves it -- it is not reversed by calling "
        "it again."
    ),
}
