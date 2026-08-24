"""Small helpers shared by registry_c.py and registry_s.py — scheduling primitives and the
Tier-3 queue write. Nothing here makes a policy or authorization decision; that lives in
policy.py and in each tool function itself.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from db.models import Appointment, PendingRequest, ServiceItem, Technician
from db.seed_common import now_utc
from tools.principal import Principal


def get_bookable_service_item(session: Session, service_item_id: int) -> ServiceItem | None:
    """None for a missing id AND for an archived one — callers never need to tell those
    apart, both mean 'not a thing you can book against.'"""
    item = session.get(ServiceItem, service_item_id)
    if item is None or item.archived:
        return None
    return item


def technician_has_skill(technician: Technician, required_skill: str | None) -> bool:
    if required_skill is None:
        return True
    return required_skill in json.loads(technician.skills_json)


def technician_has_overlap(
    session: Session,
    technician_id: int,
    start_ts: datetime,
    end_ts: datetime,
    *,
    exclude_appointment_id: int | None = None,
) -> bool:
    query = session.query(Appointment.id).filter(
        Appointment.technician_id == technician_id,
        Appointment.status == "scheduled",
        Appointment.start_ts < end_ts,
        Appointment.end_ts > start_ts,
    )
    if exclude_appointment_id is not None:
        query = query.filter(Appointment.id != exclude_appointment_id)
    return query.first() is not None


def find_available_technician(
    session: Session, service_item: ServiceItem, start_ts: datetime, end_ts: datetime
) -> Technician | None:
    """First active, skilled, free technician — deterministic (id order), not load-balanced;
    v1 has no dispatch-optimization goal, just a yes/no on whether the slot is coverable."""
    technicians = session.query(Technician).filter(Technician.active == 1).order_by(Technician.id).all()
    for tech in technicians:
        if not technician_has_skill(tech, service_item.requires_skill):
            continue
        if technician_has_overlap(session, tech.id, start_ts, end_ts):
            continue
        return tech
    return None


def render_diff(entity_label: str, before: dict[str, Any], after: dict[str, Any]) -> str:
    """A human-readable diff for a pending_requests.preview_text — never the raw args blob.
    Only fields that actually change are shown; a Tier-3 approver reviews a change, not a
    function call."""
    lines = [f"Proposed change to {entity_label}:"]
    changed = False
    for key in after:
        old, new = before.get(key), after[key]
        if old != new:
            lines.append(f"  {key}: {old!r} -> {new!r}")
            changed = True
    if not changed:
        lines.append("  (no field changes)")
    return "\n".join(lines)


def queue_request(
    session: Session,
    *,
    principal: Principal,
    tool: str,
    args: dict[str, Any],
    preview_text: str,
) -> PendingRequest:
    """Writes the Tier-3 (or provisional-cap-escalated) row and nothing else — the caller is
    responsible for NOT performing the underlying state change; queuing and executing are
    mutually exclusive within one tool call."""
    row = PendingRequest(
        requested_by_type=principal.type,
        requested_by_id=principal.id,
        tool=tool,
        args_json=json.dumps(args, default=str, sort_keys=True),
        preview_text=preview_text,
        status="pending",
        created_at=now_utc(),
    )
    session.add(row)
    session.flush()  # populate row.id for the caller's return value / entity_ref
    return row
