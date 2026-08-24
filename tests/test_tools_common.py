from __future__ import annotations

from datetime import datetime

from db.database import get_session
from db.models import PendingRequest, ServiceItem
from tools.principal import Principal
from tools.tools_common import (
    find_available_technician,
    get_bookable_service_item,
    queue_request,
    render_diff,
    technician_has_overlap,
)


def test_get_bookable_service_item_excludes_archived(edge_db):
    with get_session() as session:
        assert get_bookable_service_item(session, 1) is not None  # current Drain Cleaning
        assert get_bookable_service_item(session, 10) is None  # archived legacy duplicate
        assert get_bookable_service_item(session, 999) is None  # doesn't exist


def test_technician_has_overlap_detects_the_planted_double_booking(edge_db):
    with get_session() as session:
        assert technician_has_overlap(session, 7, datetime(2026, 8, 25, 9, 40), datetime(2026, 8, 25, 10, 0))
        assert not technician_has_overlap(session, 7, datetime(2026, 8, 25, 11, 0), datetime(2026, 8, 25, 12, 0))


def test_technician_has_overlap_excludes_given_appointment(edge_db):
    """Rescheduling appointment 3 (technician 6's only scheduled job) to its own existing
    slot shouldn't see itself as a conflict."""
    with get_session() as session:
        assert technician_has_overlap(session, 6, datetime(2026, 8, 26, 13, 0), datetime(2026, 8, 26, 14, 0))
        assert not technician_has_overlap(
            session, 6, datetime(2026, 8, 26, 13, 0), datetime(2026, 8, 26, 14, 0),
            exclude_appointment_id=3,
        )


def test_find_available_technician_skips_conflicted_and_unskilled(edge_db):
    with get_session() as session:
        item = session.get(ServiceItem, 4)  # AC Tune-Up, requires hvac
        tech = find_available_technician(session, item, datetime(2026, 8, 25, 9, 0), datetime(2026, 8, 25, 10, 0))
    # technician 7 has the skill but is double-booked at this exact window; technician 2 is
    # the next hvac-skilled, unconflicted active technician.
    assert tech is not None
    assert tech.id == 2


def test_find_available_technician_returns_none_when_no_one_qualifies(edge_db):
    with get_session() as session:
        item = session.get(ServiceItem, 4)  # hvac
        # Every hvac-capable technician is busy or nonexistent at a manufactured triple-booked
        # window is overkill to construct; instead ask for a skill nobody active has.
        item.requires_skill = "gas_fitting"
        tech = find_available_technician(session, item, datetime(2026, 8, 25, 9, 0), datetime(2026, 8, 25, 10, 0))
    assert tech is None


def test_render_diff_shows_only_changed_fields():
    text = render_diff("invoice:1", before={"status": "sent", "total_cents": 100}, after={"status": "void", "total_cents": 100})
    assert "status: 'sent' -> 'void'" in text
    assert "total_cents" not in text


def test_render_diff_no_changes():
    text = render_diff("invoice:1", before={"status": "sent"}, after={"status": "sent"})
    assert "no field changes" in text


def test_queue_request_persists_a_pending_row(edge_db):
    principal = Principal(type="customer", id=1)
    with get_session() as session:
        row = queue_request(
            session, principal=principal, tool="book_appointment",
            args={"service_item_id": 4}, preview_text="Proposed change: book AC Tune-Up",
        )
        assert row.id is not None
        session.commit()

    with get_session() as session:
        fetched = session.get(PendingRequest, row.id)
        assert fetched is not None
        assert fetched.status == "pending"
        assert fetched.tool == "book_appointment"
        assert fetched.requested_by_id == 1
