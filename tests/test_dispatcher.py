"""dispatch() in isolation, using a small fake registry — real registry/tool behavior is
covered in test_registry_c.py / test_registry_s.py. What matters here is purely the boundary:
membership, role gating, and pass-through, each with the correct audit trail."""

from __future__ import annotations

from db.database import get_session
from db.models import AuditLog
from tools.dispatcher import Decision, ToolSpec, dispatch
from tools.principal import Principal


def _echo_tool(*, principal, run_id=None, **kwargs):
    return {"decision": Decision.EXECUTED.value, "principal_id": principal.id, "kwargs": kwargs}


FAKE_REGISTRY = {
    "echo": ToolSpec(fn=_echo_tool, tier=0),
    "manager_only": ToolSpec(fn=_echo_tool, tier=2, min_role="manager"),
}


def test_dispatch_calls_the_tool_when_everything_checks_out(db_path):
    principal = Principal(type="customer", id=1)
    result = dispatch(FAKE_REGISTRY, "echo", principal, foo="bar")
    assert result["decision"] == Decision.EXECUTED.value
    assert result["principal_id"] == 1
    assert result["kwargs"] == {"foo": "bar"}


def test_dispatch_denies_and_logs_tool_not_in_registry(db_path):
    principal = Principal(type="customer", id=1)
    result = dispatch(FAKE_REGISTRY, "merge_customers", principal, run_id="run-x")
    assert result == {"decision": "denied", "reason": "not_in_registry", "tool": "merge_customers"}

    with get_session() as session:
        row = session.query(AuditLog).one()
        assert row.tool == "merge_customers"
        assert row.decision == "denied"
        assert row.reason == "not_in_registry"
        assert row.run_id == "run-x"
        assert row.principal_type == "customer"
        assert row.principal_id == 1


def test_dispatch_never_calls_the_tool_function_for_an_absent_tool(db_path, monkeypatch):
    calls = []
    monkeypatch.setitem(FAKE_REGISTRY, "echo", ToolSpec(fn=lambda **kw: calls.append(kw), tier=0))
    dispatch(FAKE_REGISTRY, "not_a_real_tool", Principal(type="customer", id=1))
    assert calls == []


def test_dispatch_denies_insufficient_role(db_path):
    dispatcher_role = Principal(type="staff", id=1, role="dispatcher")
    result = dispatch(FAKE_REGISTRY, "manager_only", dispatcher_role)
    assert result["decision"] == "denied"
    assert result["reason"] == "insufficient_role"

    with get_session() as session:
        row = session.query(AuditLog).one()
        assert row.reason == "insufficient_role"
        assert row.declared_tier == 2


def test_dispatch_allows_sufficient_role(db_path):
    manager = Principal(type="staff", id=2, role="manager")
    result = dispatch(FAKE_REGISTRY, "manager_only", manager)
    assert result["decision"] == Decision.EXECUTED.value

    owner = Principal(type="staff", id=3, role="owner")
    result = dispatch(FAKE_REGISTRY, "manager_only", owner)
    assert result["decision"] == Decision.EXECUTED.value


def test_dispatch_role_gate_never_applies_to_a_customer_principal(db_path):
    """A tool with min_role set is staff-only by construction (customers never have a role),
    so a customer principal always fails the gate rather than bypassing it."""
    customer = Principal(type="customer", id=1)
    result = dispatch(FAKE_REGISTRY, "manager_only", customer)
    assert result["decision"] == "denied"
    assert result["reason"] == "insufficient_role"
