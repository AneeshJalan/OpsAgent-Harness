"""Principal is a frozen dataclass, but role-ranking is real logic worth pinning down."""

from __future__ import annotations

from tools.principal import Principal


def test_role_at_least_respects_rank_order():
    dispatcher = Principal(type="staff", id=1, role="dispatcher")
    manager = Principal(type="staff", id=2, role="manager")
    owner = Principal(type="staff", id=3, role="owner")

    assert dispatcher.has_role_at_least("dispatcher")
    assert not dispatcher.has_role_at_least("manager")
    assert manager.has_role_at_least("dispatcher")
    assert manager.has_role_at_least("manager")
    assert not manager.has_role_at_least("owner")
    assert owner.has_role_at_least("manager")


def test_role_at_least_false_for_non_staff():
    customer = Principal(type="customer", id=1)
    system = Principal(type="system", id=None)
    assert not customer.has_role_at_least("dispatcher")
    assert not system.has_role_at_least("dispatcher")


def test_principal_is_frozen_and_hashable():
    p1 = Principal(type="staff", id=1, role="manager")
    p2 = Principal(type="staff", id=1, role="manager")
    assert p1 == p2
    assert hash(p1) == hash(p2)
