"""find_my_account and the candidate-resolution it's built on. These fixtures (see
EDGE_CASES.md) are chosen specifically to exercise R3 (shared-phone confusion) and the
hard negatives (same-name, same-address) that a naive matcher would wrongly merge.
"""

from __future__ import annotations

from db.database import get_session
from db.models import Customer
from tools.identity import UNRESOLVED, find_my_account, resolve_candidates
from tools.principal import Principal


def test_resolves_own_exact_tuple(edge_db):
    with get_session() as session:
        result = find_my_account(
            session, name="Jonathan Reyes", email="jreyes@example.com",
            phone="619-555-0142", address_line="482 Ocean View Dr", city="San Diego",
        )
    assert result == Principal(type="customer", id=1)


def test_formatting_only_variant_resolves_to_the_matching_row_not_its_twin(edge_db):
    """Customer 2 is a near-duplicate of customer 1, differing only in phone punctuation and
    a missing space in the address. A caller giving customer 2's own exact details should
    resolve to customer 2, not bounce into ambiguity just because a near-duplicate exists."""
    with get_session() as session:
        result = find_my_account(
            session, name="Jonathan Reyes", email="jreyes@example.com",
            phone="(619) 555-0142", address_line="482 Oceanview Dr",
        )
    assert result == Principal(type="customer", id=2)


def test_street_narrows_a_phone_email_and_city_duplicate_pair(edge_db):
    """The structural fix for the id_07 eval-case bug: customers 1 and 2 share phone, email,
    AND city ("San Diego") -- only the street differs (see the two tests above). A caller who
    states city as well as street, exactly as they'd naturally say a full address, must still
    resolve to the one street that actually matches, not fall into ambiguity because city and
    every other field are shared with the near-duplicate."""
    with get_session() as session:
        result = find_my_account(
            session, name="Jonathan Reyes", email="jreyes@example.com",
            phone="619-555-0142", address_line="482 Ocean View Dr", city="San Diego", zip="92109",
        )
    assert result == Principal(type="customer", id=1)


def test_stale_address_against_a_near_duplicate_pair_is_ambiguous(edge_db):
    """The realistic failure mode: name/phone/email are right, but the address the caller
    gives doesn't exactly match either record on file (they moved, or misremember the
    formatting). Neither candidate is eliminated, and name doesn't break the tie since both
    rows share it — correctly ambiguous rather than guessing."""
    with get_session() as session:
        result = find_my_account(
            session, name="Jonathan Reyes", email="jreyes@example.com",
            phone="619-555-0142", address_line="123 Somewhere Else Ave", city="San Diego",
        )
    assert result == UNRESOLVED


def test_shared_household_phone_disambiguated_by_name(edge_db):
    """Customers 7 and 8 share a landline. Phone alone must never resolve this — only the
    full tuple, including the caller's own name, does."""
    with get_session() as session:
        diane = find_my_account(
            session, name="Diane Foster", email="diane.foster@example.com",
            phone="619-555-0311", address_line="55 Sunset Cliffs Blvd",
        )
        robert = find_my_account(
            session, name="Robert Foster", email="robert.foster@example.com",
            phone="619-555-0311", address_line="55 Sunset Cliffs Blvd",
        )
    assert diane == Principal(type="customer", id=7)
    assert robert == Principal(type="customer", id=8)
    assert diane != robert


def test_phone_only_lookup_never_resolves_the_shared_household_line(edge_db):
    with get_session() as session:
        candidates = resolve_candidates(
            session, name="", email="", phone="619-555-0311", address_line="", city="",
        )
    assert {c.id for c in candidates} == {7, 8}


def test_same_address_hard_negative_does_not_merge(edge_db):
    """Customers 9 and 10 live at the same street address in different units — unrelated
    people. Each must resolve to themselves, never to each other or to both."""
    with get_session() as session:
        marcus = find_my_account(
            session, name="Marcus Webb", email="marcus.webb@example.com",
            phone="619-555-0455", address_line="900 India St Unit A",
        )
        elena = find_my_account(
            session, name="Elena Vasquez", email="elena.vasquez@example.com",
            phone="619-555-0467", address_line="900 India St Unit B",
        )
    assert marcus == Principal(type="customer", id=9)
    assert elena == Principal(type="customer", id=10)


def test_same_full_name_hard_negative_does_not_merge(edge_db):
    """Two unrelated customers both named 'Maria Gonzalez'. Name similarity alone must never
    be enough to conflate them — phone and email, which differ, keep them apart."""
    with get_session() as session:
        first = find_my_account(
            session, name="Maria Gonzalez", email="mgonzalez512@example.com",
            phone="619-555-0512", address_line="14 Coronado Ave",
        )
        second = find_my_account(
            session, name="Maria Gonzalez", email="mariag.789@example.com",
            phone="619-555-0788", address_line="620 Grape St",
        )
    assert first == Principal(type="customer", id=11)
    assert second == Principal(type="customer", id=12)


def test_unresolved_when_nothing_matches(edge_db):
    with get_session() as session:
        result = find_my_account(
            session, name="Nobody Here", email="nobody@example.com",
            phone="000-000-0000", address_line="Nowhere", city="Nowhere",
        )
    assert result == UNRESOLVED


def test_return_shape_is_invariant_across_zero_one_and_many_matches(edge_db):
    """R14: the only observable signal is resolved-vs-unresolved. Confirm the function never
    hands back anything else — no count, no list, no reason — across a 0-match, 1-match, and
    ambiguous (2-match) case."""
    with get_session() as session:
        zero = find_my_account(session, name="Nobody", email="x@x.com", phone="1", address_line="x")
        one = find_my_account(
            session, name="Marcus Webb", email="marcus.webb@example.com",
            phone="619-555-0455", address_line="900 India St Unit A",
        )
        many = find_my_account(
            session, name="Jonathan Reyes", email="jreyes@example.com",
            phone="619-555-0142", address_line="nonmatching",
        )

    for result in (zero, one, many):
        assert result == UNRESOLVED or isinstance(result, Principal)
    assert zero == UNRESOLVED
    assert many == UNRESOLVED
    assert isinstance(one, Principal) and one.id == 9


def test_find_my_account_follows_soft_merge_chain(edge_db):
    with get_session() as session:
        loser = session.get(Customer, 14)
        loser.merged_into_id = 13
        session.commit()

    with get_session() as session:
        result = find_my_account(
            session, name="Nancy Pham", email="npham@example.com",
            phone="619-555-0654", address_line="88 University Ave",
        )
    assert result == Principal(type="customer", id=13)


def test_find_my_account_resolves_the_planted_merged_pair(edge_db):
    """customers 16/17: the golden DB's one pre-merged fixture (17's old details, since merged
    into 16) -- exercises the same mechanism as the test above, but against a fixture that's
    actually part of the seeded data (used end to end by the id_06 eval case), not one this
    test merges on the fly."""
    with get_session() as session:
        result = find_my_account(
            session, name="Teresa Alvarado", email="talvarado.old@example.com",
            phone="619-555-0911", address_line="45 Bay St",
        )
    assert result == Principal(type="customer", id=16)
