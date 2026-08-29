"""agent/schemas.py: registry -> Anthropic tools array, and the two description sets. No DB, no
API calls -- this is pure introspection over the real registries, so every test runs against
REGISTRY_C/REGISTRY_S directly rather than a fake stand-in.
"""

from __future__ import annotations

import json

import pytest

from agent.schemas import DESCRIPTIONS_TERSE, DESCRIPTIONS_VERBOSE, build_schemas
from tools.registry_c import REGISTRY_C
from tools.registry_s import REGISTRY_S

ALL_TOOL_NAMES = set(REGISTRY_C) | set(REGISTRY_S)
REGISTRIES = [(REGISTRY_C, 10), (REGISTRY_S, 18)]
DESCRIPTION_SETS = [DESCRIPTIONS_TERSE, DESCRIPTIONS_VERBOSE]


@pytest.mark.parametrize("registry,expected_count", REGISTRIES)
@pytest.mark.parametrize("descriptions", DESCRIPTION_SETS)
def test_schema_names_exactly_match_registry_keys(registry, expected_count, descriptions):
    schemas = build_schemas(registry, descriptions)
    assert len(schemas) == expected_count
    assert {s["name"] for s in schemas} == set(registry)


@pytest.mark.parametrize("registry,_", REGISTRIES)
@pytest.mark.parametrize("descriptions", DESCRIPTION_SETS)
def test_no_schema_exposes_principal_or_run_id(registry, _, descriptions):
    for schema in build_schemas(registry, descriptions):
        props = schema["input_schema"]["properties"]
        assert "principal" not in props, schema["name"]
        assert "run_id" not in props, schema["name"]
        assert "principal" not in schema["input_schema"]["required"]
        assert "run_id" not in schema["input_schema"]["required"]


@pytest.mark.parametrize("registry,_", REGISTRIES)
def test_every_schema_is_strict_with_no_additional_properties(registry, _):
    for schema in build_schemas(registry, DESCRIPTIONS_TERSE):
        assert schema["strict"] is True
        assert schema["input_schema"]["additionalProperties"] is False
        assert schema["input_schema"]["type"] == "object"
        # required must be an actual subset of the declared properties -- never a stray key
        assert set(schema["input_schema"]["required"]) <= set(schema["input_schema"]["properties"])


@pytest.mark.parametrize("registry,_", REGISTRIES)
def test_every_schema_is_json_serializable(registry, _):
    schemas = build_schemas(registry, DESCRIPTIONS_TERSE)
    json.dumps(schemas)  # raises on anything that isn't a plain JSON-able value


def test_description_sets_cover_every_tool_in_both_registries():
    assert ALL_TOOL_NAMES <= set(DESCRIPTIONS_TERSE)
    assert ALL_TOOL_NAMES <= set(DESCRIPTIONS_VERBOSE)


def test_every_description_is_a_nonempty_string():
    for descriptions in DESCRIPTION_SETS:
        for name, text in descriptions.items():
            assert isinstance(text, str) and text.strip(), name


def test_terse_and_verbose_are_actually_different_text():
    """The point of having two sets is that they differ -- a description set that's secretly
    identical to the other one would make the terse-vs-verbose ablation measure nothing."""
    for name in ALL_TOOL_NAMES:
        assert DESCRIPTIONS_TERSE[name] != DESCRIPTIONS_VERBOSE[name], name
        # verbose should generally carry more explanatory text than terse
        assert len(DESCRIPTIONS_VERBOSE[name]) > len(DESCRIPTIONS_TERSE[name]), name


def test_missing_description_raises_instead_of_silently_omitting_it():
    incomplete = {k: v for k, v in DESCRIPTIONS_TERSE.items() if k != "book_appointment"}
    with pytest.raises(KeyError):
        build_schemas(REGISTRY_C, incomplete)


def test_optional_datetime_and_array_parameters_map_to_the_right_json_types():
    """Spot-checks on book_appointment (datetime + defaulted bool) and create_invoice (a
    list[dict] parameter) -- the two type-mapping cases that aren't a plain scalar."""
    by_name = {s["name"]: s for s in build_schemas(REGISTRY_C, DESCRIPTIONS_TERSE)}
    book = by_name["book_appointment"]["input_schema"]
    assert book["properties"]["start_ts"] == {"type": "string", "format": "date-time"}
    assert book["properties"]["confirmed"] == {"type": "boolean"}
    assert "confirmed" not in book["required"]  # has a default -- optional
    assert "start_ts" in book["required"]  # no default -- required

    by_name_s = {s["name"]: s for s in build_schemas(REGISTRY_S, DESCRIPTIONS_TERSE)}
    invoice = by_name_s["create_invoice"]["input_schema"]
    assert invoice["properties"]["line_items"]["type"] == "array"
    assert "appointment_id" in invoice["properties"]  # Optional[int] -- present but not required
    assert "appointment_id" not in invoice["required"]


def test_zero_argument_tool_has_an_empty_but_well_formed_schema():
    by_name = {s["name"]: s for s in build_schemas(REGISTRY_C, DESCRIPTIONS_TERSE)}
    schema = by_name["list_services"]["input_schema"]
    assert schema["properties"] == {}
    assert schema["required"] == []


def test_manager_gated_tool_schema_carries_no_trace_of_the_role_gate():
    """record_payment's min_role='manager' lives on the ToolSpec, enforced by dispatch() --
    it must not leak into the schema as a fake 'role' argument the model could try to set."""
    by_name = {s["name"]: s for s in build_schemas(REGISTRY_S, DESCRIPTIONS_TERSE)}
    props = by_name["record_payment"]["input_schema"]["properties"]
    assert set(props) == {"invoice_id", "processor_ref", "amount_cents"}


def test_line_items_declares_its_fields_so_a_strict_call_can_actually_populate_one():
    """Regression: `line_items` used to render as a property-less
    `{"type": "object", "additionalProperties": false}`, which under `strict: true` accepts only
    `{}` -- making the parameter impossible for a model to fill in, and sending the agent loop
    into an identical-retry loop against the resulting KeyError."""
    by_name = {s["name"]: s for s in build_schemas(REGISTRY_S, DESCRIPTIONS_TERSE)}
    items = by_name["create_invoice"]["input_schema"]["properties"]["line_items"]["items"]

    assert items["properties"]["unit_price_cents"] == {"type": "integer"}
    assert items["properties"]["qty"] == {"type": "integer"}
    assert items["properties"]["description"] == {"type": "string"}
    assert items["additionalProperties"] is False


def test_line_item_optionality_survives_postponed_annotations():
    """`from __future__ import annotations` hands TypedDict the *string* "NotRequired[int]", so
    `__required_keys__` reports every field as required. Optionality has to come from the
    resolved hints instead -- only the price is genuinely mandatory."""
    by_name = {s["name"]: s for s in build_schemas(REGISTRY_S, DESCRIPTIONS_TERSE)}
    items = by_name["create_invoice"]["input_schema"]["properties"]["line_items"]["items"]

    assert items["required"] == ["unit_price_cents"]
