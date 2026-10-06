"""Model-level tests for ScientificConditionsDeclaration, PhDeclaration,
PhPoint, PhRange - Increment 019, Checkpoint 1 (Domain).

No filesystem I/O here - see test_conditions_registry.py for persistence
behavior, including the two registry-owned scenarios (tz-aware declared_at
round-trip, full YAML round-trip) that the frozen specification assigns to
that file despite being numbered within this file's own T-001-T-038 range
(T-028, T-034).

Each test is labeled with its exact scenario id from
INCREMENT_019_FINAL_SPECIFICATION.md, Section 25.2, for mechanical
reconciliation against the normative matrix.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from molecular.conditions import (
    PhDeclaration,
    PhPoint,
    PhRange,
    ScientificConditionsDeclaration,
)


def _point(value) -> PhDeclaration:
    return PhDeclaration(point=PhPoint(value=value))


def _range(minimum, maximum) -> PhDeclaration:
    return PhDeclaration(range=PhRange(minimum=minimum, maximum=maximum))


def _declaration(**overrides) -> ScientificConditionsDeclaration:
    defaults = dict(
        schema_version="1.0",
        id="ph-standard-001",
        ph=_point(Decimal("7.4")),
        purpose="simulation_input",
        declared_by="Dr. Jane Doe",
        declared_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return ScientificConditionsDeclaration(**defaults)


# --- point/range exclusivity (T-001-T-006) ---


def test_declaration_accepts_valid_point_ph():  # T-001
    declaration = _declaration(ph=_point(Decimal("7.4")))

    assert declaration.ph.point.value == Decimal("7.4")
    assert declaration.ph.range is None


def test_declaration_accepts_valid_ph_range():  # T-002
    declaration = _declaration(ph=_range(Decimal("6.0"), Decimal("8.0")))

    assert declaration.ph.range.minimum == Decimal("6.0")
    assert declaration.ph.range.maximum == Decimal("8.0")
    assert declaration.ph.point is None


def test_ph_declaration_rejects_both_point_and_range():  # T-003
    with pytest.raises(ValidationError):
        PhDeclaration(point=PhPoint(value=Decimal("7.0")), range=PhRange(minimum=Decimal("6.0"), maximum=Decimal("8.0")))


def test_ph_declaration_rejects_neither_point_nor_range():  # T-004
    with pytest.raises(ValidationError):
        PhDeclaration()


def test_ph_range_rejects_minimum_greater_than_maximum():  # T-005
    with pytest.raises(ValidationError):
        PhRange(minimum=Decimal("8.0"), maximum=Decimal("6.0"))


def test_ph_range_accepts_minimum_equal_maximum():  # T-006
    range_ = PhRange(minimum=Decimal("7.0"), maximum=Decimal("7.0"))

    assert range_.minimum == range_.maximum == Decimal("7.0")


# --- Decimal precision and equality (T-007-T-008) ---


def test_decimal_value_preserves_exact_textual_precision():  # T-007
    point = PhPoint(value="7.40")

    assert str(point.value) == "7.40"
    assert point.model_dump(mode="json")["value"] == "7.40"


def test_decimal_1_1_0_1_00_textually_distinct_but_numerically_equal():  # T-008
    one = PhPoint(value="1")
    one_point_zero = PhPoint(value="1.0")
    one_point_zero_zero = PhPoint(value="1.00")

    assert str(one.value) == "1"
    assert str(one_point_zero.value) == "1.0"
    assert str(one_point_zero_zero.value) == "1.00"
    assert one.value == one_point_zero.value == one_point_zero_zero.value


# --- native rejection of bool/NaN/Infinity (T-009-T-012) ---


@pytest.mark.parametrize("flag", [True, False])
def test_boolean_rejected_natively_for_point(flag):  # T-009
    with pytest.raises(ValidationError):
        PhPoint(value=flag)


@pytest.mark.parametrize("flag", [True, False])
def test_boolean_rejected_natively_for_range_bounds(flag):  # T-009
    with pytest.raises(ValidationError):
        PhRange(minimum=flag, maximum=Decimal("8.0"))
    with pytest.raises(ValidationError):
        PhRange(minimum=Decimal("6.0"), maximum=flag)


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_rejected_natively_for_point(non_finite):  # T-010, T-011
    with pytest.raises(ValidationError):
        PhPoint(value=non_finite)


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_rejected_natively_for_range_minimum(non_finite):  # T-012
    with pytest.raises(ValidationError):
        PhRange(minimum=non_finite, maximum=Decimal("8.0"))


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_rejected_natively_for_range_maximum(non_finite):  # T-012
    with pytest.raises(ValidationError):
        PhRange(minimum=Decimal("6.0"), maximum=non_finite)


# --- whitespace, negative zero, structural non-claim on 0-14 (T-013-T-016) ---


def test_whitespace_padded_decimal_accepted_and_trimmed():  # T-013
    point = PhPoint(value=" 7.4 ")

    assert point.value == Decimal("7.4")


def test_negative_zero_accepted_and_equal_to_zero():  # T-014
    point = PhPoint(value="-0")

    assert point.value == Decimal("0")
    assert point.model_dump(mode="json")["value"] == "-0"


def test_ph_below_zero_accepted_structurally():  # T-015
    point = PhPoint(value=Decimal("-3.5"))

    assert point.value == Decimal("-3.5")


def test_ph_above_fourteen_accepted_structurally():  # T-016
    point = PhPoint(value=Decimal("99.5"))

    assert point.value == Decimal("99.5")


# --- the three independent Decimal constraints (T-017-T-022) ---


def test_value_exceeding_twelve_total_digits_rejected():  # T-017
    with pytest.raises(ValidationError, match="decimal_max_digits"):
        PhPoint(value="1234567.123456")  # 13 total digits


def test_value_exceeding_six_decimal_places_rejected():  # T-018
    with pytest.raises(ValidationError, match="decimal_max_places"):
        PhPoint(value="1.1234567")  # 7 decimal places


@pytest.mark.parametrize("value", ["999999", "999999.999999"])
def test_integer_digit_boundary_accepted(value):  # T-019
    point = PhPoint(value=value)

    assert str(point.value) == value


@pytest.mark.parametrize("value", ["1000000", "-1000000", "1234567", "1E+6"])
def test_integer_digit_boundary_rejected_via_decimal_whole_digits(value):  # T-020
    with pytest.raises(ValidationError, match="decimal_whole_digits"):
        PhPoint(value=value)


def test_exponential_notation_within_bounds_accepted():  # T-021
    point = PhPoint(value="1.23E+2")

    assert point.value == Decimal("123")


@pytest.mark.parametrize("value", ["1E+50", "1E-50"])
def test_extreme_exponent_rejected_via_decimal_max_digits(value):  # T-022
    with pytest.raises(ValidationError, match="decimal_max_digits"):
        PhPoint(value=value)


# --- label/rationale blank and length limits (T-023-T-026) ---


def test_label_whitespace_only_rejected():  # T-023
    with pytest.raises(ValidationError):
        _declaration(label="   ")


def test_rationale_whitespace_only_rejected():  # T-024
    with pytest.raises(ValidationError):
        _declaration(rationale="   ")


def test_label_exceeding_200_chars_rejected():  # T-025
    with pytest.raises(ValidationError):
        _declaration(label="x" * 201)


def test_rationale_exceeding_4000_chars_rejected():  # T-026
    with pytest.raises(ValidationError):
        _declaration(rationale="x" * 4001)


# --- declared_at, declared_by, purpose (T-027, T-029-T-031) ---


def test_naive_declared_at_rejected():  # T-027
    with pytest.raises(ValidationError):
        _declaration(declared_at=datetime(2026, 1, 1))


def test_missing_declared_by_rejected():  # T-029
    with pytest.raises(ValidationError):
        ScientificConditionsDeclaration(
            schema_version="1.0",
            id="ph-standard-001",
            ph=_point(Decimal("7.4")),
            purpose="simulation_input",
            declared_at=datetime.now(timezone.utc),
        )


def test_purpose_other_than_simulation_input_rejected():  # T-030
    with pytest.raises(ValidationError):
        _declaration(purpose="measurement")


def test_purpose_empty_string_rejected():  # T-031
    with pytest.raises(ValidationError):
        _declaration(purpose="")


# --- extra="forbid" and frozen=True (T-032-T-033) ---


def test_extra_field_rejected():  # T-032
    with pytest.raises(ValidationError):
        _declaration(unexpected_field="not allowed")


def test_frozen_field_reassignment_rejected():  # T-033
    declaration = _declaration()

    with pytest.raises(ValidationError):
        declaration.declared_by = "Someone Else"


# --- id / supersedes_id validation (T-035-T-037) ---


def test_id_with_invalid_characters_rejected():  # T-035
    with pytest.raises(ValidationError):
        _declaration(id="not a valid id!")


def test_supersedes_id_validated_like_id():  # T-036
    with pytest.raises(ValidationError):
        _declaration(id="ph-revision-001", supersedes_id="not a valid id!")


def test_supersedes_id_equal_to_id_rejected():  # T-037
    with pytest.raises(ValidationError):
        _declaration(id="ph-standard-001", supersedes_id="ph-standard-001")


# --- bounded canonical representation (T-038) ---


def test_canonical_serialized_ph_representation_is_bounded_in_length():  # T-038
    # 12 significant digits is the worst case max_digits/decimal_places allows.
    point = PhPoint(value="999999.999999")

    serialized = point.model_dump(mode="json")["value"]

    assert len(serialized) <= 20
