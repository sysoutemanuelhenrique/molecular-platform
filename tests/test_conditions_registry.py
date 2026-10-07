"""Persistence tests for ScientificConditionsDeclaration - Increment 019,
Checkpoint 1 (Registry).

Covers the registry-owned scenarios from the normative matrix
(INCREMENT_019_FINAL_SPECIFICATION.md, Section 25.2): T-028, T-034, and
T-039-T-047. T-028 and T-034 are registry-owned despite being numbered
within test_conditions.py's own T-001-T-038 range, since both require
actual YAML persistence, not just in-memory Pydantic construction.

No Application Service, retry, predecessor, or revision logic is exercised
here - register_conditions_declaration is a pure create-once write, with
no recognition of a "safe retry" under a colliding id. That recognition is
conditions_service.py's responsibility (Checkpoint 2).
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from molecular.conditions import PhDeclaration, PhPoint, ScientificConditionsDeclaration
from molecular.conditions_registry import (
    ConditionsIdentityConflictError,
    ConditionsRecordInvalidError,
    _conditions_dir,
    list_conditions_declarations,
    load_conditions_declaration,
    register_conditions_declaration,
)


def _declaration(**overrides) -> ScientificConditionsDeclaration:
    defaults = dict(
        schema_version="1.0",
        id="ph-standard-001",
        ph=PhDeclaration(point=PhPoint(value=Decimal("7.4"))),
        purpose="simulation_input",
        declared_by="Dr. Jane Doe",
        declared_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return ScientificConditionsDeclaration(**defaults)


def _register(tmp_path, **overrides) -> ScientificConditionsDeclaration:
    return register_conditions_declaration(tmp_path, _declaration(**overrides))


# --- round-trip fidelity (T-028, T-034) ---


def test_declared_at_round_trips_as_utc(tmp_path):  # T-028
    declared_at = datetime.now(timezone.utc)
    registered = _register(tmp_path, declared_at=declared_at)

    loaded = load_conditions_declaration(tmp_path, registered.id)

    assert loaded.declared_at == declared_at
    assert loaded.declared_at.tzinfo is not None
    assert loaded.declared_at.utcoffset().total_seconds() == 0


def test_full_yaml_round_trip_reproduces_an_equal_model(tmp_path):  # T-034
    registered = _register(tmp_path)

    loaded = load_conditions_declaration(tmp_path, registered.id)

    assert loaded == registered


# --- create-once and identity (T-039, T-040) ---


def test_first_creation_persists_successfully(tmp_path):  # T-039
    registered = _register(tmp_path)

    metadata_file = _conditions_dir(tmp_path) / f"{registered.id}.yaml"
    assert metadata_file.is_file()


def test_casefold_collision_raises_identity_conflict(tmp_path):  # T-040
    _register(tmp_path, id="ph-standard-001")

    with pytest.raises(ConditionsIdentityConflictError):
        _register(tmp_path, id="PH-STANDARD-001", declared_by="Someone Else")


# --- listing (T-041, T-042, T-046) ---


def test_listing_on_project_with_zero_declarations_returns_empty_tuple(tmp_path):  # T-041
    declarations = list_conditions_declarations(tmp_path)

    assert declarations == ()


def test_listing_never_implies_a_current_or_latest_declaration(tmp_path):  # T-042
    _register(tmp_path, id="ph-first-001")
    _register(tmp_path, id="ph-second-001")

    declarations = list_conditions_declarations(tmp_path)

    # The tuple itself carries no "current"/"active"/"latest" marker of any
    # kind - every element is an ordinary, independently-valid declaration.
    assert len(declarations) == 2
    assert all(isinstance(d, ScientificConditionsDeclaration) for d in declarations)


def test_listing_produces_a_stable_repeatable_ordering(tmp_path):  # T-046
    first = _register(
        tmp_path, id="ph-alpha-001", declared_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    second = _register(
        tmp_path, id="ph-beta-001", declared_at=datetime(2026, 1, 2, tzinfo=timezone.utc)
    )

    first_listing = list_conditions_declarations(tmp_path)
    second_listing = list_conditions_declarations(tmp_path)

    assert first_listing == second_listing
    assert [d.id for d in first_listing] == [first.id, second.id]


# --- load failures (T-043, T-044, T-045) ---


def test_loading_missing_declaration_raises_file_not_found(tmp_path):  # T-043
    with pytest.raises(FileNotFoundError):
        load_conditions_declaration(tmp_path, "does-not-exist")


def test_loading_malformed_yaml_raises_record_invalid_with_yaml_cause(tmp_path):  # T-044
    conditions_dir = _conditions_dir(tmp_path)
    conditions_dir.mkdir(parents=True)
    (conditions_dir / "broken-001.yaml").write_text(
        "schema_version: '1.0'\nid: [unterminated\n", encoding="utf-8"
    )

    with pytest.raises(ConditionsRecordInvalidError) as excinfo:
        load_conditions_declaration(tmp_path, "broken-001")

    assert excinfo.value.__cause__ is not None


def test_loading_schema_invalid_yaml_raises_record_invalid_with_validation_cause(tmp_path):  # T-045
    conditions_dir = _conditions_dir(tmp_path)
    conditions_dir.mkdir(parents=True)
    (conditions_dir / "incompatible-001.yaml").write_text(
        "schema_version: '2.0'\nid: incompatible-001\n", encoding="utf-8"
    )

    with pytest.raises(ConditionsRecordInvalidError) as excinfo:
        load_conditions_declaration(tmp_path, "incompatible-001")

    assert isinstance(excinfo.value.__cause__, ValidationError)


# --- no absolute paths persisted (T-047) ---


def test_no_persisted_record_contains_an_absolute_filesystem_path(tmp_path):  # T-047
    registered = _register(tmp_path)

    metadata_file = _conditions_dir(tmp_path) / f"{registered.id}.yaml"
    raw_content = metadata_file.read_text(encoding="utf-8")

    assert str(tmp_path) not in raw_content
