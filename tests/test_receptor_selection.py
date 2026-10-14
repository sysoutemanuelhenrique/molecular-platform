"""Tests for receptor_selection.py - Increment 020, Checkpoint 2.

Covers T-043-056 (preserved) and T-123 (new) from the frozen final
specification's normative matrix (Section 35.3). Each test is labeled
with its exact scenario id. No filesystem I/O here - see
test_receptor_registry.py for persistence behavior.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from molecular.receptor_selection import (
    ChainSelector,
    ComponentDisposition,
    ReceptorSelectionPolicy,
)

_HEX64 = "a" * 64
_HEX64_B = "b" * 64


def _policy(**overrides) -> ReceptorSelectionPolicy:
    defaults = dict(
        schema_version="1.0",
        selection_id="sel-001",
        protein_input_id="demo-protein",
        inspection_id=_HEX64,
        raw_checksum_observed=_HEX64_B,
        selected_model=1,
        selected_assembly=None,
        selected_chains=(ChainSelector(label_asym_id="A"),),
        water_disposition="keep",
        metal_disposition="keep",
        component_dispositions=(),
        declared_by="Dr. Jane Doe",
        label=None,
        rationale=None,
        declared_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return ReceptorSelectionPolicy(**defaults)


# --- construction / required fields (T-043) ---------------------------------


def test_minimal_valid_construction_requires_inspection_id_and_raw_checksum():  # T-043
    policy = _policy()

    assert policy.inspection_id == _HEX64
    assert policy.raw_checksum_observed == _HEX64_B

    with pytest.raises(ValidationError):
        ReceptorSelectionPolicy(
            schema_version="1.0",
            selection_id="sel-001",
            protein_input_id="demo-protein",
            # inspection_id and raw_checksum_observed omitted - both required
            selected_chains=(),
            component_dispositions=(),
            declared_by="Dr. Jane Doe",
            declared_at=datetime.now(timezone.utc),
        )


# --- field validation (T-044-055) --------------------------------------------


def test_malformed_selection_id_raises_value_error():  # T-044
    with pytest.raises(ValidationError):
        _policy(selection_id="not valid!")


def test_duplicate_label_asym_id_in_selected_chains_raises():  # T-045
    with pytest.raises(ValidationError):
        _policy(
            selected_chains=(
                ChainSelector(label_asym_id="A"),
                ChainSelector(label_asym_id="A"),
            )
        )


def test_duplicate_component_disposition_pair_raises():  # T-046
    duplicate = ComponentDisposition(
        label_asym_id="B", ccd_code="ZN", role="cofactor", disposition="keep"
    )
    with pytest.raises(ValidationError):
        _policy(component_dispositions=(duplicate, duplicate))


def test_blank_declared_by_raises():  # T-047
    with pytest.raises(ValidationError):
        _policy(declared_by="   ")


def test_blank_label_or_rationale_when_present_raises():  # T-048
    with pytest.raises(ValidationError):
        _policy(label="   ")
    with pytest.raises(ValidationError):
        _policy(rationale="   ")


def test_declared_at_without_timezone_raises():  # T-049
    with pytest.raises(ValidationError):
        _policy(declared_at=datetime(2026, 1, 1))


def test_selected_model_zero_or_negative_raises():  # T-050
    with pytest.raises(ValidationError):
        _policy(selected_model=0)
    with pytest.raises(ValidationError):
        _policy(selected_model=-1)


def test_unknown_field_is_rejected():  # T-051
    with pytest.raises(ValidationError):
        _policy(unexpected_field="nope")


def test_model_is_frozen():  # T-052
    policy = _policy()
    with pytest.raises(ValidationError):
        policy.selection_id = "sel-002"


def test_chain_selector_with_only_label_asym_id_is_valid():  # T-053
    selector = ChainSelector(label_asym_id="A")

    assert selector.label_asym_id == "A"
    assert selector.auth_asym_id is None


def test_chain_selector_with_empty_label_asym_id_raises():  # T-054
    with pytest.raises(ValidationError):
        ChainSelector(label_asym_id="")


def test_component_disposition_role_outside_closed_vocabulary_raises():  # T-055
    with pytest.raises(ValidationError):
        ComponentDisposition(
            label_asym_id="B", ccd_code="ZN", role="inhibitor", disposition="keep"
        )


# --- no removed policy-bucket fields (T-056) ---------------------------------


def test_no_altloc_or_bucket_policy_fields_exist_in_schema():  # T-056
    field_names = set(ReceptorSelectionPolicy.model_fields)

    assert "altloc_policy" not in field_names
    assert "cofactor_policy" not in field_names
    assert "crystallographic_ligand_policy" not in field_names


# --- auth_asym_id alone is never a valid selector (T-123) --------------------


def test_chain_selector_never_accepts_auth_asym_id_alone():  # T-123
    with pytest.raises(ValidationError):
        ChainSelector(auth_asym_id="A")
