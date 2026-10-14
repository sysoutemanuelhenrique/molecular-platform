"""Tests for receptor_resolution.py - Increment 020, Checkpoint 2.

Covers T-057-068 (rewritten/preserved) and T-124 (new) from the frozen
final specification's normative matrix (Section 35.4). Each test is
labeled with its exact scenario id. No filesystem I/O here - see
test_receptor_registry.py for persistence behavior.

T-061 migrates, by the frozen spec's own instruction (Section 35.4, n-02),
from a model validator to a Service-level behavior: `ChainNotFoundError`
is raised by `resolve_receptor_selection` (Checkpoint 3) before any
`ResolvedReceptorSelection` is ever constructed, never by this model.
This file's own T-061 test proves the model side of that migration - that
no cross-check against an inspection report happens at construction time
- leaving the active rejection itself to test_receptor_service.py
(Checkpoint 3), outside this checkpoint's allowlist.
"""

import inspect

import pytest
from pydantic import ValidationError

from molecular.receptor_inspection import ParserProvenance
from molecular.receptor_resolution import (
    ResolvedChainSelection,
    ResolvedComponentDisposition,
    ResolvedReceptorSelection,
)

_HEX64 = "a" * 64
_HEX64_B = "b" * 64


def _resolution(**overrides) -> ResolvedReceptorSelection:
    from datetime import datetime, timezone

    defaults = dict(
        schema_version="1.0",
        resolved_selection_id="sel-001",
        policy_selection_id="sel-001",
        protein_input_id="demo-protein",
        inspection_id=_HEX64,
        raw_checksum=_HEX64_B,
        structure_format="mmcif",
        parser=ParserProvenance(name="gemmi", version="0.7.5"),
        resolved_model=1,
        model_resolution_basis="only-model-present",
        resolved_assembly=None,
        resolved_chains=(
            ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=1),
        ),
        resolved_component_dispositions=(),
        resolved_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return ResolvedReceptorSelection(**defaults)


# --- construction / required fields (T-057) ----------------------------------


def test_minimal_valid_construction_succeeds_and_has_well_formed_inspection_id():  # T-057
    resolution = _resolution()

    assert resolution.resolved_selection_id == resolution.policy_selection_id
    assert len(resolution.inspection_id) == 64
    assert "inspection_snapshot" not in ResolvedReceptorSelection.model_fields


def test_resolved_selection_id_mismatch_with_policy_selection_id_raises():  # T-058
    with pytest.raises(ValidationError):
        _resolution(resolved_selection_id="sel-001", policy_selection_id="sel-002")


def test_malformed_inspection_id_raises_at_field_boundary():  # T-059
    with pytest.raises(ValidationError):
        _resolution(inspection_id="not-64-hex-chars")


def test_empty_or_malformed_parser_version_raises():  # T-060
    with pytest.raises(ValidationError):
        _resolution(parser=ParserProvenance(name="gemmi", version=""))


def test_model_itself_never_validates_a_chain_against_any_inspection_report():  # T-061
    """At the model layer, `resolved_chains` accepts any well-formed
    `ResolvedChainSelection` regardless of whether a real inspection
    report would confirm it - confirming that an unconfirmed chain never
    reaches `resolved_chains` is `resolve_receptor_selection`'s
    responsibility (Checkpoint 3's Application Service, Section 25.4), not
    this model's. The frozen spec's T-061 (Section 35.4) migrates this
    exact check from "model validator" to "Service behavior"; this proves
    the model side of that migration - no cross-check against an
    inspection report happens here. The active rejection itself
    (`ChainNotFoundError`) belongs to test_receptor_service.py (Checkpoint
    3), outside this checkpoint's allowlist.
    """
    resolution = _resolution(
        resolved_chains=(
            ResolvedChainSelection(
                label_asym_id="ZZZ-not-a-real-chain",
                label_entity_id="1",
                model_number=1,
            ),
        )
    )

    assert resolution.resolved_chains[0].label_asym_id == "ZZZ-not-a-real-chain"


def test_duplicate_label_asym_id_in_resolved_chains_raises():  # T-062
    with pytest.raises(ValidationError):
        _resolution(
            resolved_chains=(
                ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=1),
                ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=1),
            )
        )


def test_no_field_names_an_output_structure_file():  # T-063
    field_names = set(ResolvedReceptorSelection.model_fields)
    forbidden_markers = ("path", "file", "artifact", "pdb", "mmcif", "coordinate")

    for name in field_names:
        lowered = name.lower()
        for marker in forbidden_markers:
            assert marker not in lowered, f"field {name!r} suggests an output file reference"

    chain_field_names = set(ResolvedChainSelection.model_fields)
    for name in chain_field_names:
        for marker in forbidden_markers:
            assert marker not in name.lower()


def test_resolved_assembly_none_is_valid():  # T-064
    resolution = _resolution(resolved_assembly=None)

    assert resolution.resolved_assembly is None


def test_only_model_present_basis_is_an_accepted_closed_value():  # T-065
    resolution = _resolution(model_resolution_basis="only-model-present")

    assert resolution.model_resolution_basis == "only-model-present"


def test_water_disposition_source_implies_water_classification():  # T-066
    disposition = ResolvedComponentDisposition(
        label_asym_id="B",
        ccd_code="HOH",
        classification="water",
        disposition="remove",
        disposition_source="water_disposition",
    )

    assert disposition.classification == "water"
    assert disposition.disposition_source == "water_disposition"


def test_json_round_trip_reproduces_an_equal_model():  # T-067
    resolution = _resolution()

    reloaded = ResolvedReceptorSelection.model_validate(resolution.model_dump(mode="json"))

    assert reloaded == resolution


def test_no_semantic_key_field_exists_in_schema():  # T-068
    assert "semantic_key" not in ResolvedReceptorSelection.model_fields


# --- schema introspection proves no duplicated inventory (T-124) ------------


def test_no_field_is_or_contains_a_receptor_inspection_report():  # T-124
    from molecular.receptor_inspection import ReceptorInspectionReport

    def _collect_annotated_types(model_cls):
        types = []
        for field in model_cls.model_fields.values():
            annotation = field.annotation
            types.append(annotation)
            args = getattr(annotation, "__args__", ())
            types.extend(args)
        return types

    for model_cls in (
        ResolvedReceptorSelection,
        ResolvedChainSelection,
        ResolvedComponentDisposition,
    ):
        for annotated_type in _collect_annotated_types(model_cls):
            assert annotated_type is not ReceptorInspectionReport
            if inspect.isclass(annotated_type) and issubclass(annotated_type, type(None)) is False:
                assert not (
                    inspect.isclass(annotated_type)
                    and issubclass(annotated_type, ReceptorInspectionReport)
                )

    inventory_collection_names = {
        "models", "assemblies", "entities", "chains", "altloc_observations",
        "occupancy_observations", "missing_residues", "missing_atoms",
        "disulfide_bonds", "covalent_connections", "metal_coordinations",
        "non_polymer_components", "warnings", "unavailable_categories",
    }
    assert not (inventory_collection_names & set(ResolvedReceptorSelection.model_fields))
