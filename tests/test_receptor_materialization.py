"""Tests for receptor_materialization.py - Increment 021, Checkpoint 1.

Covers all 43 Checkpoint 1 scenarios from the frozen final specification's
normative matrix (increments/021/15-final-specification/evidence/
SCENARIOS_021.csv): T021-001-040 and T021-159-161, plus the corrections
required by the architect reconciliation at governance commit
521500c18272a379dc4fa2557754d9fa30d60cb7 (ARCH-01/02/03, MAJOR-A/B,
MINOR-1-7 - tasks/021-CHECKPOINT1-CORRECTIONS.json). Each test is labeled
with its exact scenario id or finding id. This checkpoint is domain-only -
no RAW, no Gemmi, no filesystem - so every fixture here is a directly
constructed, in-memory DTO; `AtomObservation` rows stand in for what the
Checkpoint 2 Gemmi adapter will eventually supply, never a stub that
fabricates a constant "scientific success".
"""

import subprocess
import sys
import textwrap
import unicodedata
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from molecular.inputs import AcquisitionRecord, ProteinInput, RawArtifact
from molecular.receptor_inspection import (
    InspectionWarning,
    ModelDescriptor,
    ParserProvenance,
    ReceptorInspectionReport,
    compute_inspection_id,
)
from molecular.receptor_materialization import (
    CATALOGUE_SHA256,
    AtomDecision,
    AtomObservation,
    CategoryDecision,
    ConnectionDecision,
    Engine,
    MaterializationConfigurationError,
    MaterializationExecutionError,
    MaterializationInputError,
    MaterializationPlan,
    OccurrenceIdentity,
    OutsideEntry,
    PlanBinding,
    PolicyRecord,
    RawCategoryTable,
    RawConnectionEndpointTokens,
    RawConnectionRow,
    RecordSnapshot,
    SourceContext,
    SymmetryOperatorTables,
    XEntry,
    _compute_legacy_inspection_id,
    build_materialization_plan,
    build_policy_record,
    build_source_context,
    compute_atom_decisions_key,
    compute_atom_ids_key,
    compute_category_decisions,
    compute_category_decisions_key,
    compute_connection_decisions_key,
    compute_exclusions_key,
    compute_materialization_hash,
    compute_occurrence_key,
    compute_outside_key,
    compute_plan_key,
    compute_policy_id,
    compute_request_key,
    compute_source_context_key,
    resolve_connection_decisions,
    validate_time_utc,
)
from molecular.receptor_resolution import (
    ResolvedChainSelection,
    ResolvedComponentDisposition,
    ResolvedReceptorSelection,
)
from molecular.receptor_selection import ReceptorSelectionPolicy

PROTEIN_INPUT_ID = "protein-1"
SELECTION_ID = "sel-1"
RAW_SHA = "a" * 64
PARSER = ParserProvenance(name="gemmi", version="0.7.5")
INSPECTION_ID = compute_inspection_id(
    parser_name="gemmi", parser_version="0.7.5",
    protein_input_id=PROTEIN_INPUT_ID, raw_checksum=RAW_SHA, raw_format="mmcif",
)
TS = "2026-01-01T00:00:00.000000Z"


def _hex(n: int) -> str:
    return f"{n:064x}"


def _inspection_report(**overrides) -> ReceptorInspectionReport:
    protein_input_id = overrides.pop("protein_input_id", PROTEIN_INPUT_ID)
    raw_checksum = overrides.pop("raw_checksum", RAW_SHA)
    parser = overrides.pop("parser", PARSER)
    structure_format = overrides.pop("structure_format", "mmcif")
    inspection_id = overrides.pop(
        "inspection_id",
        compute_inspection_id(
            parser_name=parser.name, parser_version=parser.version,
            protein_input_id=protein_input_id, raw_checksum=raw_checksum, raw_format=structure_format,
        ),
    )
    defaults = dict(
        inspection_id=inspection_id,
        inspection_schema_version="1.0",
        inspection_profile_version="1.0",
        protein_input_id=protein_input_id,
        raw_checksum=raw_checksum,
        raw_format_declared="mmcif",
        structure_format=structure_format,
        parser=parser,
        selection_capability="full",
        models=(ModelDescriptor(model_number=1),),
        assemblies=(),
        entities=(),
        chains=(),
        altloc_observations=(),
        occupancy_observations=(),
        missing_residues=(),
        missing_atoms=(),
        disulfide_bonds=(),
        covalent_connections=(),
        metal_coordinations=(),
        non_polymer_components=(),
        explicit_hydrogens_present=False,
        explicit_hydrogen_count=0,
        unavailable_categories=(),
        warnings=(),
    )
    defaults.update(overrides)
    return ReceptorInspectionReport(**defaults)


def _resolution(**overrides) -> ResolvedReceptorSelection:
    defaults = dict(
        schema_version="1.0",
        resolved_selection_id=SELECTION_ID,
        policy_selection_id=SELECTION_ID,
        protein_input_id=PROTEIN_INPUT_ID,
        inspection_id=INSPECTION_ID,
        raw_checksum=RAW_SHA,
        structure_format="mmcif",
        parser=PARSER,
        resolved_model=1,
        model_resolution_basis="only-model-present",
        resolved_assembly=None,
        resolved_chains=(ResolvedChainSelection(label_asym_id="A", label_entity_id="E1", model_number=1),),
        resolved_component_dispositions=(),
        resolved_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return ResolvedReceptorSelection(**defaults)


def _records(**overrides):
    defaults = dict(
        protein=RecordSnapshot(kind="protein", id=PROTEIN_INPUT_ID, sha256=_hex(101), size_bytes=10, schema_version=None),
        inspection=RecordSnapshot(kind="inspection", id=INSPECTION_ID, sha256=_hex(102), size_bytes=20, schema_version="1.0"),
        selection=RecordSnapshot(kind="selection", id=SELECTION_ID, sha256=_hex(103), size_bytes=30, schema_version="1.0"),
        resolution=RecordSnapshot(kind="resolution", id=SELECTION_ID, sha256=_hex(104), size_bytes=40, schema_version="1.0"),
    )
    defaults.update(overrides)
    return tuple(defaults[key] for key in ("protein", "inspection", "selection", "resolution"))


def _protein_input(**overrides) -> ProteinInput:
    defaults = dict(
        id=PROTEIN_INPUT_ID,
        name="Protein One",
        acquisition=AcquisitionRecord(
            method="uploaded",
            acquired_at=datetime.now(timezone.utc),
            raw_artifact=RawArtifact(path="inputs/proteins/protein-1/raw.cif", format="mmcif", checksum=RAW_SHA),
        ),
    )
    defaults.update(overrides)
    return ProteinInput(**defaults)


def _policy(**overrides) -> ReceptorSelectionPolicy:
    defaults = dict(
        schema_version="1.0",
        selection_id=SELECTION_ID,
        protein_input_id=PROTEIN_INPUT_ID,
        inspection_id=INSPECTION_ID,
        raw_checksum_observed=RAW_SHA,
        declared_by="scientist",
        declared_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return ReceptorSelectionPolicy(**defaults)


def _source_context(**overrides):
    inspection_report = overrides.pop("inspection_report", None) or _inspection_report()
    resolution = overrides.pop("resolved_selection", None) or _resolution()
    records = overrides.pop("records", None) or _records()
    protein_input = overrides.pop("protein_input", None) or _protein_input()
    policy = overrides.pop("policy", None) or _policy()
    return build_source_context(
        protein_input=protein_input,
        inspection_report=inspection_report,
        policy=policy,
        resolved_selection=resolution,
        raw_size_bytes=overrides.pop("raw_size_bytes", 1000),
        records=records,
    )


def _atom(**overrides) -> AtomObservation:
    defaults = dict(
        ordinal=0, atom_id="A1", model=1, label_asym_id="A", entity_id="E1",
        entity_type="polymer", ccd="ALA", label_seq_id="1", auth_asym_id="A",
        auth_seq_id="1", insertion_token=None,
    )
    defaults.update(overrides)
    return AtomObservation(**defaults)


# --- T021-001-003: closed, strict, immutable DTOs ------------------------


def test_strict_str_rejects_bytes_and_int_without_coercion():  # T021-001
    with pytest.raises(ValidationError):
        AtomObservation(ordinal=0, atom_id=123, model=1, label_asym_id="A", entity_id="E1", entity_type="polymer", ccd="ALA")
    with pytest.raises(ValidationError):
        AtomObservation(ordinal=0, atom_id=b"bytes", model=1, label_asym_id="A", entity_id="E1", entity_type="polymer", ccd="ALA")


def test_strict_int_rejects_bool_and_bool_field_rejects_int():  # T021-002
    with pytest.raises(ValidationError):
        AtomObservation(ordinal=True, atom_id="A1", model=1, label_asym_id="A", entity_id="E1", entity_type="polymer", ccd="ALA")
    with pytest.raises(ValidationError):
        AtomDecision(ordinal=1, atom_id="A1", occurrence_key=_hex(1), retained=1, reason="selected-asym")


def test_closed_immutable_rejects_extra_field_and_mutation():  # T021-003
    decision = AtomDecision(ordinal=1, atom_id="A1", occurrence_key=_hex(1), retained=True, reason="selected-asym")
    with pytest.raises(ValidationError):
        decision.retained = False
    with pytest.raises(ValidationError):
        AtomDecision(ordinal=1, atom_id="A1", occurrence_key=_hex(1), retained=True, reason="selected-asym", extra_field=1)


# --- T021-004-008: TimeUTC and source cross-checks ------------------------


def test_time_utc_rejects_naive_and_impossible_dates():  # T021-004
    assert validate_time_utc(TS) == TS
    with pytest.raises(ValueError):
        validate_time_utc("2026-02-30T00:00:00.000000Z")
    with pytest.raises(ValueError):
        validate_time_utc("2026-01-01 00:00:00")
    with pytest.raises(ValueError):
        validate_time_utc(datetime(2026, 1, 1))


def test_versions_schema_profile_catalogue_engine_divergent_rejected():  # T021-005
    with pytest.raises(ValidationError):
        Engine(name="gemmi", version="0.7.6")
    with pytest.raises(ValidationError):
        Engine(name="other", version="0.7.5")
    with pytest.raises(ValidationError):
        _inspection_report(inspection_schema_version="2.0")
    with pytest.raises(MaterializationConfigurationError):
        _source_context(resolved_selection=_resolution(parser=ParserProvenance(name="gemmi", version="0.7.6")))


def test_source_links_cross_ids_inconsistent_rejected():  # T021-006
    with pytest.raises(MaterializationConfigurationError):
        _source_context(policy=_policy(selection_id="different-sel"))
    with pytest.raises(MaterializationConfigurationError):
        _source_context(resolved_selection=_resolution(raw_checksum=_hex(99)))


def test_inspection_id_payload_must_reproduce_itself():  # T021-007
    tampered_id = _hex(77)
    tampered_report = _inspection_report(inspection_id=tampered_id)
    with pytest.raises(MaterializationConfigurationError):
        _source_context(
            inspection_report=tampered_report,
            resolved_selection=_resolution(inspection_id=tampered_id),
            policy=_policy(inspection_id=tampered_id),
        )


def test_truncated_inspection_rejected_for_non_full_or_parser_limitation():  # T021-008
    with pytest.raises(MaterializationConfigurationError):
        _source_context(inspection_report=_inspection_report(selection_capability="inspection-only"))
    limited = _inspection_report(warnings=(InspectionWarning(code="parser-limitation", detail="x"),))
    with pytest.raises(MaterializationConfigurationError):
        _source_context(inspection_report=limited)


# --- MAJOR-A (corrections 521500c): full §5 cross-checks over the typed
# ProteinInput/ReceptorSelectionPolicy, not just the three previous scalars.


def test_major_a_inspection_parser_divergent_from_resolution_rejected():
    tampered_report = _inspection_report(parser=ParserProvenance(name="gemmi", version="0.7.4"))
    with pytest.raises(MaterializationConfigurationError) as exc_info:
        _source_context(
            inspection_report=tampered_report,
            resolved_selection=_resolution(inspection_id=tampered_report.inspection_id),
            policy=_policy(inspection_id=tampered_report.inspection_id),
        )
    assert exc_info.value.code == "ENGINE_VERSION"


def test_major_a_resolution_format_divergent_from_inspection_rejected():
    with pytest.raises(MaterializationConfigurationError):
        _source_context(resolved_selection=_resolution(structure_format="pdb"))


def test_major_a_acquisition_checksum_divergent_from_observed_rejected():
    with pytest.raises(MaterializationConfigurationError) as exc_info:
        _source_context(protein_input=_protein_input(
            acquisition=AcquisitionRecord(
                method="uploaded", acquired_at=datetime.now(timezone.utc),
                raw_artifact=RawArtifact(path="x", format="mmcif", checksum=_hex(55)),
            )
        ))
    assert exc_info.value.code == "CHECKSUM_MISMATCH"


def test_major_a_policy_protein_input_id_divergent_rejected():
    with pytest.raises(MaterializationConfigurationError) as exc_info:
        _source_context(policy=_policy(protein_input_id="other-protein"))
    assert exc_info.value.code == "IDENTITY_MISMATCH"


# --- MAJOR-A residual (reconciliation 155661115851c2a5fb70f5d905425baeb4d23c9f):
# acquisition/declared/detected format cross-check, with the 020 alias map.


def _source_context_with_formats(acquisition_format: str, declared_format: str):
    return _source_context(
        protein_input=_protein_input(
            acquisition=AcquisitionRecord(
                method="uploaded", acquired_at=datetime.now(timezone.utc),
                raw_artifact=RawArtifact(path="x", format=acquisition_format, checksum=RAW_SHA),
            )
        ),
        inspection_report=_inspection_report(raw_format_declared=declared_format),
    )


def test_major_a_residual_pdb_acquisition_and_declared_with_mmcif_detected_rejected():
    # The 020 inspection's own declared-format-mismatch warning (detected
    # mmcif vs declared pdb) is never read as authorization to materialize
    # with contradictory provenance.
    with pytest.raises(MaterializationConfigurationError) as exc_info:
        _source_context_with_formats(acquisition_format="pdb", declared_format="pdb")
    assert exc_info.value.code == "INVALID_RECORD"


def test_major_a_residual_acquisition_and_declared_format_disagree_rejected():
    with pytest.raises(MaterializationConfigurationError) as exc_info:
        _source_context_with_formats(acquisition_format="mmcif", declared_format="pdb")
    assert exc_info.value.code == "INVALID_RECORD"
    with pytest.raises(MaterializationConfigurationError):  # reverse direction
        _source_context_with_formats(acquisition_format="pdb", declared_format="mmcif")


def test_major_a_residual_unknown_format_label_rejected_on_either_side():
    with pytest.raises(MaterializationConfigurationError) as exc_info:
        _source_context_with_formats(acquisition_format="xyz", declared_format="xyz")
    assert exc_info.value.code == "INVALID_RECORD"
    with pytest.raises(MaterializationConfigurationError):  # unknown only on the declared side
        _source_context_with_formats(acquisition_format="mmcif", declared_format="xyz")


def test_major_a_residual_020_aliases_accepted_both_directions():
    # cif -> mmcif is the 020 alias; both directions must be accepted, and
    # the 020 strip().lower() normalization of the label itself is honored.
    sc1 = _source_context_with_formats(acquisition_format="cif", declared_format="mmcif")
    assert sc1.raw_format == "mmcif"
    sc2 = _source_context_with_formats(acquisition_format="mmcif", declared_format="cif")
    assert sc2.raw_format == "mmcif"
    sc3 = _source_context_with_formats(acquisition_format="  MMCIF  ", declared_format="CIF")
    assert sc3.raw_format == "mmcif"


def test_major_a_residual_coherent_mmcif_control_accepted():
    sc = _source_context_with_formats(acquisition_format="mmcif", declared_format="mmcif")
    assert sc.raw_format == "mmcif"


# --- T021-009-011: model/assembly/empty-selection guards ------------------


def test_selected_model_is_used_exactly_not_first():  # T021-009
    resolution = _resolution(
        resolved_model=5,
        resolved_chains=(ResolvedChainSelection(label_asym_id="A", label_entity_id="E1", model_number=5),),
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (_atom(ordinal=0, model=1), _atom(ordinal=1, atom_id="A2", model=5))
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )
    decisions = {d.ordinal: d for d in plan.atom_decisions}
    assert decisions[0].reason == "model-filter" and not decisions[0].retained
    assert decisions[1].reason == "selected-asym" and decisions[1].retained


def test_resolved_assembly_not_null_rejected():  # T021-010
    with pytest.raises(MaterializationInputError):
        _source_context(resolved_selection=_resolution(resolved_assembly="1"))


def test_empty_resolved_chains_rejected():  # T021-011
    with pytest.raises(MaterializationInputError):
        _source_context(resolved_selection=_resolution(resolved_chains=()))


# --- T021-012-013: label-vs-auth namespace and NFC collisions ------------


def test_native_identity_uses_label_never_auth_namespace():  # T021-012
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    atom = _atom(auth_asym_id="a")  # same chain, case-distinct auth id
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(atom,)
    )
    assert plan.atom_decisions[0].retained and plan.atom_decisions[0].reason == "selected-asym"


def test_nfc_collision_between_distinct_auth_tokens_rejected():  # T021-013
    composed = unicodedata.normalize("NFC", "Á")
    decomposed = "Á"
    assert composed != decomposed and unicodedata.normalize("NFC", decomposed) == composed

    resolution = _resolution(
        resolved_chains=(
            ResolvedChainSelection(label_asym_id="P1", label_entity_id="E1", model_number=1),
            ResolvedChainSelection(label_asym_id="P2", label_entity_id="E1", model_number=1),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0, atom_id="a1", label_asym_id="P1", auth_asym_id=composed),
        _atom(ordinal=1, atom_id="a2", label_asym_id="P2", auth_asym_id=decomposed),
    )
    with pytest.raises(MaterializationInputError):
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
        )


# --- T021-014-020: S/D/U table over polymer and non-polymer asyms --------


def test_polymer_outside_selection_becomes_candidate_chain_complement_exclusion():  # T021-014
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    atoms = (_atom(ordinal=0), _atom(ordinal=1, atom_id="b1", label_asym_id="B", entity_id="E2"))
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )
    assert len(plan.plan.exclusions) == 1
    assert plan.plan.exclusions[0].label_asym_id == "B"
    b_decision = next(d for d in plan.atom_decisions if d.atom_id == "b1")
    assert b_decision.reason == "chain-complement" and not b_decision.retained


def test_selection_disposition_conflicts_rejected_on_both_sides():  # T021-015
    resolution_remove_in_s = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(
                label_asym_id="A", ccd_code="ALA", classification="modified-polymer-residue",
                disposition="remove", disposition_source="component_dispositions",
            ),
        )
    )
    sc1 = _source_context(resolved_selection=resolution_remove_in_s)
    with pytest.raises(MaterializationInputError):
        build_materialization_plan(
            source_context=sc1, resolved_selection=resolution_remove_in_s,
            unaddressed_components="preserve", atoms=(_atom(ordinal=0),),
        )

    resolution_keep_outside_s = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(
                label_asym_id="B", ccd_code="GLY", classification="modified-polymer-residue",
                disposition="keep", disposition_source="component_dispositions",
            ),
        )
    )
    sc2 = _source_context(resolved_selection=resolution_keep_outside_s)
    atoms2 = (_atom(ordinal=0), _atom(ordinal=1, atom_id="b1", label_asym_id="B", entity_id="E2", ccd="GLY"))
    with pytest.raises(MaterializationInputError):
        build_materialization_plan(
            source_context=sc2, resolved_selection=resolution_keep_outside_s,
            unaddressed_components="preserve", atoms=atoms2,
        )


def test_nonuniform_decisions_across_ccd_variants_in_same_asym_rejected():  # T021-016
    resolution = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(label_asym_id="L", ccd_code="NAG", classification="other-non-polymer", disposition="remove", disposition_source="component_dispositions"),
            ResolvedComponentDisposition(label_asym_id="L", ccd_code="BMA", classification="other-non-polymer", disposition="keep", disposition_source="component_dispositions"),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0),
        _atom(ordinal=1, atom_id="n1", label_asym_id="L", entity_id="EL", entity_type="branched", ccd="NAG", label_seq_id="1"),
        _atom(ordinal=2, atom_id="n2", label_asym_id="L", entity_id="EL", entity_type="branched", ccd="BMA", label_seq_id="2"),
    )
    with pytest.raises(MaterializationInputError):
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
        )


def test_branched_glycan_with_mixed_ccd_decisions_is_never_split():  # T021-017
    resolution = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(label_asym_id="L", ccd_code="NAG", classification="other-non-polymer", disposition="remove", disposition_source="component_dispositions"),
            ResolvedComponentDisposition(label_asym_id="L", ccd_code="BMA", classification="other-non-polymer", disposition="keep", disposition_source="component_dispositions"),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0),
        _atom(ordinal=1, atom_id="n1", label_asym_id="L", entity_id="EL", entity_type="branched", ccd="NAG", label_seq_id="1"),
        _atom(ordinal=2, atom_id="n2", label_asym_id="L", entity_id="EL", entity_type="branched", ccd="BMA", label_seq_id="2"),
    )
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
        )
    assert exc_info.value.code == "NONUNIFORM_ASYM"


def test_all_altloc_and_hd_variants_of_an_occurrence_share_one_uniform_decision():  # T021-018
    resolution = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(label_asym_id="W", ccd_code="HOH", classification="water", disposition="keep", disposition_source="water_disposition"),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0),
        _atom(ordinal=1, atom_id="w1a", label_asym_id="W", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="100"),
        _atom(ordinal=2, atom_id="w1b", label_asym_id="W", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="100"),
    )
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )
    water_decisions = [d for d in plan.atom_decisions if d.atom_id in ("w1a", "w1b")]
    assert len(water_decisions) == 2
    assert all(d.retained for d in water_decisions)
    assert all(d.reason == "disposition-keep" for d in water_decisions)


def test_disposition_key_in_another_model_never_affects_m():  # T021-019
    """ARCH-02 (corrections 521500c): both directions of "a (label_asym_id,
    CCD) disposition key whose CCD is absent from M has no effect on M"
    (Section 7), not merely the model-filter mask T021-019 originally
    exercised alone."""

    resolution = _resolution(resolved_model=1)
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0, model=1),
        _atom(ordinal=1, atom_id="x1", model=2, label_asym_id="X", entity_id="EX", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="1"),
    )
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )
    x_decision = next(d for d in plan.atom_decisions if d.atom_id == "x1")
    assert x_decision.reason == "model-filter" and not x_decision.retained

    # ARCH-02 direction 1: A is selected (S={A}); a "remove" disposition on
    # (A, GLY) where GLY only exists in M=2 must have NO effect on A/ALA
    # retained in M=1.
    resolution_remove_other_model = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(
                label_asym_id="A", ccd_code="GLY", classification="modified-polymer-residue",
                disposition="remove", disposition_source="component_dispositions",
            ),
        )
    )
    sc_remove = _source_context(resolved_selection=resolution_remove_other_model)
    two_model_atoms = (
        _atom(ordinal=0, atom_id="a1", model=1, ccd="ALA"),
        _atom(ordinal=1, atom_id="a2", model=2, ccd="GLY"),
    )
    plan_remove = build_materialization_plan(
        source_context=sc_remove, resolved_selection=resolution_remove_other_model,
        unaddressed_components="preserve", atoms=two_model_atoms,
    )
    decisions_remove = {d.atom_id: d for d in plan_remove.atom_decisions}
    assert decisions_remove["a1"].retained and decisions_remove["a1"].reason == "selected-asym"

    # Control: the same disposition key present in M (ALA, not GLY) still
    # conflicts, as before the fix.
    resolution_remove_in_m = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(
                label_asym_id="A", ccd_code="ALA", classification="modified-polymer-residue",
                disposition="remove", disposition_source="component_dispositions",
            ),
        )
    )
    sc_remove_in_m = _source_context(resolved_selection=resolution_remove_in_m)
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc_remove_in_m, resolved_selection=resolution_remove_in_m,
            unaddressed_components="preserve", atoms=two_model_atoms,
        )
    assert exc_info.value.code == "SELECTION_CONFLICT"

    # ARCH-02 direction 2: B is outside S; a "keep" disposition on
    # (B, MSE) where MSE only exists in M=2 must have NO effect - B/ALA in
    # M=1 is still excluded wholly by chain-complement.
    resolution_keep_other_model = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(
                label_asym_id="B", ccd_code="MSE", classification="modified-polymer-residue",
                disposition="keep", disposition_source="component_dispositions",
            ),
        )
    )
    sc_keep = _source_context(resolved_selection=resolution_keep_other_model)
    b_atoms = (
        _atom(ordinal=0, atom_id="a1", model=1, ccd="ALA"),
        _atom(ordinal=1, atom_id="b1", model=1, label_asym_id="B", entity_id="E2", ccd="ALA", auth_asym_id="B"),
        _atom(ordinal=2, atom_id="b2", model=2, label_asym_id="B", entity_id="E2", ccd="MSE", auth_asym_id="B"),
    )
    plan_keep = build_materialization_plan(
        source_context=sc_keep, resolved_selection=resolution_keep_other_model,
        unaddressed_components="preserve", atoms=b_atoms,
    )
    decisions_keep = {d.atom_id: d for d in plan_keep.atom_decisions}
    assert not decisions_keep["b1"].retained and decisions_keep["b1"].reason == "chain-complement"

    # Control: the same disposition key present in M (ALA, not MSE) still
    # conflicts, as before the fix.
    resolution_keep_in_m = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(
                label_asym_id="B", ccd_code="ALA", classification="modified-polymer-residue",
                disposition="keep", disposition_source="component_dispositions",
            ),
        )
    )
    sc_keep_in_m = _source_context(resolved_selection=resolution_keep_in_m)
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc_keep_in_m, resolved_selection=resolution_keep_in_m,
            unaddressed_components="preserve", atoms=b_atoms,
        )
    assert exc_info.value.code == "SELECTION_CONFLICT"


def test_unaddressed_components_policy_controls_outcome_with_no_disposition():  # T021-020
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    atoms = (_atom(ordinal=0), _atom(ordinal=1, atom_id="w1", label_asym_id="W", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="1"))

    plan_preserve = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )
    w_decision = next(d for d in plan_preserve.atom_decisions if d.atom_id == "w1")
    assert w_decision.retained and w_decision.reason == "unaddressed-preserve"

    with pytest.raises(MaterializationInputError):
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="require-disposition", atoms=atoms
        )

    resolution_s = _resolution(
        resolved_chains=(
            ResolvedChainSelection(label_asym_id="A", label_entity_id="E1", model_number=1),
            ResolvedChainSelection(label_asym_id="W", label_entity_id="EW", model_number=1),
        )
    )
    sc_s = _source_context(resolved_selection=resolution_s)
    plan_s = build_materialization_plan(
        source_context=sc_s, resolved_selection=resolution_s, unaddressed_components="require-disposition", atoms=atoms
    )
    w_decision_s = next(d for d in plan_s.atom_decisions if d.atom_id == "w1")
    assert w_decision_s.retained and w_decision_s.reason == "selected-asym"


# --- T021-021-028: acknowledgement gates over X and O --------------------


def _plan_with_exclusion():
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    atoms = (_atom(ordinal=0), _atom(ordinal=1, atom_id="b1", label_asym_id="B", entity_id="E2"))
    return build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )


def test_ack_x_required_before_declare_but_never_before_preview():  # T021-021
    plan = _plan_with_exclusion()
    assert plan.plan.exclusions  # preview succeeds with no acknowledgement at all
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
            acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
        )


def test_ack_x_stale_hash_rejected():  # T021-022
    plan = _plan_with_exclusion()
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=_hex(999),
            acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
        )


def test_ack_x_empty_rejects_nonnull_acknowledgement():  # T021-023
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
    )
    assert not plan.plan.exclusions
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=plan.plan.exclusions_key,
            acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
        )


def test_nucleic_chain_in_recognized_x_is_excluded_wholly_like_any_polymer():  # T021-024
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0),
        _atom(ordinal=1, atom_id="n1", label_asym_id="N", entity_id="EN", ccd="DA", label_seq_id="1"),
        _atom(ordinal=2, atom_id="n2", label_asym_id="N", entity_id="EN", ccd="DA", label_seq_id="2"),
    )
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )
    assert plan.plan.exclusions and plan.plan.exclusions[0].label_asym_id == "N"
    assert plan.plan.exclusions[0].atom_count == 2
    record = build_policy_record(
        materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=plan.plan.exclusions_key,
        acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
    )
    assert record.excluded_polymer_asym_count == 1


def _plan_with_outside():
    resolution = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(label_asym_id="W", ccd_code="HOH", classification="water", disposition="keep", disposition_source="water_disposition"),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (_atom(ordinal=0), _atom(ordinal=1, atom_id="w1", label_asym_id="W", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="1"))
    return build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )


def test_ack_o_required_before_declare():  # T021-025
    plan = _plan_with_outside()
    assert plan.plan.outside
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
            acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
        )


def test_ack_o_stale_or_incomplete_hash_rejected():  # T021-026
    plan = _plan_with_outside()
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
            acknowledge_outside_key=_hex(999), declared_by="scientist", declared_at=TS,
        )


def test_ack_o_empty_rejects_nonnull_acknowledgement():  # T021-027
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
    )
    assert not plan.plan.outside
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
            acknowledge_outside_key=plan.plan.outside_key, declared_by="scientist", declared_at=TS,
        )


def test_outside_set_change_with_same_visible_names_changes_hash_and_invalidates_ack():  # T021-028
    resolution = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(label_asym_id="W", ccd_code="HOH", classification="water", disposition="keep", disposition_source="water_disposition"),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    atoms_v1 = (_atom(ordinal=0), _atom(ordinal=1, atom_id="w1", label_asym_id="W", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="1"))
    atoms_v2 = (_atom(ordinal=0), _atom(ordinal=1, atom_id="w1-changed", label_asym_id="W", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="1"))
    plan1 = build_materialization_plan(source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms_v1)
    plan2 = build_materialization_plan(source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms_v2)
    assert plan1.plan.outside_key != plan2.plan.outside_key
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan2, acknowledge_plan_key=plan2.plan_key, acknowledge_exclusions_key=None,
            acknowledge_outside_key=plan1.plan.outside_key, declared_by="scientist", declared_at=TS,
        )


# --- T021-029-035: declared_by, policy_id/request_key canonicalization ---


def test_declared_by_blank_or_control_chars_rejected_unicode_preserved():  # T021-029
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
    )
    with pytest.raises(ValidationError):
        build_policy_record(materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None, acknowledge_outside_key=None, declared_by="   ", declared_at=TS)
    with pytest.raises(ValidationError):
        build_policy_record(materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None, acknowledge_outside_key=None, declared_by="bad\x00name", declared_at=TS)
    # strict=True rejects bytes outright for declared_by - exercised directly
    # at the DTO boundary (reviewer T021-029 note); build_policy_record's own
    # compute_policy_id hashes declared_by before the DTO is constructed, so
    # a raw TypeError from json.dumps - not ValidationError - is what a bytes
    # value hitting that function first actually raises.
    with pytest.raises(TypeError):
        build_policy_record(materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None, acknowledge_outside_key=None, declared_by=b"bytes-name", declared_at=TS)
    record = build_policy_record(materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None, acknowledge_outside_key=None, declared_by="Cientista Ü", declared_at=TS)
    assert record.declared_by == "Cientista Ü"


def test_policy_idempotence_preserves_record_and_timestamp_on_repeat():  # T021-030
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
    )
    first = build_policy_record(materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None, acknowledge_outside_key=None, declared_by="scientist", declared_at=TS)
    second = build_policy_record(
        materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None, acknowledge_outside_key=None,
        declared_by="scientist", declared_at="2099-01-01T00:00:00.000000Z", existing=first,
    )
    assert second.policy_id == first.policy_id
    assert second.declared_at == first.declared_at


def test_same_vector_under_two_hash_kinds_yields_different_keys():  # T021-031
    vector = [{"a": 1}]
    key1 = compute_materialization_hash({"kind": "kind-one", "version": "1.0", "items": vector})
    key2 = compute_materialization_hash({"kind": "kind-two", "version": "1.0", "items": vector})
    assert key1 != key2


def test_canonical_json_sorts_object_keys_but_preserves_list_order():  # T021-032
    assert compute_materialization_hash({"a": 1, "b": 2}) == compute_materialization_hash({"b": 2, "a": 1})
    assert compute_materialization_hash({"x": [1, 2]}) != compute_materialization_hash({"x": [2, 1]})


def test_hash_rejects_float_and_nan_payload():  # T021-033
    with pytest.raises(ValueError):
        compute_materialization_hash({"kind": "k", "version": "1.0", "value": 1.0})
    with pytest.raises(ValueError):
        compute_materialization_hash({"kind": "k", "version": "1.0", "value": float("nan")})


def test_policy_id_changes_with_actor_or_ack_but_never_with_timestamp():  # T021-034
    base = dict(plan_key=_hex(1), acknowledge_plan_key=_hex(1), acknowledge_exclusions_key=None, acknowledge_outside_key=None, declared_by="scientist")
    id1 = compute_policy_id(**base)
    id2 = compute_policy_id(**{**base, "declared_by": "other"})
    id3 = compute_policy_id(**{**base, "acknowledge_exclusions_key": _hex(2)})
    assert id1 != id2 and id1 != id3
    assert "declared_at" not in compute_policy_id.__code__.co_varnames


def test_request_key_changes_with_policy_or_version_fields_but_isolates_application_version():  # T021-035
    engine = Engine(name="gemmi", version="0.7.5")
    base = dict(
        policy_id=_hex(1), policy_file_sha256=_hex(2), source_context_key=_hex(3), plan_key=_hex(4),
        profile="1.0", algorithm="1.0", catalogue_sha256=CATALOGUE_SHA256, engine=engine,
    )
    key1 = compute_request_key(**base, application_version="1.0.0")
    key2 = compute_request_key(**base, application_version="1.0.1")
    key3 = compute_request_key(**{**base, "policy_file_sha256": _hex(9)}, application_version="1.0.0")
    assert key1 != key2
    assert key1 != key3


# --- T021-036-040: occurrence identity, author coincidence, boundaries ---


def test_water_occurrence_distinguished_by_auth_seq_or_rejected_when_insufficient():  # T021-036
    identity1 = OccurrenceIdentity(model=1, label_asym_id="W", entity_id="EW", ccd="HOH", label_seq_id=None, auth_asym_id="W", auth_seq_id="100", insertion_token=None)
    identity2 = OccurrenceIdentity(model=1, label_asym_id="W", entity_id="EW", ccd="HOH", label_seq_id=None, auth_asym_id="W", auth_seq_id="101", insertion_token=None)
    assert compute_occurrence_key(identity1) != compute_occurrence_key(identity2)
    with pytest.raises(ValidationError):
        OccurrenceIdentity(model=1, label_asym_id="W", entity_id="EW", ccd="HOH", label_seq_id=None, auth_asym_id="W", auth_seq_id=None, insertion_token=None)


def test_outside_author_id_coincidence_classification():  # T021-037
    resolution = _resolution(
        resolved_chains=(
            ResolvedChainSelection(label_asym_id="A", label_entity_id="E1", model_number=1),
            ResolvedChainSelection(label_asym_id="A2", label_entity_id="E1", model_number=1),
            ResolvedChainSelection(label_asym_id="A3", label_entity_id="E1", model_number=1),
        ),
        resolved_component_dispositions=tuple(
            ResolvedComponentDisposition(label_asym_id=asym, ccd_code="HOH", classification="water", disposition="keep", disposition_source="water_disposition")
            for asym in ("W1", "W2", "W3", "W4", "W5")
        ),
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0, atom_id="a1", label_asym_id="A", auth_asym_id="A"),
        _atom(ordinal=1, atom_id="a2", label_asym_id="A2", auth_asym_id="A2"),
        _atom(ordinal=2, atom_id="a3", label_asym_id="A3", auth_asym_id="M"),
        _atom(ordinal=3, atom_id="c1", label_asym_id="C", entity_id="E2", auth_asym_id="C"),
        _atom(ordinal=4, atom_id="c2", label_asym_id="C2", entity_id="E2", auth_asym_id="M"),
        _atom(ordinal=5, atom_id="w1", label_asym_id="W1", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="1", auth_asym_id="A"),
        _atom(ordinal=6, atom_id="w2", label_asym_id="W2", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="2", auth_asym_id="C"),
        _atom(ordinal=7, atom_id="w3", label_asym_id="W3", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="3", auth_asym_id="Z"),
        _atom(ordinal=8, atom_id="w4", label_asym_id="W4", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="4", auth_asym_id=None),
        _atom(ordinal=9, atom_id="w5", label_asym_id="W5", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="5", auth_asym_id="M"),
    )
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )
    outside_by_asym = {entry.identity.label_asym_id: entry for entry in plan.plan.outside}
    assert outside_by_asym["W1"].author_id_coincidence == "retained-polymer"
    assert outside_by_asym["W2"].author_id_coincidence == "excluded-polymer"
    assert outside_by_asym["W3"].author_id_coincidence == "none"
    assert outside_by_asym["W4"].author_id_coincidence == "unavailable"
    assert outside_by_asym["W5"].author_id_coincidence == "multiple"


def test_preview_never_asserts_consent_but_declare_requires_it():  # T021-038
    plan = _plan_with_exclusion()
    assert "acknowledge_plan_key" not in PlanBinding.model_fields
    assert "commit_state" not in PlanBinding.model_fields
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
            acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
        )


def test_unknown_or_unrecognized_entity_type_in_m_rejected():  # T021-039
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    atoms = (_atom(ordinal=0, entity_type="unknown"),)
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
        )
    assert exc_info.value.code == "UNSUPPORTED_REQUEST"


def test_domain_module_imports_no_gemmi_and_performs_no_file_io():  # T021-040
    import re

    import molecular.receptor_materialization as module

    assert not hasattr(module, "gemmi")
    assert not hasattr(module, "rdkit")
    source = module.__file__ and open(module.__file__, encoding="utf-8").read()
    assert not re.search(r"^\s*(import gemmi|from gemmi|import rdkit|from rdkit)\b", source, re.MULTILINE)
    assert not re.search(r"\bopen\s*\(", source)
    assert not re.search(r"\bPath\s*\(", source)


def test_arch01_domain_module_is_importable_and_pure_functional_with_gemmi_and_rdkit_blocked():
    """ARCH-01 (corrections 521500c): the static text check above does not
    catch a transitive import (`receptor_materialization` -> `receptor_
    inspection` -> `receptor_parsing` -> `import gemmi`), which the
    reviewer reproduced directly. The real proof is dynamic: a fresh
    subprocess with Gemmi *and* RDKit genuinely blocked (no stub object at
    all) must still import this module and run its pure DTO/hash/planner
    logic to completion."""

    script = textwrap.dedent(
        """
        import sys
        sys.modules["gemmi"] = None
        sys.modules["rdkit"] = None

        import molecular.receptor_materialization as rm

        class FakeChain:
            def __init__(self, label_asym_id):
                self.label_asym_id = label_asym_id

        class FakeResolution:
            resolved_model = 1
            resolved_chains = (FakeChain("A"),)
            resolved_component_dispositions = ()

        atom = rm.AtomObservation(
            ordinal=0, atom_id="A1", model=1, label_asym_id="A", entity_id="E1",
            entity_type="polymer", ccd="ALA", label_seq_id="1",
        )
        h = rm.compute_materialization_hash({"kind": "k", "version": "1.0"})
        assert len(h) == 64

        sc = rm.SourceContext(
            protein_input_id="protein-1", selection_id="sel-1", inspection_id="a" * 64, model=1,
            records=(
                rm.RecordSnapshot(kind="protein", id="protein-1", sha256="1" * 64, size_bytes=1, schema_version=None),
                rm.RecordSnapshot(kind="inspection", id="a" * 64, sha256="2" * 64, size_bytes=1, schema_version="1.0"),
                rm.RecordSnapshot(kind="selection", id="sel-1", sha256="3" * 64, size_bytes=1, schema_version="1.0"),
                rm.RecordSnapshot(kind="resolution", id="sel-1", sha256="4" * 64, size_bytes=1, schema_version="1.0"),
            ),
            raw_sha256="b" * 64, raw_size_bytes=10, parser=rm.Engine(),
        )
        plan = rm.build_materialization_plan(
            source_context=sc, resolved_selection=FakeResolution(), unaddressed_components="preserve", atoms=(atom,)
        )
        assert plan.plan_key
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


# --- T021-159-161: application_version isolation, decision sensitivity, catalogue DTO ---


def test_application_version_alone_does_not_change_plan_or_policy_id():  # T021-159
    engine = Engine(name="gemmi", version="0.7.5")
    base = dict(
        policy_id=_hex(1), policy_file_sha256=_hex(2), source_context_key=_hex(3), plan_key=_hex(4),
        profile="1.0", algorithm="1.0", catalogue_sha256=CATALOGUE_SHA256, engine=engine,
    )
    key1 = compute_request_key(**base, application_version="1.0.0")
    key2 = compute_request_key(**base, application_version="1.0.1")
    assert key1 != key2
    assert "application_version" not in PlanBinding.model_fields
    assert "application_version" not in PolicyRecord.model_fields


def test_profile_algorithm_or_decision_changes_invalidate_plan_key_and_old_ack():  # T021-160
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    atoms_v1 = (_atom(ordinal=0),)
    atoms_v2 = (_atom(ordinal=0, atom_id="A1-renamed"),)
    plan1 = build_materialization_plan(source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms_v1)
    plan2 = build_materialization_plan(source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms_v2)
    assert plan1.plan_key != plan2.plan_key
    with pytest.raises(MaterializationInputError):
        build_policy_record(
            materialization_plan=plan2, acknowledge_plan_key=plan1.plan_key, acknowledge_exclusions_key=None,
            acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
        )


def test_category_omit_with_empty_arrays_is_valid_without_a_sentinel():  # T021-161
    decision = CategoryDecision(
        category="_pdbx_validate_close_contact", action="omit",
        omitted_row_ordinals=(0, 1), preserved_tags=(), omitted_tags=(), generated_tags=(),
        source_rows_key=_hex(1),
    )
    assert decision.action == "omit" and decision.omitted_tags == ()

    with pytest.raises(ValidationError):
        CategoryDecision(
            category="_struct_ncs_oper", action="guard_then_omit",
            omitted_tags=(), preserved_tags=(), generated_tags=(), source_rows_key=_hex(2),
        )

    guarded = CategoryDecision(
        category="_struct_ncs_oper", action="guard_then_omit",
        omitted_tags=("id", "matrix[1][1]"), preserved_tags=(), generated_tags=(), source_rows_key=_hex(3),
    )
    assert guarded.omitted_tags == ("id", "matrix[1][1]")


# --- ARCH-03 (corrections 521500c): filter may legitimately generate tags ---


def test_arch03_filter_action_may_generate_tags_like_catalogue_cell_and_symmetry():
    for category, preserved in (("_cell", ("length_a",)), ("_symmetry", ("space_group_name_h-m",))):
        decision = CategoryDecision(
            category=category, action="filter", kept_row_ordinals=(0,),
            preserved_tags=preserved, generated_tags=("entry_id",), source_rows_key=_hex(4),
        )
        assert decision.generated_tags == ("entry_id",)

    # filter must still refuse an overlap between preserve/omit/generate.
    with pytest.raises(ValidationError):
        CategoryDecision(
            category="_cell", action="filter", preserved_tags=("length_a",),
            generated_tags=("length_a",), source_rows_key=_hex(5),
        )


# --- MAJOR-B (corrections 521500c): PolicyRecord self-check and existing
# integrity/equivalence verification before reuse. -------------------------


def _materialization_plan_for_policy_tests() -> MaterializationPlan:
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    return build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
    )


def test_major_b_policy_record_rejects_arbitrary_self_declared_policy_id():
    plan = _materialization_plan_for_policy_tests()
    record = build_policy_record(
        materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
        acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
    )
    with pytest.raises(ValidationError):
        PolicyRecord(**{**{field: getattr(record, field) for field in PolicyRecord.model_fields}, "policy_id": _hex(999)})


def test_major_b_tampered_existing_with_same_policy_id_is_rejected_not_silently_reused():
    plan = _materialization_plan_for_policy_tests()
    first = build_policy_record(
        materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
        acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
    )
    tampered = PolicyRecord(
        **{**{field: getattr(first, field) for field in PolicyRecord.model_fields}, "unaddressed_components": "require-disposition"}
    )
    with pytest.raises(MaterializationConfigurationError) as exc_info:
        build_policy_record(
            materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
            acknowledge_outside_key=None, declared_by="scientist", declared_at="2099-01-01T00:00:00.000000Z",
            existing=tampered,
        )
    assert exc_info.value.code == "SOURCE_CHANGED"

    # A legitimate, byte-identical repeat (declared_at aside) still reuses
    # the existing record and its original timestamp (T021-030 preserved).
    second = build_policy_record(
        materialization_plan=plan, acknowledge_plan_key=plan.plan_key, acknowledge_exclusions_key=None,
        acknowledge_outside_key=None, declared_by="scientist", declared_at="2099-01-01T00:00:00.000000Z",
        existing=first,
    )
    assert second.declared_at == first.declared_at


# --- MINOR-1 (corrections 521500c): planner guards over the received atom
# vector itself - uniqueness, unambiguous asym/entity linkage, S resolved
# in M. ----------------------------------------------------------------------


def test_minor1_duplicate_ordinal_rejected():
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve",
            atoms=(_atom(ordinal=0, atom_id="a1"), _atom(ordinal=0, atom_id="a2")),
        )
    assert exc_info.value.code == "INVALID_ID"


def test_minor1_duplicate_atom_id_rejected():
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve",
            atoms=(_atom(ordinal=0, atom_id="a1"), _atom(ordinal=1, atom_id="a1", label_seq_id="2", auth_seq_id="2")),
        )
    assert exc_info.value.code == "INVALID_ID"


def test_minor1_contradictory_entity_id_within_same_asym_rejected():
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve",
            atoms=(
                _atom(ordinal=0, atom_id="a1", entity_id="E1"),
                _atom(ordinal=1, atom_id="a2", entity_id="E9", label_seq_id="2", auth_seq_id="2"),
            ),
        )
    assert exc_info.value.code == "INVALID_ID"


def test_minor1_selected_asym_absent_from_m_rejected():
    resolution = _resolution(
        resolved_chains=(
            ResolvedChainSelection(label_asym_id="A", label_entity_id="E1", model_number=1),
            ResolvedChainSelection(label_asym_id="Q", label_entity_id="E1", model_number=1),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve",
            atoms=(_atom(ordinal=0, atom_id="a1"),),
        )
    assert exc_info.value.code == "INVALID_ID"


def test_minor1_valid_multimodel_vector_with_unique_ordinals_and_ids_is_accepted():
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve",
        atoms=(
            _atom(ordinal=0, atom_id="a1", model=1),
            _atom(ordinal=1, atom_id="a1-m2", model=2),
        ),
    )
    assert {d.ordinal for d in plan.atom_decisions} == {0, 1}
    assert [d.ordinal for d in plan.atom_decisions] == sorted(d.ordinal for d in plan.atom_decisions)


# --- MINOR-2 (corrections 521500c): NFC injectivity across every auth
# token in M's correspondence, water/ligand included - not just polymers. --


def test_minor2_water_nfd_auth_id_colliding_with_polymer_nfc_auth_id_is_rejected():
    composed, decomposed = unicodedata.normalize("NFC", "Á"), "Á"
    resolution = _resolution(
        resolved_component_dispositions=(
            ResolvedComponentDisposition(label_asym_id="W", ccd_code="HOH", classification="water", disposition="keep", disposition_source="water_disposition"),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0, atom_id="a1", auth_asym_id=composed),
        _atom(ordinal=1, atom_id="w1", label_asym_id="W", entity_id="EW", entity_type="water", ccd="HOH", label_seq_id=None, auth_seq_id="1", auth_asym_id=decomposed),
    )
    with pytest.raises(MaterializationInputError) as exc_info:
        build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
        )
    assert exc_info.value.code == "UNSUPPORTED_REQUEST"


def test_minor2_same_literal_auth_token_across_distinct_asyms_remains_legitimate():
    # Unchanged behaviour: this is exactly T021-037's "multiple" case,
    # reasserted here to pin that MINOR-2's broadened check never treats a
    # repeated literal token as a collision.
    resolution = _resolution(
        resolved_chains=(
            ResolvedChainSelection(label_asym_id="A", label_entity_id="E1", model_number=1),
            ResolvedChainSelection(label_asym_id="A2", label_entity_id="E1", model_number=1),
        )
    )
    sc = _source_context(resolved_selection=resolution)
    atoms = (
        _atom(ordinal=0, atom_id="a1", label_asym_id="A", auth_asym_id="M"),
        _atom(ordinal=1, atom_id="a2", label_asym_id="A2", auth_asym_id="M"),
    )
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms
    )
    assert {d.atom_id for d in plan.atom_decisions} == {"a1", "a2"}


# --- MINOR-3 (corrections 521500c): CategoryDecision local invariants ----


def test_minor3_omit_and_guard_then_omit_never_retain_a_row():
    with pytest.raises(ValidationError):
        CategoryDecision(category="_struct_conf", action="omit", kept_row_ordinals=(0,), source_rows_key=_hex(6))
    with pytest.raises(ValidationError):
        CategoryDecision(
            category="_struct_ncs_oper", action="guard_then_omit", kept_row_ordinals=(0,),
            omitted_tags=("id",), source_rows_key=_hex(7),
        )


def test_minor3_negative_row_ordinals_rejected():
    with pytest.raises(ValidationError):
        CategoryDecision(category="_struct_conf", action="filter", kept_row_ordinals=(-1,), source_rows_key=_hex(8))


def test_minor3_noncanonical_category_name_rejected():
    with pytest.raises(ValidationError):
        CategoryDecision(category="_made_up_category", action="omit", source_rows_key=_hex(9))


def test_minor3_omission_family_prefix_only_valid_with_omit_action():
    with pytest.raises(ValidationError):
        CategoryDecision(category="_pdbx_validate_close_contact", action="filter", source_rows_key=_hex(10))
    accepted = CategoryDecision(category="_pdbx_validate_close_contact", action="omit", source_rows_key=_hex(11))
    assert accepted.category == "_pdbx_validate_close_contact"


def test_minor3_source_rows_key_required_for_filter_omit_and_guard_then_omit():
    with pytest.raises(ValidationError):
        CategoryDecision(category="_struct_conf", action="omit")
    # an empty source category still carries a non-null hash of zero rows.
    empty_source_omit = CategoryDecision(category="_struct_conf", action="omit", source_rows_key=_hex(12))
    assert empty_source_omit.source_rows_key == _hex(12)


def test_minor3_row_ordinals_without_source_rows_key_rejected():
    with pytest.raises(ValidationError):
        CategoryDecision(category="_entry", action="generate", generated_tags=("id",), kept_row_ordinals=(0,))


# --- MINOR-3 residual (reconciliation 155661115851c2a5fb70f5d905425baeb4d23c9f):
# intra-tuple ordinal uniqueness and the delimited family-name grammar.


def test_minor3_residual_duplicate_ordinal_within_the_same_tuple_rejected():
    with pytest.raises(ValidationError):
        CategoryDecision(category="_struct_conf", action="filter", kept_row_ordinals=(0, 0), source_rows_key=_hex(30))
    with pytest.raises(ValidationError):
        CategoryDecision(category="_struct_conf", action="omit", omitted_row_ordinals=(0, 0), source_rows_key=_hex(31))
    # the pre-existing cross-tuple overlap and type/negative controls remain in force.
    with pytest.raises(ValidationError):
        CategoryDecision(category="_struct_conf", action="filter", kept_row_ordinals=(0,), omitted_row_ordinals=(0,), source_rows_key=_hex(32))
    with pytest.raises(ValidationError):
        CategoryDecision(category="_struct_conf", action="filter", kept_row_ordinals=(-1,), source_rows_key=_hex(33))
    valid = CategoryDecision(category="_struct_conf", action="filter", kept_row_ordinals=(0, 1), source_rows_key=_hex(34))
    assert valid.kept_row_ordinals == (0, 1)


def test_minor3_residual_malformed_omission_family_names_rejected():
    for malformed_category in (
        "_em_example\n",       # bare prefix family name followed by a trailing LF
        "_em_",                 # bare prefix, no category characters at all
        "_pdbx_nmr_ x",         # embedded space
        "_em_entity.id",        # an item name (contains '.'), not a category name
        "_EM_EXAMPLE",          # uppercase
        "_pdbx_validate_\x00x",  # control character
    ):
        with pytest.raises(ValidationError):
            CategoryDecision(category=malformed_category, action="omit", source_rows_key=_hex(35))


def test_minor3_residual_valid_omission_family_names_in_all_three_families_accepted():
    for category in ("_em_example", "_pdbx_nmr_example", "_pdbx_validate_close_contact", "_pdbx_validate_torsion_2"):
        decision = CategoryDecision(category=category, action="omit", source_rows_key=_hex(36))
        assert decision.category == category
    # controls that must remain valid: omit with empty arrays + source,
    # filter with a generated tag, generate without a source (null allowed).
    omit_control = CategoryDecision(category="_struct_conf", action="omit", source_rows_key=_hex(37))
    assert omit_control.source_rows_key == _hex(37)
    filter_control = CategoryDecision(category="_cell", action="filter", preserved_tags=("length_a",), generated_tags=("entry_id",), source_rows_key=_hex(38))
    assert filter_control.generated_tags == ("entry_id",)
    generate_control = CategoryDecision(category="_entry", action="generate", generated_tags=("id",))
    assert generate_control.source_rows_key is None


# --- MINOR-4 (corrections 521500c): catalogue pinned in PlanBinding and
# PolicyRecord; version/engine/decision sensitivity. ------------------------


def test_minor4_divergent_catalogue_rejected_in_planbinding_and_policyrecord():
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
    ).plan
    with pytest.raises(ValidationError):
        PlanBinding(**{**{field: getattr(plan, field) for field in PlanBinding.model_fields}, "catalogue_sha256": _hex(13)})

    record = build_policy_record(
        materialization_plan=build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
        ),
        acknowledge_plan_key=build_materialization_plan(
            source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
        ).plan_key,
        acknowledge_exclusions_key=None, acknowledge_outside_key=None, declared_by="scientist", declared_at=TS,
    )
    with pytest.raises(ValidationError):
        PolicyRecord(**{**{field: getattr(record, field) for field in PolicyRecord.model_fields}, "catalogue_sha256": _hex(14)})


def test_minor4_engine_and_profile_divergence_rejected_by_closed_literals():
    with pytest.raises(ValidationError):
        Engine(name="gemmi", version="0.7.4")
    resolution = _resolution()
    sc = _source_context(resolved_selection=resolution)
    plan = build_materialization_plan(
        source_context=sc, resolved_selection=resolution, unaddressed_components="preserve", atoms=(_atom(ordinal=0),)
    ).plan
    with pytest.raises(ValidationError):
        PlanBinding(**{**{field: getattr(plan, field) for field in PlanBinding.model_fields}, "profile": "2.0"})
    with pytest.raises(ValidationError):
        PlanBinding(**{**{field: getattr(plan, field) for field in PlanBinding.model_fields}, "algorithm": "2.0"})


def test_minor4_request_key_envelope_is_sensitive_to_profile_algorithm_catalogue_and_engine():
    # T021-035/160 (MINOR-4): request_key's own envelope must change when
    # profile/algorithm/catalogue/engine change, not only policy_file_sha256
    # and application_version.
    base = dict(
        policy_id=_hex(20), policy_file_sha256=_hex(21), source_context_key=_hex(22), plan_key=_hex(23),
        profile="1.0", algorithm="1.0", catalogue_sha256=CATALOGUE_SHA256, engine=Engine(),
        application_version="1.0.0",
    )
    baseline_key = compute_request_key(**base)
    assert compute_request_key(**{**base, "profile": "9.9"}) != baseline_key
    assert compute_request_key(**{**base, "algorithm": "9.9"}) != baseline_key
    assert compute_request_key(**{**base, "catalogue_sha256": _hex(999)}) != baseline_key
    assert compute_request_key(**{**base, "engine": Engine.model_construct(name="other", version="9.9")}) != baseline_key


# --- MINOR-5 (corrections 521500c): a non-mmCIF inspection is a record
# configuration problem, never UNSUPPORTED_PLATFORM (host/profile only). ---


def test_minor5_non_mmcif_format_uses_invalid_record_not_unsupported_platform():
    pdb_report = _inspection_report(structure_format="pdb")
    with pytest.raises(MaterializationConfigurationError) as exc_info:
        _source_context(
            inspection_report=pdb_report,
            resolved_selection=_resolution(structure_format="pdb", inspection_id=pdb_report.inspection_id),
            policy=_policy(inspection_id=pdb_report.inspection_id),
        )
    assert exc_info.value.code == "INVALID_RECORD"
    assert exc_info.value.code != "UNSUPPORTED_PLATFORM"


# --- MINOR-6 (corrections 521500c): the closed error-code set is enforced
# by an explicit `raise`, surviving `python -O` (where `assert` is a no-op). -


def test_minor6_unknown_error_code_rejected_even_under_python_dash_o():
    script = textwrap.dedent(
        """
        import molecular.receptor_materialization as rm
        for cls in (rm.MaterializationInputError, rm.MaterializationConfigurationError):
            try:
                cls("NOT_A_CODE", "message")
            except ValueError:
                pass
            else:
                raise SystemExit("accepted an unknown code: " + cls.__name__)
        print("OK")
        """
    )
    normal = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert normal.returncode == 0, normal.stderr
    assert normal.stdout.strip() == "OK"

    optimized = subprocess.run([sys.executable, "-O", "-B", "-c", script], capture_output=True, text=True)
    assert optimized.returncode == 0, optimized.stderr
    assert optimized.stdout.strip() == "OK"


def test_minor6_valid_codes_still_preserve_code_and_phase():
    error = MaterializationInputError("INVALID_ID", "message", phase="plan")
    assert error.code == "INVALID_ID" and error.phase == "plan"
    config_error = MaterializationConfigurationError("ENGINE_VERSION", "message", phase="metadata")
    assert config_error.code == "ENGINE_VERSION" and config_error.phase == "metadata"


# --- MINOR-7 residual (corrections 521500c + residual reconciliation
# 155661115851c2a5fb70f5d905425baeb4d23c9f): independent *literal* SHA-256
# digests for all 13 documentary vectors named in
# increments/021/24-checkpoint1-residual-reconciliation/evidence/
# DOCUMENTARY_GOLDEN_VECTORS.json, copied inline here (no fixture file, no
# governance read at test time, per the task's explicit instruction). Each
# literal is asserted against hashlib(canonical bytes) as an internal
# consistency check, and separately against the real production entry
# point - never against a value computed by production itself. ------------

_GOLDEN_CANONICAL_JSON_SHA256 = "14332640256b6a761e453eb0beba1ba7cb1c005ab6d7c01a8eb51dfbe0cc4994"
_GOLDEN_LEGACY_INSPECTION_ID_SHA256 = "1c217d99e7b2b09d9c31a0c8f9daf056c48aa916bdf957cae83df449ad68e80c"
_GOLDEN_SOURCE_CONTEXT_KEY_SHA256 = "783c1d30f99898be83fef40c069294d9219c25db45218474c1fabad7a14520df"
_GOLDEN_OCCURRENCE_KEY_SHA256 = "022d3a1316ffa9b93217161e5e004eef03c3bf500184d06042134bd016269a4e"
_GOLDEN_ATOM_IDS_KEY_SHA256 = "1696038f38afba311d8944dc19e698f2bcbeb2453bd3fb0c5a32b4ccd29445a8"
_GOLDEN_EXCLUSIONS_KEY_SHA256 = "01c461514e8a9915d3fd9feadd2565d11ced78dafb04bc61ba9151d9aa2fe06d"
_GOLDEN_OUTSIDE_KEY_SHA256 = "cafba614ae1dbc9c41f7e727926198d995d9ae3611e09664d7957b407c739e32"
_GOLDEN_ATOM_DECISIONS_KEY_SHA256 = "cc0fb129f56d061a165e94774d323be510a46fa7f22a1cd816734bb0c5849850"
_GOLDEN_CONNECTION_DECISIONS_KEY_SHA256 = "67c0c45369f339bdebbbf2fbc1ee26e20f699f9fe2b93f2edf88c68e3ed85889"
_GOLDEN_CATEGORY_DECISIONS_KEY_SHA256 = "7fc88d1a1d5deec637359849766c989e04832753ce943bb7a9bdccdfd03661df"
_GOLDEN_PLAN_KEY_SHA256 = "00b8734e5e21f85208f9578ef62ffb6e8cf077dc26b771443186bb049805e2da"
_GOLDEN_POLICY_ID_SHA256 = "3bd864722608e2a83ecf71627cd0400380d0ad0828016bb8705c2cd62aae8d02"
_GOLDEN_REQUEST_KEY_SHA256 = "5b1de4a13571f147a84520042b4830235e7b4a611ab7b6b1b972c60027510095"
_GOLDEN_SOURCE_CONTEXT_KEY = _GOLDEN_SOURCE_CONTEXT_KEY_SHA256


def _assert_hand_bytes_hash_to_literal(hand_written_bytes: bytes, literal: str) -> None:
    import hashlib

    assert hashlib.sha256(hand_written_bytes).hexdigest() == literal


def test_minor7_golden_canonical_json_bytes_independent_of_production():
    hand_written_bytes = '{"b":true,"kind":"k","l":[2,1],"n":null,"s":"éÅ","version":"1.0"}'.encode("utf-8")
    _assert_hand_bytes_hash_to_literal(hand_written_bytes, _GOLDEN_CANONICAL_JSON_SHA256)
    payload = {"version": "1.0", "kind": "k", "s": "éÅ", "l": [2, 1], "n": None, "b": True}
    assert compute_materialization_hash(payload) == _GOLDEN_CANONICAL_JSON_SHA256


def test_minor7_golden_legacy_inspection_id_calls_the_new_helper_against_independent_literal():
    """MINOR-7 residual: the superseded version of this test compared the
    OLD 020 helper (`receptor_inspection.compute_inspection_id`) against a
    module constant (`INSPECTION_ID`) also computed by calling that same
    old helper - a tautology that never exercised the NEW pure helper
    (`_compute_legacy_inspection_id`) at all. This version calls the new
    helper directly with the documented vector inputs and compares it to
    an independent literal computed with no production code; the old
    helper is then compared only as an additional, optional control."""

    new_helper_result = _compute_legacy_inspection_id(
        parser_name="gemmi", parser_version="0.7.5", protein_input_id="protein-1",
        raw_checksum="a" * 64, raw_format="mmcif",
    )
    assert new_helper_result == _GOLDEN_LEGACY_INSPECTION_ID_SHA256

    old_helper_result = compute_inspection_id(
        parser_name="gemmi", parser_version="0.7.5", protein_input_id="protein-1",
        raw_checksum="a" * 64, raw_format="mmcif",
    )
    assert old_helper_result == _GOLDEN_LEGACY_INSPECTION_ID_SHA256


def _golden_source_context() -> SourceContext:
    return SourceContext(
        protein_input_id="protein-1", selection_id="sel-1",
        inspection_id="2b367d0134006b35c4f878984e6b0cc613f5a27db5e00bc20b42044381b1b3f6", model=1,
        records=(
            RecordSnapshot(kind="protein", id="protein-1", sha256="1bffa22ef057ae13d72de8812a363194cd35c0fa1b136bd24faa37b322a843b1", size_bytes=66, schema_version=None),
            RecordSnapshot(kind="inspection", id="2b367d0134006b35c4f878984e6b0cc613f5a27db5e00bc20b42044381b1b3f6", sha256="64b982b230587eb5983408d68105abd62c45c879830f53754232694639230d83", size_bytes=124, schema_version="1.0"),
            RecordSnapshot(kind="selection", id="sel-1", sha256="409ea1325548b16b3de5ef6312e43cf70d247a3a30565a5612e01fa2c08a3ba5", size_bytes=64, schema_version="1.0"),
            RecordSnapshot(kind="resolution", id="sel-1", sha256="02fb0364edff743c45709d19f707afd25e242f569109b951992b16abbcbbcf3d", size_bytes=65, schema_version="1.0"),
        ),
        raw_sha256="d020364a30da702323fe02b6b5d10f1861dbb7c6817c61b34a2648037b3b5927", raw_size_bytes=61,
        raw_format="mmcif", inspection_schema="1.0", inspection_profile="1.0",
        parser=Engine(name="gemmi", version="0.7.5"),
    )


def test_minor7_golden_source_context_key_envelope():
    assert compute_source_context_key(_golden_source_context()) == _GOLDEN_SOURCE_CONTEXT_KEY_SHA256


def test_minor7_golden_occurrence_key_envelope_independent_of_production():
    hand_written_bytes = (
        b'{"identity":{"auth_asym_id":"W","auth_seq_id":"17","ccd":"HOH","entity_id":"EW",'
        b'"insertion_token":null,"label_asym_id":"W","label_seq_id":null,"model":1},'
        b'"kind":"receptor-materialization-occurrence","version":"1.0"}'
    )
    _assert_hand_bytes_hash_to_literal(hand_written_bytes, _GOLDEN_OCCURRENCE_KEY_SHA256)
    identity = OccurrenceIdentity(model=1, label_asym_id="W", entity_id="EW", ccd="HOH", label_seq_id=None, auth_asym_id="W", auth_seq_id="17", insertion_token=None)
    assert compute_occurrence_key(identity) == _GOLDEN_OCCURRENCE_KEY_SHA256


def test_minor7_golden_atom_ids_key_preserves_given_order_independent_of_production():
    hand_written_bytes = b'{"atoms":[{"id":"b1","ordinal":1}],"kind":"receptor-materialization-atom-ids","version":"1.0"}'
    _assert_hand_bytes_hash_to_literal(hand_written_bytes, _GOLDEN_ATOM_IDS_KEY_SHA256)
    atoms = (_atom(ordinal=1, atom_id="b1"),)
    assert compute_atom_ids_key(atoms) == _GOLDEN_ATOM_IDS_KEY_SHA256


def test_atom_ids_key_preserves_given_order_against_ordinal_and_id_sorting():
    """R-1 (final focal review edb8764b4d4ee56e49ff849b65b950e5e430fae0;
    architect response 771ae24f6289e41a2f8c7e17906bf294925bd375): the
    single-atom golden above stopped exercising order preservation at all
    once the residual-corrections round replaced the original two-atom,
    out-of-order vector with it - a concrete test regression, not a
    production defect. This restores a genuine two-atom vector whose
    *given* order ((ordinal=3,"a") then (ordinal=1,"Z")) differs from both
    ordinal-ascending and ASCII-id-ascending sort order, so a mutant that
    sorted by either key inside `compute_atom_ids_key` would be caught.
    Both of those wrong orderings happen to coincide on the same
    reordering here - (1,"Z") before (3,"a") - so the single reversed
    literal below discriminates both mutants at once. Expected literals
    are copied from the architect's documentary vector
    (increments/021/28-checkpoint1-r1-test-correction/evidence/
    R1_ORDER_GOLDEN_VECTOR.json), never computed by calling production."""

    given_order_bytes = b'{"atoms":[{"id":"a","ordinal":3},{"id":"Z","ordinal":1}],"kind":"receptor-materialization-atom-ids","version":"1.0"}'
    _assert_hand_bytes_hash_to_literal(given_order_bytes, "96fd3eacd89d16753dbdc5ec22bb79db0740e9f355e6322cb0c4641bc0f3106a")
    reversed_order_bytes = b'{"atoms":[{"id":"Z","ordinal":1},{"id":"a","ordinal":3}],"kind":"receptor-materialization-atom-ids","version":"1.0"}'
    _assert_hand_bytes_hash_to_literal(reversed_order_bytes, "bf820fe14730b4b8ede9d8e007e4c3d32f71def5d8a4747c657d6d76dac562d1")

    given_order = (_atom(ordinal=3, atom_id="a"), _atom(ordinal=1, atom_id="Z"))
    assert compute_atom_ids_key(given_order) == "96fd3eacd89d16753dbdc5ec22bb79db0740e9f355e6322cb0c4641bc0f3106a"

    # A mutant sorting by ordinal, or by atom_id, would both produce this
    # same wrong digest instead - never the literal above.
    reversed_order = (_atom(ordinal=1, atom_id="Z"), _atom(ordinal=3, atom_id="a"))
    assert compute_atom_ids_key(reversed_order) == "bf820fe14730b4b8ede9d8e007e4c3d32f71def5d8a4747c657d6d76dac562d1"


def test_minor7_golden_exclusions_key_envelope():
    xentry = XEntry(model=1, label_asym_id="B", entity_id="E2", atom_count=1, atom_ids_key=_GOLDEN_ATOM_IDS_KEY_SHA256)
    assert compute_exclusions_key(source_context_key=_GOLDEN_SOURCE_CONTEXT_KEY, model=1, entries=(xentry,)) == _GOLDEN_EXCLUSIONS_KEY_SHA256


def test_minor7_golden_outside_key_envelope():
    identity = OccurrenceIdentity(model=1, label_asym_id="W", entity_id="EW", ccd="HOH", label_seq_id=None, auth_asym_id="W", auth_seq_id="17", insertion_token=None)
    oentry = OutsideEntry(
        identity=identity, occurrence_key=_GOLDEN_OCCURRENCE_KEY_SHA256, decision_source="unaddressed-preserve",
        atom_count=1, atom_ids_key="8eeed5d55f10287690bf61adfaffbec360b1dd75af6cb380872e1c3c7fd1eb13",
        author_id_coincidence="none",
    )
    assert compute_outside_key(source_context_key=_GOLDEN_SOURCE_CONTEXT_KEY, model=1, entries=(oentry,)) == _GOLDEN_OUTSIDE_KEY_SHA256


def test_minor7_golden_atom_decisions_key_envelope():
    decisions = (
        AtomDecision(ordinal=0, atom_id="a1", occurrence_key="c7be110af90a4a6028070621a83de375e8fc75c67f16d67f7df84115e66e147d", retained=True, reason="selected-asym"),
        AtomDecision(ordinal=1, atom_id="b1", occurrence_key="c130a69653371b5fd775a994dc3fde9e2f892a147478289054f8cc1b5367bd57", retained=False, reason="chain-complement"),
        AtomDecision(ordinal=2, atom_id="w1", occurrence_key=_GOLDEN_OCCURRENCE_KEY_SHA256, retained=True, reason="unaddressed-preserve"),
    )
    assert compute_atom_decisions_key(source_context_key=_GOLDEN_SOURCE_CONTEXT_KEY, decisions=decisions) == _GOLDEN_ATOM_DECISIONS_KEY_SHA256


def test_minor7_golden_empty_connection_decisions_key_envelope_independent_of_production():
    hand_written_bytes = (
        b'{"decisions":[],"kind":"receptor-materialization-connection-decisions",'
        b'"source_context_key":"' + _GOLDEN_SOURCE_CONTEXT_KEY.encode() + b'","version":"1.0"}'
    )
    _assert_hand_bytes_hash_to_literal(hand_written_bytes, _GOLDEN_CONNECTION_DECISIONS_KEY_SHA256)
    assert compute_connection_decisions_key(source_context_key=_GOLDEN_SOURCE_CONTEXT_KEY) == _GOLDEN_CONNECTION_DECISIONS_KEY_SHA256


def test_minor7_golden_category_decisions_key_envelope():
    decision = CategoryDecision(
        category="_atom_site", action="filter", kept_row_ordinals=(0, 2), omitted_row_ordinals=(1,),
        preserved_tags=("id",), omitted_tags=(), generated_tags=(),
        source_rows_key="69d7a6a8fe0baba92f59dad6a6930ea5e9ad78ce6372833af7cfe1a4a073c2a6",
    )
    assert compute_category_decisions_key(source_context_key=_GOLDEN_SOURCE_CONTEXT_KEY, decisions=(decision,)) == _GOLDEN_CATEGORY_DECISIONS_KEY_SHA256


def test_minor7_golden_plan_key_envelope():
    xentry = XEntry(model=1, label_asym_id="B", entity_id="E2", atom_count=1, atom_ids_key=_GOLDEN_ATOM_IDS_KEY_SHA256)
    identity = OccurrenceIdentity(model=1, label_asym_id="W", entity_id="EW", ccd="HOH", label_seq_id=None, auth_asym_id="W", auth_seq_id="17", insertion_token=None)
    oentry = OutsideEntry(
        identity=identity, occurrence_key=_GOLDEN_OCCURRENCE_KEY_SHA256, decision_source="unaddressed-preserve",
        atom_count=1, atom_ids_key="8eeed5d55f10287690bf61adfaffbec360b1dd75af6cb380872e1c3c7fd1eb13",
        author_id_coincidence="none",
    )
    plan = PlanBinding(
        catalogue_sha256=CATALOGUE_SHA256, engine=Engine(name="gemmi", version="0.7.5"), sources=_golden_source_context(),
        source_context_key=_GOLDEN_SOURCE_CONTEXT_KEY, selected_asym_ids=("A",), unaddressed_components="preserve",
        exclusions=(xentry,), outside=(oentry,),
        exclusions_key=_GOLDEN_EXCLUSIONS_KEY_SHA256, outside_key=_GOLDEN_OUTSIDE_KEY_SHA256,
        atom_decisions_key=_GOLDEN_ATOM_DECISIONS_KEY_SHA256, connection_decisions_key=_GOLDEN_CONNECTION_DECISIONS_KEY_SHA256,
        category_decisions_key=_GOLDEN_CATEGORY_DECISIONS_KEY_SHA256, unavailable_categories=(),
        warnings=("outside-selected-asym-retained",),
    )
    assert compute_plan_key(plan) == _GOLDEN_PLAN_KEY_SHA256


def test_minor7_golden_policy_id_envelope():
    result = compute_policy_id(
        plan_key=_GOLDEN_PLAN_KEY_SHA256, acknowledge_plan_key=_GOLDEN_PLAN_KEY_SHA256,
        acknowledge_exclusions_key=_GOLDEN_EXCLUSIONS_KEY_SHA256, acknowledge_outside_key=_GOLDEN_OUTSIDE_KEY_SHA256,
        declared_by="documentary scientist",
    )
    assert result == _GOLDEN_POLICY_ID_SHA256


def test_minor7_golden_request_key_envelope():
    result = compute_request_key(
        policy_id=_GOLDEN_POLICY_ID_SHA256, policy_file_sha256="9111f3e2210af7359155992b55b689f67f77a195e29e6a979c28a55e679a1d63",
        source_context_key=_GOLDEN_SOURCE_CONTEXT_KEY, plan_key=_GOLDEN_PLAN_KEY_SHA256,
        profile="1.0", algorithm="1.0", catalogue_sha256=CATALOGUE_SHA256, engine=Engine(name="gemmi", version="0.7.5"),
        application_version="0.0.0-documentary",
    )
    assert result == _GOLDEN_REQUEST_KEY_SHA256


# ==========================================================================
# Increment 021, Checkpoint 2: struct_conn/symmetry resolution (Section
# 9/10/10.1) and the 94-category catalogue engine (Section 11). Scenario
# ids below are the CP2 subset of increments/021/34-checkpoint2-
# preparation/evidence/CP2_SCENARIO_SCOPE.csv (79 entries) - each test is
# labeled with the exact id(s) it genuinely proves; a scenario this
# checkpoint does not implement is never tagged here (it stays explicitly
# pending, per the task's own "não convertido em skip aprovado" rule).
# ==========================================================================


def _no_tables() -> SymmetryOperatorTables:
    return SymmetryOperatorTables(symmetry_equiv=None, space_group_symop=None)


def _endpoint(*, asym="A", ccd="ALA", seq="1", atom_id_label="CA", auth_seq="1", symmetry=None) -> RawConnectionEndpointTokens:
    return RawConnectionEndpointTokens(
        label_asym_id=asym, label_comp_id=ccd, label_seq_id=seq, label_atom_id=atom_id_label,
        auth_seq_id=auth_seq, insertion_token=None, altloc_aliases=(None, None, None), symmetry_token=symmetry,
    )


def _connection_scenario(
    *, retained1, retained2, symmetry1, symmetry2, conn_type="disulf",
    tables=None, asym1="A", asym2="B", reason1=None, reason2=None, source_context_key=None,
):
    tables = tables if tables is not None else _no_tables()
    a1 = _atom(ordinal=0, atom_id="a1", label_asym_id=asym1, auth_asym_id=asym1, label_atom_id="CA")
    a2 = _atom(ordinal=1, atom_id="a2", label_asym_id=asym2, auth_asym_id=asym2, entity_id="E2", label_atom_id="CA")
    atoms_in_model = (a1, a2)
    d1 = AtomDecision(
        ordinal=0, atom_id="a1", occurrence_key=_hex(1), retained=retained1,
        reason=reason1 or ("selected-asym" if retained1 else "chain-complement"),
    )
    d2 = AtomDecision(
        ordinal=1, atom_id="a2", occurrence_key=_hex(2), retained=retained2,
        reason=reason2 or ("selected-asym" if retained2 else "chain-complement"),
    )
    atom_decisions_by_ordinal = {0: d1, 1: d2}
    row = RawConnectionRow(
        ordinal=0, id="c1", conn_type=conn_type,
        ptnr1=_endpoint(asym=asym1, symmetry=symmetry1), ptnr2=_endpoint(asym=asym2, symmetry=symmetry2),
        ptnr3_present=False, source_tokens=(("id", "c1"),),
    )
    return resolve_connection_decisions(
        source_context_key=source_context_key or _hex(9), atoms_in_model=atoms_in_model,
        atom_decisions_by_ordinal=atom_decisions_by_ordinal, raw_connections=(row,), tables=tables,
    )


def test_k2_both_identity_preserves():  # T021-045
    decisions = _connection_scenario(retained1=True, retained2=True, symmetry1="1_555", symmetry2="1_555")
    assert decisions[0].action == "preserve"
    assert decisions[0].reason == "both-retained"
    assert decisions[0].symmetry_class == "both-identity"


def _same_nonidentity_tables():
    return SymmetryOperatorTables(symmetry_equiv={"2": "-x,-y,z"}, space_group_symop=None)


def test_k2_same_nonidentity_rejects():  # T021-046
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(
            retained1=True, retained2=True, symmetry1="2_555", symmetry2="2_555", tables=_same_nonidentity_tables(),
        )
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"


def test_k2_different_rejects():  # T021-047
    tables = SymmetryOperatorTables(symmetry_equiv={"2": "-x,-y,z", "3": "-x,y,-z"}, space_group_symop=None)
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(retained1=True, retained2=True, symmetry1="2_555", symmetry2="3_555", tables=tables)
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"


def test_k2_one_null_rejects():  # T021-048
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(retained1=True, retained2=True, symmetry1=None, symmetry2="1_555")
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"


def test_k2_both_null_rejects():  # T021-049
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(retained1=True, retained2=True, symmetry1=None, symmetry2=None)
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"


def test_k1_both_identity_metalc_omits_whole_partner_removed():  # T021-050 (metalc/hydrog half)
    decisions = _connection_scenario(
        retained1=True, retained2=False, symmetry1="1_555", symmetry2="1_555", conn_type="metalc",
    )
    assert decisions[0].action == "omit"
    assert decisions[0].reason == "whole-partner-removed-metalc"
    assert decisions[0].symmetry_class == "both-identity"


def test_k1_both_identity_covalent_type_rejects():  # T021-050 (covalente recusa half)
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(
            retained1=True, retained2=False, symmetry1="1_555", symmetry2="1_555", conn_type="disulf",
        )
    assert excinfo.value.code == "COVALENT_CUT"


@pytest.mark.parametrize(
    "symmetry1,symmetry2,tables,scenario_id",
    [
        ("2_555", "2_555", _same_nonidentity_tables(), "T021-051"),
        ("2_555", "3_555", SymmetryOperatorTables(symmetry_equiv={"2": "-x,-y,z", "3": "-x,y,-z"}, space_group_symop=None), "T021-052"),
        (None, "1_555", None, "T021-053"),
        (None, None, None, "T021-054"),
    ],
)
def test_k1_non_identity_symmetry_rejects(symmetry1, symmetry2, tables, scenario_id):  # T021-051/052/053/054
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(
            retained1=True, retained2=False, symmetry1=symmetry1, symmetry2=symmetry2,
            conn_type="metalc", tables=tables or _no_tables(),
        )
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY", scenario_id


@pytest.mark.parametrize(
    "symmetry1,symmetry2,tables,expected_class,scenario_id",
    [
        ("1_555", "1_555", None, "both-identity", "T021-055"),
        ("2_555", "2_555", _same_nonidentity_tables(), "same-nonidentity", "T021-056"),
        ("2_555", "3_555", SymmetryOperatorTables(symmetry_equiv={"2": "-x,-y,z", "3": "-x,y,-z"}, space_group_symop=None), "different", "T021-057"),
        (None, "1_555", None, "one-null", "T021-058"),
        (None, None, None, "both-null", "T021-059"),
    ],
)
def test_k0_omits_with_audit_for_every_symmetry_class(symmetry1, symmetry2, tables, expected_class, scenario_id):  # T021-055-059
    decisions = _connection_scenario(
        retained1=False, retained2=False, symmetry1=symmetry1, symmetry2=symmetry2,
        conn_type="metalc", tables=tables or _no_tables(),
    )
    assert decisions[0].action == "omit", scenario_id
    assert decisions[0].reason == "both-removed", scenario_id
    assert decisions[0].symmetry_class == expected_class, scenario_id


def test_symmetry_preconditions_malformed_rejects_at_k_positive_but_omits_at_k0():  # T021-060
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(retained1=True, retained2=False, symmetry1="01_555", symmetry2="1_555", conn_type="metalc")
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"

    decisions = _connection_scenario(retained1=False, retained2=False, symmetry1="01_555", symmetry2="1_555")
    assert decisions[0].action == "omit"
    assert decisions[0].symmetry_class == "unsupported-discarded"


def test_identity_no_table_1_555_convention_without_fabricating_operators():  # T021-141
    decisions = _connection_scenario(retained1=True, retained2=True, symmetry1="1_555", symmetry2="1_555", tables=_no_tables())
    assert decisions[0].identity_basis == ("pdbx-1_555-convention", "pdbx-1_555-convention")
    assert decisions[0].action == "preserve"


def test_identity_contradiction_with_tables_present_never_falls_back_to_convention():  # T021-142
    contradictory_tables = SymmetryOperatorTables(symmetry_equiv={"1": None}, space_group_symop=None)
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(
            retained1=True, retained2=True, symmetry1="1_555", symmetry2="1_555", tables=contradictory_tables,
        )
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"


def test_other_operator_without_table_rejects_k_positive_never_uses_1_555_exception():  # T021-143
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(retained1=True, retained2=False, symmetry1="2_555", symmetry2="1_555", conn_type="metalc")
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"


def test_different_image_reason_hydrog_1_555_vs_9_555_rejects_without_claiming_geometric_impossibility():  # T021-144
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _connection_scenario(retained1=True, retained2=False, symmetry1="1_555", symmetry2="9_555", conn_type="hydrog")
    error = excinfo.value
    assert error.code == "UNSUPPORTED_SYMMETRY"
    assert "geometr" not in str(error).lower()


def test_third_partner_present_rejects():  # T021-044
    a1 = _atom(ordinal=0, atom_id="a1", label_asym_id="A", label_atom_id="CA")
    a2 = _atom(ordinal=1, atom_id="a2", label_asym_id="B", entity_id="E2", label_atom_id="CA")
    atom_decisions_by_ordinal = {
        0: AtomDecision(ordinal=0, atom_id="a1", occurrence_key=_hex(1), retained=True, reason="selected-asym"),
        1: AtomDecision(ordinal=1, atom_id="a2", occurrence_key=_hex(2), retained=True, reason="selected-asym"),
    }
    row = RawConnectionRow(
        ordinal=0, id="c1", conn_type="disulf",
        ptnr1=_endpoint(asym="A", symmetry="1_555"), ptnr2=_endpoint(asym="B", symmetry="1_555"),
        ptnr3_present=True, source_tokens=(("id", "c1"),),
    )
    with pytest.raises(MaterializationExecutionError) as excinfo:
        resolve_connection_decisions(
            source_context_key=_hex(9), atoms_in_model=(a1, a2), atom_decisions_by_ordinal=atom_decisions_by_ordinal,
            raw_connections=(row,), tables=_no_tables(),
        )
    assert excinfo.value.code == "UNSUPPORTED_PARTNER"


# --- T021-164-173: IdentityBasis pairs (all K=0) --------------------------


_WITH_OPERATOR_2 = SymmetryOperatorTables(symmetry_equiv={"2": "-x,-y,z"}, space_group_symop=None)


@pytest.mark.parametrize(
    "symmetry1,symmetry2,tables,expected_basis,scenario_id",
    [
        ("1_555", "1_555", _no_tables(), ("pdbx-1_555-convention", "pdbx-1_555-convention"), "T021-164"),
        ("1_555", None, _no_tables(), ("pdbx-1_555-convention", "not-applicable"), "T021-166"),
        ("2_555", "2_555", _WITH_OPERATOR_2, ("source-operator", "source-operator"), "T021-168"),
        ("2_555", None, _WITH_OPERATOR_2, ("source-operator", "not-applicable"), "T021-169"),
        (None, "1_555", _no_tables(), ("not-applicable", "pdbx-1_555-convention"), "T021-170"),
        (None, "2_555", _WITH_OPERATOR_2, ("not-applicable", "source-operator"), "T021-171"),
        (None, None, _no_tables(), ("not-applicable", "not-applicable"), "T021-172"),
    ],
)
def test_identity_basis_pairs_k0_reachable_pairs(symmetry1, symmetry2, tables, expected_basis, scenario_id):  # T021-164/166/168-172
    # C-07: a *real* plan_key, derived from a real SourceContext/
    # ResolvedReceptorSelection/atom vector through build_materialization_plan
    # end to end - never just the same already-built ConnectionDecision
    # object hashed twice (a tautology that proves nothing about the real
    # hash chain). Two independently-built plans from identical inputs must
    # still agree, and partner order (ptnr1-then-ptnr2) is preserved, never
    # silently normalized/sorted.
    atoms = (
        _atom(ordinal=0, atom_id="a1", label_asym_id="A", label_atom_id="CA"),
        _atom(ordinal=1, atom_id="a2", label_asym_id="B", entity_id="E2", label_atom_id="CA"),
    )

    def build_plan():
        source_context = _source_context()
        source_context_key = compute_source_context_key(source_context)
        decisions = _connection_scenario(
            retained1=False, retained2=False, symmetry1=symmetry1, symmetry2=symmetry2, tables=tables,
            source_context_key=source_context_key,
        )
        assert decisions[0].action == "omit", scenario_id
        assert decisions[0].identity_basis == expected_basis, scenario_id
        plan = build_materialization_plan(
            source_context=source_context, resolved_selection=_resolution(), unaddressed_components="preserve",
            atoms=atoms, connection_decisions=decisions,
        )
        return plan, decisions

    plan_a, decisions_a = build_plan()
    plan_b, decisions_b = build_plan()
    assert plan_a.plan_key == plan_b.plan_key, scenario_id
    assert plan_a.plan.connection_decisions_key == plan_b.plan.connection_decisions_key, scenario_id
    assert plan_a.connection_decisions == decisions_a == decisions_b, scenario_id
    assert plan_a.connection_decisions[0].identity_basis == expected_basis, scenario_id


@pytest.mark.parametrize(
    "identity_basis,scenario_id",
    [
        (("pdbx-1_555-convention", "source-operator"), "T021-165"),
        (("source-operator", "pdbx-1_555-convention"), "T021-167"),
    ],
)
def test_identity_basis_impossible_pairs_never_construct_an_accepted_connection_decision(identity_basis, scenario_id):  # T021-165/167
    # These two pairs can never arise from `_resolve_symmetry_endpoint`
    # itself (both endpoints always share one structure's `tables.any_present`),
    # so this is the DTO's own defense-in-depth validator - exercised by
    # direct construction, never produced by the real resolution pipeline.
    with pytest.raises(ValidationError):
        ConnectionDecision(
            ordinal=0, id="c1", row_key=_hex(1), conn_type_token="metalc",
            endpoint_atom_ids=(("a1",), ("a2",)), identity_basis=identity_basis,
            retained_endpoints=0, symmetry_class="unsupported-discarded", action="omit", reason="both-removed",
        )


def test_identity_basis_precedence_applies_10_1_without_k_shortcuts():  # T021-173
    # A non-null, malformed code (fails the n_klm grammar) must resolve to
    # not-applicable/insufficient-proof regardless of K or table presence -
    # never short-circuited by "it's at K=0 so it doesn't matter".
    decisions = _connection_scenario(retained1=False, retained2=False, symmetry1="not-a-code", symmetry2=None)
    assert decisions[0].identity_basis == ("not-applicable", "not-applicable")
    assert decisions[0].symmetry_class == "unsupported-discarded"


# --- Section 11: the catalogue engine -------------------------------------


def _category_table(**columns):
    return RawCategoryTable(columns=columns)


def _full_atom_site_table(
    *, ids, type_symbols, label_atom_ids, comp_ids, asym_ids, entity_ids, seq_ids, auth_asym_ids, auth_seq_ids,
    model_nums=None,
):
    n = len(ids)
    model_nums = model_nums or ("1",) * n
    return _category_table(
        id=ids, type_symbol=type_symbols, label_atom_id=label_atom_ids, label_alt_id=(None,) * n,
        label_comp_id=comp_ids, label_asym_id=asym_ids, label_entity_id=entity_ids, label_seq_id=seq_ids,
        auth_asym_id=auth_asym_ids, auth_seq_id=auth_seq_ids, pdbx_pdb_ins_code=(None,) * n,
        pdbx_pdb_model_num=model_nums, cartn_x=("1.0",) * n, cartn_y=("1.0",) * n, cartn_z=("1.0",) * n,
        occupancy=("1.0",) * n, b_iso_or_equiv=("10.0",) * n,
    )


def _minimal_catalogue_scenario():
    atoms = (
        _atom(ordinal=0, atom_id="a-ca", label_asym_id="A", ccd="ALA", label_seq_id="1", label_atom_id="CA"),
        _atom(ordinal=1, atom_id="a-sg", label_asym_id="A", ccd="CYS", label_seq_id="2", label_atom_id="SG"),
        _atom(ordinal=2, atom_id="b-sd", label_asym_id="B", ccd="MET", label_seq_id="1", auth_asym_id="B", entity_id="E2", label_atom_id="SD"),
    )
    source_context = _source_context()
    resolution = _resolution()
    plan = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms,
    )
    atom_decisions_by_ordinal = {d.ordinal: d for d in plan.atom_decisions}
    atoms_by_ordinal = {a.ordinal: a for a in atoms}
    retained_entity_ids = frozenset(
        atoms_by_ordinal[o].entity_id for o, d in atom_decisions_by_ordinal.items() if d.retained
    )
    retained_asym_ids = frozenset(
        atoms_by_ordinal[o].label_asym_id for o, d in atom_decisions_by_ordinal.items() if d.retained
    )
    return plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids


def _polymer_conditional_categories():
    """C-05: E1/chain A is a retained polymer entity in every
    `_minimal_catalogue_scenario()` fixture below - its three
    conditionally-mandatory companion categories, with rows matching the
    atoms' own (entity_id, label_seq_id, ccd) triples exactly."""

    return {
        "_entity_poly": _category_table(entity_id=("E1",), type=("polypeptide(L)",)),
        "_entity_poly_seq": _category_table(entity_id=("E1", "E1"), num=("1", "2"), mon_id=("ALA", "CYS")),
        "_pdbx_poly_seq_scheme": _category_table(
            asym_id=("A", "A"), entity_id=("E1", "E1"), seq_id=("1", "2"), mon_id=("ALA", "CYS"),
        ),
    }


def test_catalogue_atom_mask_filters_by_real_atom_decisions():  # T021-061
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    categories = {
        "_atom_site": _category_table(
            id=("a-ca", "a-sg", "b-sd"), type_symbol=("C", "S", "S"), label_atom_id=("CA", "SG", "SD"),
            label_alt_id=(None, None, None), label_comp_id=("ALA", "CYS", "MET"), label_asym_id=("A", "A", "B"),
            label_entity_id=("E1", "E1", "E2"), label_seq_id=("1", "2", "1"), auth_asym_id=("A", "A", "B"),
            auth_seq_id=("1", "2", "1"), pdbx_pdb_ins_code=(None, None, None), pdbx_pdb_model_num=("1", "1", "1"),
            cartn_x=("1.0", "2.0", "3.0"), cartn_y=("1.0", "2.0", "3.0"), cartn_z=("1.0", "2.0", "3.0"),
            occupancy=("1.0", "1.0", "1.0"), b_iso_or_equiv=("10.0", "10.0", "10.0"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("L-peptide linking",) * 3),
        **_polymer_conditional_categories(),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_atom_site"].kept_row_ordinals == (0, 1)  # T021-061
    assert by_cat["_struct_asym"].kept_row_ordinals == (0,)  # T021-063
    assert by_cat["_entity"].kept_row_ordinals == (0,)  # T021-064
    assert by_cat["_chem_comp"].kept_row_ordinals == (0, 1)  # T021-067 (CCD closure)
    assert by_cat["_entry"].action == "generate"  # T021-082


def test_catalogue_anisotrop_requires_complete_tensor_and_matching_atom_site_id():  # T021-062
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    atom_site = _category_table(
        id=("a-ca",), type_symbol=("C",), label_atom_id=("CA",), label_alt_id=(None,), label_comp_id=("ALA",),
        label_asym_id=("A",), label_entity_id=("E1",), label_seq_id=("1",), auth_asym_id=("A",), auth_seq_id=("1",),
        pdbx_pdb_ins_code=(None,), pdbx_pdb_model_num=("1",), cartn_x=("1.0",), cartn_y=("1.0",), cartn_z=("1.0",),
        occupancy=("1.0",), b_iso_or_equiv=("10.0",),
    )
    base_categories = {
        "_atom_site": atom_site, "_entity": _category_table(id=("E1",), type=("polymer",)),
        "_struct_asym": _category_table(id=("A",), entity_id=("E1",)), "_chem_comp": _category_table(id=("ALA",), type=("x",)),
        "_entity_poly": _category_table(entity_id=("E1",), type=("polypeptide(L)",)),
        "_entity_poly_seq": _category_table(entity_id=("E1",), num=("1",), mon_id=("ALA",)),
        "_pdbx_poly_seq_scheme": _category_table(asym_id=("A",), entity_id=("E1",), seq_id=("1",), mon_id=("ALA",)),
    }

    def run(anisotrop_columns):
        categories = dict(base_categories, _atom_site_anisotrop=_category_table(**anisotrop_columns))
        return compute_category_decisions(
            source_context_key=plan.plan.source_context_key, categories=categories,
            retained_entity_ids=frozenset({"E1"}), retained_asym_ids=frozenset({"A"}), resolved_model=1,
            atoms_by_ordinal={0: atoms_by_ordinal[0]}, atom_decisions_by_ordinal={0: atom_decisions_by_ordinal[0]},
            connection_decisions=(),
        )

    decisions = run({
        "id": ("a-ca",), "u[1][1]": ("0.1",), "u[1][2]": ("0.0",), "u[1][3]": ("0.0",),
        "u[2][2]": ("0.1",), "u[2][3]": ("0.0",), "u[3][3]": ("0.1",),
    })
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_atom_site_anisotrop"].kept_row_ordinals == (0,)

    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"id": ("a-ca",), "u[1][1]": ("0.1",), "u[1][2]": (None,), "u[1][3]": ("0.0",),
             "u[2][2]": ("0.1",), "u[2][3]": ("0.0",), "u[3][3]": ("0.1",)})
    assert excinfo.value.code == "INVALID_CIF"


def test_catalogue_anisotrop_rejects_informed_field_disagreement_and_undeclared_partial_tensor():  # R-CP2-02
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    atom_site = _category_table(
        id=("a-ca",), type_symbol=("C",), label_atom_id=("CA",), label_alt_id=(None,), label_comp_id=("ALA",),
        label_asym_id=("A",), label_entity_id=("E1",), label_seq_id=("1",), auth_asym_id=("A",), auth_seq_id=("1",),
        pdbx_pdb_ins_code=(None,), pdbx_pdb_model_num=("1",), cartn_x=("1.0",), cartn_y=("1.0",), cartn_z=("1.0",),
        occupancy=("1.0",), b_iso_or_equiv=("10.0",),
    )
    base_categories = {
        "_atom_site": atom_site, "_entity": _category_table(id=("E1",), type=("polymer",)),
        "_struct_asym": _category_table(id=("A",), entity_id=("E1",)), "_chem_comp": _category_table(id=("ALA",), type=("x",)),
        "_entity_poly": _category_table(entity_id=("E1",), type=("polypeptide(L)",)),
        "_entity_poly_seq": _category_table(entity_id=("E1",), num=("1",), mon_id=("ALA",)),
        "_pdbx_poly_seq_scheme": _category_table(asym_id=("A",), entity_id=("E1",), seq_id=("1",), mon_id=("ALA",)),
    }
    complete_u = {
        "u[1][1]": ("0.1",), "u[1][2]": ("0.0",), "u[1][3]": ("0.0",),
        "u[2][2]": ("0.1",), "u[2][3]": ("0.0",), "u[3][3]": ("0.1",),
    }

    def run(anisotrop_columns):
        categories = dict(base_categories, _atom_site_anisotrop=_category_table(**anisotrop_columns))
        return compute_category_decisions(
            source_context_key=plan.plan.source_context_key, categories=categories,
            retained_entity_ids=frozenset({"E1"}), retained_asym_ids=frozenset({"A"}), resolved_model=1,
            atoms_by_ordinal={0: atoms_by_ordinal[0]}, atom_decisions_by_ordinal={0: atom_decisions_by_ordinal[0]},
            connection_decisions=(),
        )

    # R-CP2-02: pdbx_label_seq_id=99 disagrees with the referenced real
    # atom's label_seq_id=1 - the previous mapping only ever checked
    # atom/asym/comp labels, never seq_id.
    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"id": ("a-ca",), "pdbx_label_seq_id": ("99",), **complete_u})
    assert excinfo.value.code == "INVALID_CIF"

    # R-CP2-02: type_symbol='O' disagrees with the referenced real atom's
    # type_symbol='C' - type_symbol was never checked at all before.
    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"id": ("a-ca",), "type_symbol": ("O",), **complete_u})
    assert excinfo.value.code == "INVALID_CIF"

    # Genuinely informed, *agreeing* mirrored fields (seq_id, alt_id,
    # auth_asym_id/auth_seq_id, type_symbol) must still pass - this is
    # integrity checking, never inventing stricter-than-source agreement.
    decisions = run({
        "id": ("a-ca",), "type_symbol": ("C",), "pdbx_label_seq_id": ("1",), "pdbx_label_alt_id": (None,),
        "pdbx_auth_asym_id": ("A",), "pdbx_auth_seq_id": ("1",), **complete_u,
    })
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_atom_site_anisotrop"].kept_row_ordinals == (0,)

    # R-CP2-02: a row with a genuine numeric value in u[1][1], but whose
    # category never declares the other five U columns at all (not merely
    # null for this row), must still be "partial" and rejected even
    # though a complete B tensor is also present - the previous
    # `_tensor_family_state` returned "absent" for U whenever any of its
    # six columns was undeclared, letting this slip through as if only B
    # existed.
    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({
            "id": ("a-ca",), "u[1][1]": ("0.1",),
            "b[1][1]": ("1.0",), "b[1][2]": ("0.0",), "b[1][3]": ("0.0",),
            "b[2][2]": ("1.0",), "b[2][3]": ("0.0",), "b[3][3]": ("1.0",),
        })
    assert excinfo.value.code == "INVALID_CIF"


def test_catalogue_struct_conn_used_types_and_elements_from_real_connection_decisions():  # T021-075/076/077
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    tables = _no_tables()
    row = RawConnectionRow(
        ordinal=0, id="conn1", conn_type="disulf",
        ptnr1=_endpoint(asym="A", ccd="ALA", seq="1", atom_id_label="CA", symmetry="1_555"),
        ptnr2=_endpoint(asym="A", ccd="CYS", seq="2", atom_id_label="SG", symmetry="1_555"),
        ptnr3_present=False, source_tokens=(("id", "conn1"),),
    )
    connection_decisions = resolve_connection_decisions(
        source_context_key=plan.plan.source_context_key, atoms_in_model=tuple(atoms_by_ordinal.values()),
        atom_decisions_by_ordinal=atom_decisions_by_ordinal, raw_connections=(row,), tables=tables,
    )
    assert connection_decisions[0].action == "preserve"

    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        "_struct_conn": _category_table(
            id=("conn1",), conn_type_id=("disulf",),
            ptnr1_label_asym_id=("A",), ptnr1_label_comp_id=("ALA",), ptnr1_label_atom_id=("CA",),
            ptnr2_label_asym_id=("A",), ptnr2_label_comp_id=("CYS",), ptnr2_label_atom_id=("SG",),
        ),
        "_struct_conn_type": _category_table(id=("disulf", "metalc")),
        "_atom_type": _category_table(symbol=("C", "N", "S")),
        **_polymer_conditional_categories(),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=connection_decisions,
    )
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_struct_conn"].kept_row_ordinals == (0,)  # T021-075
    assert by_cat["_struct_conn_type"].kept_row_ordinals == (0,)  # T021-076 (disulf only, metalc never fabricated)
    assert by_cat["_atom_type"].kept_row_ordinals == (0, 2)  # T021-077 (C, S used by retained atoms; N unused)


def test_catalogue_cell_symmetry_singleton_and_operator_tables_preserved_verbatim():  # T021-078/079
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        "_cell": _category_table(length_a=("10.0",), length_b=("10.0",), length_c=("10.0",)),
        "_symmetry": _category_table(**{"space_group_name_h-m": ("'P 1'",)}),
        "_symmetry_equiv": _category_table(id=("1", "2"), pos_as_xyz=("x,y,z", "-x,-y,z")),
        **_polymer_conditional_categories(),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_cell"].kept_row_ordinals == (0,)  # T021-078
    assert by_cat["_cell"].generated_tags == ("entry_id",)
    assert by_cat["_symmetry"].kept_row_ordinals == (0,)  # T021-078
    assert by_cat["_symmetry_equiv"].kept_row_ordinals == (0, 1)  # T021-079 (verbatim, no filtering)


def test_catalogue_omitted_family_and_unknown_category_and_tag():  # T021-083/084/085
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    base = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
    }

    def run(categories):
        return compute_category_decisions(
            source_context_key=plan.plan.source_context_key, categories=dict(base, **categories),
            retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
            atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
            connection_decisions=(),
        )

    decisions = run({"_pdbx_validate_rmsd_bond": _category_table(id=("1",))})
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_pdbx_validate_rmsd_bond"].action == "omit"  # T021-083

    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"_my_unlisted_category": _category_table(id=("1",))})
    assert excinfo.value.code == "UNSUPPORTED_CATEGORY"  # T021-084

    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer"), bogus=("x", "y"))})
    assert excinfo.value.code == "UNSUPPORTED_TAG"  # T021-085


def test_catalogue_ncs_scenarios():  # T021-150/151/152/153/154
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    base = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
    }
    identity_matrix = dict(
        **{f"matrix[{i}][{j}]": ("1.0" if i == j else "0.0",) for i in (1, 2, 3) for j in (1, 2, 3)},
        **{f"vector[{i}]": ("0.0",) for i in (1, 2, 3)},
    )
    nonidentity_matrix = dict(identity_matrix)
    nonidentity_matrix["matrix[1][1]"] = ("0.0",)
    nonidentity_matrix["matrix[1][2]"] = ("-1.0",)
    nonidentity_matrix["matrix[2][1]"] = ("1.0",)
    nonidentity_matrix["matrix[2][2]"] = ("0.0",)

    def run(code, matrix):
        categories = dict(base, _struct_ncs_oper=_category_table(id=("1",), code=(code,), **matrix))
        return compute_category_decisions(
            source_context_key=plan.plan.source_context_key, categories=categories,
            retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
            atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
            connection_decisions=(),
        )

    with pytest.raises(MaterializationExecutionError) as excinfo:
        run("generate", nonidentity_matrix)
    assert excinfo.value.code == "UNSUPPORTED_NCS"  # T021-150

    with pytest.raises(MaterializationExecutionError) as excinfo:
        run(None, nonidentity_matrix)
    assert excinfo.value.code == "UNSUPPORTED_NCS"  # T021-151

    decisions = run("given", nonidentity_matrix)
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_struct_ncs_oper"].action == "guard_then_omit"  # T021-152
    assert by_cat["_struct_ncs_oper"].omitted_row_ordinals == (0,)

    decisions = run("generate", identity_matrix)
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_struct_ncs_oper"].omitted_row_ordinals == (0,)  # T021-153

    partial_matrix = dict(identity_matrix)
    partial_matrix["matrix[1][1]"] = (None,)
    with pytest.raises(MaterializationExecutionError) as excinfo:
        run("given", partial_matrix)
    assert excinfo.value.code == "INVALID_CIF"  # T021-154 (partial matrix)

    with pytest.raises(MaterializationExecutionError) as excinfo:
        run("some-unknown-code", identity_matrix)
    assert excinfo.value.code == "INVALID_CIF"  # T021-154 (unknown code)


def test_catalogue_special_position_guard_requires_unambiguous_context():  # T021-158
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    base = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
    }

    def run(special_columns):
        categories = dict(base, _pdbx_struct_special_symmetry=_category_table(**special_columns))
        return compute_category_decisions(
            source_context_key=plan.plan.source_context_key, categories=categories,
            retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
            atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
            connection_decisions=(),
        )

    decisions = run({"id": ("1",), "label_asym_id": ("A",), "label_seq_id": ("1",)})
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_pdbx_struct_special_symmetry"].action == "guard_then_omit"

    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"id": ("1",), "auth_asym_id": ("Z",), "label_seq_id": ("1",)})
    assert excinfo.value.code == "INVALID_CIF"  # ambiguity/unresolvable context refuses, per T021-158

    # R-CP2-01: a genuinely unknown label_asym_id must refuse, never pass
    # silently just because the column itself is non-null - the previous
    # guard returned immediately for *any* non-null label_asym_id.
    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"id": ("1",), "label_asym_id": ("NO_SUCH_ASYM",), "label_seq_id": ("1",)})
    assert excinfo.value.code == "INVALID_CIF"

    # R-CP2-01: a real asym with no seq identity at all (neither
    # label_seq_id nor auth_seq_id) cannot be resolved to any residue.
    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"id": ("1",), "label_asym_id": ("A",)})
    assert excinfo.value.code == "INVALID_CIF"

    # R-CP2-01: a real asym with a seq_id that matches no real residue in
    # any model (never merely "not in M") is an invalid reference, not a
    # zero-overlap resolution.
    with pytest.raises(MaterializationExecutionError) as excinfo:
        run({"id": ("1",), "label_asym_id": ("A",), "label_seq_id": ("999",)})
    assert excinfo.value.code == "INVALID_CIF"


def test_compute_special_position_summary_counts_overlap_removed_and_other_model():  # R-CP2-01
    # A richer scenario than _minimal_catalogue_scenario(): two altloc
    # conformers of the same polymer residue (A/seq2), a second, identical
    # model-1 atom in another model (model 2) to exercise "other model"
    # without an error, a non-polymer water on its own asym resolved via
    # auth_asym_id/auth_seq_id/insertion (never label_seq_id), and a
    # non-selected polymer asym (B) whose annotation is real but never
    # retained.
    atoms = (
        _atom(ordinal=0, atom_id="a1", label_asym_id="A", entity_id="E1", label_seq_id="1", auth_seq_id="1", ccd="ALA"),
        _atom(ordinal=1, atom_id="a2", label_asym_id="A", entity_id="E1", label_seq_id="2", auth_seq_id="2", ccd="CYS", altloc="A"),
        _atom(ordinal=2, atom_id="a3", label_asym_id="A", entity_id="E1", label_seq_id="2", auth_seq_id="2", ccd="CYS", altloc="B"),
        _atom(ordinal=3, atom_id="b1", label_asym_id="B", entity_id="E2", label_seq_id="1", auth_seq_id="1", ccd="MET"),
        _atom(ordinal=4, atom_id="a1-m2", model=2, label_asym_id="A", entity_id="E1", label_seq_id="1", auth_seq_id="1", ccd="ALA"),
        _atom(
            ordinal=5, atom_id="c1", label_asym_id="C", entity_id="E3", entity_type="water", label_seq_id=None,
            auth_asym_id="C", auth_seq_id="50", insertion_token="A", ccd="HOH",
        ),
    )
    source_context = _source_context()
    resolution = _resolution()
    plan = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms,
    )
    atom_decisions_by_ordinal = {d.ordinal: d for d in plan.atom_decisions}
    atoms_by_ordinal = {a.ordinal: a for a in atoms}
    # Sanity check on the fixture's own retention shape before trusting
    # the counts below: A (selected) and the water (unaddressed-preserve)
    # retained; B (unselected polymer) and the model-2 duplicate (model
    # filter) not retained.
    assert [atom_decisions_by_ordinal[o].retained for o in range(6)] == [
        True, True, True, False, False, True,
    ]

    special_table = _category_table(
        id=("1", "2", "3", "4", "5", "6", "7"),
        label_asym_id=("A", "A", "A", "B", "A", "A", None),
        label_seq_id=("1", "2", "1", "1", "1", "2", None),
        auth_asym_id=(None, None, None, None, None, None, "C"),
        auth_seq_id=(None, None, None, None, None, None, "50"),
        pdb_ins_code=(None, None, None, None, None, None, "A"),
        pdb_model_num=(None, None, None, None, "2", None, None),
        label_alt_id=(None, None, None, None, None, "A", None),
    )

    from molecular.receptor_materialization import compute_special_position_summary

    summary = compute_special_position_summary(
        categories={"_pdbx_struct_special_symmetry": special_table},
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
    )
    assert summary.source_annotation_count == 7
    # Retained: row1 (ordinal0), row2 (ordinals1+2), row3 (ordinal0 again
    # - overlap with row1, never double-counted as a *second* retained
    # atom), row6 (ordinal1 only, altloc='A' narrows out ordinal2), row7
    # (ordinal5, resolved via auth_asym_id/auth_seq_id/insertion). Never
    # retained: row4 (asym B, real but unselected) and row5 (asym A/seq1,
    # but scoped to model 2 by pdb_model_num - a real residue, simply not
    # in M - never an error).
    assert summary.retained_annotation_count == 5
    assert summary.retained_unique_atom_count == 4  # {0, 1, 2, 5} - never 6 or 7

    # SCHEMAS Section 8: zero counts when the category is absent.
    empty_summary = compute_special_position_summary(
        categories={}, atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
    )
    assert empty_summary.source_annotation_count == 0
    assert empty_summary.retained_annotation_count == 0
    assert empty_summary.retained_unique_atom_count == 0

    # The plan-level warning now requires a real retained intersection -
    # an all-removed/other-model-only category must never trigger it.
    removed_only_table = _category_table(
        id=("1",), label_asym_id=("B",), label_seq_id=("1",), auth_asym_id=(None,), auth_seq_id=(None,),
        pdb_ins_code=(None,), pdb_model_num=(None,), label_alt_id=(None,),
    )
    removed_only_summary = compute_special_position_summary(
        categories={"_pdbx_struct_special_symmetry": removed_only_table},
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
    )
    assert removed_only_summary.retained_annotation_count == 0
    plan_removed_only = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms,
        special_positions=removed_only_summary,
    )
    assert "special-position-annotations-omitted" not in plan_removed_only.plan.warnings

    plan_with_warning = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms,
        special_positions=summary,
    )
    assert "special-position-annotations-omitted" in plan_with_warning.plan.warnings
    assert plan_with_warning.special_positions is summary


# --- Remaining C-07 scenarios: raw tokens, coordinates, insertion codes,
# the individual RETAINED_* categories, context categories, audit
# completeness, numeric grammar, and altloc endpoint sets. -----------------


def test_atom_site_coordinates_and_occupancy_zero_preserved_without_repair():  # T021-042
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    categories = {
        "_atom_site": _category_table(
            id=("a-ca", "a-sg", "b-sd"), type_symbol=("C", "S", "S"), label_atom_id=("CA", "SG", "SD"),
            label_alt_id=(None, None, None), label_comp_id=("ALA", "CYS", "MET"), label_asym_id=("A", "A", "B"),
            label_entity_id=("E1", "E1", "E2"), label_seq_id=("1", "2", "1"), auth_asym_id=("A", "A", "B"),
            auth_seq_id=("1", "2", "1"), pdbx_pdb_ins_code=(None, None, None), pdbx_pdb_model_num=("1", "1", "1"),
            cartn_x=("1.0", "2.0", "3.0"), cartn_y=("1.0", "2.0", "3.0"), cartn_z=("1.0", "2.0", "3.0"),
            occupancy=("0.0", "0.5", "1.0"), b_iso_or_equiv=(None, "11.0", "10.0"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    # occupancy=0.0 and a null B-factor are both legitimate, never repaired
    # into a default - the row still survives ATOM_MASK filtering untouched.
    assert by_cat["_atom_site"].kept_row_ordinals == (0, 1)


def test_atom_site_null_coordinate_rejects():  # T021-042 (negative half)
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    categories = {
        "_atom_site": _category_table(
            id=("a-ca", "a-sg", "b-sd"), type_symbol=("C", "S", "S"), label_atom_id=("CA", "SG", "SD"),
            label_alt_id=(None, None, None), label_comp_id=("ALA", "CYS", "MET"), label_asym_id=("A", "A", "B"),
            label_entity_id=("E1", "E1", "E2"), label_seq_id=("1", "2", "1"), auth_asym_id=("A", "A", "B"),
            auth_seq_id=("1", "2", "1"), pdbx_pdb_ins_code=(None, None, None), pdbx_pdb_model_num=("1", "1", "1"),
            cartn_x=(None, "2.0", "3.0"), cartn_y=("1.0", "2.0", "3.0"), cartn_z=("1.0", "2.0", "3.0"),
            occupancy=("1.0", "1.0", "1.0"), b_iso_or_equiv=("10.0", "10.0", "10.0"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
    }
    with pytest.raises(MaterializationExecutionError) as excinfo:
        compute_category_decisions(
            source_context_key=plan.plan.source_context_key, categories=categories,
            retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
            atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
            connection_decisions=(),
        )
    assert excinfo.value.code == "INVALID_CIF"


def test_struct_conn_endpoint_insertion_code_distinguishes_residues():  # T021-043
    # Residues 52 and 52A (same auth_seq_id, different insertion code) must
    # resolve to distinct occurrences when label_seq_id is null - never
    # collapsed via legacy PDB-style concatenated residue identifiers.
    a52 = _atom(ordinal=0, atom_id="a52", label_asym_id="A", label_seq_id=None, auth_seq_id="52", label_atom_id="CA", ccd="ALA")
    a52a = AtomObservation(
        ordinal=1, atom_id="a52a", model=1, label_asym_id="A", entity_id="E1", entity_type="polymer", ccd="ALA",
        label_seq_id=None, auth_asym_id="A", auth_seq_id="52", insertion_token="A", label_atom_id="CA",
    )
    endpoint_52 = RawConnectionEndpointTokens(
        label_asym_id="A", label_comp_id="ALA", label_seq_id=None, label_atom_id="CA",
        auth_seq_id="52", insertion_token=None, altloc_aliases=(None, None, None), symmetry_token="1_555",
    )
    endpoint_52a = RawConnectionEndpointTokens(
        label_asym_id="A", label_comp_id="ALA", label_seq_id=None, label_atom_id="CA",
        auth_seq_id="52", insertion_token="A", altloc_aliases=(None, None, None), symmetry_token="1_555",
    )
    from molecular.receptor_materialization import _resolve_endpoint_atoms
    resolved_52 = _resolve_endpoint_atoms(endpoint_52, (a52, a52a))
    resolved_52a = _resolve_endpoint_atoms(endpoint_52a, (a52, a52a))
    assert resolved_52 == (a52,)
    assert resolved_52a == (a52a,)


def test_remaining_retained_categories_individually():  # T021-065/066/068/069/070/071/072/073/074
    atoms = (
        _atom(ordinal=0, atom_id="a-ca", label_asym_id="A", ccd="ALA", label_seq_id="1", label_atom_id="CA"),
        _atom(
            ordinal=1, atom_id="l-lig", label_asym_id="L", ccd="HEM", label_seq_id=None, auth_asym_id="L",
            entity_id="E2", entity_type="non-polymer", label_atom_id="FE",
        ),
        _atom(
            ordinal=2, atom_id="b-man", label_asym_id="BR", ccd="MAN", label_seq_id="1", auth_asym_id="BR",
            entity_id="E3", entity_type="branched", label_atom_id="C1",
        ),
    )
    source_context = _source_context()
    resolution = _resolution(
        resolved_chains=(
            ResolvedChainSelection(label_asym_id="A", label_entity_id="E1", model_number=1),
            ResolvedChainSelection(label_asym_id="BR", label_entity_id="E3", model_number=1),
        ),
    )
    plan = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms,
    )
    atom_decisions_by_ordinal = {d.ordinal: d for d in plan.atom_decisions}
    atoms_by_ordinal = {a.ordinal: a for a in atoms}
    retained_entity_ids = frozenset(
        atoms_by_ordinal[o].entity_id for o, d in atom_decisions_by_ordinal.items() if d.retained
    )
    retained_asym_ids = frozenset(
        atoms_by_ordinal[o].label_asym_id for o, d in atom_decisions_by_ordinal.items() if d.retained
    )
    assert atom_decisions_by_ordinal[1].retained is True  # L/HEM unaddressed non-polymer, U=preserve policy
    assert atom_decisions_by_ordinal[1].reason == "unaddressed-preserve"

    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "l-lig", "b-man"), type_symbols=("C", "FE", "C"), label_atom_ids=("CA", "FE", "C1"),
            comp_ids=("ALA", "HEM", "MAN"), asym_ids=("A", "L", "BR"), entity_ids=("E1", "E2", "E3"),
            seq_ids=("1", "1", "1"), auth_asym_ids=("A", "L", "BR"), auth_seq_ids=("1", "1", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2", "E3"), type=("polymer", "non-polymer", "branched")),
        "_struct_asym": _category_table(id=("A", "L", "BR"), entity_id=("E1", "E2", "E3")),
        "_chem_comp": _category_table(id=("ALA", "HEM", "MAN"), type=("x",) * 3),
        "_entity_poly": _category_table(entity_id=("E1",), type=("polypeptide(L)",)),  # T021-065
        "_entity_poly_seq": _category_table(entity_id=("E1",), num=("1",), mon_id=("ALA",)),  # T021-066
        "_pdbx_poly_seq_scheme": _category_table(asym_id=("A",), entity_id=("E1",), seq_id=("1",), mon_id=("ALA",)),  # T021-069
        "_pdbx_entity_nonpoly": _category_table(entity_id=("E2",), comp_id=("HEM",)),  # T021-068
        "_pdbx_nonpoly_scheme": _category_table(asym_id=("L",), entity_id=("E2",), mon_id=("HEM",), ndb_seq_num=("1",)),  # T021-070
        "_pdbx_entity_branch": _category_table(entity_id=("E3",), type=("oligosaccharide",)),  # T021-071
        "_pdbx_entity_branch_list": _category_table(entity_id=("E3",), num=("1",), comp_id=("MAN",)),  # T021-072
        "_pdbx_entity_branch_link": _category_table(
            link_id=("1",), entity_id=("E3",), entity_branch_list_num_1=("1",), comp_id_1=("MAN",), atom_id_1=("C1",),
            entity_branch_list_num_2=("1",), comp_id_2=("MAN",), atom_id_2=("C2",),
        ),  # T021-073 (presence only; no required_tags enforce link endpoints match num_1 != num_2 here)
        "_pdbx_branch_scheme": _category_table(asym_id=("BR",), entity_id=("E3",), mon_id=("MAN",), num=("1",)),  # T021-074
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_entity_poly"].kept_row_ordinals == (0,)  # T021-065
    assert by_cat["_entity_poly_seq"].kept_row_ordinals == (0,)  # T021-066
    assert by_cat["_pdbx_poly_seq_scheme"].kept_row_ordinals == (0,)  # T021-069
    assert by_cat["_pdbx_entity_nonpoly"].kept_row_ordinals == (0,)  # T021-068 (E2 is unaddressed-preserved)
    assert by_cat["_pdbx_nonpoly_scheme"].kept_row_ordinals == (0,)  # T021-070
    assert by_cat["_pdbx_entity_branch"].kept_row_ordinals == (0,)  # T021-071
    assert by_cat["_pdbx_entity_branch_list"].kept_row_ordinals == (0,)  # T021-072
    assert by_cat["_pdbx_entity_branch_link"].kept_row_ordinals == (0,)  # T021-073
    assert by_cat["_pdbx_branch_scheme"].kept_row_ordinals == (0,)  # T021-074


def test_missing_conditional_branched_categories_rejects():  # T021-086
    atoms = (
        _atom(
            ordinal=0, atom_id="b-man", label_asym_id="BR", ccd="MAN", label_seq_id="1", auth_asym_id="BR",
            entity_id="E3", entity_type="branched", label_atom_id="C1",
        ),
    )
    source_context = _source_context()
    resolution = _resolution(
        resolved_chains=(ResolvedChainSelection(label_asym_id="BR", label_entity_id="E3", model_number=1),),
    )
    plan = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms,
    )
    atom_decisions_by_ordinal = {d.ordinal: d for d in plan.atom_decisions}
    atoms_by_ordinal = {a.ordinal: a for a in atoms}
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("b-man",), type_symbols=("C",), label_atom_ids=("C1",), comp_ids=("MAN",), asym_ids=("BR",),
            entity_ids=("E3",), seq_ids=("1",), auth_asym_ids=("BR",), auth_seq_ids=("1",),
        ),
        "_entity": _category_table(id=("E3",), type=("branched",)),
        "_struct_asym": _category_table(id=("BR",), entity_id=("E3",)),
        "_chem_comp": _category_table(id=("MAN",), type=("x",)),
        # _pdbx_entity_branch/_pdbx_entity_branch_list/_pdbx_branch_scheme deliberately absent
    }
    with pytest.raises(MaterializationExecutionError) as excinfo:
        compute_category_decisions(
            source_context_key=plan.plan.source_context_key, categories=categories,
            retained_entity_ids=frozenset({"E3"}), retained_asym_ids=frozenset({"BR"}), resolved_model=1,
            atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
            connection_decisions=(),
        )
    assert excinfo.value.code == "INVALID_CIF"


def test_numeric_token_grammar_su_exponent():  # T021-155
    from molecular.receptor_materialization import _validate_numeric_token
    from decimal import Decimal
    assert _validate_numeric_token("1.2e3(4)") == Decimal("1.2e3")
    assert _validate_numeric_token("1.2(4)e3") is None
    assert _validate_numeric_token("1_000") is None
    assert _validate_numeric_token("1.0") == Decimal("1.0")
    assert _validate_numeric_token("NaN") is None
    assert _validate_numeric_token(None) is None


def test_altloc_endpoint_without_alias_returns_full_altloc_set():  # T021-156
    a_alt = AtomObservation(
        ordinal=0, atom_id="a-alt", model=1, label_asym_id="A", entity_id="E1", entity_type="polymer", ccd="CYS",
        label_seq_id="1", auth_asym_id="A", auth_seq_id="1", label_atom_id="SG", altloc="A",
    )
    b_alt = AtomObservation(
        ordinal=1, atom_id="b-alt", model=1, label_asym_id="A", entity_id="E1", entity_type="polymer", ccd="CYS",
        label_seq_id="1", auth_asym_id="A", auth_seq_id="1", label_atom_id="SG", altloc="B",
    )
    tokens = RawConnectionEndpointTokens(
        label_asym_id="A", label_comp_id="CYS", label_seq_id="1", label_atom_id="SG",
        auth_seq_id="1", insertion_token=None, altloc_aliases=(None, None, None), symmetry_token="1_555",
    )
    from molecular.receptor_materialization import _resolve_endpoint_atoms
    resolved = _resolve_endpoint_atoms(tokens, (a_alt, b_alt))
    assert set(resolved) == {a_alt, b_alt}  # one endpoint, both altloc variants as a single set


def test_altloc_endpoint_missing_variant_rejects():  # T021-157 (reachable half: declared altloc absent from occurrence)
    a_alt = AtomObservation(
        ordinal=0, atom_id="a-alt", model=1, label_asym_id="A", entity_id="E1", entity_type="polymer", ccd="CYS",
        label_seq_id="1", auth_asym_id="A", auth_seq_id="1", label_atom_id="SG", altloc="A",
    )
    tokens = RawConnectionEndpointTokens(
        label_asym_id="A", label_comp_id="CYS", label_seq_id="1", label_atom_id="SG",
        auth_seq_id="1", insertion_token=None, altloc_aliases=("B", None, None), symmetry_token="1_555",
    )
    from molecular.receptor_materialization import _resolve_endpoint_atoms
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _resolve_endpoint_atoms(tokens, (a_alt,))
    assert excinfo.value.code == "AMBIGUOUS_ENDPOINT"


def test_modified_residue_context_resolves_by_label_and_enters_ccd_closure():  # T021-080
    atoms = (
        _atom(ordinal=0, atom_id="a-mse", label_asym_id="A", ccd="MSE", label_seq_id="1", label_atom_id="SE"),
        _atom(ordinal=1, atom_id="b-ala", label_asym_id="B", ccd="ALA", label_seq_id="1", auth_asym_id="B", entity_id="E2", label_atom_id="CA"),
    )
    source_context = _source_context()
    resolution = _resolution()
    plan = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms,
    )
    atom_decisions_by_ordinal = {d.ordinal: d for d in plan.atom_decisions}
    atoms_by_ordinal = {a.ordinal: a for a in atoms}
    retained_entity_ids = frozenset(atoms_by_ordinal[o].entity_id for o, d in atom_decisions_by_ordinal.items() if d.retained)
    retained_asym_ids = frozenset(atoms_by_ordinal[o].label_asym_id for o, d in atom_decisions_by_ordinal.items() if d.retained)
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-mse", "b-ala"), type_symbols=("Se", "C"), label_atom_ids=("SE", "CA"), comp_ids=("MSE", "ALA"),
            asym_ids=("A", "B"), entity_ids=("E1", "E2"), seq_ids=("1", "1"), auth_asym_ids=("A", "B"), auth_seq_ids=("1", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("MSE", "ALA", "MET"), type=("x",) * 3),
        "_entity_poly": _category_table(entity_id=("E1",), type=("polypeptide(L)",)),
        "_entity_poly_seq": _category_table(entity_id=("E1",), num=("1",), mon_id=("MSE",)),
        "_pdbx_poly_seq_scheme": _category_table(asym_id=("A",), entity_id=("E1",), seq_id=("1",), mon_id=("MSE",)),
        "_pdbx_struct_mod_residue": _category_table(
            id=("1",), label_asym_id=("A",), label_comp_id=("MSE",), label_seq_id=("1",),
            auth_asym_id=("A",), auth_comp_id=("MSE",), auth_seq_id=("1",), parent_comp_id=("MET",),
        ),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_pdbx_struct_mod_residue"].kept_row_ordinals == (0,)  # resolved by label_asym_id in retained set
    # parent_comp_id='MET' enters the CCD closure even though MET is never
    # itself instantiated as an atom - chem_comp still only keeps real CCDs
    # actually present (MSE) plus the documentary parent (MET); ALA is
    # excluded (chain B not retained).
    assert set(categories["_chem_comp"].columns["id"][i] for i in by_cat["_chem_comp"].kept_row_ordinals) == {"MSE", "MET"}


def test_modified_residue_preserved_parent_without_ccd_row_rejects():  # T021-080 (governance addendum 2026-11-24 §3.1)
    # Same shape as the test above, but _chem_comp never declares MET -
    # exactly 5I55's real situation (see test_real_fixture_5i55_rejects_
    # missing_parent_ccd_closure_row). A preserved parent_comp_id that
    # enters the closure without a matching row must refuse, never pass
    # silently just because MET was never physically instantiated.
    atoms = (
        _atom(ordinal=0, atom_id="a-mse", label_asym_id="A", ccd="MSE", label_seq_id="1", label_atom_id="SE"),
    )
    source_context = _source_context()
    resolution = _resolution()
    plan = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="preserve", atoms=atoms,
    )
    atom_decisions_by_ordinal = {d.ordinal: d for d in plan.atom_decisions}
    atoms_by_ordinal = {a.ordinal: a for a in atoms}
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-mse",), type_symbols=("Se",), label_atom_ids=("SE",), comp_ids=("MSE",),
            asym_ids=("A",), entity_ids=("E1",), seq_ids=("1",), auth_asym_ids=("A",), auth_seq_ids=("1",),
        ),
        "_entity": _category_table(id=("E1",), type=("polymer",)),
        "_struct_asym": _category_table(id=("A",), entity_id=("E1",)),
        "_chem_comp": _category_table(id=("MSE",), type=("x",)),  # MET deliberately absent
        "_entity_poly": _category_table(entity_id=("E1",), type=("polypeptide(L)",)),
        "_entity_poly_seq": _category_table(entity_id=("E1",), num=("1",), mon_id=("MSE",)),
        "_pdbx_poly_seq_scheme": _category_table(asym_id=("A",), entity_id=("E1",), seq_id=("1",), mon_id=("MSE",)),
        "_pdbx_struct_mod_residue": _category_table(
            id=("1",), label_asym_id=("A",), label_comp_id=("MSE",), label_seq_id=("1",),
            auth_asym_id=("A",), auth_comp_id=("MSE",), auth_seq_id=("1",), parent_comp_id=("MET",),
        ),
    }
    with pytest.raises(MaterializationExecutionError) as excinfo:
        compute_category_decisions(
            source_context_key=plan.plan.source_context_key, categories=categories,
            retained_entity_ids=frozenset({"E1"}), retained_asym_ids=frozenset({"A"}), resolved_model=1,
            atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
            connection_decisions=(),
        )
    assert excinfo.value.code == "INVALID_CIF"
    assert "MET" in str(excinfo.value)


def test_modified_residue_not_preserved_never_requires_parent_ccd_row():  # T021-080 (governance addendum 2026-11-24 §3.1)
    # The mod_residue row's context (asym B) is NOT retained - it never
    # enters the CCD closure, so a missing parent_comp_id='MET' row must
    # NOT block materialization (never confuse a non-surviving reference
    # with a preserved parent).
    atoms = (
        _atom(ordinal=0, atom_id="a-ala", label_asym_id="A", ccd="ALA", label_seq_id="1", label_atom_id="CA"),
        _atom(ordinal=1, atom_id="b-mse", label_asym_id="B", ccd="MSE", label_seq_id="1", auth_asym_id="B", entity_id="E2", label_atom_id="SE"),
    )
    source_context = _source_context()
    resolution = _resolution()  # only chain A selected
    plan = build_materialization_plan(
        source_context=source_context, resolved_selection=resolution, unaddressed_components="require-disposition", atoms=atoms,
    )
    atom_decisions_by_ordinal = {d.ordinal: d for d in plan.atom_decisions}
    atoms_by_ordinal = {a.ordinal: a for a in atoms}
    assert atom_decisions_by_ordinal[1].retained is False  # B/MSE excluded - chain-complement
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ala", "b-mse"), type_symbols=("C", "Se"), label_atom_ids=("CA", "SE"), comp_ids=("ALA", "MSE"),
            asym_ids=("A", "B"), entity_ids=("E1", "E2"), seq_ids=("1", "1"), auth_asym_ids=("A", "B"), auth_seq_ids=("1", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "MSE"), type=("x", "x")),  # MET absent - irrelevant here
        "_entity_poly": _category_table(entity_id=("E1",), type=("polypeptide(L)",)),
        "_entity_poly_seq": _category_table(entity_id=("E1",), num=("1",), mon_id=("ALA",)),
        "_pdbx_poly_seq_scheme": _category_table(asym_id=("A",), entity_id=("E1",), seq_id=("1",), mon_id=("ALA",)),
        "_pdbx_struct_mod_residue": _category_table(
            id=("1",), label_asym_id=("B",), label_comp_id=("MSE",), label_seq_id=("1",),
            auth_asym_id=("B",), auth_comp_id=("MSE",), auth_seq_id=("1",), parent_comp_id=("MET",),
        ),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=frozenset({"E1"}), retained_asym_ids=frozenset({"A"}), resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_pdbx_struct_mod_residue"].kept_row_ordinals == ()  # not preserved - context (B) excluded
    assert by_cat["_chem_comp"].kept_row_ordinals == (0,)  # only ALA; MSE/MET never required


def test_missing_residue_and_atom_context_resolve_by_label_asym_id():  # T021-081
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
        "_pdbx_unobs_or_zero_occ_residues": _category_table(
            id=("1", "2"), label_asym_id=("A", "B"), auth_asym_id=("A", "B"), auth_seq_id=("3", "2"),
        ),
        "_pdbx_unobs_or_zero_occ_atoms": _category_table(
            id=("1", "2"), label_asym_id=("A", "B"), label_atom_id=("OXT", "OXT"), auth_asym_id=("A", "B"), auth_seq_id=("1", "1"),
        ),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    # Absence of coordinates is itself the annotation - it never requires
    # a matching real atom_site row to survive; only the chain (A) context
    # must resolve/be retained.
    assert by_cat["_pdbx_unobs_or_zero_occ_residues"].kept_row_ordinals == (0,)  # A kept, B (excluded chain) omitted
    assert by_cat["_pdbx_unobs_or_zero_occ_atoms"].kept_row_ordinals == (0,)


def test_catalogue_cell_preserves_esd_reciprocal_and_volume_fields():  # T021-145
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
        "_cell": _category_table(
            length_a=("10.0",), length_b=("10.0",), length_c=("10.0",), length_a_esd=("0.01",),
            volume=("1000.0",), volume_esd=("1.0",), reciprocal_length_a=("0.1",),
        ),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_cell"].kept_row_ordinals == (0,)
    assert set(by_cat["_cell"].preserved_tags) >= {"length_a_esd", "volume", "volume_esd", "reciprocal_length_a"}


def test_unobs_atoms_label_alt_id_recognized_and_preserved():  # T021-146
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
        "_pdbx_unobs_or_zero_occ_atoms": _category_table(
            id=("1",), label_asym_id=("A",), label_atom_id=("OXT",), label_alt_id=("B",), auth_asym_id=("A",), auth_seq_id=("1",),
        ),
    }
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    assert by_cat["_pdbx_unobs_or_zero_occ_atoms"].kept_row_ordinals == (0,)
    assert "label_alt_id" in by_cat["_pdbx_unobs_or_zero_occ_atoms"].preserved_tags


def test_catalogue_additional_observed_categories_omitted_with_full_audit():  # T021-148
    plan, atoms_by_ordinal, atom_decisions_by_ordinal, retained_entity_ids, retained_asym_ids = _minimal_catalogue_scenario()
    extra_omit_categories = [
        "_citation", "_citation_author", "_software", "_struct", "_struct_keywords", "_exptl", "_exptl_crystal",
        "_diffrn", "_diffrn_detector", "_diffrn_radiation", "_reflns", "_refine", "_refine_hist",
        "_database_2", "_audit_author", "_struct_biol",
    ]
    categories = {
        "_atom_site": _full_atom_site_table(
            ids=("a-ca", "a-sg", "b-sd"), type_symbols=("C", "S", "S"), label_atom_ids=("CA", "SG", "SD"),
            comp_ids=("ALA", "CYS", "MET"), asym_ids=("A", "A", "B"), entity_ids=("E1", "E1", "E2"),
            seq_ids=("1", "2", "1"), auth_asym_ids=("A", "A", "B"), auth_seq_ids=("1", "2", "1"),
        ),
        "_entity": _category_table(id=("E1", "E2"), type=("polymer", "polymer")),
        "_struct_asym": _category_table(id=("A", "B"), entity_id=("E1", "E2")),
        "_chem_comp": _category_table(id=("ALA", "CYS", "MET"), type=("x",) * 3),
        **_polymer_conditional_categories(),
    }
    for name in extra_omit_categories:
        categories[name] = _category_table(dummy_tag=("x",))
    decisions = compute_category_decisions(
        source_context_key=plan.plan.source_context_key, categories=categories,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=1,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=(),
    )
    by_cat = {d.category: d for d in decisions}
    assert len(extra_omit_categories) == 16
    for name in extra_omit_categories:
        assert by_cat[name].action == "omit", name
        assert by_cat[name].omitted_row_ordinals == (0,), name
        assert by_cat[name].source_rows_key is not None, name  # fully audited, never a fictional "all" sentinel
