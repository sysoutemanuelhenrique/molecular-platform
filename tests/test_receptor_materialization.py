"""Tests for receptor_materialization.py - Increment 021, Checkpoint 1.

Covers all 43 Checkpoint 1 scenarios from the frozen final specification's
normative matrix (increments/021/15-final-specification/evidence/
SCENARIOS_021.csv): T021-001-040 and T021-159-161. Each test is labeled
with its exact scenario id. This checkpoint is domain-only - no RAW, no
Gemmi, no filesystem - so every fixture here is a directly constructed,
in-memory DTO; `AtomObservation` rows stand in for what the Checkpoint 2
Gemmi adapter will eventually supply, never a stub that fabricates a
constant "scientific success".
"""

import unicodedata
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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
    Engine,
    MaterializationConfigurationError,
    MaterializationInputError,
    MaterializationPlan,
    OccurrenceIdentity,
    PlanBinding,
    PolicyRecord,
    RecordSnapshot,
    build_materialization_plan,
    build_policy_record,
    build_source_context,
    compute_materialization_hash,
    compute_occurrence_key,
    compute_policy_id,
    compute_request_key,
    validate_time_utc,
)
from molecular.receptor_resolution import (
    ResolvedChainSelection,
    ResolvedComponentDisposition,
    ResolvedReceptorSelection,
)

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


def _source_context(**overrides):
    inspection_report = overrides.pop("inspection_report", None) or _inspection_report()
    resolution = overrides.pop("resolved_selection", None) or _resolution()
    records = overrides.pop("records", None) or _records()
    return build_source_context(
        protein_input_id=overrides.pop("protein_input_id", PROTEIN_INPUT_ID),
        inspection_report=inspection_report,
        policy_selection_id=overrides.pop("policy_selection_id", SELECTION_ID),
        policy_inspection_id=overrides.pop("policy_inspection_id", INSPECTION_ID),
        policy_raw_checksum_observed=overrides.pop("policy_raw_checksum_observed", RAW_SHA),
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
        _source_context(policy_selection_id="different-sel")
    with pytest.raises(MaterializationConfigurationError):
        _source_context(resolved_selection=_resolution(raw_checksum=_hex(99)))


def test_inspection_id_payload_must_reproduce_itself():  # T021-007
    tampered_id = _hex(77)
    tampered_report = _inspection_report(inspection_id=tampered_id)
    with pytest.raises(MaterializationConfigurationError):
        _source_context(
            inspection_report=tampered_report,
            resolved_selection=_resolution(inspection_id=tampered_id),
            policy_inspection_id=tampered_id,
        )


def test_truncated_inspection_rejected_for_non_full_or_parser_limitation():  # T021-008
    with pytest.raises(MaterializationConfigurationError):
        _source_context(inspection_report=_inspection_report(selection_capability="inspection-only"))
    limited = _inspection_report(warnings=(InspectionWarning(code="parser-limitation", detail="x"),))
    with pytest.raises(MaterializationConfigurationError):
        _source_context(inspection_report=limited)


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
    )
    assert decision.action == "omit" and decision.omitted_tags == ()

    with pytest.raises(ValidationError):
        CategoryDecision(category="_struct_ncs_oper", action="guard_then_omit", omitted_tags=(), preserved_tags=(), generated_tags=())

    guarded = CategoryDecision(
        category="_struct_ncs_oper", action="guard_then_omit",
        omitted_tags=("id", "matrix[1][1]"), preserved_tags=(), generated_tags=(),
    )
    assert guarded.omitted_tags == ("id", "matrix[1][1]")
