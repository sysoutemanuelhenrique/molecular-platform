"""Persistence tests for receptor_registry.py - Increment 020, Checkpoint 2.

Covers T-069-080 (preserved/rewritten) and T-125-130 (new) from the frozen
final specification's normative matrix (Section 35.5). Each test is
labeled with its exact scenario id.

T-125-130's own prose ("chamada de inspect_receptor") describes the
future Application Service's orchestration (Checkpoint 3) - this
checkpoint builds only the registry layer `inspect_receptor` will
eventually call. `_get_or_create_inspection` below is test-local
scaffolding (not new production API) that composes exactly the two
registry functions Checkpoint 1 already built
(`load_receptor_inspection`/`register_receptor_inspection`) into the
create-or-reuse pattern a future `inspect_receptor` will perform,
mirroring how test_receptor_inspection.py's own T-121 (Checkpoint 1)
worked around a capability gap with a documented, test-local
construction rather than anticipating later-checkpoint production code.

No Application Service, retry-vs-conflict decision for ReceptorSelectionPolicy,
or chain/model/assembly resolution logic is exercised here - those are
receptor_service.py's responsibility (Checkpoint 3).
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import molecular.receptor_inspection as receptor_inspection_module
from molecular.receptor_inspection import (
    INSPECTION_PROFILE_VERSION,
    INSPECTION_SCHEMA_VERSION,
    ParserProvenance,
    ReceptorInspectionReport,
    compute_inspection_id,
)
from molecular.receptor_registry import (
    ReceptorInspectionNotFoundError,
    ReceptorResolutionIdentityConflictError,
    ReceptorResolutionNotFoundError,
    ReceptorResolutionRecordInvalidError,
    ReceptorSelectionIdentityConflictError,
    ReceptorSelectionNotFoundError,
    ReceptorSelectionRecordInvalidError,
    _receptor_inspections_dir,
    _receptor_selections_dir,
    _resolved_receptor_selections_dir,
    list_receptor_selections,
    load_receptor_inspection,
    load_receptor_selection,
    load_resolved_receptor_selection,
    register_receptor_inspection,
    register_receptor_selection,
    register_resolved_receptor_selection,
)
from molecular.receptor_resolution import ResolvedChainSelection, ResolvedReceptorSelection
from molecular.receptor_selection import ChainSelector, ReceptorSelectionPolicy

_HEX64_A = "a" * 64
_HEX64_B = "b" * 64
_DEMO_CHECKSUM = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# --- fixture helpers ----------------------------------------------------


def _policy(**overrides) -> ReceptorSelectionPolicy:
    defaults = dict(
        schema_version="1.0",
        selection_id="sel-001",
        protein_input_id="demo-protein",
        inspection_id=_HEX64_A,
        raw_checksum_observed=_HEX64_B,
        selected_model=1,
        selected_assembly=None,
        selected_chains=(ChainSelector(label_asym_id="A"),),
        water_disposition="keep",
        metal_disposition="keep",
        component_dispositions=(),
        declared_by="Dr. Jane Doe",
        declared_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return ReceptorSelectionPolicy(**defaults)


def _resolution(**overrides) -> ResolvedReceptorSelection:
    defaults = dict(
        schema_version="1.0",
        resolved_selection_id="sel-001",
        policy_selection_id="sel-001",
        protein_input_id="demo-protein",
        inspection_id=_HEX64_A,
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


def _inspection_report(
    *,
    protein_input_id="demo-protein",
    raw_checksum=_DEMO_CHECKSUM,
    parser_version="0.7.5",
    raw_format="mmcif",
    inspection_id=None,
    inspection_profile_version=None,
) -> ReceptorInspectionReport:
    profile_version = inspection_profile_version or INSPECTION_PROFILE_VERSION
    computed_id = inspection_id or compute_inspection_id(
        parser_name="gemmi",
        parser_version=parser_version,
        protein_input_id=protein_input_id,
        raw_checksum=raw_checksum,
        raw_format=raw_format,
    )
    return ReceptorInspectionReport(
        inspection_id=computed_id,
        inspection_schema_version=INSPECTION_SCHEMA_VERSION,
        inspection_profile_version=profile_version,
        protein_input_id=protein_input_id,
        raw_checksum=raw_checksum,
        raw_format_declared=raw_format,
        structure_format=raw_format,
        parser=ParserProvenance(name="gemmi", version=parser_version),
        selection_capability="full",
        models=(),
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


def _get_or_create_inspection(project_dir, report: ReceptorInspectionReport):
    """Test-local create-or-reuse helper (see module docstring): composes
    the two registry primitives Checkpoint 1 already built into the exact
    pattern a future `inspect_receptor` (Checkpoint 3) will perform -
    never new production surface."""

    try:
        existing = load_receptor_inspection(project_dir, report.inspection_id)
        return existing, False
    except ReceptorInspectionNotFoundError:
        return register_receptor_inspection(project_dir, report), True


# --- ReceptorSelectionPolicy persistence (T-069-075) ---------------------


def test_register_receptor_selection_creates_metadata_file(tmp_path):  # T-069
    policy = register_receptor_selection(tmp_path, _policy())

    metadata_file = _receptor_selections_dir(tmp_path) / f"{policy.selection_id}.yaml"
    assert metadata_file.is_file()


def test_casefold_collision_raises_selection_identity_conflict(tmp_path):  # T-070
    register_receptor_selection(tmp_path, _policy(selection_id="sel-001"))

    with pytest.raises(ReceptorSelectionIdentityConflictError):
        register_receptor_selection(
            tmp_path, _policy(selection_id="SEL-001", declared_by="Someone Else")
        )


def test_load_receptor_selection_missing_raises_not_found(tmp_path):  # T-071
    with pytest.raises(ReceptorSelectionNotFoundError):
        load_receptor_selection(tmp_path, "does-not-exist")


def test_load_receptor_selection_malformed_yaml_raises_record_invalid(tmp_path):  # T-072
    selections_dir = _receptor_selections_dir(tmp_path)
    selections_dir.mkdir(parents=True)
    (selections_dir / "broken-001.yaml").write_text(
        "schema_version: [unterminated\n", encoding="utf-8"
    )

    with pytest.raises(ReceptorSelectionRecordInvalidError) as excinfo:
        load_receptor_selection(tmp_path, "broken-001")

    assert excinfo.value.__cause__ is not None


def test_load_receptor_selection_schema_invalid_raises_record_invalid(tmp_path):  # T-073
    selections_dir = _receptor_selections_dir(tmp_path)
    selections_dir.mkdir(parents=True)
    (selections_dir / "incompatible-001.yaml").write_text(
        "schema_version: '2.0'\nselection_id: incompatible-001\n", encoding="utf-8"
    )

    with pytest.raises(ReceptorSelectionRecordInvalidError) as excinfo:
        load_receptor_selection(tmp_path, "incompatible-001")

    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_list_receptor_selections_empty_project_returns_empty_tuple(tmp_path):  # T-074
    assert list_receptor_selections(tmp_path) == ()


def test_list_receptor_selections_orders_by_declared_at_then_id(tmp_path):  # T-075
    first = register_receptor_selection(
        tmp_path,
        _policy(selection_id="sel-alpha", declared_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
    )
    second = register_receptor_selection(
        tmp_path,
        _policy(selection_id="sel-beta", declared_at=datetime(2026, 1, 2, tzinfo=timezone.utc)),
    )

    listing = list_receptor_selections(tmp_path)

    assert [policy.selection_id for policy in listing] == [first.selection_id, second.selection_id]


# --- ResolvedReceptorSelection persistence (T-076) -----------------------


def test_register_resolved_receptor_selection_creates_metadata_file(tmp_path):  # T-076
    resolution = register_resolved_receptor_selection(tmp_path, _resolution())

    metadata_file = (
        _resolved_receptor_selections_dir(tmp_path) / f"{resolution.resolved_selection_id}.yaml"
    )
    assert metadata_file.is_file()


def test_casefold_collision_raises_resolution_identity_conflict(tmp_path):  # CP2-REGISTRY-EXTRA-01
    # Not a numbered T-xxx scenario: the frozen matrix exercises this exact
    # identity-conflict mechanism for inspections (shared check_storage_
    # identity_conflict primitive) but does not spell out a dedicated
    # scenario id for the resolved-selection side specifically. Added as
    # engineering-discipline coverage, mirroring how Checkpoint 1's own
    # correction added 16 non-normative CP1-MAJOR-labeled tests.
    register_resolved_receptor_selection(
        tmp_path, _resolution(resolved_selection_id="sel-001", policy_selection_id="sel-001")
    )

    with pytest.raises(ReceptorResolutionIdentityConflictError):
        register_resolved_receptor_selection(
            tmp_path, _resolution(resolved_selection_id="SEL-001", policy_selection_id="SEL-001")
        )


def test_load_resolved_receptor_selection_missing_raises_not_found(tmp_path):  # CP2-REGISTRY-EXTRA-02
    with pytest.raises(ReceptorResolutionNotFoundError):
        load_resolved_receptor_selection(tmp_path, "does-not-exist")


def test_load_resolved_receptor_selection_malformed_yaml_raises_record_invalid(tmp_path):  # CP2-REGISTRY-EXTRA-03
    resolved_dir = _resolved_receptor_selections_dir(tmp_path)
    resolved_dir.mkdir(parents=True)
    (resolved_dir / "broken-001.yaml").write_text(
        "schema_version: [unterminated\n", encoding="utf-8"
    )

    with pytest.raises(ReceptorResolutionRecordInvalidError) as excinfo:
        load_resolved_receptor_selection(tmp_path, "broken-001")

    assert excinfo.value.__cause__ is not None


# --- inspection registry create-once mechanics (T-077, T-078) ------------


def test_second_identical_inspection_registration_is_a_create_once_hit_no_rewrite(tmp_path):  # T-077
    report = _inspection_report()
    register_receptor_inspection(tmp_path, report)
    metadata_file = _receptor_inspections_dir(tmp_path) / f"{report.inspection_id}.yaml"
    before_content = metadata_file.read_text(encoding="utf-8")
    before_mtime = metadata_file.stat().st_mtime_ns

    reused_report, created = _get_or_create_inspection(tmp_path, report)

    after_content = metadata_file.read_text(encoding="utf-8")
    after_mtime = metadata_file.stat().st_mtime_ns
    assert created is False
    assert reused_report == report
    assert after_content == before_content
    assert after_mtime == before_mtime


def test_different_inspection_key_produces_a_separate_file_prior_intact(tmp_path):  # T-078
    first = _inspection_report(raw_checksum=_HEX64_A)
    second = _inspection_report(raw_checksum=_HEX64_B)
    assert first.inspection_id != second.inspection_id

    register_receptor_inspection(tmp_path, first)
    register_receptor_inspection(tmp_path, second)

    assert load_receptor_inspection(tmp_path, first.inspection_id) == first
    assert load_receptor_inspection(tmp_path, second.inspection_id) == second


# --- no staging, atomic writes (T-079, T-080) -----------------------------


def test_no_staging_directory_created_under_any_of_the_three_paths(tmp_path):  # T-079
    register_receptor_inspection(tmp_path, _inspection_report())
    register_receptor_selection(tmp_path, _policy())
    register_resolved_receptor_selection(tmp_path, _resolution())

    for directory in (
        _receptor_inspections_dir(tmp_path),
        _receptor_selections_dir(tmp_path),
        _resolved_receptor_selections_dir(tmp_path),
    ):
        entries = {entry.name for entry in directory.iterdir()}
        assert ".staging" not in entries


def test_atomic_write_leaves_no_stray_temp_file_for_any_of_the_three_paths(tmp_path):  # T-080
    register_receptor_inspection(tmp_path, _inspection_report())
    register_receptor_selection(tmp_path, _policy())
    register_resolved_receptor_selection(tmp_path, _resolution())

    for directory in (
        _receptor_inspections_dir(tmp_path),
        _receptor_selections_dir(tmp_path),
        _resolved_receptor_selections_dir(tmp_path),
    ):
        for entry in directory.iterdir():
            if entry.is_file():
                assert entry.suffix == ".yaml"


# --- inspection create-or-reuse semantics (T-125-129) --------------------


def test_first_inspection_registration_for_unseen_key_creates(tmp_path):  # T-125
    report = _inspection_report()

    result, created = _get_or_create_inspection(tmp_path, report)

    assert created is True
    assert result == report
    metadata_file = _receptor_inspections_dir(tmp_path) / f"{report.inspection_id}.yaml"
    assert metadata_file.is_file()


def test_second_identical_call_reports_reused_never_a_timestamp_inference(tmp_path):  # T-126
    report = _inspection_report()
    _get_or_create_inspection(tmp_path, report)
    metadata_file = _receptor_inspections_dir(tmp_path) / f"{report.inspection_id}.yaml"
    before_content = metadata_file.read_text(encoding="utf-8")

    result, created = _get_or_create_inspection(tmp_path, report)

    assert created is False
    assert result == report
    assert metadata_file.read_text(encoding="utf-8") == before_content


def test_different_raw_checksum_yields_a_different_inspection_id_prior_intact(tmp_path):  # T-127
    first = _inspection_report(raw_checksum=_HEX64_A)
    second = _inspection_report(raw_checksum=_HEX64_B)
    assert first.inspection_id != second.inspection_id

    _get_or_create_inspection(tmp_path, first)
    _get_or_create_inspection(tmp_path, second)

    assert load_receptor_inspection(tmp_path, first.inspection_id) == first
    assert load_receptor_inspection(tmp_path, second.inspection_id) == second


def test_different_parser_version_yields_a_different_inspection_id_no_conflict(tmp_path):  # T-128
    first = _inspection_report(parser_version="0.7.5")
    second = _inspection_report(parser_version="0.7.6")
    assert first.inspection_id != second.inspection_id

    _, created_first = _get_or_create_inspection(tmp_path, first)
    _, created_second = _get_or_create_inspection(tmp_path, second)

    assert created_first is True
    assert created_second is True
    assert load_receptor_inspection(tmp_path, first.inspection_id) == first


def test_different_inspection_profile_version_yields_a_different_inspection_id(
    tmp_path, monkeypatch
):  # T-129
    # `ReceptorInspectionReport.inspection_profile_version` is typed as
    # `Literal["1.0"]` in Checkpoint 1's receptor_inspection.py (outside
    # this checkpoint's allowlist) - a real profile bump would ship with an
    # updated Literal there too, so this scenario is reconciled at the
    # level it actually claims (Section 15.1/23.3): compute_inspection_id
    # reads the live INSPECTION_PROFILE_VERSION global, producing a
    # different id under a patched profile version, with the prior
    # inspection - registered under the original id - left fully intact
    # (same behavior as T-128).
    first = _inspection_report()
    _get_or_create_inspection(tmp_path, first)

    monkeypatch.setattr(receptor_inspection_module, "INSPECTION_PROFILE_VERSION", "2.0")
    patched_id = receptor_inspection_module.compute_inspection_id(
        parser_name="gemmi",
        parser_version="0.7.5",
        protein_input_id="demo-protein",
        raw_checksum=_DEMO_CHECKSUM,
        raw_format="mmcif",
    )

    assert patched_id != first.inspection_id
    assert load_receptor_inspection(tmp_path, first.inspection_id) == first


# --- multiple resolutions never duplicate the inspection (T-130) ---------


def test_two_distinct_resolutions_sharing_one_inspection_id_write_it_once(tmp_path):  # T-130
    report = _inspection_report()
    register_receptor_inspection(tmp_path, report)
    metadata_file = _receptor_inspections_dir(tmp_path) / f"{report.inspection_id}.yaml"
    before_mtime = metadata_file.stat().st_mtime_ns

    register_resolved_receptor_selection(
        tmp_path,
        _resolution(
            resolved_selection_id="sel-one",
            policy_selection_id="sel-one",
            inspection_id=report.inspection_id,
        ),
    )
    register_resolved_receptor_selection(
        tmp_path,
        _resolution(
            resolved_selection_id="sel-two",
            policy_selection_id="sel-two",
            inspection_id=report.inspection_id,
        ),
    )

    after_mtime = metadata_file.stat().st_mtime_ns
    assert after_mtime == before_mtime
    assert load_receptor_inspection(tmp_path, report.inspection_id) == report
