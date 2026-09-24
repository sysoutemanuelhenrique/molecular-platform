"""Direct tests for the application services.

register_ligand_from_pubchem tests exercise the orchestration directly -
no CliRunner, no typer - proving resolution order, activity logging
counts, and cleanup guarantees. Business rules already covered by
test_pubchem.py (HTTP/acquisition behavior) and test_registry.py
(registration/identity-conflict behavior) are not repeated here.

enumerate_state_tautomers tests (Increment 016, Section 20.4 of the frozen
spec) use the private _resolve_engine/_enumerate injection seams with fake
implementations - never real RDKit - to exercise orchestration, ordering,
and compensation; the real RDKit adapter is covered separately, in
test_rdkit_tautomer.py.
"""

import importlib.util
import subprocess
import sys

import pytest
import yaml

import molecular.application as application_module
from molecular.application import (
    InvalidPubChemCidError,
    enumerate_state_tautomers,
    register_ligand_from_pubchem,
)
from molecular.context import (
    AmbiguousExperimentError,
    ExperimentNotFoundError,
    InvalidExperimentConfigError,
)
from molecular.execution import TautomerEnumerationPolicy
from molecular.pubchem import PubChemAcquisitionError, PubChemNotFoundError
from molecular.rdkit_tautomer import (
    IncompleteEnumerationError,
    InvalidSDFError,
    MultipleMoleculesError,
    RDKitEngineResolution,
    RDKitTautomerCandidate,
    RDKitTautomerEnumerationResult,
    RDKitUnavailableError,
)
from molecular.execution_registry import load_execution_attempt
from molecular.scientific import ScientificEntity
from molecular.scientific_registry import load_state, register_entity, register_state
from molecular.transformation_registry import list_state_transformations


def _init_valid_experiment(tmp_path, experiment_id="exp001", dirname=None):
    project = tmp_path / (dirname or experiment_id)
    project.mkdir(parents=True, exist_ok=True)
    (project / "experiment.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": "1.1", "experiment_id": experiment_id}, sort_keys=False
        ),
        encoding="utf-8",
    )
    return project


def _fake_fetch_returning(tmp_path, content=b"fake-pubchem-sdf-bytes"):
    """Fake fetch_pubchem_sdf_2d replacement returning a real temp file inside
    its own isolated directory, mirroring the real function's contract so
    cleanup_temp_artifact's rmtree-the-parent behavior only removes that
    isolated directory - without touching the network."""

    def _fetch(cid, timeout=None):
        temp_dir = tmp_path / f"fake_pubchem_temp_{cid}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"pubchem_{cid}_fake.sdf"
        temp_path.write_bytes(content)
        return temp_path

    return _fetch


def _activity_lines(project_dir):
    log_file = project_dir / "logs" / "activity.log"
    if not log_file.exists():
        return []
    return log_file.read_text(encoding="utf-8").splitlines()


# --- 1. success ---


def test_register_ligand_from_pubchem_success(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path)
    monkeypatch.setattr(
        application_module, "fetch_pubchem_sdf_2d", _fake_fetch_returning(tmp_path)
    )

    ligand = register_ligand_from_pubchem(
        "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
    )

    assert ligand.id == "curcumin"
    assert (project / "inputs" / "ligands" / "curcumin.yaml").exists()


# --- 2. invalid CID ---


def test_register_ligand_from_pubchem_rejects_invalid_cid(tmp_path):
    _init_valid_experiment(tmp_path)

    with pytest.raises(InvalidPubChemCidError, match=r"'0' is not a valid PubChem CID\."):
        register_ligand_from_pubchem("exp001", "curcumin", 0, "Curcumin", base_dir=tmp_path)


# --- 3. invalid CID never calls PubChem ---


def test_register_ligand_from_pubchem_invalid_cid_skips_network_call(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path)

    def _fail_if_called(cid, timeout=None):
        raise AssertionError("fetch_pubchem_sdf_2d must not be called for an invalid CID")

    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", _fail_if_called)

    with pytest.raises(InvalidPubChemCidError):
        register_ligand_from_pubchem("exp001", "curcumin", -5, "Curcumin", base_dir=tmp_path)


# --- 4. missing experiment ---


def test_register_ligand_from_pubchem_missing_experiment(tmp_path):
    with pytest.raises(ExperimentNotFoundError):
        register_ligand_from_pubchem(
            "nao_existe", "curcumin", 969516, "Curcumin", base_dir=tmp_path
        )


# --- 5. invalid experiment ---


def test_register_ligand_from_pubchem_invalid_experiment_config(tmp_path):
    project = tmp_path / "exp001"
    project.mkdir()
    (project / "experiment.yaml").write_text(
        yaml.safe_dump({"schema_version": "9.9", "experiment_id": "exp001"}),
        encoding="utf-8",
    )

    with pytest.raises(InvalidExperimentConfigError):
        register_ligand_from_pubchem(
            "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
        )


# --- 6. ambiguous experiment ---


def test_register_ligand_from_pubchem_ambiguous_experiment(tmp_path):
    _init_valid_experiment(tmp_path, experiment_id="exp001", dirname="a")
    _init_valid_experiment(tmp_path, experiment_id="exp001", dirname="b")

    with pytest.raises(AmbiguousExperimentError):
        register_ligand_from_pubchem(
            "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
        )


# --- 7. missing experiment + invalid CID -> ExperimentNotFoundError wins ---


def test_register_ligand_from_pubchem_missing_experiment_wins_over_invalid_cid(
    tmp_path, monkeypatch
):
    def _fail_if_called(cid, timeout=None):
        raise AssertionError("fetch_pubchem_sdf_2d must not be called")

    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", _fail_if_called)

    with pytest.raises(ExperimentNotFoundError):
        register_ligand_from_pubchem("nao_existe", "curcumin", 0, "Curcumin", base_dir=tmp_path)


# --- 8. PubChemNotFoundError ---


def test_register_ligand_from_pubchem_not_found(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path)

    def _raise_not_found(cid, timeout=None):
        raise PubChemNotFoundError(f"PubChem has no record for CID {cid}.")

    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", _raise_not_found)

    with pytest.raises(PubChemNotFoundError):
        register_ligand_from_pubchem(
            "exp001", "curcumin", 999999999, "Curcumin", base_dir=tmp_path
        )

    lines = _activity_lines(project)
    assert len(lines) == 1
    assert "outcome=error" in lines[0]


# --- 9. PubChemAcquisitionError ---


def test_register_ligand_from_pubchem_acquisition_error(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path)

    def _raise_acquisition_error(cid, timeout=None):
        raise PubChemAcquisitionError("PubChem request timed out after 30.0s for CID 969516.")

    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", _raise_acquisition_error)

    with pytest.raises(PubChemAcquisitionError):
        register_ligand_from_pubchem(
            "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
        )

    lines = _activity_lines(project)
    assert len(lines) == 1
    assert "outcome=error" in lines[0]


# --- 10. input_id conflict ---


def test_register_ligand_from_pubchem_rejects_duplicate_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path)
    monkeypatch.setattr(
        application_module, "fetch_pubchem_sdf_2d", _fake_fetch_returning(tmp_path)
    )

    register_ligand_from_pubchem("exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path)

    with pytest.raises(FileExistsError, match="already registered"):
        register_ligand_from_pubchem(
            "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
        )


# --- 11. cleanup on success ---


def test_register_ligand_from_pubchem_cleans_up_temp_on_success(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path)
    monkeypatch.setattr(
        application_module, "fetch_pubchem_sdf_2d", _fake_fetch_returning(tmp_path)
    )

    register_ligand_from_pubchem("exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path)

    assert not (tmp_path / "fake_pubchem_temp_969516").exists()


# --- 12. cleanup if registration fails ---


def test_register_ligand_from_pubchem_cleans_up_temp_on_registration_failure(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path)
    monkeypatch.setattr(
        application_module, "fetch_pubchem_sdf_2d", _fake_fetch_returning(tmp_path)
    )

    register_ligand_from_pubchem("exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path)

    with pytest.raises(FileExistsError):
        register_ligand_from_pubchem(
            "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
        )

    assert not (tmp_path / "fake_pubchem_temp_969516").exists()


# --- 13. provenance is exact ---


def test_register_ligand_from_pubchem_provenance_is_exact(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path)
    monkeypatch.setattr(
        application_module, "fetch_pubchem_sdf_2d", _fake_fetch_returning(tmp_path)
    )

    ligand = register_ligand_from_pubchem(
        "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
    )

    assert ligand.external_id == "CID 969516"
    assert ligand.acquisition.method == "downloaded"
    assert ligand.acquisition.source_name == "PubChem"
    assert ligand.acquisition.source_identifier == "CID 969516"
    assert ligand.acquisition.source_uri == (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/969516/SDF?record_type=2d"
    )
    assert ligand.acquisition.raw_artifact.format == "sdf"


# --- 14. activity success exactly once ---


def test_register_ligand_from_pubchem_records_activity_success_once(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path)
    monkeypatch.setattr(
        application_module, "fetch_pubchem_sdf_2d", _fake_fetch_returning(tmp_path)
    )

    register_ligand_from_pubchem("exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path)

    lines = _activity_lines(project)
    assert len(lines) == 1
    assert "command=ligand-fetch-pubchem" in lines[0]
    assert "outcome=success" in lines[0]


# --- 15. activity error exactly once after resolution ---


def test_register_ligand_from_pubchem_records_activity_error_once_after_resolution(tmp_path):
    project = _init_valid_experiment(tmp_path)

    with pytest.raises(InvalidPubChemCidError):
        register_ligand_from_pubchem("exp001", "curcumin", 0, "Curcumin", base_dir=tmp_path)

    lines = _activity_lines(project)
    assert len(lines) == 1
    assert "command=ligand-fetch-pubchem" in lines[0]
    assert "outcome=error" in lines[0]


# --- 16. resolution failure -> zero activity ---


def test_register_ligand_from_pubchem_resolution_failure_records_no_activity(tmp_path):
    with pytest.raises(ExperimentNotFoundError):
        register_ligand_from_pubchem(
            "nao_existe", "curcumin", 969516, "Curcumin", base_dir=tmp_path
        )

    assert not (tmp_path / "nao_existe").exists()
    assert list(tmp_path.iterdir()) == []


# --- 17. activity logging failure does not alter success ---


def test_register_ligand_from_pubchem_activity_failure_does_not_alter_success(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path)
    monkeypatch.setattr(
        application_module, "fetch_pubchem_sdf_2d", _fake_fetch_returning(tmp_path)
    )

    def _raise_oserror(*args, **kwargs):
        raise OSError("simulated activity log failure")

    monkeypatch.setattr(application_module, "record_activity", _raise_oserror)

    ligand = register_ligand_from_pubchem(
        "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
    )

    assert ligand.id == "curcumin"


# --- 18. activity logging failure does not replace the original exception ---


def test_register_ligand_from_pubchem_activity_failure_does_not_replace_original_exception(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path)

    def _raise_oserror(*args, **kwargs):
        raise OSError("simulated activity log failure")

    monkeypatch.setattr(application_module, "record_activity", _raise_oserror)

    with pytest.raises(InvalidPubChemCidError):
        register_ligand_from_pubchem("exp001", "curcumin", 0, "Curcumin", base_dir=tmp_path)


# --- identity invariant: experiment_id resolution, never base_dir/experiment_id ---


def test_register_ligand_from_pubchem_resolves_by_experiment_id_not_directory_name(
    tmp_path, monkeypatch
):
    project = _init_valid_experiment(
        tmp_path, experiment_id="exp001", dirname="some_other_folder_name"
    )
    monkeypatch.setattr(
        application_module, "fetch_pubchem_sdf_2d", _fake_fetch_returning(tmp_path)
    )

    ligand = register_ligand_from_pubchem(
        "exp001", "curcumin", 969516, "Curcumin", base_dir=tmp_path
    )

    assert ligand.id == "curcumin"
    assert (project / "inputs" / "ligands" / "curcumin.yaml").exists()
    assert not (tmp_path / "exp001").exists()


# ==========================================================================
# enumerate_state_tautomers (Increment 016, Section 20.4)
# ==========================================================================


def _entity(entity_id="curcumin", name="Curcumin"):
    return ScientificEntity(schema_version="1.0", id=entity_id, name=name)


def _write_sdf_source(tmp_path, name="structure.sdf", content=b"fake-sdf-bytes"):
    source_file = tmp_path / name
    source_file.write_bytes(content)
    return source_file


def _setup_input_state(tmp_path, entity_id="curcumin", state_id="state-a", file_format="sdf"):
    register_entity(tmp_path, _entity(entity_id=entity_id))
    return register_state(
        tmp_path,
        state_id=state_id,
        entity_id=entity_id,
        source_file=_write_sdf_source(tmp_path),
        format=file_format,
    )


def _policy(**overrides):
    fields = {"schema_version": "1.0", "max_states": None, "source": "scientist"}
    fields.update(overrides)
    return TautomerEnumerationPolicy(**fields)


_TWELVE_PARAMETERS = {
    "algorithm": "rdkit.MolStandardize.TautomerEnumerator",
    "input_format": "sdf",
    "output_format": "smiles",
    "max_tautomers": 1000,
    "max_transforms": 1000,
    "remove_bond_stereo": True,
    "remove_sp3_stereo": True,
    "reassign_stereo": True,
    "remove_isotopic_hs": False,
    "deduplication": "canonical-isomeric-smiles",
    "ordering": "canonical-isomeric-smiles-lexicographic",
    "selection": "complete-then-policy-limit",
}


def _fake_resolve_engine(**overrides):
    fields = dict(engine_name="rdkit", engine_version="2026.03.6", parameters=_TWELVE_PARAMETERS)
    fields.update(overrides)

    def _resolve():
        return RDKitEngineResolution(**fields)

    return _resolve


def _fake_resolve_engine_raising(exc):
    def _resolve():
        raise exc

    return _resolve


def _fake_enumerate_returning(smiles_list, enumerated_unique_count=None):
    unique = enumerated_unique_count if enumerated_unique_count is not None else len(smiles_list)

    def _enumerate(*, source_path, source_format):
        return RDKitTautomerEnumerationResult(
            enumerated_unique_count=unique,
            new_unique_count=len(smiles_list),
            candidates=tuple(
                RDKitTautomerCandidate(canonical_isomeric_smiles=smiles) for smiles in smiles_list
            ),
        )

    return _enumerate


def _fake_enumerate_raising(exc):
    def _enumerate(*, source_path, source_format):
        raise exc

    return _enumerate


def _fail_nth_call(real_fn, n, exception_factory):
    """Wraps real_fn so the n-th call (1-indexed) raises instead of
    delegating - every other call behaves exactly like real_fn."""

    state = {"count": 0}

    def _wrapper(*args, **kwargs):
        state["count"] += 1
        if state["count"] == n:
            raise exception_factory()
        return real_fn(*args, **kwargs)

    return _wrapper


def _noop_nth_call(real_fn, n):
    """Wraps real_fn so the n-th call (1-indexed) silently does nothing
    (simulating a best-effort removal that fails to confirm, exactly like
    _remove_state_best_effort/_remove_state_transformation_best_effort
    behave when the underlying OSError is swallowed) - every other call
    behaves exactly like real_fn."""

    state = {"count": 0}

    def _wrapper(*args, **kwargs):
        state["count"] += 1
        if state["count"] == n:
            return None
        return real_fn(*args, **kwargs)

    return _wrapper


def _capture_activity_calls(monkeypatch):
    """Monkeypatches application_module.record_activity to capture every
    call's (command, outcome, details) instead of writing to the log
    file - lets a test assert directly on the typed
    TautomerEnumerationActivityDetails instance actually passed, rather
    than parsing the text log line."""

    calls = []

    def _capture(project_dir, command, outcome, *, details=None):
        calls.append({"command": command, "outcome": outcome, "details": details})

    monkeypatch.setattr(application_module, "record_activity", _capture)
    return calls


def _rdkit_extra_is_installed():
    return importlib.util.find_spec("rdkit") is not None


def _attempt_ids(tmp_path):
    attempts_dir = tmp_path / "scientific" / "execution_attempts"
    if not attempts_dir.is_dir():
        return set()
    return {p.stem for p in attempts_dir.glob("*.yaml")}


def _new_attempt(tmp_path, before_ids):
    """enumerate_state_tautomers raises the original exception after
    recording it against its attempt (mirroring register_derived_state's
    convention) - so a failure test must catch that exception, then load
    the persisted attempt back by diffing the execution_attempts
    directory, rather than inspecting a return value."""

    new_ids = _attempt_ids(tmp_path) - before_ids
    assert len(new_ids) == 1, f"expected exactly one new attempt, got {new_ids}"
    return load_execution_attempt(tmp_path, next(iter(new_ids)))


# --- full success cycle ---


def test_enumerate_state_tautomers_full_success_cycle(tmp_path):
    _setup_input_state(tmp_path)

    attempt = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning(["C1", "C2", "C3"]),
    )

    assert attempt.status == "succeeded"
    assert attempt.input_state_id == "state-a"
    assert attempt.schema_version == "1.1"
    assert len(attempt.operation_ids) == 3

    transformations = list_state_transformations(tmp_path, parent_state_id="state-a")
    assert len(transformations) == 3
    assert set(t.id for t in transformations) == set(attempt.operation_ids)


def test_enumerate_state_tautomers_operation_ids_in_deterministic_publication_order(tmp_path):
    _setup_input_state(tmp_path)

    attempt = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc", "Cd"]),
    )

    transformations = [
        t
        for t in list_state_transformations(tmp_path, parent_state_id="state-a")
    ]
    by_id = {t.id: t for t in transformations}
    ranks_in_operation_id_order = [
        next(p.value for p in by_id[op_id].parameters if p.name == "candidate_rank")
        for op_id in attempt.operation_ids
    ]
    assert ranks_in_operation_id_order == [0, 1, 2, 3]


def test_enumerate_state_tautomers_published_states_inherit_parent_entity_id(tmp_path):
    parent = _setup_input_state(tmp_path)

    attempt = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning(["C1", "C2", "C3"]),
    )

    transformations = list_state_transformations(tmp_path, parent_state_id="state-a")
    assert len(transformations) == 3
    for transformation in transformations:
        child = load_state(tmp_path, transformation.child_state_id)
        assert child.entity_id == parent.entity_id


# --- resolution-phase failures ---


def test_enumerate_state_tautomers_engine_resolution_failure(tmp_path):
    _setup_input_state(tmp_path)
    before = _attempt_ids(tmp_path)

    with pytest.raises(RDKitUnavailableError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine_raising(RDKitUnavailableError("no engine")),
            _enumerate=_fake_enumerate_returning([]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "resolution_failed"
    assert attempt.error.phase == "resolution"


def test_enumerate_state_tautomers_missing_input_state(tmp_path):
    with pytest.raises(FileNotFoundError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="does-not-exist",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning([]),
        )

    attempt = _new_attempt(tmp_path, set())
    assert attempt.status == "resolution_failed"
    assert attempt.error.phase == "resolution"


def test_enumerate_state_tautomers_rejects_non_sdf_format(tmp_path):
    _setup_input_state(tmp_path, file_format="smiles")
    before = _attempt_ids(tmp_path)

    with pytest.raises(ValueError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning([]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "resolution_failed"
    assert attempt.error.phase == "resolution"


def test_enumerate_state_tautomers_rejects_checksum_mismatch(tmp_path):
    _setup_input_state(tmp_path)

    metadata_file = tmp_path / "scientific" / "states" / "state-a.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    data["representation"]["checksum"] = "0" * 64
    metadata_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    before = _attempt_ids(tmp_path)

    with pytest.raises(ValueError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning([]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "resolution_failed"
    assert attempt.error.phase == "resolution"


# --- execution-phase engine failures ---


@pytest.mark.parametrize(
    "exc",
    [
        InvalidSDFError("empty SDF"),
        MultipleMoleculesError(3),
        IncompleteEnumerationError(status="max-tautomers-reached", tautomer_count=2),
    ],
    ids=["invalid-sdf", "multiple-molecules", "incomplete-enumeration"],
)
def test_enumerate_state_tautomers_engine_execution_failure(tmp_path, exc):
    _setup_input_state(tmp_path)
    before = _attempt_ids(tmp_path)

    with pytest.raises(type(exc)):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_raising(exc),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "execution_failed"
    assert attempt.error.phase == "execution"
    assert list_state_transformations(tmp_path, parent_state_id="state-a") == []


# --- zero new states is a valid success ---


def test_enumerate_state_tautomers_zero_new_states_is_success(tmp_path):
    _setup_input_state(tmp_path)

    attempt = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning([], enumerated_unique_count=0),
    )

    assert attempt.status == "succeeded"
    assert attempt.operation_ids == ()
    assert list_state_transformations(tmp_path, parent_state_id="state-a") == []
    assert not (tmp_path / "scientific" / "states" / "state-a-child.yaml").exists()


# --- max_states truncation ---


@pytest.mark.parametrize("max_states,expected_published,expected_truncated", [
    (None, 4, False),
    (1, 1, True),
    (2, 2, True),
])
def test_enumerate_state_tautomers_max_states_truncation(
    tmp_path, max_states, expected_published, expected_truncated
):
    _setup_input_state(tmp_path)

    attempt = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(max_states=max_states),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc", "Cd"]),
    )

    assert len(attempt.operation_ids) == expected_published

    transformations = list_state_transformations(tmp_path, parent_state_id="state-a")
    assert len(transformations) == expected_published
    for transformation in transformations:
        truncated_param = next(p.value for p in transformation.parameters if p.name == "truncated_by_policy")
        published_count_param = next(
            p.value for p in transformation.parameters if p.name == "published_count"
        )
        assert truncated_param == expected_truncated
        assert published_count_param == expected_published


# --- rollback / compensation ---


def test_enumerate_state_tautomers_rollback_on_kth_register_state_failure(tmp_path, monkeypatch):
    _setup_input_state(tmp_path)
    before = _attempt_ids(tmp_path)

    real_register_state = application_module.register_state
    monkeypatch.setattr(
        application_module,
        "register_state",
        _fail_nth_call(real_register_state, 3, lambda: OSError("simulated failure")),
    )

    with pytest.raises(OSError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc", "Cd"]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "execution_failed"
    assert attempt.error.residual_state_ids == ()
    assert attempt.error.residual_operation_ids == ()
    assert list_state_transformations(tmp_path, parent_state_id="state-a") == []

    states_dir = tmp_path / "scientific" / "states"
    remaining = [p for p in states_dir.glob("*.yaml")] if states_dir.exists() else []
    assert remaining == [] or all(p.stem == "state-a" for p in remaining)


def test_enumerate_state_tautomers_rollback_preserves_prior_calls_records(tmp_path, monkeypatch):
    """A rollback triggered by a second, later call must never remove
    records published by an earlier, already-succeeded call."""

    _setup_input_state(tmp_path)

    first = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning(["Cfirst1", "Cfirst2"]),
    )
    assert first.status == "succeeded"
    first_transformations = list_state_transformations(tmp_path, parent_state_id="state-a")
    assert len(first_transformations) == 2

    real_register_state = application_module.register_state
    monkeypatch.setattr(
        application_module,
        "register_state",
        _fail_nth_call(real_register_state, 2, lambda: OSError("simulated failure")),
    )

    before = _attempt_ids(tmp_path)
    with pytest.raises(OSError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning(["Csecond1", "Csecond2"]),
        )

    second = _new_attempt(tmp_path, before)
    assert second.status == "execution_failed"

    remaining_transformations = list_state_transformations(tmp_path, parent_state_id="state-a")
    assert {t.id for t in remaining_transformations} == {t.id for t in first_transformations}


def test_enumerate_state_tautomers_rollback_on_kth_register_transformation_failure(
    tmp_path, monkeypatch
):
    _setup_input_state(tmp_path)
    before = _attempt_ids(tmp_path)

    real_register_transformation = application_module.register_state_transformation
    monkeypatch.setattr(
        application_module,
        "register_state_transformation",
        _fail_nth_call(real_register_transformation, 2, lambda: OSError("simulated failure")),
    )

    with pytest.raises(OSError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc"]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "execution_failed"
    assert list_state_transformations(tmp_path, parent_state_id="state-a") == []

    states_dir = tmp_path / "scientific" / "states"
    remaining = [p.stem for p in states_dir.glob("*.yaml")] if states_dir.exists() else []
    assert remaining == ["state-a"]


def test_enumerate_state_tautomers_rollback_on_mark_succeeded_failure(tmp_path, monkeypatch):
    _setup_input_state(tmp_path)

    real_mark_succeeded = application_module.mark_succeeded

    def _raise_mark_succeeded(*args, **kwargs):
        raise OSError("simulated mark_succeeded failure")

    monkeypatch.setattr(application_module, "mark_succeeded", _raise_mark_succeeded)

    with pytest.raises(OSError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc"]),
        )

    assert list_state_transformations(tmp_path, parent_state_id="state-a") == []
    states_dir = tmp_path / "scientific" / "states"
    remaining = [p.stem for p in states_dir.glob("*.yaml")] if states_dir.exists() else []
    assert remaining == ["state-a"]


def test_enumerate_state_tautomers_residual_ids_recorded_when_rollback_cannot_confirm_removal(
    tmp_path, monkeypatch
):
    _setup_input_state(tmp_path)
    before = _attempt_ids(tmp_path)

    real_register_state = application_module.register_state
    monkeypatch.setattr(
        application_module,
        "register_state",
        _fail_nth_call(real_register_state, 2, lambda: OSError("simulated failure")),
    )
    # The first candidate's state was fully published before the failure;
    # simulate compensation being unable to confirm its removal.
    monkeypatch.setattr(application_module, "_remove_state_best_effort", lambda *a, **k: None)

    with pytest.raises(OSError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc"]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "execution_failed"
    assert len(attempt.error.residual_state_ids) == 1

    states_dir = tmp_path / "scientific" / "states"
    remaining = {p.stem for p in states_dir.glob("*.yaml")}
    assert attempt.error.residual_state_ids[0] in remaining


# --- activity log semantics: published_*_ids means "actually registered by
# this execution", identically on success and error - never residual ids
# (Increment 016 adversarial code review reconciliation, MINOR-2) ---


def test_activity_log_success_records_exactly_the_published_ids(tmp_path, monkeypatch):
    _setup_input_state(tmp_path)
    calls = _capture_activity_calls(monkeypatch)

    attempt = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc"]),
    )

    success_calls = [c for c in calls if c["outcome"] == "success"]
    assert len(success_calls) == 1
    details = success_calls[0]["details"]

    transformations = list_state_transformations(tmp_path, parent_state_id="state-a")
    expected_state_ids = {t.child_state_id for t in transformations}

    assert set(details.published_state_ids) == expected_state_ids
    assert set(details.published_transformation_ids) == set(attempt.operation_ids)
    assert details.attempt_id == attempt.id
    assert details.input_state_id == "state-a"


def test_activity_log_error_after_partial_publication_records_everything_actually_registered(
    tmp_path, monkeypatch
):
    """Failure after partial publication (candidate 0 fully published,
    candidate 1's state created but its transformation registration
    fails): the error-outcome activity log must record BOTH ids that were
    actually registered by this execution before it ended - the same
    meaning the success path already gives these field names - never just
    the residual/unconfirmed-removal subset."""

    _setup_input_state(tmp_path)
    calls = _capture_activity_calls(monkeypatch)

    real_register_state_transformation = application_module.register_state_transformation
    monkeypatch.setattr(
        application_module,
        "register_state_transformation",
        _fail_nth_call(real_register_state_transformation, 2, lambda: OSError("simulated failure")),
    )

    before = _attempt_ids(tmp_path)
    with pytest.raises(OSError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc"]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "execution_failed"

    error_calls = [c for c in calls if c["outcome"] == "error"]
    assert len(error_calls) == 1
    details = error_calls[0]["details"]

    # candidate 0: state + transformation both registered; candidate 1:
    # only its state was registered before the transformation call failed.
    assert len(details.published_state_ids) == 2
    assert len(details.published_transformation_ids) == 1


def test_activity_log_still_shows_published_ids_when_compensation_removes_everything(
    tmp_path, monkeypatch
):
    """When compensation successfully confirms removal of everything it
    touched, residuals are empty - but the activity log must still show
    what was actually published during execution, not an empty set."""

    _setup_input_state(tmp_path)
    calls = _capture_activity_calls(monkeypatch)

    real_register_state_transformation = application_module.register_state_transformation
    monkeypatch.setattr(
        application_module,
        "register_state_transformation",
        _fail_nth_call(real_register_state_transformation, 2, lambda: OSError("simulated failure")),
    )

    before = _attempt_ids(tmp_path)
    with pytest.raises(OSError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc"]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.error.residual_state_ids == ()
    assert attempt.error.residual_operation_ids == ()

    error_calls = [c for c in calls if c["outcome"] == "error"]
    details = error_calls[0]["details"]
    assert len(details.published_state_ids) == 2
    assert len(details.published_transformation_ids) == 1


def test_activity_log_does_not_redefine_published_as_residual_when_residuals_exist(
    tmp_path, monkeypatch
):
    """With 4 fully-published candidates rolled back by a mark_succeeded
    failure, and compensation unable to confirm removal of exactly one
    transformation: the authoritative residual set (1 id) must differ
    from - and be strictly smaller than - what the activity log reports
    as published (4 ids). The log must never collapse "published" down to
    just "residual"."""

    _setup_input_state(tmp_path)
    calls = _capture_activity_calls(monkeypatch)

    def _raise_mark_succeeded(*args, **kwargs):
        raise OSError("simulated mark_succeeded failure")

    monkeypatch.setattr(application_module, "mark_succeeded", _raise_mark_succeeded)

    real_remove_transformation = application_module._remove_state_transformation_best_effort
    monkeypatch.setattr(
        application_module,
        "_remove_state_transformation_best_effort",
        _noop_nth_call(real_remove_transformation, 1),
    )

    before = _attempt_ids(tmp_path)
    with pytest.raises(OSError):
        enumerate_state_tautomers(
            tmp_path,
            input_state_id="state-a",
            policy=_policy(),
            _resolve_engine=_fake_resolve_engine(),
            _enumerate=_fake_enumerate_returning(["Ca", "Cb", "Cc", "Cd"]),
        )

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "execution_failed"

    # exactly one transformation's removal could not be confirmed; every
    # state's removal succeeded (removing a state never depends on its
    # transformation's own removal having been confirmed first).
    assert len(attempt.error.residual_operation_ids) == 1
    assert attempt.error.residual_state_ids == ()

    error_calls = [c for c in calls if c["outcome"] == "error"]
    assert len(error_calls) == 1
    details = error_calls[0]["details"]

    # all 4 candidates were fully published before mark_succeeded failed.
    assert len(details.published_state_ids) == 4
    assert len(details.published_transformation_ids) == 4

    # the residual set is a strict subset of what was published - the log
    # never redefines "published" as "residual".
    assert set(attempt.error.residual_operation_ids) < set(details.published_transformation_ids)
    assert len(details.published_transformation_ids) > len(attempt.error.residual_operation_ids)


# --- independence of successive attempts ---


def test_enumerate_state_tautomers_repeated_call_creates_independent_attempt(tmp_path):
    _setup_input_state(tmp_path)

    first = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning(["C1"]),
    )
    second = enumerate_state_tautomers(
        tmp_path,
        input_state_id="state-a",
        policy=_policy(),
        _resolve_engine=_fake_resolve_engine(),
        _enumerate=_fake_enumerate_returning(["C2"]),
    )

    assert first.id != second.id
    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert first.operation_ids != second.operation_ids


# --- packaging: no rdkit extra installed ---


@pytest.mark.skipif(
    _rdkit_extra_is_installed(), reason="only meaningful in a base install without the rdkit extra"
)
def test_enumerate_state_tautomers_without_rdkit_extra_produces_resolution_failed(tmp_path):
    """No monkeypatch, real defaults (Section 6.2/20.5 packaging test) -
    only runs for real in an environment where rdkit is genuinely absent."""

    _setup_input_state(tmp_path)

    before = _attempt_ids(tmp_path)

    with pytest.raises(RDKitUnavailableError):
        enumerate_state_tautomers(tmp_path, input_state_id="state-a", policy=_policy())

    attempt = _new_attempt(tmp_path, before)
    assert attempt.status == "resolution_failed"
    assert attempt.error.phase == "resolution"
    assert attempt.error.message


# --- packaging and dependency tests (Section 20.5) ---


def test_base_installation_surface_imports_cleanly():
    """Regression guard for Section 6.2/20.5: the pre-existing module
    surface must always import without error, whether or not the rdkit
    extra happens to be installed in this environment."""

    import molecular.application  # noqa: F401
    import molecular.cli  # noqa: F401
    import molecular.execution  # noqa: F401
    import molecular.execution_registry  # noqa: F401


@pytest.mark.skipif(
    _rdkit_extra_is_installed(),
    reason=(
        "only meaningful in a base install without the rdkit extra - when rdkit is "
        "genuinely installed, molecular.cli's existing import of molecular.application "
        "legitimately (and correctly) pulls rdkit into sys.modules on a real import, "
        "which is not the same thing as this codebase's own modules writing `import "
        "rdkit` outside rdkit_tautomer.py"
    ),
)
def test_cli_help_never_imports_rdkit():
    """Section 20.5: no chemistry-unrelated CLI command may import rdkit.
    Run in a fresh subprocess - sys.modules is shared, process-wide state,
    so an in-process check would be corrupted by any earlier test in this
    same pytest session that already imported rdkit (e.g.
    test_rdkit_tautomer.py)."""

    probe = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from molecular.cli import app\n"
        "CliRunner().invoke(app, ['--help'])\n"
        "print('rdkit' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.skipif(not _rdkit_extra_is_installed(), reason="requires the rdkit extra")
def test_real_resolved_snapshot_engine_version_matches_rdbase_version(tmp_path):
    """Section 20.5: the version reported by the environment
    (rdBase.rdkitVersion) must coincide with engine_version persisted in
    a real ResolvedExecutionSnapshot from an integration run - the full
    pipeline, using enumerate_state_tautomers' real defaults, not fakes."""

    from rdkit import rdBase

    parent = _setup_input_state(tmp_path)
    # register_state's fake-SDF-bytes fixture is not valid chemistry -
    # replace it with a minimal real SDF so the real adapter can parse it.
    from rdkit import Chem

    real_sdf = tmp_path / "real.sdf"
    mol = Chem.MolFromSmiles("CC(=O)CC(C)=O")
    writer = Chem.SDWriter(str(real_sdf))
    writer.write(mol)
    writer.close()

    register_state(
        tmp_path, state_id="state-real", entity_id=parent.entity_id, source_file=real_sdf, format="sdf"
    )

    attempt = enumerate_state_tautomers(tmp_path, input_state_id="state-real", policy=_policy())

    assert attempt.status == "succeeded"
    assert attempt.resolved.engine_version == rdBase.rdkitVersion
