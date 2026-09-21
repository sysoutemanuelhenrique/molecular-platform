"""Direct tests for the PubChem ligand acquisition application service.

These tests exercise register_ligand_from_pubchem directly - no CliRunner,
no typer - to prove the orchestration itself: resolution order, activity
logging counts, and cleanup guarantees. Business rules already covered by
test_pubchem.py (HTTP/acquisition behavior) and test_registry.py
(registration/identity-conflict behavior) are not repeated here.
"""

import pytest
import yaml

import molecular.application as application_module
from molecular.application import InvalidPubChemCidError, register_ligand_from_pubchem
from molecular.context import (
    AmbiguousExperimentError,
    ExperimentNotFoundError,
    InvalidExperimentConfigError,
)
from molecular.pubchem import PubChemAcquisitionError, PubChemNotFoundError


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
