import decimal
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

import molecular.application as application_module
import molecular.cli as cli_module
from molecular.atomic import (
    RawArtifactIdentityUnstableError,
    RawArtifactSharingViolationError,
    UnsafeRawArtifactPathError,
)
from molecular.cli import app
from molecular.promotion_registry import PromotionPublicationUnavailableError
from molecular.promotion_service import (
    LigandPromotionAttemptFailedError,
    LigandPromotionAttemptPersistenceError,
    PromotionConsistencyError,
    PromotionIdentityConflictError,
    PromotionReservationConflictError,
    PromotionRevisionRequiredError,
    UnsupportedPromotionFormatError,
)
from molecular.rdkit_tautomer import (
    DisconnectedStructureError,
    IncompleteEnumerationError,
    InvalidPromotionSDFError,
    InvalidSDFError,
    MultipleMoleculesError,
    PromotionRecordCountError,
    RDKitUnavailableError,
)
from molecular.scientific import ScientificEntity


runner = CliRunner()


def test_init_creates_experiment_structure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "exp001"])

    assert result.exit_code == 0

    project = tmp_path / "exp001"

    assert project.exists()
    assert (project / "experiment.yaml").exists()
    assert (project / "inputs" / "receptor").exists()
    assert (project / "inputs" / "ligands").exists()
    assert (project / "inputs" / "proteins").exists()
    assert (project / "artifacts").exists()
    assert (project / "results").exists()
    assert (project / "logs").exists()


def test_validate_accepts_valid_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        'schema_version: "1.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", "exp001"])

    assert result.exit_code == 0
    assert "Experiment 'exp001' is valid (schema 1.0)." in result.output


def test_validate_writes_activity_log_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        'schema_version: "1.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    log_file = project / "logs" / "activity.log"
    assert not log_file.exists()

    runner.invoke(app, ["validate", "exp001"])

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "command=validate" in lines[0]
    assert "outcome=success" in lines[0]


def test_validate_treats_invalid_schema_with_known_id_as_configuration_error(tmp_path, monkeypatch):
    """validate resolves by logical experiment_id like every other command.
    Here the YAML parses fine and clearly declares experiment_id 'exp001' -
    only schema_version fails validation. That identity is confiably known
    (read from the YAML itself, never from the directory name), so the
    command must report a config error tied to it, not claim the experiment
    doesn't exist."""

    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        'schema_version: "2.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", "exp001"])

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output
    assert "Experiment 'exp001' is invalid." in result.output
    assert "schema_version" in result.output


def test_validate_does_not_record_activity_when_candidate_cannot_be_resolved(tmp_path, monkeypatch):
    """No context can be resolved for a broken candidate, so there is no
    project_dir to log against - the failure leaves no activity log entry."""

    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        'schema_version: "2.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    runner.invoke(app, ["validate", "exp001"])

    assert not (project / "logs" / "activity.log").exists()

def test_validate_rejects_missing_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["validate", "nao_existe"])

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'nao_existe' was not found." in result.output


def test_validate_missing_experiment_does_not_create_project_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["validate", "nao_existe"])

    assert not (tmp_path / "nao_existe").exists()


def test_validate_treats_malformed_yaml_as_not_found(tmp_path, monkeypatch):
    """A candidate with malformed YAML can't yield an experiment_id either,
    so - same as an invalid schema - it resolves to 'not found'."""

    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        "experiment_id: [unclosed\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", "exp001"])

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'exp001' was not found." in result.output


def test_plan_valid_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        'schema_version: "1.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["plan", "exp001"])

    assert result.exit_code == 0
    assert "EXPERIMENT PLAN" in result.output
    assert "Experiment: exp001" in result.output
    assert "Schema: 1.0" in result.output
    assert "Status: VALID" in result.output
    assert "No execution steps are defined yet." in result.output


def test_plan_and_validate_append_activity_entries_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        'schema_version: "1.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    runner.invoke(app, ["validate", "exp001"])
    runner.invoke(app, ["plan", "exp001"])

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert "command=validate" in lines[0]
    assert "command=plan" in lines[1]


def test_plan_rejects_missing_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["plan", "nao_existe"])

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'nao_existe' was not found." in result.output


def test_plan_treats_invalid_schema_with_known_id_as_configuration_error(tmp_path, monkeypatch):
    """Same resolution rule as validate: a candidate whose YAML declares a
    usable experiment_id but fails schema validation surfaces as a config
    error tied to that known identity, not as 'not found'."""

    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        'schema_version: "2.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["plan", "exp001"])

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output
    assert "Experiment 'exp001' is invalid." in result.output
    assert "schema_version" in result.output


def test_plan_does_not_record_activity_when_candidate_cannot_be_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        'schema_version: "2.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    runner.invoke(app, ["plan", "exp001"])

    assert not (project / "logs" / "activity.log").exists()


def _init_valid_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()
    (project / "experiment.yaml").write_text(
        'schema_version: "1.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )
    return project


def test_ligand_add_registers_uploaded_ligand(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code == 0
    assert "Ligand 'curcumin' registered successfully." in result.output
    assert (project / "inputs" / "ligands" / "curcumin.yaml").exists()


def test_ligand_add_registers_downloaded_ligand_with_source_info(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--external-id",
            "CID 969516",
            "--acquisition-method",
            "downloaded",
            "--source-name",
            "PubChem",
            "--source-identifier",
            "CID 969516",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code == 0
    assert "Ligand 'curcumin' registered successfully." in result.output


def test_protein_add_registers_downloaded_protein(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = tmp_path / "3la4.cif"
    raw_file.write_bytes(b"fake-cif-bytes")

    result = runner.invoke(
        app,
        [
            "protein",
            "add",
            "exp001",
            "urease",
            "--name",
            "Jack bean urease",
            "--external-id",
            "3LA4",
            "--acquisition-method",
            "downloaded",
            "--source-name",
            "RCSB PDB",
            "--source-identifier",
            "3LA4",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code == 0
    assert "Protein 'urease' registered successfully." in result.output
    assert (project / "inputs" / "proteins" / "urease.yaml").exists()


def test_ligand_add_writes_activity_log_entry(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert "command=ligand-add" in lines[0]
    assert "outcome=success" in lines[0]


def test_ligand_add_rejects_missing_raw_file(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(tmp_path / "missing.sdf"),
        ],
    )

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=ligand-add" in lines[0]
    assert "outcome=error" in lines[0]


def test_ligand_add_rejects_duplicate_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    args = [
        "ligand",
        "add",
        "exp001",
        "curcumin",
        "--name",
        "Curcumin",
        "--acquisition-method",
        "uploaded",
        "--raw-file",
        str(raw_file),
    ]

    runner.invoke(app, args)
    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output


def test_ligand_add_rejects_downloaded_without_source_info(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "downloaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output
    assert "Ligand 'curcumin' is invalid." in result.output


def test_ligand_add_rejects_invalid_acquisition_method(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "synthesized",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output


def test_ligand_add_rejects_missing_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "nao_existe",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'nao_existe' was not found." in result.output


def test_ligand_add_handles_storage_error_without_traceback(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "register_ligand", _raise_storage_error)

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code == 1
    assert not isinstance(result.exception, OSError)
    assert "STORAGE ERROR" in result.output
    assert "simulated disk failure" in result.output

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "command=ligand-add" in lines[0]
    assert "outcome=error" in lines[0]


def _fake_fetch_returning(tmp_path, content=b"fake-pubchem-sdf-bytes"):
    """Build a fake fetch_pubchem_sdf_2d replacement that returns a real temp
    file inside its own isolated directory (mirroring the real function's
    contract, so cleanup_temp_artifact's rmtree-the-parent behavior only
    removes that isolated directory), without touching the network."""

    def _fetch(cid, timeout=None):
        temp_dir = tmp_path / f"fake_pubchem_temp_{cid}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"pubchem_{cid}_fake.sdf"
        temp_path.write_bytes(content)
        return temp_path

    return _fetch


def test_ligand_fetch_pubchem_registers_successfully(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    fake_fetch = _fake_fetch_returning(tmp_path)
    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", fake_fetch)

    result = runner.invoke(
        app,
        [
            "ligand",
            "fetch-pubchem",
            "exp001",
            "curcumin",
            "--cid",
            "969516",
            "--name",
            "Curcumin",
        ],
    )

    assert result.exit_code == 0
    assert "Ligand 'curcumin' registered successfully." in result.output

    metadata_file = project / "inputs" / "ligands" / "curcumin.yaml"
    assert metadata_file.exists()

    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    acquisition = metadata["acquisition"]
    assert metadata["external_id"] == "CID 969516"
    assert acquisition["method"] == "downloaded"
    assert acquisition["source_name"] == "PubChem"
    assert acquisition["source_identifier"] == "CID 969516"
    assert acquisition["source_uri"] == (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/969516/SDF?record_type=2d"
    )
    assert acquisition["raw_artifact"]["format"] == "sdf"

    stored_file = project / "inputs" / "ligands" / "curcumin" / "pubchem_969516_fake.sdf"
    assert stored_file.exists()
    assert stored_file.read_bytes() == b"fake-pubchem-sdf-bytes"

    # the temporary directory created by the (fake) fetch must be fully cleaned up
    assert not (tmp_path / "fake_pubchem_temp_969516").exists()

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "command=ligand-fetch-pubchem" in lines[0]
    assert "outcome=success" in lines[0]


class _FakePubchemResponse:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._data


def test_ligand_fetch_pubchem_persists_deterministic_raw_artifact_name(tmp_path, monkeypatch):
    """End-to-end through the real fetch_pubchem_sdf_2d (only urlopen is faked),
    proving the persisted RAW artifact name is deterministic and cleanup is total."""

    project = _init_valid_experiment(tmp_path, monkeypatch)

    def _fake_urlopen(url, timeout=None):
        return _FakePubchemResponse(b"fake-sdf-bytes-for-969516")

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    result = runner.invoke(
        app,
        [
            "ligand",
            "fetch-pubchem",
            "exp001",
            "curcumin",
            "--cid",
            "969516",
            "--name",
            "Curcumin",
        ],
    )

    assert result.exit_code == 0

    metadata_file = project / "inputs" / "ligands" / "curcumin.yaml"
    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    raw_path = metadata["acquisition"]["raw_artifact"]["path"]

    assert raw_path == "inputs/ligands/curcumin/pubchem_cid_969516_2d.sdf"

    stored_file = project / raw_path
    assert stored_file.exists()
    assert stored_file.read_bytes() == b"fake-sdf-bytes-for-969516"

    # complete cleanup: no leftover pubchem temp directory anywhere in the OS temp dir
    leftover = list(Path(tempfile.gettempdir()).glob("pubchem_969516_*"))
    assert leftover == []


def test_ligand_fetch_pubchem_rejects_invalid_cid_without_network_call(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    def _fail_if_called(cid, timeout=None):
        raise AssertionError("fetch_pubchem_sdf_2d must not be called for an invalid CID")

    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", _fail_if_called)

    result = runner.invoke(
        app,
        [
            "ligand",
            "fetch-pubchem",
            "exp001",
            "curcumin",
            "--cid",
            "0",
            "--name",
            "Curcumin",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=ligand-fetch-pubchem" in lines[0]
    assert "outcome=error" in lines[0]


def test_ligand_fetch_pubchem_handles_not_found(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    def _raise_not_found(cid, timeout=None):
        raise cli_module.PubChemNotFoundError(f"PubChem has no record for CID {cid}.")

    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", _raise_not_found)

    result = runner.invoke(
        app,
        [
            "ligand",
            "fetch-pubchem",
            "exp001",
            "curcumin",
            "--cid",
            "999999999",
            "--name",
            "Curcumin",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "PubChem has no record for CID 999999999." in result.output

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=ligand-fetch-pubchem" in lines[0]
    assert "outcome=error" in lines[0]


def test_ligand_fetch_pubchem_handles_acquisition_error(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    def _raise_acquisition_error(cid, timeout=None):
        raise cli_module.PubChemAcquisitionError("PubChem request timed out after 30.0s for CID 969516.")

    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", _raise_acquisition_error)

    result = runner.invoke(
        app,
        [
            "ligand",
            "fetch-pubchem",
            "exp001",
            "curcumin",
            "--cid",
            "969516",
            "--name",
            "Curcumin",
        ],
    )

    assert result.exit_code == 1
    assert "ACQUISITION ERROR" in result.output
    assert "timed out" in result.output

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=ligand-fetch-pubchem" in lines[0]
    assert "outcome=error" in lines[0]


def test_ligand_fetch_pubchem_rejects_missing_experiment_without_network_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _fail_if_called(cid, timeout=None):
        raise AssertionError("fetch_pubchem_sdf_2d must not be called for a missing experiment")

    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", _fail_if_called)

    result = runner.invoke(
        app,
        [
            "ligand",
            "fetch-pubchem",
            "nao_existe",
            "curcumin",
            "--cid",
            "969516",
            "--name",
            "Curcumin",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "Experiment 'nao_existe' was not found." in result.output
    assert not (tmp_path / "nao_existe").exists()


def test_ligand_fetch_pubchem_rejects_duplicate_id_and_cleans_up_temp_file(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    existing_raw_file = tmp_path / "existing.sdf"
    existing_raw_file.write_bytes(b"already-registered-bytes")
    runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(existing_raw_file),
        ],
    )

    fake_fetch = _fake_fetch_returning(tmp_path)
    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", fake_fetch)

    result = runner.invoke(
        app,
        [
            "ligand",
            "fetch-pubchem",
            "exp001",
            "curcumin",
            "--cid",
            "969516",
            "--name",
            "Curcumin",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "already registered" in result.output

    # the original registration must be untouched (no overwrite)
    stored_file = project / "inputs" / "ligands" / "curcumin" / "existing.sdf"
    assert stored_file.read_bytes() == b"already-registered-bytes"

    # the temp directory from the failed fetch-pubchem attempt must still be cleaned up
    assert not (tmp_path / "fake_pubchem_temp_969516").exists()


def test_plan_treats_malformed_yaml_as_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    project = tmp_path / "exp001"
    project.mkdir()

    experiment_file = project / "experiment.yaml"
    experiment_file.write_text(
        "experiment_id: [unclosed\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["plan", "exp001"])

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'exp001' was not found." in result.output


def _write_experiment_yaml(project: Path, **fields) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / "experiment.yaml").write_text(
        yaml.safe_dump(fields, sort_keys=False),
        encoding="utf-8",
    )


def test_init_writes_schema_version_1_1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "exp001"])

    metadata = yaml.safe_load((tmp_path / "exp001" / "experiment.yaml").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "1.1"


def test_init_writes_created_at_in_utc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "exp001"])

    metadata = yaml.safe_load((tmp_path / "exp001" / "experiment.yaml").read_text(encoding="utf-8"))
    assert metadata["created_at"] is not None
    assert metadata["created_at"].endswith("Z") or "+00:00" in metadata["created_at"]


def test_init_persists_metadata_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app,
        [
            "init",
            "exp001",
            "--title",
            "Curcumin x Jack bean urease",
            "--responsible",
            "Emanuel Henrique",
            "--summary",
            "Docking study.",
        ],
    )

    metadata = yaml.safe_load((tmp_path / "exp001" / "experiment.yaml").read_text(encoding="utf-8"))
    assert metadata["title"] == "Curcumin x Jack bean urease"
    assert metadata["responsible"] == "Emanuel Henrique"
    assert metadata["summary"] == "Docking study."


def test_init_without_metadata_flags_leaves_fields_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "exp001"])

    assert result.exit_code == 0
    metadata = yaml.safe_load((tmp_path / "exp001" / "experiment.yaml").read_text(encoding="utf-8"))
    assert metadata["title"] is None
    assert metadata["responsible"] is None
    assert metadata["summary"] is None


def test_init_rejects_empty_title(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "exp001", "--title", ""])

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output


def test_experiment_get_resolves_when_directory_name_matches_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(
        tmp_path / "exp001",
        schema_version="1.1",
        experiment_id="exp001",
        title="Curcumin x Jack bean urease",
        responsible="Emanuel Henrique",
        summary="Docking study.",
        created_at="2026-01-01T00:00:00+00:00",
    )

    result = runner.invoke(app, ["experiment", "get", "exp001"])

    assert result.exit_code == 0
    assert "EXPERIMENT" in result.output
    assert "ID: exp001" in result.output
    assert "Directory: exp001" in result.output
    assert "Title: Curcumin x Jack bean urease" in result.output
    assert "Responsible: Emanuel Henrique" in result.output
    assert "Status: VALID" in result.output
    assert "Summary:" in result.output
    assert "Docking study." in result.output
    assert "Ligands: 0" in result.output
    assert "Proteins: 0" in result.output

    log_file = tmp_path / "exp001" / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=experiment-get" in lines[0]
    assert "outcome=success" in lines[0]


def test_experiment_get_resolves_by_experiment_id_when_directory_name_differs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(
        tmp_path / "some_folder",
        schema_version="1.1",
        experiment_id="exp001",
        title="Curcumin x Jack bean urease",
    )

    result = runner.invoke(app, ["experiment", "get", "exp001"])

    assert result.exit_code == 0
    assert "ID: exp001" in result.output
    assert "Directory: some_folder" in result.output

    log_file = tmp_path / "some_folder" / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=experiment-get" in lines[0]
    assert "outcome=success" in lines[0]


def test_experiment_get_counts_only_yaml_metadata_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "exp001", schema_version="1.1", experiment_id="exp001")

    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")
    runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    result = runner.invoke(app, ["experiment", "get", "exp001"])

    assert result.exit_code == 0
    assert "Ligands: 1" in result.output
    assert "Proteins: 0" in result.output


def test_experiment_get_handles_legacy_experiment_without_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "exp001", schema_version="1.0", experiment_id="exp001")

    result = runner.invoke(app, ["experiment", "get", "exp001"])

    assert result.exit_code == 0
    assert "ID: exp001" in result.output
    assert "Title: \n" in result.output or result.output.count("Title: ") == 1
    assert "Summary:" not in result.output


def test_experiment_get_returns_input_error_when_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["experiment", "get", "nao_existe"])

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'nao_existe' was not found." in result.output
    assert not (tmp_path / "nao_existe").exists()


def test_experiment_get_returns_ambiguity_error_without_choosing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "folder_a", schema_version="1.1", experiment_id="dup001")
    _write_experiment_yaml(tmp_path / "folder_b", schema_version="1.1", experiment_id="dup001")

    result = runner.invoke(app, ["experiment", "get", "dup001"])

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output
    assert "dup001" in result.output
    assert "folder_a" in result.output
    assert "folder_b" in result.output

    assert not (tmp_path / "folder_a" / "logs" / "activity.log").exists()
    assert not (tmp_path / "folder_b" / "logs" / "activity.log").exists()


def test_experiment_get_treats_invalid_schema_with_known_id_as_configuration_error(
    tmp_path, monkeypatch
):
    """A directory whose experiment.yaml parses and clearly declares
    experiment_id 'exp001', but fails schema validation on schema_version,
    has a confiably known identity - read from the YAML, never the
    directory name. Asking for it by that id surfaces a configuration
    error tied to it, not a false 'not found'. No activity log is written
    either way, since no ExperimentContext was ever resolved."""

    monkeypatch.chdir(tmp_path)
    project = tmp_path / "exp001"
    project.mkdir()
    (project / "experiment.yaml").write_text(
        'schema_version: "2.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["experiment", "get", "exp001"])

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output
    assert "Experiment 'exp001' is invalid." in result.output
    assert "schema_version" in result.output
    assert not (project / "logs" / "activity.log").exists()


def test_experiment_get_treats_malformed_yaml_as_not_found(tmp_path, monkeypatch):
    """Unlike a schema violation with a recoverable id, YAML that doesn't
    parse at all leaves no confiable identity, so it still resolves to
    'not found'."""

    monkeypatch.chdir(tmp_path)
    project = tmp_path / "exp001"
    project.mkdir()
    (project / "experiment.yaml").write_text("experiment_id: [unclosed\n", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "get", "exp001"])

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'exp001' was not found." in result.output
    assert not (project / "logs" / "activity.log").exists()


# --- FIX 2: identity must never be shortcut by a matching directory name ---


def test_experiment_get_directory_name_does_not_satisfy_wrong_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(
        tmp_path / "exp001",
        schema_version="1.1",
        experiment_id="totally-different-experiment",
    )

    result = runner.invoke(app, ["experiment", "get", "exp001"])

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'exp001' was not found." in result.output


def test_experiment_get_finds_correct_id_even_when_another_directory_shares_the_name(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(
        tmp_path / "exp001",
        schema_version="1.1",
        experiment_id="totally-different-experiment",
    )
    _write_experiment_yaml(
        tmp_path / "real_exp001",
        schema_version="1.1",
        experiment_id="exp001",
    )

    result = runner.invoke(app, ["experiment", "get", "exp001"])

    assert result.exit_code == 0
    assert "ID: exp001" in result.output
    assert "Directory: real_exp001" in result.output


def test_experiment_get_id_comparison_is_case_sensitive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "exp001", schema_version="1.1", experiment_id="exp001")

    result = runner.invoke(app, ["experiment", "get", "EXP001"])

    assert result.exit_code != 0
    assert "INPUT ERROR" in result.output
    assert "Experiment 'EXP001' was not found." in result.output


def test_experiment_get_ambiguity_not_shortcut_by_matching_directory_name(tmp_path, monkeypatch):
    """Even when one of the ambiguous directories is literally named after
    the identifier, the directory-name shortcut no longer exists, so
    ambiguity must still be raised rather than silently returning that one."""

    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "dup001", schema_version="1.1", experiment_id="dup001")
    _write_experiment_yaml(tmp_path / "folder_b", schema_version="1.1", experiment_id="dup001")

    result = runner.invoke(app, ["experiment", "get", "dup001"])

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output
    assert "dup001" in result.output
    assert "folder_b" in result.output

    assert not (tmp_path / "dup001" / "logs" / "activity.log").exists()
    assert not (tmp_path / "folder_b" / "logs" / "activity.log").exists()


# --- FIX 1: a single broken filesystem candidate must not crash discovery ---


def test_experiment_list_skips_experiment_yaml_that_is_a_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "good_exp", schema_version="1.1", experiment_id="good")
    (tmp_path / "weird_exp" / "experiment.yaml").mkdir(parents=True)

    result = runner.invoke(app, ["experiment", "list"])

    assert result.exit_code == 0
    assert not isinstance(result.exception, OSError)
    assert "good" in result.output
    assert "1 directories skipped" in result.output
    assert "weird_exp" in result.output


def test_experiment_get_skips_experiment_yaml_that_is_a_directory_and_still_finds_valid_one(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "good_exp", schema_version="1.1", experiment_id="good")
    (tmp_path / "weird_exp" / "experiment.yaml").mkdir(parents=True)

    result = runner.invoke(app, ["experiment", "get", "good"])

    assert result.exit_code == 0
    assert not isinstance(result.exception, OSError)
    assert "ID: good" in result.output


def test_experiment_list_skips_candidate_raising_permission_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "good_exp", schema_version="1.1", experiment_id="good")
    _write_experiment_yaml(tmp_path / "locked_exp", schema_version="1.1", experiment_id="locked")

    real_load_experiment = cli_module.load_experiment

    def _flaky_load_experiment(path):
        if "locked_exp" in str(path):
            raise PermissionError("simulated permission error")
        return real_load_experiment(path)

    monkeypatch.setattr(cli_module, "load_experiment", _flaky_load_experiment)

    result = runner.invoke(app, ["experiment", "list"])

    assert result.exit_code == 0
    assert not isinstance(result.exception, PermissionError)
    assert "good" in result.output
    assert "1 directories skipped" in result.output
    assert "locked_exp" in result.output


def test_experiment_list_handles_unreadable_workspace_gracefully(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _raise_oserror(self):
        raise OSError("simulated unreadable directory")

    monkeypatch.setattr(Path, "iterdir", _raise_oserror)

    result = runner.invoke(app, ["experiment", "list"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, OSError)
    assert "STORAGE ERROR" in result.output


def test_experiment_list_shows_id_from_yaml_and_location(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(
        tmp_path / "folder_a",
        schema_version="1.1",
        experiment_id="exp001",
        title="Curcumin x Jack bean urease",
        responsible="Emanuel Henrique",
    )

    result = runner.invoke(app, ["experiment", "list"])

    assert result.exit_code == 0
    assert "EXPERIMENTS" in result.output
    assert "exp001" in result.output
    assert "Curcumin x Jack bean urease" in result.output
    assert "Emanuel Henrique" in result.output
    assert "folder_a" in result.output


def test_experiment_list_ignores_non_experiment_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "exp001", schema_version="1.1", experiment_id="exp001")
    (tmp_path / "not_an_experiment").mkdir()

    result = runner.invoke(app, ["experiment", "list"])

    assert result.exit_code == 0
    assert "not_an_experiment" not in result.output


def test_experiment_list_reports_aggregated_warning_for_invalid_experiments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "exp001", schema_version="1.1", experiment_id="exp001")

    broken = tmp_path / "broken_exp"
    broken.mkdir()
    (broken / "experiment.yaml").write_text("experiment_id: [unclosed\n", encoding="utf-8")

    result = runner.invoke(app, ["experiment", "list"])

    assert result.exit_code == 0
    assert "exp001" in result.output
    assert "1 directories skipped" in result.output
    assert "broken_exp" in result.output
    # the invalid experiment must not appear as if it were a valid row
    assert "broken_exp" not in result.output.split("skipped")[0]


def test_experiment_list_shows_empty_message_when_no_experiments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["experiment", "list"])

    assert result.exit_code == 0
    assert "No experiments found in the current directory." in result.output


def test_experiment_list_does_not_write_activity_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "exp001", schema_version="1.1", experiment_id="exp001")

    runner.invoke(app, ["experiment", "list"])

    assert not (tmp_path / "exp001" / "logs" / "activity.log").exists()


# --- Incremento 008: all commands resolve through the same ExperimentContext,
# so directory != experiment_id must work uniformly, not just for `experiment get`. ---


def test_validate_resolves_by_experiment_id_when_directory_name_differs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "some_folder", schema_version="1.1", experiment_id="exp001")

    result = runner.invoke(app, ["validate", "exp001"])

    assert result.exit_code == 0
    assert "Experiment 'exp001' is valid (schema 1.1)." in result.output

    log_file = tmp_path / "some_folder" / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=validate" in lines[0]
    assert "outcome=success" in lines[0]


def test_validate_rejects_ambiguous_experiment_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "folder_a", schema_version="1.1", experiment_id="dup001")
    _write_experiment_yaml(tmp_path / "folder_b", schema_version="1.1", experiment_id="dup001")

    result = runner.invoke(app, ["validate", "dup001"])

    assert result.exit_code != 0
    assert "CONFIGURATION ERROR" in result.output
    assert "dup001" in result.output
    assert "folder_a" in result.output
    assert "folder_b" in result.output

    assert not (tmp_path / "folder_a" / "logs" / "activity.log").exists()
    assert not (tmp_path / "folder_b" / "logs" / "activity.log").exists()


def test_plan_resolves_by_experiment_id_when_directory_name_differs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "some_folder", schema_version="1.1", experiment_id="exp001")

    result = runner.invoke(app, ["plan", "exp001"])

    assert result.exit_code == 0
    assert "Experiment: exp001" in result.output

    log_file = tmp_path / "some_folder" / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=plan" in lines[0]
    assert "outcome=success" in lines[0]


def test_ligand_add_registers_when_directory_name_differs_from_experiment_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "some_folder", schema_version="1.1", experiment_id="exp001")
    raw_file = tmp_path / "curcumin.sdf"
    raw_file.write_bytes(b"fake-sdf-bytes")

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code == 0
    assert "Ligand 'curcumin' registered successfully." in result.output

    project = tmp_path / "some_folder"
    assert (project / "inputs" / "ligands" / "curcumin.yaml").exists()

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "command=ligand-add" in lines[0]
    assert "outcome=success" in lines[0]


def test_protein_add_registers_when_directory_name_differs_from_experiment_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "some_folder", schema_version="1.1", experiment_id="exp001")
    raw_file = tmp_path / "3la4.cif"
    raw_file.write_bytes(b"fake-cif-bytes")

    result = runner.invoke(
        app,
        [
            "protein",
            "add",
            "exp001",
            "urease",
            "--name",
            "Jack bean urease",
            "--external-id",
            "3LA4",
            "--acquisition-method",
            "downloaded",
            "--source-name",
            "RCSB PDB",
            "--source-identifier",
            "3LA4",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code == 0
    assert "Protein 'urease' registered successfully." in result.output

    project = tmp_path / "some_folder"
    assert (project / "inputs" / "proteins" / "urease.yaml").exists()


def test_ligand_fetch_pubchem_registers_when_directory_name_differs_from_experiment_id(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _write_experiment_yaml(tmp_path / "some_folder", schema_version="1.1", experiment_id="exp001")

    fake_fetch = _fake_fetch_returning(tmp_path)
    monkeypatch.setattr(application_module, "fetch_pubchem_sdf_2d", fake_fetch)

    result = runner.invoke(
        app,
        [
            "ligand",
            "fetch-pubchem",
            "exp001",
            "curcumin",
            "--cid",
            "969516",
            "--name",
            "Curcumin",
        ],
    )

    assert result.exit_code == 0
    assert "Ligand 'curcumin' registered successfully." in result.output

    project = tmp_path / "some_folder"
    assert (project / "inputs" / "ligands" / "curcumin.yaml").exists()


# =====================================================================
# Increment 014 - Scientific CLI Foundation
# =====================================================================


def _write_source_file(tmp_path, name="mol.sdf", content=b"fake-bytes"):
    source_file = tmp_path / name
    source_file.write_bytes(content)
    return source_file


def _last_log_line(project: Path) -> str:
    lines = (project / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()
    return lines[-1]


def _add_entity(tmp_path, monkeypatch, entity_id="curcumin", name="Curcumin"):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    runner.invoke(app, ["entity", "add", "exp001", entity_id, "--name", name])
    return project


def _add_entity_and_state(
    tmp_path, monkeypatch, entity_id="curcumin", state_id="curcumin-neutral"
):
    project = _add_entity(tmp_path, monkeypatch, entity_id=entity_id)
    source_file = _write_source_file(tmp_path, name="state.sdf")
    runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            state_id,
            "--entity",
            entity_id,
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )
    return project


# --- regression: F-CLI-05 malformed id bugfix (ligand add / protein add) ---


def test_ligand_add_rejects_malformed_id_without_traceback(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="lig.sdf")

    result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "bad id",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert not (project / "inputs" / "ligands" / "bad id.yaml").exists()


def test_protein_add_rejects_malformed_id_without_traceback(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="prot.cif")

    result = runner.invoke(
        app,
        [
            "protein",
            "add",
            "exp001",
            "bad id",
            "--name",
            "Urease",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert not (project / "inputs" / "proteins" / "bad id.yaml").exists()


# --- ligand show ---


def test_ligand_show_displays_registered_ligand(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="curcumin.sdf", content=b"curcumin-bytes")
    runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--external-id",
            "CID 969516",
            "--acquisition-method",
            "downloaded",
            "--source-name",
            "PubChem",
            "--source-identifier",
            "CID 969516",
            "--raw-file",
            str(raw_file),
        ],
    )

    result = runner.invoke(app, ["ligand", "show", "exp001", "curcumin"])

    assert result.exit_code == 0
    assert "ID: curcumin" in result.output
    assert "Name: Curcumin" in result.output
    assert "External ID: CID 969516" in result.output
    assert "Acquisition method: downloaded" in result.output
    assert "Source name: PubChem" in result.output
    assert "inputs/ligands/curcumin/curcumin.sdf" in result.output
    assert "Format: sdf" in result.output
    assert "Checksum:" in result.output


def test_ligand_show_rejects_malformed_id_without_calling_load_ligand(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("load_ligand must not be called for a malformed id")

    monkeypatch.setattr(cli_module, "load_ligand", _spy)

    result = runner.invoke(app, ["ligand", "show", "exp001", "bad id"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert called["value"] is False


def test_ligand_show_returns_input_error_for_unknown_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["ligand", "show", "exp001", "unknown-ligand"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_ligand_show_rejects_missing_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["ligand", "show", "exp001", "curcumin"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_ligand_show_writes_activity_log_success(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="curcumin.sdf")
    runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    runner.invoke(app, ["ligand", "show", "exp001", "curcumin"])

    line = _last_log_line(project)
    assert "command=ligand-show" in line
    assert "outcome=success" in line


def test_ligand_show_writes_activity_log_error(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(app, ["ligand", "show", "exp001", "unknown-ligand"])

    line = _last_log_line(project)
    assert "command=ligand-show" in line
    assert "outcome=error" in line


def test_ligand_show_does_not_mutate_registered_ligand(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="curcumin.sdf")
    runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )
    metadata_file = project / "inputs" / "ligands" / "curcumin.yaml"
    before = metadata_file.read_text(encoding="utf-8")

    runner.invoke(app, ["ligand", "show", "exp001", "curcumin"])

    assert metadata_file.read_text(encoding="utf-8") == before


# --- protein show ---


def test_protein_show_displays_registered_protein(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="urease.cif", content=b"urease-bytes")
    runner.invoke(
        app,
        [
            "protein",
            "add",
            "exp001",
            "urease",
            "--name",
            "Jack bean urease",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    result = runner.invoke(app, ["protein", "show", "exp001", "urease"])

    assert result.exit_code == 0
    assert "ID: urease" in result.output
    assert "Name: Jack bean urease" in result.output
    assert "inputs/proteins/urease/urease.cif" in result.output


def test_protein_show_rejects_malformed_id_without_calling_load_protein(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("load_protein must not be called for a malformed id")

    monkeypatch.setattr(cli_module, "load_protein", _spy)

    result = runner.invoke(app, ["protein", "show", "exp001", "bad id"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert called["value"] is False


def test_protein_show_returns_input_error_for_unknown_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["protein", "show", "exp001", "unknown-protein"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_protein_show_writes_activity_log_success(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="urease.cif")
    runner.invoke(
        app,
        [
            "protein",
            "add",
            "exp001",
            "urease",
            "--name",
            "Urease",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    runner.invoke(app, ["protein", "show", "exp001", "urease"])

    line = _last_log_line(project)
    assert "command=protein-show" in line
    assert "outcome=success" in line


# --- entity add ---


def test_entity_add_registers_entity(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["entity", "add", "exp001", "curcumin", "--name", "Curcumin"])

    assert result.exit_code == 0
    assert "Entity 'curcumin' registered successfully." in result.output
    assert (project / "scientific" / "entities" / "curcumin.yaml").exists()


def test_entity_add_rejects_missing_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["entity", "add", "exp001", "curcumin", "--name", "Curcumin"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_entity_add_rejects_malformed_id_as_configuration_error(tmp_path, monkeypatch):
    """Unlike state/conformer add, entity add builds a ScientificEntity
    model directly - a malformed id fails pydantic field validation before
    register_entity is ever called, surfacing as CONFIGURATION ERROR, not
    the bare-ValueError/INPUT ERROR path used by state/conformer add."""

    project = _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["entity", "add", "exp001", "bad id", "--name", "Curcumin"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "CONFIGURATION ERROR" in result.output
    assert not (project / "scientific" / "entities" / "bad id.yaml").exists()


def test_entity_add_rejects_exact_duplicate(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)

    result = runner.invoke(app, ["entity", "add", "exp001", "curcumin", "--name", "Other"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_entity_add_rejects_casefold_collision(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)

    result = runner.invoke(app, ["entity", "add", "exp001", "Curcumin", "--name", "Other"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_entity_add_handles_storage_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "register_entity", _raise_storage_error)

    result = runner.invoke(app, ["entity", "add", "exp001", "curcumin", "--name", "Curcumin"])

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output
    assert "simulated disk failure" in result.output


def test_entity_add_writes_activity_log_success(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(app, ["entity", "add", "exp001", "curcumin", "--name", "Curcumin"])

    line = _last_log_line(project)
    assert "command=entity-add" in line
    assert "outcome=success" in line


def test_entity_add_writes_activity_log_error_on_duplicate(tmp_path, monkeypatch):
    project = _add_entity(tmp_path, monkeypatch)

    runner.invoke(app, ["entity", "add", "exp001", "curcumin", "--name", "Other"])

    line = _last_log_line(project)
    assert "command=entity-add" in line
    assert "outcome=error" in line


# --- entity show ---


def test_entity_show_displays_registered_entity(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)

    result = runner.invoke(app, ["entity", "show", "exp001", "curcumin"])

    assert result.exit_code == 0
    assert "Schema version: 1.0" in result.output
    assert "ID: curcumin" in result.output
    assert "Name: Curcumin" in result.output


def test_entity_show_rejects_malformed_id_without_calling_load_entity(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("load_entity must not be called for a malformed id")

    monkeypatch.setattr(cli_module, "load_entity", _spy)

    result = runner.invoke(app, ["entity", "show", "exp001", "bad id"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert called["value"] is False


def test_entity_show_returns_input_error_for_unknown_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["entity", "show", "exp001", "unknown-entity"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_entity_show_writes_activity_log_success(tmp_path, monkeypatch):
    project = _add_entity(tmp_path, monkeypatch)

    runner.invoke(app, ["entity", "show", "exp001", "curcumin"])

    line = _last_log_line(project)
    assert "command=entity-show" in line
    assert "outcome=success" in line


def test_entity_show_writes_activity_log_error_for_unknown_id(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(app, ["entity", "show", "exp001", "unknown-entity"])

    line = _last_log_line(project)
    assert "command=entity-show" in line
    assert "outcome=error" in line


def test_entity_show_does_not_mutate_registered_entity(tmp_path, monkeypatch):
    project = _add_entity(tmp_path, monkeypatch)
    metadata_file = project / "scientific" / "entities" / "curcumin.yaml"
    before = metadata_file.read_text(encoding="utf-8")

    runner.invoke(app, ["entity", "show", "exp001", "curcumin"])

    assert metadata_file.read_text(encoding="utf-8") == before


# --- state add ---


def test_state_add_registers_state(tmp_path, monkeypatch):
    project = _add_entity(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="state.sdf", content=b"state-bytes")

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 0
    assert "State 'curcumin-neutral' registered successfully." in result.output
    assert (project / "scientific" / "states" / "curcumin-neutral.yaml").exists()
    assert (
        project / "scientific" / "artifacts" / "states" / "curcumin-neutral" / "state.sdf"
    ).exists()


def test_state_add_rejects_missing_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source_file = _write_source_file(tmp_path)

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_add_rejects_malformed_state_id_without_traceback(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path)

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "bad id",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output


def test_state_add_rejects_missing_entity(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path)

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "does-not-exist",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_add_rejects_missing_source_file(tmp_path, monkeypatch):
    project = _add_entity(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(project.parent / "does-not-exist.sdf"),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_add_rejects_empty_format(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path)

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "",
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output


def test_state_add_rejects_exact_duplicate(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="other.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_add_rejects_casefold_collision(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="other.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "Curcumin-Neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_add_handles_storage_error(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path)

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "register_state", _raise_storage_error)

    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output


def test_state_add_writes_activity_log_success(tmp_path, monkeypatch):
    project = _add_entity(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path)

    runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    line = _last_log_line(project)
    assert "command=state-add" in line
    assert "outcome=success" in line


def test_state_add_writes_activity_log_error(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="other.sdf")

    runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    line = _last_log_line(project)
    assert "command=state-add" in line
    assert "outcome=error" in line


# --- state show ---


def test_state_show_displays_registered_state(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)

    result = runner.invoke(app, ["state", "show", "exp001", "curcumin-neutral"])

    assert result.exit_code == 0
    assert "ID: curcumin-neutral" in result.output
    assert "Entity: curcumin" in result.output
    assert "scientific/artifacts/states/curcumin-neutral/state.sdf" in result.output
    assert "Representation format: sdf" in result.output
    assert "Representation checksum:" in result.output


def test_state_show_rejects_malformed_id_without_calling_load_state(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("load_state must not be called for a malformed id")

    monkeypatch.setattr(cli_module, "load_state", _spy)

    result = runner.invoke(app, ["state", "show", "exp001", "bad id"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert called["value"] is False


def test_state_show_returns_input_error_for_unknown_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["state", "show", "exp001", "unknown-state"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_show_writes_activity_log_success(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)

    runner.invoke(app, ["state", "show", "exp001", "curcumin-neutral"])

    line = _last_log_line(project)
    assert "command=state-show" in line
    assert "outcome=success" in line


def test_state_show_does_not_mutate_registered_state(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    metadata_file = project / "scientific" / "states" / "curcumin-neutral.yaml"
    before = metadata_file.read_text(encoding="utf-8")

    runner.invoke(app, ["state", "show", "exp001", "curcumin-neutral"])

    assert metadata_file.read_text(encoding="utf-8") == before


# --- state list ---


def test_state_list_returns_matching_states(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch, state_id="curcumin-a")
    source_b = _write_source_file(tmp_path, name="b.sdf")
    runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-b",
            "--entity",
            "curcumin",
            "--source-file",
            str(source_b),
            "--format",
            "sdf",
        ],
    )

    result = runner.invoke(app, ["state", "list", "exp001", "--entity", "curcumin"])

    assert result.exit_code == 0
    assert "curcumin-a" in result.output
    assert "curcumin-b" in result.output


def test_state_list_rejects_malformed_entity_id_without_calling_list_entity_states(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("list_entity_states must not be called for a malformed id")

    monkeypatch.setattr(cli_module, "list_entity_states", _spy)

    result = runner.invoke(app, ["state", "list", "exp001", "--entity", "bad id"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert called["value"] is False


def test_state_list_returns_empty_for_valid_unknown_entity_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["state", "list", "exp001", "--entity", "unknown-entity"])

    assert result.exit_code == 0
    assert "No states found for entity 'unknown-entity'." in result.output


def test_state_list_does_not_write_activity_log(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    log_file = project / "logs" / "activity.log"
    lines_before = log_file.read_text(encoding="utf-8").splitlines()

    runner.invoke(app, ["state", "list", "exp001", "--entity", "curcumin"])

    lines_after = log_file.read_text(encoding="utf-8").splitlines()
    assert lines_after == lines_before


# --- state derive ---


def test_state_derive_registers_child_state_and_transformation_with_method_description(
    tmp_path, monkeypatch
):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf", content=b"child-bytes")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "pH 7.4 protonation, assigned manually",
        ],
    )

    assert result.exit_code == 0
    assert "State 'curcumin-prot' registered successfully." in result.output
    assert "Transformation 't1' recorded successfully." in result.output
    assert (project / "scientific" / "states" / "curcumin-prot.yaml").exists()
    assert (project / "scientific" / "transformations" / "t1.yaml").exists()


def test_state_derive_registers_with_producer(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "tautomerization",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-taut",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--producer-name",
            "manual",
            "--producer-version",
            "1.0",
        ],
    )

    assert result.exit_code == 0

    data = yaml.safe_load(
        (project / "scientific" / "transformations" / "t1.yaml").read_text(encoding="utf-8")
    )
    assert data["producer"]["name"] == "manual"
    assert data["producer"]["version"] == "1.0"


def test_state_derive_rejects_invalid_operation(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "combustion",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_derive_rejects_producer_version_without_producer_name(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--producer-version",
            "1.0",
            "--method-description",
            "manual",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "--producer-version requires --producer-name" in result.output


def test_state_derive_rejects_missing_provenance(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "usable provenance" in result.output


def test_state_derive_rejects_malformed_param(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
            "--param",
            "no-equals-sign",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_derive_coerces_param_types(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
            "--param",
            "ph=7.4",
            "--param",
            "count=3",
            "--param",
            "verified=true",
            "--param",
            "label=manual",
        ],
    )

    assert result.exit_code == 0

    data = yaml.safe_load(
        (project / "scientific" / "transformations" / "t1.yaml").read_text(encoding="utf-8")
    )
    values = {item["name"]: item["value"] for item in data["parameters"]}

    assert values["ph"] == 7.4
    assert values["count"] == 3
    assert values["verified"] is True
    assert values["label"] == "manual"


def test_state_derive_rejects_duplicate_param_names(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
            "--param",
            "ph=7.4",
            "--param",
            "ph=8.0",
        ],
    )

    assert result.exit_code == 1
    assert "CONFIGURATION ERROR" in result.output


def test_state_derive_rejects_malformed_parent_state_id_without_traceback(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "bad id",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output


def test_state_derive_rejects_missing_parent_state(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "does-not-exist",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_derive_rejects_duplicate_child_state_id(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    other_source = _write_source_file(tmp_path, name="other.sdf")
    runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-prot",
            "--entity",
            "curcumin",
            "--source-file",
            str(other_source),
            "--format",
            "sdf",
        ],
    )
    child_source = _write_source_file(tmp_path, name="child.sdf")

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_state_derive_rejects_storage_error(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "register_derived_state", _raise_storage_error)

    result = runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
        ],
    )

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output


def test_state_derive_does_not_duplicate_activity_logging(tmp_path, monkeypatch):
    """register_derived_state already logs started/success internally under
    command=register-derived-state - the CLI must not add a second,
    redundant entry under a command=state-derive name."""

    project = _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")

    runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "manual",
        ],
    )

    log_text = (project / "logs" / "activity.log").read_text(encoding="utf-8")
    assert "command=register-derived-state" in log_text
    assert "outcome=started" in log_text
    assert "outcome=success" in log_text
    assert "command=state-derive" not in log_text


def test_state_derive_help_does_not_claim_to_perform_chemistry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["state", "derive", "--help"])

    assert result.exit_code == 0
    assert "does not perform protonation or tautomerization" in result.output
    assert "protonated" not in result.output.lower()
    assert "generated" not in result.output.lower()


# --- conformer add ---


def test_conformer_add_registers_conformer(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf", content=b"conf-bytes")

    result = runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-neutral-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 0
    assert "Conformer 'curcumin-neutral-conf-001' registered successfully." in result.output
    assert (
        project / "scientific" / "conformers" / "curcumin-neutral-conf-001.yaml"
    ).exists()


def test_conformer_add_rejects_malformed_id_without_traceback(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")

    result = runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "bad id",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output


def test_conformer_add_rejects_missing_state(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")

    result = runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "does-not-exist",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conformer_add_rejects_missing_source_file(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(project.parent / "does-not-exist.sdf"),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conformer_add_rejects_empty_format(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")

    result = runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "",
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output


def test_conformer_add_rejects_exact_duplicate(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")
    runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    result = runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conformer_add_rejects_casefold_collision(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")
    runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    result = runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "Curcumin-Conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conformer_add_handles_storage_error(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "register_conformer", _raise_storage_error)

    result = runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output


def test_conformer_add_writes_activity_log_success(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")

    runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    line = _last_log_line(project)
    assert "command=conformer-add" in line
    assert "outcome=success" in line


def test_conformer_add_help_states_it_does_not_generate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["conformer", "add", "--help"])

    assert result.exit_code == 0
    assert "does not generate a conformer" in result.output


# --- conformer show ---


def test_conformer_show_displays_registered_conformer(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")
    runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    result = runner.invoke(app, ["conformer", "show", "exp001", "curcumin-conf-001"])

    assert result.exit_code == 0
    assert "ID: curcumin-conf-001" in result.output
    assert "State: curcumin-neutral" in result.output
    assert "scientific/artifacts/conformers/curcumin-conf-001/conf.sdf" in result.output


def test_conformer_show_rejects_malformed_id_without_calling_load_conformer(
    tmp_path, monkeypatch
):
    _add_entity_and_state(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("load_conformer must not be called for a malformed id")

    monkeypatch.setattr(cli_module, "load_conformer", _spy)

    result = runner.invoke(app, ["conformer", "show", "exp001", "bad id"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert called["value"] is False


def test_conformer_show_returns_input_error_for_unknown_id(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)

    result = runner.invoke(app, ["conformer", "show", "exp001", "unknown-conformer"])

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conformer_show_writes_activity_log_success(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")
    runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    runner.invoke(app, ["conformer", "show", "exp001", "curcumin-conf-001"])

    line = _last_log_line(project)
    assert "command=conformer-show" in line
    assert "outcome=success" in line


# --- conformer list ---


def test_conformer_list_returns_matching_conformers(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    source_a = _write_source_file(tmp_path, name="a.sdf")
    source_b = _write_source_file(tmp_path, name="b.sdf")
    runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "conf-a",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_a),
            "--format",
            "sdf",
        ],
    )
    runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "conf-b",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_b),
            "--format",
            "sdf",
        ],
    )

    result = runner.invoke(app, ["conformer", "list", "exp001", "--state", "curcumin-neutral"])

    assert result.exit_code == 0
    assert "conf-a" in result.output
    assert "conf-b" in result.output


def test_conformer_list_rejects_malformed_state_id_without_calling_list_state_conformers(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("list_state_conformers must not be called for a malformed id")

    monkeypatch.setattr(cli_module, "list_state_conformers", _spy)

    result = runner.invoke(app, ["conformer", "list", "exp001", "--state", "bad id"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert called["value"] is False


def test_conformer_list_returns_empty_for_valid_unknown_state_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["conformer", "list", "exp001", "--state", "unknown-state"])

    assert result.exit_code == 0
    assert "No conformers found for state 'unknown-state'." in result.output


def test_conformer_list_does_not_write_activity_log(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    log_file = project / "logs" / "activity.log"
    lines_before = log_file.read_text(encoding="utf-8").splitlines()

    runner.invoke(app, ["conformer", "list", "exp001", "--state", "curcumin-neutral"])

    lines_after = log_file.read_text(encoding="utf-8").splitlines()
    assert lines_after == lines_before


# --- provenance show ---


def test_provenance_show_displays_no_provenance_message_for_valid_unknown_state(
    tmp_path, monkeypatch
):
    _add_entity_and_state(tmp_path, monkeypatch)

    result = runner.invoke(app, ["provenance", "show", "exp001", "--state", "curcumin-neutral"])

    assert result.exit_code == 0
    assert "No registered state-transformation provenance exists for 'curcumin-neutral'" in (
        result.output
    )
    assert "Acquisition provenance is tracked separately" in result.output


def test_provenance_show_displays_transformation_records(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)
    child_source = _write_source_file(tmp_path, name="child.sdf")
    runner.invoke(
        app,
        [
            "state",
            "derive",
            "exp001",
            "--operation",
            "protonation",
            "--parent-state",
            "curcumin-neutral",
            "--child-state",
            "curcumin-prot",
            "--transformation-id",
            "t1",
            "--source-file",
            str(child_source),
            "--format",
            "sdf",
            "--method-description",
            "pH 7.4 manual",
            "--param",
            "ph=7.4",
        ],
    )

    result = runner.invoke(app, ["provenance", "show", "exp001", "--state", "curcumin-prot"])

    assert result.exit_code == 0
    assert "Transformation: t1" in result.output
    assert "Parent state: curcumin-neutral" in result.output
    assert "Child state: curcumin-prot" in result.output
    assert "Operation: protonation" in result.output
    assert "Method description: pH 7.4 manual" in result.output
    assert "Parameter: ph=7.4" in result.output


def test_provenance_show_rejects_malformed_state_id_without_calling_list_state_transformations(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("list_state_transformations must not be called for a malformed id")

    monkeypatch.setattr(cli_module, "list_state_transformations", _spy)

    result = runner.invoke(app, ["provenance", "show", "exp001", "--state", "bad id"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "INPUT ERROR" in result.output
    assert called["value"] is False


def test_provenance_show_writes_activity_log_success(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)

    runner.invoke(app, ["provenance", "show", "exp001", "--state", "curcumin-neutral"])

    line = _last_log_line(project)
    assert "command=provenance-show" in line
    assert "outcome=success" in line


def test_provenance_show_never_echoes_activity_log_content(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)

    result = runner.invoke(app, ["provenance", "show", "exp001", "--state", "curcumin-neutral"])

    assert "outcome=" not in result.output
    assert (project / "logs" / "activity.log").exists()


# --- manual acquisition -> scientific model bridge (Decision A) ---


def _extract_labeled_line(output: str, label: str) -> str:
    """Extract the value of a `<label>: <value>` line printed by the CLI.

    Splits only on the label prefix, never on ':', so a Windows absolute
    path (which itself contains a drive-letter colon) is returned intact.
    """

    prefix = f"{label}:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise AssertionError(f"'{prefix}' line not found in output:\n{output}")


def test_ligand_show_prints_a_resolved_source_path_usable_directly_in_source_file(
    tmp_path, monkeypatch
):
    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="curcumin.sdf", content=b"curcumin-bytes")
    runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin-lig",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    show_result = runner.invoke(app, ["ligand", "show", "exp001", "curcumin-lig"])
    assert show_result.exit_code == 0
    assert "Stored path: inputs/ligands/curcumin-lig/curcumin.sdf" in show_result.output

    resolved_path = _extract_labeled_line(show_result.output, "Resolved source path")

    assert Path(resolved_path).is_absolute()
    assert Path(resolved_path).is_file()
    assert Path(resolved_path) == (
        project / "inputs" / "ligands" / "curcumin-lig" / "curcumin.sdf"
    ).resolve()


def test_protein_show_prints_a_resolved_source_path_pointing_to_an_existing_file(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="urease.cif", content=b"urease-bytes")
    runner.invoke(
        app,
        [
            "protein",
            "add",
            "exp001",
            "urease",
            "--name",
            "Urease",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    show_result = runner.invoke(app, ["protein", "show", "exp001", "urease"])

    resolved_path = _extract_labeled_line(show_result.output, "Resolved source path")

    assert Path(resolved_path).is_absolute()
    assert Path(resolved_path).is_file()


def test_manual_bridge_from_ligand_to_state_duplicates_the_artifact(tmp_path, monkeypatch):
    """Demonstrates the manual, ergonomics-only bridge from ligand
    acquisition to the scientific model (Decision A), reproducing the real
    user flow: the Resolved source path printed by `ligand show` is used
    verbatim - by copy-paste, not reconstructed independently by the test -
    as `state add`'s --source-file. This creates a second, independently
    stored copy; nothing links the two structurally, and provenance show
    never claims otherwise."""

    project = _init_valid_experiment(tmp_path, monkeypatch)
    raw_file = _write_source_file(tmp_path, name="curcumin.sdf", content=b"curcumin-bytes")

    runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin-lig",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
        ],
    )

    show_result = runner.invoke(app, ["ligand", "show", "exp001", "curcumin-lig"])
    resolved_ligand_path = _extract_labeled_line(show_result.output, "Resolved source path")
    assert Path(resolved_ligand_path).is_file()

    runner.invoke(app, ["entity", "add", "exp001", "curcumin", "--name", "Curcumin"])

    state_result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            # Exactly the string `ligand show` printed above - the whole
            # point of this test is that no path is reconstructed here.
            "--source-file",
            resolved_ligand_path,
            "--format",
            "sdf",
        ],
    )
    assert state_result.exit_code == 0

    state_artifact = (
        project / "scientific" / "artifacts" / "states" / "curcumin-neutral" / "curcumin.sdf"
    )
    assert state_artifact.exists()
    assert str(state_artifact.resolve()) != resolved_ligand_path
    assert state_artifact.read_bytes() == Path(resolved_ligand_path).read_bytes()

    provenance_result = runner.invoke(
        app, ["provenance", "show", "exp001", "--state", "curcumin-neutral"]
    )
    assert "No registered state-transformation provenance exists" in provenance_result.output


# =====================================================================
# Increment 014 hardening (post adversarial review)
# =====================================================================


# --- F-014-C-01: read commands must translate OSError/YAMLError/
# ValidationError into clean banners, never an unhandled traceback ---


def test_ligand_show_handles_storage_error_without_traceback(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "load_ligand", _raise_storage_error)

    result = runner.invoke(app, ["ligand", "show", "exp001", "curcumin"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output

    line = _last_log_line(project)
    assert "command=ligand-show" in line
    assert "outcome=error" in line


def test_ligand_show_handles_malformed_yaml_without_traceback(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    def _raise_yaml_error(*args, **kwargs):
        raise yaml.YAMLError("simulated corrupt yaml")

    monkeypatch.setattr(cli_module, "load_ligand", _raise_yaml_error)

    result = runner.invoke(app, ["ligand", "show", "exp001", "curcumin"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "CONFIGURATION ERROR" in result.output

    line = _last_log_line(project)
    assert "command=ligand-show" in line
    assert "outcome=error" in line


def test_protein_show_handles_storage_error_without_traceback(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "load_protein", _raise_storage_error)

    result = runner.invoke(app, ["protein", "show", "exp001", "urease"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output


def test_entity_show_handles_validation_error_without_traceback(tmp_path, monkeypatch):
    """A stored entities/<id>.yaml that no longer matches the model schema
    must not surface as an unhandled pydantic traceback."""

    project = _add_entity(tmp_path, monkeypatch)

    def _raise_validation_error(*args, **kwargs):
        ScientificEntity(schema_version="1.0", id="", name="Curcumin")

    monkeypatch.setattr(cli_module, "load_entity", _raise_validation_error)

    result = runner.invoke(app, ["entity", "show", "exp001", "curcumin"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "CONFIGURATION ERROR" in result.output

    line = _last_log_line(project)
    assert "command=entity-show" in line
    assert "outcome=error" in line


def test_entity_show_handles_storage_error_without_traceback(tmp_path, monkeypatch):
    _add_entity(tmp_path, monkeypatch)

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "load_entity", _raise_storage_error)

    result = runner.invoke(app, ["entity", "show", "exp001", "curcumin"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output


def test_state_show_handles_storage_error_without_traceback(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch)

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "load_state", _raise_storage_error)

    result = runner.invoke(app, ["state", "show", "exp001", "curcumin-neutral"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output


def test_state_list_handles_storage_error_without_traceback_and_without_activity_log(
    tmp_path, monkeypatch
):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    log_file = project / "logs" / "activity.log"
    lines_before = log_file.read_text(encoding="utf-8").splitlines()

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "list_entity_states", _raise_storage_error)

    result = runner.invoke(app, ["state", "list", "exp001", "--entity", "curcumin"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output

    lines_after = log_file.read_text(encoding="utf-8").splitlines()
    assert lines_after == lines_before


def test_conformer_show_handles_storage_error_without_traceback(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    source_file = _write_source_file(tmp_path, name="conf.sdf")
    runner.invoke(
        app,
        [
            "conformer",
            "add",
            "exp001",
            "curcumin-conf-001",
            "--state",
            "curcumin-neutral",
            "--source-file",
            str(source_file),
            "--format",
            "sdf",
        ],
    )

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "load_conformer", _raise_storage_error)

    result = runner.invoke(app, ["conformer", "show", "exp001", "curcumin-conf-001"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output


def test_conformer_list_handles_storage_error_without_traceback_and_without_activity_log(
    tmp_path, monkeypatch
):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    log_file = project / "logs" / "activity.log"
    lines_before = log_file.read_text(encoding="utf-8").splitlines()

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "list_state_conformers", _raise_storage_error)

    result = runner.invoke(app, ["conformer", "list", "exp001", "--state", "curcumin-neutral"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output

    lines_after = log_file.read_text(encoding="utf-8").splitlines()
    assert lines_after == lines_before


def test_provenance_show_handles_storage_error_without_traceback(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(cli_module, "list_state_transformations", _raise_storage_error)

    result = runner.invoke(app, ["provenance", "show", "exp001", "--state", "curcumin-neutral"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output

    line = _last_log_line(project)
    assert "command=provenance-show" in line
    assert "outcome=error" in line


# --- F-014-C-02 tests already added above: resolved source path tests
# and the rewritten manual bridge test ---


# --- F-014-C-03: state derive pre-flight failures must log activity,
# without ever duplicating register_derived_state's own logging ---


def _derive_args(tmp_path, **overrides):
    child_source = _write_source_file(tmp_path, name="child.sdf")
    args = {
        "operation": "protonation",
        "parent-state": "curcumin-neutral",
        "child-state": "curcumin-prot",
        "transformation-id": "t1",
        "source-file": str(child_source),
        "format": "sdf",
        "method-description": "manual",
    }
    args.update(overrides)

    flat = ["state", "derive", "exp001"]
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                flat.extend([f"--{key}", item])
        else:
            flat.extend([f"--{key}", value])
    return flat


def test_state_derive_logs_activity_error_for_invalid_operation(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("register_derived_state must not be called")

    monkeypatch.setattr(cli_module, "register_derived_state", _spy)

    result = runner.invoke(app, _derive_args(tmp_path, operation="combustion"))

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert called["value"] is False

    line = _last_log_line(project)
    assert "command=state-derive" in line
    assert "outcome=error" in line


def test_state_derive_logs_activity_error_for_producer_version_without_name(
    tmp_path, monkeypatch
):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("register_derived_state must not be called")

    monkeypatch.setattr(cli_module, "register_derived_state", _spy)

    result = runner.invoke(
        app, _derive_args(tmp_path, **{"producer-version": "1.0"})
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert called["value"] is False

    line = _last_log_line(project)
    assert "command=state-derive" in line
    assert "outcome=error" in line


def test_state_derive_logs_activity_error_for_missing_provenance(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("register_derived_state must not be called")

    monkeypatch.setattr(cli_module, "register_derived_state", _spy)

    result = runner.invoke(
        app, _derive_args(tmp_path, **{"method-description": None})
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert called["value"] is False

    line = _last_log_line(project)
    assert "command=state-derive" in line
    assert "outcome=error" in line


def test_state_derive_logs_activity_error_for_malformed_param(tmp_path, monkeypatch):
    project = _add_entity_and_state(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("register_derived_state must not be called")

    monkeypatch.setattr(cli_module, "register_derived_state", _spy)

    result = runner.invoke(
        app, _derive_args(tmp_path, param="no-equals-sign")
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert called["value"] is False

    line = _last_log_line(project)
    assert "command=state-derive" in line
    assert "outcome=error" in line


def test_state_derive_logs_activity_error_for_invalid_producer(tmp_path, monkeypatch):
    """A blank --producer-name fails TransformationProducer's own field
    validation before register_derived_state is ever called."""

    project = _add_entity_and_state(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("register_derived_state must not be called")

    monkeypatch.setattr(cli_module, "register_derived_state", _spy)

    result = runner.invoke(
        app,
        _derive_args(
            tmp_path, **{"producer-name": "   ", "method-description": None}
        ),
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "CONFIGURATION ERROR" in result.output
    assert called["value"] is False

    line = _last_log_line(project)
    assert "command=state-derive" in line
    assert "outcome=error" in line


def test_state_derive_logs_activity_error_for_invalid_parameter(tmp_path, monkeypatch):
    """A --param with a blank name (e.g. "=5") fails TransformationParameter's
    own field validation before register_derived_state is ever called."""

    project = _add_entity_and_state(tmp_path, monkeypatch)
    called = {"value": False}

    def _spy(*args, **kwargs):
        called["value"] = True
        raise AssertionError("register_derived_state must not be called")

    monkeypatch.setattr(cli_module, "register_derived_state", _spy)

    result = runner.invoke(app, _derive_args(tmp_path, param="=5"))

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "CONFIGURATION ERROR" in result.output
    assert called["value"] is False

    line = _last_log_line(project)
    assert "command=state-derive" in line
    assert "outcome=error" in line


def test_state_derive_does_not_log_state_derive_when_register_derived_state_is_called(
    tmp_path, monkeypatch
):
    """Once register_derived_state has actually been invoked, any failure
    from that point on (here: duplicate parameter names, detected only
    when StateTransformationRecord is assembled inside the Application
    Service) must rely solely on its own internal logging - the CLI must
    never add a second command=state-derive entry."""

    project = _add_entity_and_state(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        _derive_args(tmp_path, param=["ph=7.4", "ph=8.0"]),
    )

    assert result.exit_code == 1
    assert "CONFIGURATION ERROR" in result.output

    log_text = (project / "logs" / "activity.log").read_text(encoding="utf-8")
    assert "command=register-derived-state" in log_text
    assert "command=state-derive" not in log_text


# --- F-014-C-05: protein show activity-log-error test parity with ligand ---


def test_protein_show_writes_activity_log_error(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(app, ["protein", "show", "exp001", "unknown-protein"])

    line = _last_log_line(project)
    assert "command=protein-show" in line
    assert "outcome=error" in line

# =====================================================================
# Increment 017 - Tautomer Enumeration CLI
# =====================================================================


_RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


def _tautomer_cli_args(*extra, input_state="state-a"):
    return [
        "state",
        "enumerate-tautomers",
        "exp001",
        "--input-state",
        input_state,
        *extra,
    ]


def _fake_tautomer_success_attempt(operation_ids=(), *, attempt_id="attempt-017"):
    return SimpleNamespace(
        id=attempt_id,
        input_state_id="state-a",
        status="succeeded",
        resolved=SimpleNamespace(engine_name="rdkit", engine_version="2026.03.6"),
        operation_ids=tuple(operation_ids),
    )


def _patch_tautomer_success(monkeypatch, *, operation_ids=(), child_ids=None):
    observed = {}
    attempt = _fake_tautomer_success_attempt(operation_ids)
    children = child_ids or {
        operation_id: f"state-{index}"
        for index, operation_id in enumerate(operation_ids, start=1)
    }

    def _enumerate(project_dir, *, input_state_id, policy):
        observed["project_dir"] = project_dir
        observed["input_state_id"] = input_state_id
        observed["policy"] = policy
        observed["calls"] = observed.get("calls", 0) + 1
        return attempt

    def _load_transformation(project_dir, operation_id):
        observed.setdefault("loaded_operation_ids", []).append(operation_id)
        return SimpleNamespace(
            id=operation_id,
            child_state_id=children[operation_id],
        )

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(cli_module, "load_state_transformation", _load_transformation)
    return observed


def _failed_wrapper(
    original_error,
    *,
    attempt_id="attempt-failed-017",
    input_state_id="state-a",
    status="execution_failed",
):
    error = application_module.TautomerEnumerationAttemptFailedError(
        attempt_id=attempt_id,
        input_state_id=input_state_id,
        status=status,
        original_error=original_error,
    )
    error.__cause__ = original_error
    return error


def _fake_failed_attempt(
    *,
    phase="execution",
    message="scientific failure",
    residual_state_ids=(),
    residual_operation_ids=(),
):
    return SimpleNamespace(
        error=SimpleNamespace(
            phase=phase,
            message=message,
            residual_state_ids=tuple(residual_state_ids),
            residual_operation_ids=tuple(residual_operation_ids),
        )
    )


def _patch_tautomer_failure(
    monkeypatch,
    original_error,
    *,
    status="execution_failed",
    attempt_id="attempt-failed-017",
    phase="execution",
    residual_state_ids=(),
    residual_operation_ids=(),
):
    wrapper = _failed_wrapper(
        original_error,
        attempt_id=attempt_id,
        status=status,
    )

    def _enumerate(*args, **kwargs):
        raise wrapper

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(
        cli_module,
        "load_execution_attempt",
        lambda project_dir, loaded_id: _fake_failed_attempt(
            phase=phase,
            message=str(original_error),
            residual_state_ids=residual_state_ids,
            residual_operation_ids=residual_operation_ids,
        ),
    )
    return wrapper


def _register_state_from_source(
    tmp_path,
    monkeypatch,
    source_file,
    *,
    state_id="state-a",
    file_format="sdf",
):
    project = _add_entity(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            state_id,
            "--entity",
            "curcumin",
            "--source-file",
            str(source_file),
            "--format",
            file_format,
        ],
    )
    assert result.exit_code == 0
    return project


def _write_rdkit_sdf(tmp_path, smiles, *, name="input.sdf", copies=1):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    path = tmp_path / name
    writer = Chem.SDWriter(str(path))
    try:
        for _ in range(copies):
            writer.write(molecule)
    finally:
        writer.close()
    return path


def test_tautomer_cli_01_help_documents_required_surface():
    result = runner.invoke(app, ["state", "enumerate-tautomers", "--help"])

    assert result.exit_code == 0
    assert "--input-state" in result.output
    assert "--max-states" in result.output
    assert "EXPERIMENT" in result.output.upper()


def test_tautomer_cli_02_missing_input_state_is_usage_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["state", "enumerate-tautomers", "exp001"])

    assert result.exit_code != 0
    assert "--input-state" in result.output


def test_tautomer_cli_03_missing_max_states_builds_none_policy(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_tautomer_success(monkeypatch)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 0
    assert observed["calls"] == 1
    assert observed["input_state_id"] == "state-a"
    assert observed["policy"].schema_version == "1.0"
    assert observed["policy"].source == "scientist"
    assert observed["policy"].max_states is None


def test_tautomer_cli_04_valid_max_states_builds_policy(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_tautomer_success(monkeypatch)

    result = runner.invoke(app, _tautomer_cli_args("--max-states", "2"))

    assert result.exit_code == 0
    assert observed["calls"] == 1
    assert observed["policy"].max_states == 2


def test_tautomer_cli_05_zero_max_states_rejected_before_application(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    calls = {"count": 0}

    def _must_not_run(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("application service must not be called")

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _must_not_run)
    result = runner.invoke(app, _tautomer_cli_args("--max-states", "0"))

    assert result.exit_code == 1
    assert "CONFIGURATION ERROR" in result.output
    assert calls["count"] == 0


def test_tautomer_cli_06_negative_max_states_rejected_before_application(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    calls = {"count": 0}

    def _must_not_run(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("application service must not be called")

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _must_not_run)
    result = runner.invoke(app, _tautomer_cli_args("--max-states", "-1"))

    assert result.exit_code == 1
    assert "CONFIGURATION ERROR" in result.output
    assert calls["count"] == 0


def test_tautomer_cli_07_malformed_input_state_rejected_before_application(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    calls = {"count": 0}

    def _must_not_run(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("application service must not be called")

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _must_not_run)
    result = runner.invoke(app, _tautomer_cli_args(input_state="bad id"))

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert calls["count"] == 0


@pytest.mark.skipif(_RDKIT_AVAILABLE, reason="requires an environment without RDKit")
def test_tautomer_cli_08a_missing_state_without_rdkit_is_input_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, _tautomer_cli_args(input_state="missing-state"))

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "resolution_failed" in result.output


@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="requires RDKit")
def test_tautomer_cli_08b_missing_state_with_rdkit_is_input_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, _tautomer_cli_args(input_state="missing-state"))

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "resolution_failed" in result.output


def test_tautomer_cli_09_non_sdf_representation_is_resolution_input_error(tmp_path, monkeypatch):
    source = _write_source_file(tmp_path, name="state.smi", content=b"C\n")
    _register_state_from_source(
        tmp_path,
        monkeypatch,
        source,
        file_format="smiles",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "Status: resolution_failed" in result.output
    assert "expected 'sdf'" in result.output


def test_tautomer_cli_10_checksum_mismatch_is_resolution_input_error(tmp_path, monkeypatch):
    source = _write_source_file(tmp_path, name="state.sdf", content=b"original")
    project = _register_state_from_source(tmp_path, monkeypatch, source)
    artifact = project / "scientific" / "artifacts" / "states" / "state-a" / "state.sdf"
    artifact.write_bytes(b"changed-after-registration")

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "Status: resolution_failed" in result.output
    assert "checksum mismatch" in result.output


@pytest.mark.skipif(_RDKIT_AVAILABLE, reason="requires an environment without RDKit")
def test_tautomer_cli_11_rdkit_absence_is_configuration_error(tmp_path, monkeypatch):
    source = _write_source_file(tmp_path, name="state.sdf", content=b"not-read-without-rdkit")
    _register_state_from_source(tmp_path, monkeypatch, source)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "CONFIGURATION ERROR" in result.output
    assert "Status: resolution_failed" in result.output


def test_tautomer_cli_12_zero_new_states_is_success(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_tautomer_success(monkeypatch, operation_ids=())

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 0
    assert observed["calls"] == 1
    assert "TAUTOMER ENUMERATION" in result.output
    assert "Status: succeeded" in result.output
    assert "New states published: 0" in result.output
    assert "transformation '" not in result.output


@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="requires RDKit")
def test_tautomer_cli_13_one_tautomer_real_end_to_end(tmp_path, monkeypatch):
    source = _write_rdkit_sdf(
        tmp_path,
        "Nc1nc2[nH]cnc2c(=O)[nH]1",
        name="guanine.sdf",
    )
    _register_state_from_source(tmp_path, monkeypatch, source)

    result = runner.invoke(app, _tautomer_cli_args("--max-states", "1"))

    assert result.exit_code == 0
    assert "Status: succeeded" in result.output
    assert "New states published: 1" in result.output
    assert "(transformation '" in result.output


def test_tautomer_cli_14_multiple_states_success_output(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_tautomer_success(
        monkeypatch,
        operation_ids=("op-1", "op-2", "op-3"),
        child_ids={"op-1": "state-1", "op-2": "state-2", "op-3": "state-3"},
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 0
    assert observed["calls"] == 1
    assert "New states published: 3" in result.output
    assert "State 'state-1'  (transformation 'op-1')" in result.output
    assert "State 'state-3'  (transformation 'op-3')" in result.output


def test_tautomer_cli_15_max_states_truncation_surface(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_tautomer_success(
        monkeypatch,
        operation_ids=("op-1", "op-2"),
        child_ids={"op-1": "state-1", "op-2": "state-2"},
    )

    result = runner.invoke(app, _tautomer_cli_args("--max-states", "2"))

    assert result.exit_code == 0
    assert observed["policy"].max_states == 2
    assert "New states published: 2" in result.output
    assert "op-1" in result.output
    assert "op-2" in result.output


def test_tautomer_cli_16_output_preserves_operation_id_order(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_success(
        monkeypatch,
        operation_ids=("op-b", "op-a"),
        child_ids={"op-b": "state-b", "op-a": "state-a-child"},
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 0
    assert result.output.index("op-b") < result.output.index("op-a")
    assert result.output.index("state-b") < result.output.index("state-a-child")


def test_tautomer_cli_17_invalid_sdf_is_execution_error(tmp_path, monkeypatch):
    source = _write_source_file(tmp_path, name="broken.sdf", content=b"this is not an SDF")
    _register_state_from_source(tmp_path, monkeypatch, source)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "Status: execution_failed" in result.output
    assert "Traceback" not in result.output


def test_tautomer_cli_18_multiple_molecules_is_execution_error(tmp_path, monkeypatch):
    source = _write_rdkit_sdf(tmp_path, "CCO", name="three.sdf", copies=3)
    _register_state_from_source(tmp_path, monkeypatch, source)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "Status: execution_failed" in result.output
    assert "Traceback" not in result.output


def test_tautomer_cli_19_incomplete_enumeration_maps_to_execution_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    original = IncompleteEnumerationError(
        status="max-tautomers-reached",
        tautomer_count=2,
    )
    _patch_tautomer_failure(monkeypatch, original)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "TAUTOMER ENUMERATION FAILED" in result.output


def test_tautomer_cli_20_generic_value_error_maps_to_input_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    original = ValueError("representation format mismatch")
    _patch_tautomer_failure(
        monkeypatch,
        original,
        status="resolution_failed",
        phase="resolution",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "EXECUTION ERROR" not in result.output


def test_tautomer_cli_21_file_exists_maps_to_input_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(monkeypatch, FileExistsError("generated identity collision"))

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_tautomer_cli_22_oserror_maps_to_storage_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(monkeypatch, OSError("disk full"))

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output


def test_tautomer_cli_23_failed_wrapper_displays_its_attempt_id(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(
        monkeypatch,
        InvalidSDFError("bad sdf"),
        attempt_id="attempt-exact-23",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "Attempt ID: attempt-exact-23" in result.output


def test_tautomer_cli_25_loads_exact_attempt_id_from_wrapper(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    original = InvalidSDFError("bad sdf")
    wrapper = _failed_wrapper(original, attempt_id="attempt-exact-25")
    loaded = []

    def _enumerate(*args, **kwargs):
        raise wrapper

    def _load(project_dir, attempt_id):
        loaded.append(attempt_id)
        return _fake_failed_attempt(message=str(original))

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(cli_module, "load_execution_attempt", _load)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert loaded == ["attempt-exact-25"]


def test_tautomer_cli_28_uses_no_attempt_directory_diff_or_direct_rdkit_import():
    command_source = inspect.getsource(cli_module.state_enumerate_tautomers)
    module_source = inspect.getsource(cli_module)

    assert "execution_attempts" not in command_source
    assert ".iterdir(" not in command_source
    assert ".glob(" not in command_source
    assert "load_execution_attempt(context.project_dir, error.attempt_id)" in command_source
    assert "from rdkit " not in module_source
    assert "from rdkit." not in module_source
    assert "import rdkit" not in module_source


def test_tautomer_cli_29_no_residuals_omits_unconfirmed_section(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(monkeypatch, InvalidSDFError("bad sdf"))

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "Unconfirmed removals" not in result.output


def test_tautomer_cli_30_state_residuals_are_unconfirmed(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(
        monkeypatch,
        OSError("publication failed"),
        residual_state_ids=("state-residual",),
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "Unconfirmed removals: 1 state(s), 0 transformation(s)" in result.output
    assert "State 'state-residual'" in result.output


def test_tautomer_cli_31_operation_residuals_are_unconfirmed(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(
        monkeypatch,
        OSError("publication failed"),
        residual_operation_ids=("op-residual",),
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "Unconfirmed removals: 0 state(s), 1 transformation(s)" in result.output
    assert "Transformation 'op-residual'" in result.output


def test_tautomer_cli_32_both_residual_kinds_are_unconfirmed(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(
        monkeypatch,
        OSError("publication failed"),
        residual_state_ids=("state-residual",),
        residual_operation_ids=("op-residual",),
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "Unconfirmed removals: 1 state(s), 1 transformation(s)" in result.output
    assert "State 'state-residual'" in result.output
    assert "Transformation 'op-residual'" in result.output


def test_tautomer_cli_33_attempt_load_failure_uses_fallback_template(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    original = InvalidSDFError("bad sdf")
    wrapper = _failed_wrapper(original, attempt_id="attempt-fallback-33")

    def _enumerate(*args, **kwargs):
        raise wrapper

    def _fail_load(*args, **kwargs):
        raise FileNotFoundError("attempt disappeared")

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(cli_module, "load_execution_attempt", _fail_load)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "Attempt ID: attempt-fallback-33" in result.output
    assert "Input state: state-a" in result.output
    assert "Status: execution_failed" in result.output
    assert "Error: bad sdf" in result.output
    assert "(additional attempt details could not be loaded)" in result.output
    assert "Phase:" not in result.output
    assert "Unconfirmed removals" not in result.output


@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="requires RDKit")
def test_tautomer_cli_34_native_max_tautomer_log_does_not_leak_to_output_channels(
    tmp_path, monkeypatch
):
    source = _write_rdkit_sdf(
        tmp_path,
        "Nc1nc2[nH]cnc2c(=O)[nH]1",
        name="guanine-max.sdf",
    )
    _register_state_from_source(tmp_path, monkeypatch, source)

    script = (
        "import molecular.rdkit_tautomer as r; "
        "r._MAX_TAUTOMERS = 2; "
        "from molecular.cli import app; "
        "app()"
    )
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = (
        source_root
        if not environment.get("PYTHONPATH")
        else source_root + os.pathsep + environment["PYTHONPATH"]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "state",
            "enumerate-tautomers",
            "exp001",
            "--input-state",
            "state-a",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "EXECUTION ERROR" in completed.stdout
    assert "Tautomer enumeration stopped" not in completed.stdout
    assert "Tautomer enumeration stopped" not in completed.stderr


def test_tautomer_cli_38_activity_log_is_not_duplicated_by_cli(tmp_path, monkeypatch):
    source = _write_source_file(tmp_path, name="broken-log.sdf", content=b"not an sdf")
    project = _register_state_from_source(tmp_path, monkeypatch, source)

    runner.invoke(app, _tautomer_cli_args())

    lines = (project / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()
    enumeration_lines = [
        line for line in lines if "command=enumerate-state-tautomers" in line
    ]
    assert len(enumeration_lines) == 2
    assert "outcome=started" in enumeration_lines[0]
    assert "outcome=error" in enumeration_lines[1]


def test_tautomer_cli_40_existing_state_show_command_still_works(tmp_path, monkeypatch):
    _add_entity_and_state(tmp_path, monkeypatch, state_id="state-a")

    result = runner.invoke(app, ["state", "show", "exp001", "state-a"])

    assert result.exit_code == 0
    assert "ID: state-a" in result.output


def test_tautomer_cli_39a_ki_propagates_keyboard_interrupt(tmp_path, monkeypatch):
    context = SimpleNamespace(project_dir=tmp_path)
    interruption = KeyboardInterrupt()

    monkeypatch.setattr(
        cli_module,
        "_resolve_experiment_or_exit",
        lambda identifier, **kwargs: context,
    )

    def _interrupt(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _interrupt)

    with pytest.raises(KeyboardInterrupt) as caught:
        cli_module.state_enumerate_tautomers(
            "exp001",
            input_state="state-a",
            max_states=None,
        )

    assert caught.value is interruption


def test_tautomer_cli_39a_se_propagates_system_exit(tmp_path, monkeypatch):
    context = SimpleNamespace(project_dir=tmp_path)
    interruption = SystemExit(17)

    monkeypatch.setattr(
        cli_module,
        "_resolve_experiment_or_exit",
        lambda identifier, **kwargs: context,
    )

    def _interrupt(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _interrupt)

    with pytest.raises(SystemExit) as caught:
        cli_module.state_enumerate_tautomers(
            "exp001",
            input_state="state-a",
            max_states=None,
        )

    assert caught.value is interruption


def test_tautomer_cli_48_persistence_wrapper_maps_to_storage_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    original = InvalidSDFError("scientific failure")
    persistence = OSError("could not persist terminal status")
    wrapper = application_module.TautomerEnumerationAttemptPersistenceError(
        attempt_id="attempt-persistence-48",
        input_state_id="state-a",
        intended_status="execution_failed",
        phase="execution",
        original_error=original,
        persistence_error=persistence,
        residual_state_ids=("state-residual",),
        residual_operation_ids=("op-residual",),
    )
    wrapper.__cause__ = persistence

    def _enumerate(*args, **kwargs):
        raise wrapper

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "STORAGE ERROR" in result.output
    assert "Attempt ID: attempt-persistence-48" in result.output
    assert "Input state: state-a" in result.output
    assert "Phase: execution" in result.output
    assert "Intended status: execution_failed" in result.output
    assert "Terminal status persisted: no" in result.output
    assert "Original error: scientific failure" in result.output
    assert "Persistence error: could not persist terminal status" in result.output
    assert "Unconfirmed removals: 1 state(s), 1 transformation(s)" in result.output


def test_tautomer_cli_49_persistence_path_performs_no_retry_reload_or_compensation(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path, monkeypatch)
    original = InvalidSDFError("scientific failure")
    persistence = OSError("terminal persistence failed")
    wrapper = application_module.TautomerEnumerationAttemptPersistenceError(
        attempt_id="attempt-persistence-49",
        input_state_id="state-a",
        intended_status="execution_failed",
        phase="execution",
        original_error=original,
        persistence_error=persistence,
        residual_state_ids=(),
        residual_operation_ids=(),
    )
    counters = {
        "load": 0,
        "mark_resolution_failed": 0,
        "mark_execution_failed": 0,
        "compensate": 0,
    }

    def _enumerate(*args, **kwargs):
        raise wrapper

    def _count(name):
        def _spy(*args, **kwargs):
            counters[name] += 1
            raise AssertionError(f"{name} must not be called by the CLI")

        return _spy

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(cli_module, "load_execution_attempt", _count("load"))
    monkeypatch.setattr(
        application_module,
        "mark_resolution_failed",
        _count("mark_resolution_failed"),
    )
    monkeypatch.setattr(
        application_module,
        "mark_execution_failed",
        _count("mark_execution_failed"),
    )
    monkeypatch.setattr(
        application_module,
        "_compensate_published_candidates",
        _count("compensate"),
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output
    assert counters == {
        "load": 0,
        "mark_resolution_failed": 0,
        "mark_execution_failed": 0,
        "compensate": 0,
    }


# =====================================================================
# Increment 017 post-microreview hardening
# =====================================================================


def _result_streams(result):
    streams = [result.output]
    for name in ("stdout", "stderr"):
        try:
            value = getattr(result, name)
        except (AttributeError, ValueError):
            continue
        if value not in streams:
            streams.append(value)
    return streams


def _assert_path_redacted(result, *absolute_paths):
    for stream in _result_streams(result):
        for absolute_path in absolute_paths:
            assert absolute_path not in stream
    assert "[redacted-path]" in result.output


def _validation_error_with_path(path_value):
    try:
        cli_module.TautomerEnumerationPolicy(
            schema_version="1.0",
            max_states=path_value,
            source="scientist",
        )
    except cli_module.ValidationError as error:
        return error
    raise AssertionError("expected ValidationError")


def test_tautomer_cli_missing_representation_does_not_expose_absolute_path(tmp_path, monkeypatch):
    source = _write_source_file(tmp_path, name="state.sdf", content=b"registered")
    project = _register_state_from_source(tmp_path, monkeypatch, source)
    artifact = project / "scientific" / "artifacts" / "states" / "state-a" / "state.sdf"
    artifact.unlink()
    absolute_path = str(artifact)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "Attempt ID:" in result.output
    assert "Input state: state-a" in result.output
    assert "Traceback" not in result.output
    for stream in _result_streams(result):
        assert absolute_path not in stream


@pytest.mark.parametrize(
    "absolute_path",
    [
        r"C:\Users\Pichau\molecular-platform\inputs\ligand.sdf",
        "/home/user/molecular-platform/inputs/ligand.sdf",
    ],
    ids=["windows", "posix"],
)
def test_tautomer_cli_redacts_invalid_sdf_absolute_paths(
    tmp_path, monkeypatch, absolute_path
):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(
        monkeypatch,
        InvalidSDFError(f"could not parse SDF at {absolute_path}"),
        attempt_id="attempt-redact-invalid-sdf",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "Attempt ID: attempt-redact-invalid-sdf" in result.output
    assert "Input state: state-a" in result.output
    assert "Traceback" not in result.output
    _assert_path_redacted(result, absolute_path)


def test_tautomer_cli_redacts_persisted_scientific_error_message(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    persisted_path = "/srv/molecular/private/run/input.sdf"
    original = InvalidSDFError("scientific parse failure")
    wrapper = _failed_wrapper(
        original,
        attempt_id="attempt-redact-persisted",
    )

    def _enumerate(*args, **kwargs):
        raise wrapper

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(
        cli_module,
        "load_execution_attempt",
        lambda project_dir, attempt_id: _fake_failed_attempt(
            message=f"persisted parser detail at {persisted_path}"
        ),
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "Attempt ID: attempt-redact-persisted" in result.output
    assert "Input state: state-a" in result.output
    assert "Traceback" not in result.output
    _assert_path_redacted(result, persisted_path)


def test_tautomer_cli_redacts_persistence_wrapper_error_paths(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    original_path = r"C:\Users\Pichau\molecular-platform\scientific\input.sdf"
    persistence_path = "/var/lib/molecular/execution_attempts/attempt.yaml"
    wrapper = application_module.TautomerEnumerationAttemptPersistenceError(
        attempt_id="attempt-redact-persistence",
        input_state_id="state-a",
        intended_status="execution_failed",
        phase="execution",
        original_error=InvalidSDFError(f"invalid SDF at {original_path}"),
        persistence_error=OSError(f"could not write {persistence_path}"),
        residual_state_ids=("state-residual",),
        residual_operation_ids=("op-residual",),
    )

    def _enumerate(*args, **kwargs):
        raise wrapper

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output
    assert "Attempt ID: attempt-redact-persistence" in result.output
    assert "Input state: state-a" in result.output
    assert "Phase: execution" in result.output
    assert "Intended status: execution_failed" in result.output
    assert "State 'state-residual'" in result.output
    assert "Transformation 'op-residual'" in result.output
    assert "Traceback" not in result.output
    _assert_path_redacted(result, original_path, persistence_path)


def test_tautomer_cli_redacts_fallback_error_when_attempt_load_fails(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    original_path = "/home/user/molecular-platform/scientific/input.sdf"
    load_path = r"C:\Users\Pichau\molecular-platform\execution_attempts\missing.yaml"
    wrapper = _failed_wrapper(
        FileNotFoundError(f"representation missing at {original_path}"),
        attempt_id="attempt-redact-fallback",
        status="resolution_failed",
    )

    def _enumerate(*args, **kwargs):
        raise wrapper

    def _fail_load(*args, **kwargs):
        raise FileNotFoundError(f"attempt metadata missing at {load_path}")

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(cli_module, "load_execution_attempt", _fail_load)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "Attempt ID: attempt-redact-fallback" in result.output
    assert "Input state: state-a" in result.output
    assert "Status: resolution_failed" in result.output
    assert "(additional attempt details could not be loaded)" in result.output
    assert "Traceback" not in result.output
    _assert_path_redacted(result, original_path, load_path)


def test_tautomer_cli_redacts_absolute_path_in_malformed_input_state(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    absolute_path = "/home/user/private/state.sdf"

    result = runner.invoke(
        app,
        _tautomer_cli_args(input_state=absolute_path),
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "Traceback" not in result.output
    _assert_path_redacted(result, absolute_path)


@pytest.mark.parametrize(
    "error_kind,banner,absolute_path",
    [
        (
            "file-not-found",
            "STORAGE ERROR",
            "/home/user/molecular-platform/scientific/transformations/op-1.yaml",
        ),
        (
            "oserror",
            "STORAGE ERROR",
            r"C:\Users\Pichau\molecular-platform\scientific\transformations\op-1.yaml",
        ),
        (
            "validation",
            "CONFIGURATION ERROR",
            "/srv/molecular/private/transformation.yaml",
        ),
        (
            "yaml",
            "CONFIGURATION ERROR",
            r"C:\Users\Pichau\molecular-platform\scientific\transformations\bad.yaml",
        ),
    ],
)
def test_tautomer_cli_handles_post_success_transformation_load_failures_once(
    tmp_path, monkeypatch, error_kind, banner, absolute_path
):
    _init_valid_experiment(tmp_path, monkeypatch)
    attempt = _fake_tautomer_success_attempt(
        ("op-1",),
        attempt_id="attempt-post-success-load",
    )
    calls = {"enumerate": 0, "load_transformation": 0, "load_attempt": 0}

    def _enumerate(*args, **kwargs):
        calls["enumerate"] += 1
        return attempt

    def _load_transformation(*args, **kwargs):
        calls["load_transformation"] += 1
        if error_kind == "file-not-found":
            raise FileNotFoundError(f"missing transformation at {absolute_path}")
        if error_kind == "oserror":
            raise OSError(f"could not read transformation at {absolute_path}")
        if error_kind == "validation":
            raise _validation_error_with_path(absolute_path)
        raise yaml.YAMLError(f"invalid YAML at {absolute_path}")

    def _load_attempt(*args, **kwargs):
        calls["load_attempt"] += 1
        raise AssertionError("post-success presentation must not reload the attempt")

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(cli_module, "load_state_transformation", _load_transformation)
    monkeypatch.setattr(cli_module, "load_execution_attempt", _load_attempt)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert banner in result.output
    assert "Attempt ID: attempt-post-success-load" in result.output
    assert "Input state: state-a" in result.output
    assert "Scientific enumeration completed: yes" in result.output
    assert "Stored result presentation: failed" in result.output
    assert "Transformation: 'op-1'" in result.output
    assert "Traceback" not in result.output
    assert "New states published:" not in result.output
    assert "Status: succeeded" not in result.output
    assert calls == {
        "enumerate": 1,
        "load_transformation": 1,
        "load_attempt": 0,
    }
    assert attempt.status == "succeeded"
    _assert_path_redacted(result, absolute_path)


def test_tautomer_cli_resolves_all_children_before_printing_success(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    attempt = _fake_tautomer_success_attempt(
        ("op-1", "op-2"),
        attempt_id="attempt-no-partial-success",
    )
    load_calls = []
    failure_path = "/home/user/molecular-platform/scientific/transformations/op-2.yaml"

    def _enumerate(*args, **kwargs):
        return attempt

    def _load(project_dir, operation_id):
        load_calls.append(operation_id)
        if operation_id == "op-1":
            return SimpleNamespace(id="op-1", child_state_id="state-one")
        raise FileNotFoundError(f"missing transformation at {failure_path}")

    monkeypatch.setattr(cli_module, "enumerate_state_tautomers", _enumerate)
    monkeypatch.setattr(cli_module, "load_state_transformation", _load)

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert load_calls == ["op-1", "op-2"]
    assert "STORAGE ERROR" in result.output
    assert "Attempt ID: attempt-no-partial-success" in result.output
    assert "Input state: state-a" in result.output
    assert "State 'state-one'" not in result.output
    assert "New states published:" not in result.output
    assert "Status: succeeded" not in result.output
    assert "Traceback" not in result.output
    assert attempt.status == "succeeded"
    _assert_path_redacted(result, failure_path)


# =====================================================================
# Increment 017 final sanitizer precision review
# =====================================================================


@pytest.mark.parametrize(
    "case_id,absolute_path,rendered_path,sensitive_fragments",
    [
        (
            "windows-drive",
            r"C:\Users\Jane\private\ligand.sdf",
            r"C:\Users\Jane\private\ligand.sdf",
            ("Jane", "private", "ligand.sdf"),
        ),
        (
            "windows-drive-spaces",
            r"C:\Users\Jane Doe\private data\ligand.sdf",
            r"C:\Users\Jane Doe\private data\ligand.sdf",
            ("Jane Doe", "private data", "ligand.sdf"),
        ),
        (
            "windows-forward-slashes",
            "C:/Users/Jane Doe/private data/ligand.sdf",
            "C:/Users/Jane Doe/private data/ligand.sdf",
            ("Jane Doe", "private data", "ligand.sdf"),
        ),
        (
            "unc",
            r"\\server\share\patient\record.sdf",
            r"\\server\share\patient\record.sdf",
            ("server", "share", "patient", "record.sdf"),
        ),
        (
            "unc-spaces",
            r"\\server\clinical data\Jane Doe\record.sdf",
            r"\\server\clinical data\Jane Doe\record.sdf",
            ("server", "clinical data", "Jane Doe", "record.sdf"),
        ),
        (
            "posix",
            "/home/jane/private/ligand.sdf",
            "/home/jane/private/ligand.sdf",
            ("jane", "private", "ligand.sdf"),
        ),
        (
            "posix-spaces",
            "/home/jane/private data/ligand sample.sdf",
            "/home/jane/private data/ligand sample.sdf",
            ("jane", "private data", "ligand sample.sdf"),
        ),
        (
            "single-quoted",
            "/home/jane/private data/ligand sample.sdf",
            "'/home/jane/private data/ligand sample.sdf'",
            ("jane", "private data", "ligand sample.sdf"),
        ),
        (
            "double-quoted",
            r"C:\Users\Jane Doe\private data\ligand.sdf",
            '"C:\\Users\\Jane Doe\\private data\\ligand.sdf"',
            ("Jane Doe", "private data", "ligand.sdf"),
        ),
        (
            "repr-escaped",
            r"C:\Users\Jane Doe\private data\ligand.sdf",
            repr(r"C:\Users\Jane Doe\private data\ligand.sdf"),
            ("Jane Doe", "private data", "ligand.sdf"),
        ),
        (
            "trailing-punctuation",
            "/home/jane/private data/ligand sample.sdf",
            "/home/jane/private data/ligand sample.sdf,",
            ("jane", "private data", "ligand sample.sdf"),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and value in {
        "windows-drive",
        "windows-drive-spaces",
        "windows-forward-slashes",
        "unc",
        "unc-spaces",
        "posix",
        "posix-spaces",
        "single-quoted",
        "double-quoted",
        "repr-escaped",
        "trailing-punctuation",
    } else None,
)
def test_tautomer_cli_precise_redaction_covers_unambiguous_absolute_paths(
    tmp_path,
    monkeypatch,
    case_id,
    absolute_path,
    rendered_path,
    sensitive_fragments,
):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(
        monkeypatch,
        InvalidSDFError(f"could not parse SDF at {rendered_path}"),
        attempt_id=f"attempt-precise-{case_id}",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert f"Attempt ID: attempt-precise-{case_id}" in result.output
    assert "Input state: state-a" in result.output
    assert "[redacted-path]" in result.output
    assert "Traceback" not in result.output

    for stream in _result_streams(result):
        assert absolute_path not in stream
        assert rendered_path not in stream
        for fragment in sensitive_fragments:
            assert fragment not in stream


@pytest.mark.parametrize(
    "chemical_text",
    [
        "C(=O)/N=C/C",
        "H+/OH- equilibrium",
        r"N/C=C\C",
        "SMILES C/C=C/C",
        "ratio H2O/EtOH",
        "pH+/OH- equilibrium",
    ],
)
def test_tautomer_path_sanitizer_preserves_chemical_slash_expressions(chemical_text):
    sanitized = cli_module._sanitize_tautomer_error_text(chemical_text)

    assert sanitized == chemical_text
    assert chemical_text in sanitized
    assert "[redacted-path]" not in sanitized


def test_tautomer_cli_mixed_message_redacts_only_path_and_preserves_chemistry(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path, monkeypatch)
    absolute_path = r"C:\Users\Jane Doe\private data\ligand.sdf"
    chemical_text = "C(=O)/N=C/C"
    _patch_tautomer_failure(
        monkeypatch,
        InvalidSDFError(
            f"structure {chemical_text} could not be parsed from {absolute_path}"
        ),
        attempt_id="attempt-mixed-path-chemistry",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "Attempt ID: attempt-mixed-path-chemistry" in result.output
    assert "Input state: state-a" in result.output
    assert "[redacted-path]" in result.output
    assert chemical_text in result.output
    assert "Traceback" not in result.output

    for stream in _result_streams(result):
        assert absolute_path not in stream
        assert "Jane Doe" not in stream
        assert "private data" not in stream
        assert "ligand.sdf" not in stream
        assert chemical_text in stream or stream == ""


# =====================================================================
# Increment 017 URL and multi-path sanitizer boundary review
# =====================================================================


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "https://example.org/path/ligand.sdf",
            "https://example.org/path/ligand.sdf",
        ),
        (
            "https://example.org/path/ligand.sdf?download=1",
            "https://example.org/path/ligand.sdf?download=1",
        ),
        (
            "https://example.org/path/ligand.sdf#record",
            "https://example.org/path/ligand.sdf#record",
        ),
        (
            "http://example.org/a/b/file.sdf",
            "http://example.org/a/b/file.sdf",
        ),
        (
            "https://example.org/path/ligand.sdf and /home/jane/a.sdf",
            "https://example.org/path/ligand.sdf and [redacted-path]",
        ),
        (
            r"C:\Data\a.sdf then https://example.org/path/ligand.sdf",
            "[redacted-path] then https://example.org/path/ligand.sdf",
        ),
        (
            r"C:\Data\a.sdf then https://example.org/path/ligand.sdf and /home/jane/b.sdf",
            "[redacted-path] then https://example.org/path/ligand.sdf and [redacted-path]",
        ),
    ],
    ids=[
        "https",
        "https-query",
        "https-fragment",
        "http",
        "url-before-path",
        "url-after-path",
        "url-between-paths",
    ],
)
def test_tautomer_path_sanitizer_preserves_urls_exactly(message, expected):
    sanitized = cli_module._sanitize_tautomer_error_text(message)

    assert sanitized == expected

    for token in message.split():
        if token.startswith(("http://", "https://")):
            assert token in sanitized


@pytest.mark.parametrize(
    "message,expected,original_paths,sensitive_fragments",
    [
        (
            "/home/jane/a.sdf and /opt/data/b.sdf",
            "[redacted-path] and [redacted-path]",
            ("/home/jane/a.sdf", "/opt/data/b.sdf"),
            ("jane", "a.sdf", "data", "b.sdf"),
        ),
        (
            r"C:\Users\Jane\a.sdf and D:\Data\b.sdf",
            "[redacted-path] and [redacted-path]",
            (r"C:\Users\Jane\a.sdf", r"D:\Data\b.sdf"),
            ("Jane", "a.sdf", "Data", "b.sdf"),
        ),
        (
            "C:/Users/Jane/a.sdf and /home/jane/b.sdf",
            "[redacted-path] and [redacted-path]",
            ("C:/Users/Jane/a.sdf", "/home/jane/b.sdf"),
            ("Jane", "a.sdf", "jane", "b.sdf"),
        ),
        (
            r"\\server\share\a.sdf, then C:\Data\b.sdf",
            "[redacted-path], then [redacted-path]",
            (r"\\server\share\a.sdf", r"C:\Data\b.sdf"),
            ("server", "share", "a.sdf", "Data", "b.sdf"),
        ),
        (
            r"C:\Users\Jane Doe\a sample.sdf and /home/jane/private data/b sample.sdf",
            "[redacted-path] and [redacted-path]",
            (
                r"C:\Users\Jane Doe\a sample.sdf",
                "/home/jane/private data/b sample.sdf",
            ),
            ("Jane Doe", "a sample.sdf", "jane", "private data", "b sample.sdf"),
        ),
        (
            r"C:\A\a.sdf; /opt/b.sdf and \\server\share\c.sdf",
            "[redacted-path]; [redacted-path] and [redacted-path]",
            (r"C:\A\a.sdf", "/opt/b.sdf", r"\\server\share\c.sdf"),
            ("a.sdf", "b.sdf", "server", "share", "c.sdf"),
        ),
        (
            "/home/jane/a.sdf then /opt/data/b.sdf",
            "[redacted-path] then [redacted-path]",
            ("/home/jane/a.sdf", "/opt/data/b.sdf"),
            ("jane", "a.sdf", "data", "b.sdf"),
        ),
        (
            "/home/jane/a.sdf, /opt/data/b.sdf",
            "[redacted-path], [redacted-path]",
            ("/home/jane/a.sdf", "/opt/data/b.sdf"),
            ("jane", "a.sdf", "data", "b.sdf"),
        ),
        (
            "/home/jane/a.sdf; /opt/data/b.sdf",
            "[redacted-path]; [redacted-path]",
            ("/home/jane/a.sdf", "/opt/data/b.sdf"),
            ("jane", "a.sdf", "data", "b.sdf"),
        ),
        (
            "(/home/jane/a.sdf) (/opt/data/b.sdf)",
            "([redacted-path]) ([redacted-path])",
            ("/home/jane/a.sdf", "/opt/data/b.sdf"),
            ("jane", "a.sdf", "data", "b.sdf"),
        ),
        (
            "/home/jane/a.sdf\n/opt/data/b.sdf",
            "[redacted-path]\n[redacted-path]",
            ("/home/jane/a.sdf", "/opt/data/b.sdf"),
            ("jane", "a.sdf", "data", "b.sdf"),
        ),
    ],
    ids=[
        "two-posix-and",
        "two-windows-and",
        "windows-posix-and",
        "unc-windows-comma-then",
        "paths-with-spaces",
        "three-paths",
        "separator-then",
        "separator-comma",
        "separator-semicolon",
        "separator-parentheses",
        "separator-newline",
    ],
)
def test_tautomer_cli_redacts_multiple_paths_one_occurrence_at_a_time(
    tmp_path,
    monkeypatch,
    message,
    expected,
    original_paths,
    sensitive_fragments,
):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_tautomer_failure(
        monkeypatch,
        InvalidSDFError(message),
        attempt_id="attempt-multiple-paths",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "Attempt ID: attempt-multiple-paths" in result.output
    assert "Input state: state-a" in result.output
    assert f"Error: {expected}" in result.output
    assert result.output.count("[redacted-path]") == expected.count("[redacted-path]")
    assert "Traceback" not in result.output

    for stream in _result_streams(result):
        for absolute_path in original_paths:
            assert absolute_path not in stream
        for fragment in sensitive_fragments:
            assert fragment not in stream


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "C(=O)/N=C/C at /home/jane/a.sdf then H+/OH- equilibrium",
            "C(=O)/N=C/C at [redacted-path] then H+/OH- equilibrium",
        ),
        (
            "https://example.org/a.sdf and /home/jane/b.sdf with SMILES C/C=C/C",
            "https://example.org/a.sdf and [redacted-path] with SMILES C/C=C/C",
        ),
        (
            r"C:\Data\a.sdf then https://example.org/a.sdf and /home/jane/b.sdf",
            "[redacted-path] then https://example.org/a.sdf and [redacted-path]",
        ),
        (
            "https://example.org/a.sdf and http://example.org/b.sdf?x=1#record",
            "https://example.org/a.sdf and http://example.org/b.sdf?x=1#record",
        ),
        (
            "ratio H2O/EtOH remains unchanged",
            "ratio H2O/EtOH remains unchanged",
        ),
    ],
    ids=[
        "chemistry-path-chemistry",
        "url-path-chemistry",
        "path-url-path",
        "two-urls-no-path",
        "no-path",
    ],
)
def test_tautomer_path_sanitizer_preserves_mixed_message_boundaries(message, expected):
    sanitized = cli_module._sanitize_tautomer_error_text(message)

    assert sanitized == expected


# =====================================================================
# Increment 017 hotfix: a connector-like word ("and"/"then") that is
# literally part of a single path's directory name must not defeat
# redaction of that path, while two genuinely distinct paths joined by
# the same words must still be redacted separately.
# =====================================================================


@pytest.mark.parametrize(
    "path,sensitive_fragments",
    [
        (
            r"C:\Users\Jane and Joe\private\ligand.sdf",
            ("Jane", "Joe", "private", "ligand.sdf"),
        ),
        (
            r"C:\Users\Jane then Joe\private\ligand.sdf",
            ("Jane", "Joe", "private", "ligand.sdf"),
        ),
        (
            r"\\server\share\Jane and Joe\ligand.sdf",
            ("server", "share", "Jane", "Joe", "ligand.sdf"),
        ),
        (
            r"\\server\share\Jane then Joe\ligand.sdf",
            ("server", "share", "Jane", "Joe", "ligand.sdf"),
        ),
        (
            "/home/Jane and Joe/private/ligand.sdf",
            ("Jane", "Joe", "private", "ligand.sdf"),
        ),
        (
            "/home/Jane then Joe/private/ligand.sdf",
            ("Jane", "Joe", "private", "ligand.sdf"),
        ),
    ],
    ids=[
        "windows-drive-and",
        "windows-drive-then",
        "unc-and",
        "unc-then",
        "posix-and",
        "posix-then",
    ],
)
def test_tautomer_path_sanitizer_redacts_connector_word_inside_directory_name(
    path, sensitive_fragments
):
    sanitized = cli_module._sanitize_tautomer_error_text(path)

    assert sanitized == "[redacted-path]"
    assert sanitized.count("[redacted-path]") == 1
    for fragment in sensitive_fragments:
        assert fragment not in sanitized


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/Jane and Joe/ligand.sdf",
        "data/Jane then Joe/ligand.sdf",
    ],
    ids=["relative-and", "relative-then"],
)
def test_tautomer_path_sanitizer_does_not_redact_relative_path_by_connector_word_alone(
    relative_path,
):
    sanitized = cli_module._sanitize_tautomer_error_text(relative_path)

    assert sanitized == relative_path
    assert "[redacted-path]" not in sanitized


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "path '" + r"C:\Users\Jane and Joe\ligand.sdf" + "'",
            "path '[redacted-path]'",
        ),
        (
            'path "' + "/home/Jane then Joe/ligand.sdf" + '"',
            'path "[redacted-path]"',
        ),
        (
            "repr: WindowsPath('" + r"C:\Users\Jane and Joe\ligand.sdf" + "')",
            "repr: WindowsPath('[redacted-path]')",
        ),
        (
            "structure could not be parsed from /home/Jane and Joe/private/ligand.sdf.",
            "structure could not be parsed from [redacted-path].",
        ),
    ],
    ids=[
        "single-quoted",
        "double-quoted",
        "repr-wrapped",
        "trailing-punctuation",
    ],
)
def test_tautomer_path_sanitizer_connector_path_respects_quoting_and_punctuation(
    message, expected
):
    sanitized = cli_module._sanitize_tautomer_error_text(message)

    assert sanitized == expected


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "https://example.org/a.sdf then /home/Jane and Joe/private/ligand.sdf",
            "https://example.org/a.sdf then [redacted-path]",
        ),
        (
            "/home/Jane and Joe/private/ligand.sdf then see https://example.org/a.sdf",
            "[redacted-path] then see https://example.org/a.sdf",
        ),
        (
            "C(=O)/N=C/C found in /home/Jane and Joe/private/ligand.sdf "
            "while pH+/OH- equilibrium holds",
            "C(=O)/N=C/C found in [redacted-path] while pH+/OH- equilibrium holds",
        ),
    ],
    ids=[
        "url-before-connector-path",
        "url-after-connector-path",
        "chemistry-before-and-after-connector-path",
    ],
)
def test_tautomer_path_sanitizer_connector_path_preserves_urls_and_chemistry(
    message, expected
):
    sanitized = cli_module._sanitize_tautomer_error_text(message)

    assert sanitized == expected


@pytest.mark.parametrize(
    "message,expected,sensitive_fragments",
    [
        (
            "C:/Users/Jane/a.sdf and /home/Jane/b.sdf",
            "[redacted-path] and [redacted-path]",
            ("Jane", "a.sdf", "b.sdf"),
        ),
        (
            "C:/Users/Jane/a.sdf then /home/Jane/b.sdf",
            "[redacted-path] then [redacted-path]",
            ("Jane", "a.sdf", "b.sdf"),
        ),
        (
            r"C:\Users\Jane\a.sdf and /home/jane/b.sdf then \\server\share\c.sdf",
            "[redacted-path] and [redacted-path] then [redacted-path]",
            ("Jane", "a.sdf", "jane", "b.sdf", "server", "share", "c.sdf"),
        ),
    ],
    ids=[
        "two-distinct-paths-and",
        "two-distinct-paths-then",
        "three-distinct-paths-mixed-connectors",
    ],
)
def test_tautomer_path_sanitizer_still_separates_distinct_paths_across_connectors(
    message, expected, sensitive_fragments
):
    sanitized = cli_module._sanitize_tautomer_error_text(message)

    assert sanitized == expected
    assert sanitized.count("[redacted-path]") == expected.count("[redacted-path]")
    for fragment in sensitive_fragments:
        assert fragment not in sanitized
    assert " and " in sanitized or "and" not in message
    assert " then " in sanitized or "then" not in message


def test_tautomer_cli_redacts_path_with_connector_in_directory_name(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path, monkeypatch)
    absolute_path = r"C:\Users\Jane and Joe\private data\ligand.sdf"
    chemical_text = "H+/OH- equilibrium"
    _patch_tautomer_failure(
        monkeypatch,
        InvalidSDFError(
            f"structure could not be parsed from {absolute_path} ({chemical_text})"
        ),
        attempt_id="attempt-connector-directory-name",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "Attempt ID: attempt-connector-directory-name" in result.output
    assert "Input state: state-a" in result.output
    assert result.output.count("[redacted-path]") == 1
    assert chemical_text in result.output
    assert "Traceback" not in result.output

    for stream in _result_streams(result):
        assert absolute_path not in stream
        assert "Jane" not in stream
        assert "Joe" not in stream
        assert "private data" not in stream
        assert "ligand.sdf" not in stream
        assert chemical_text in stream or stream == ""


# =====================================================================
# Increment 018 Checkpoint 5 - `ligand promote` CLI
# =====================================================================


def _fake_promotion_result(
    *,
    promotion_id="eee91465c48294045e34842e20df2142a5f159a8188b104fa701eb6fe4c5e795",
    entity_id="curcumin",
    state_id="curcumin-initial",
    entity_created=True,
    state_created=True,
    reused_existing_promotion=False,
    representation_path="scientific/artifacts/states/curcumin-initial/snapshot.sdf",
):
    return SimpleNamespace(
        promotion_id=promotion_id,
        entity_id=entity_id,
        state_id=state_id,
        entity_created=entity_created,
        state_created=state_created,
        reused_existing_promotion=reused_existing_promotion,
        representation=SimpleNamespace(path=representation_path),
    )


def _fake_resolved(
    *,
    engine_name="rdkit",
    engine_version="2026.03.6",
    snapshot_checksum="0" * 64,
    warnings=(),
):
    return SimpleNamespace(
        engine_name=engine_name,
        engine_version=engine_version,
        snapshot_checksum=snapshot_checksum,
        warnings=tuple(warnings),
    )


def _fake_promotion_attempt(*, attempt_id="attempt-promo-001", status="succeeded", result=None, resolved=None):
    return SimpleNamespace(
        id=attempt_id,
        status=status,
        result=result if result is not None else _fake_promotion_result(),
        resolved=resolved if resolved is not None else _fake_resolved(),
    )


def _promote_cli_args(
    experiment="exp001",
    input_id="curcumin-pubchem",
    *,
    entity="curcumin",
    state="curcumin-initial",
    name="Curcumin",
    revision=None,
):
    args = [
        "ligand",
        "promote",
        experiment,
        input_id,
        "--entity",
        entity,
        "--state",
        state,
        "--name",
        name,
    ]
    if revision is not None:
        args += ["--revision", revision]
    return args


def _patch_promote_success(monkeypatch, *, attempt=None):
    observed = {"calls": 0}
    fake_attempt = attempt if attempt is not None else _fake_promotion_attempt()

    def _promote(
        project_dir, *, input_id, entity_id, state_id, entity_name, policy, intentional_revision=None
    ):
        observed["calls"] += 1
        observed["project_dir"] = project_dir
        observed["input_id"] = input_id
        observed["entity_id"] = entity_id
        observed["state_id"] = state_id
        observed["entity_name"] = entity_name
        observed["policy"] = policy
        observed["intentional_revision"] = intentional_revision
        return fake_attempt

    monkeypatch.setattr(cli_module, "promote_ligand_input", _promote)
    return observed


def _promotion_failed_wrapper(
    original_error,
    *,
    attempt_id="attempt-promo-failed",
    input_id="curcumin-pubchem",
    status="execution_failed",
):
    error = LigandPromotionAttemptFailedError(
        attempt_id=attempt_id,
        input_id=input_id,
        status=status,
        original_error=original_error,
    )
    error.__cause__ = original_error
    return error


def _patch_promote_failure(
    monkeypatch,
    original_error,
    *,
    attempt_id="attempt-promo-failed",
    status="execution_failed",
):
    wrapper = _promotion_failed_wrapper(original_error, attempt_id=attempt_id, status=status)
    observed = {"calls": 0}

    def _promote(*args, **kwargs):
        observed["calls"] += 1
        raise wrapper

    monkeypatch.setattr(cli_module, "promote_ligand_input", _promote)
    return observed, wrapper


def _fake_loaded_failed_attempt(
    *,
    phase="execution",
    message="scientific failure",
    residual_entity_ids=(),
    residual_state_ids=(),
    residual_reservation_keys=(),
):
    return SimpleNamespace(
        error=SimpleNamespace(
            phase=phase,
            message=message,
            residual_entity_ids=tuple(residual_entity_ids),
            residual_state_ids=tuple(residual_state_ids),
            residual_reservation_keys=tuple(residual_reservation_keys),
        )
    )


def _patch_promote_persistence_failure(
    monkeypatch,
    *,
    intended_status="execution_failed",
    attempt_id="attempt-promo-persist",
    input_id="curcumin-pubchem",
    phase="execution",
    original_error=None,
    persistence_error=None,
    residual_entity_ids=(),
    residual_state_ids=(),
    residual_reservation_keys=(),
):
    original_error = original_error if original_error is not None else RuntimeError("original scientific error")
    persistence_error = (
        persistence_error if persistence_error is not None else OSError("could not write attempt status")
    )
    wrapper = LigandPromotionAttemptPersistenceError(
        attempt_id=attempt_id,
        input_id=input_id,
        phase=phase,
        intended_status=intended_status,
        original_error=original_error,
        persistence_error=persistence_error,
        residual_entity_ids=residual_entity_ids,
        residual_state_ids=residual_state_ids,
        residual_reservation_keys=residual_reservation_keys,
    )
    wrapper.__cause__ = persistence_error

    def _promote(*args, **kwargs):
        raise wrapper

    monkeypatch.setattr(cli_module, "promote_ligand_input", _promote)
    return wrapper


# --- item 111: help and signature ---


def test_ligand_promote_appears_in_ligand_help():
    result = runner.invoke(app, ["ligand", "--help"])

    assert result.exit_code == 0
    assert "promote" in result.output


def test_ligand_promote_help_shows_frozen_signature():
    result = runner.invoke(app, ["ligand", "promote", "--help"])

    assert result.exit_code == 0
    assert "experiment" in result.output
    assert "input_id" in result.output
    assert "--entity" in result.output
    assert "--state" in result.output
    assert "--name" in result.output
    assert "--revision" in result.output
    # Section 15.1: no individual RDKit flag is ever exposed.
    assert "--sanitize" not in result.output
    assert "--remove-hs" not in result.output
    assert "--strict-parsing" not in result.output
    assert "--snapshot" not in result.output


# --- items 112/113: success - new promotion, mixed reuse, idempotent ---


def test_ligand_promote_new_success_renders_frozen_fields(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_promote_success(
        monkeypatch,
        attempt=_fake_promotion_attempt(
            attempt_id="attempt-promo-new",
            result=_fake_promotion_result(entity_created=True, state_created=True),
            resolved=_fake_resolved(warnings=("sd-name-property-absent",)),
        ),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 0
    assert observed["calls"] == 1
    assert "LIGAND PROMOTION" in result.output
    assert "Attempt ID: attempt-promo-new" in result.output
    assert (
        "Promotion ID: eee91465c48294045e34842e20df2142a5f159a8188b104fa701eb6fe4c5e795"
        in result.output
    )
    assert "Input: curcumin-pubchem" in result.output
    assert "Entity: curcumin (created)" in result.output
    assert "Initial state: curcumin-initial (created)" in result.output
    assert "Representation: scientific/artifacts/states/curcumin-initial/snapshot.sdf" in result.output
    assert f"RAW checksum verified: {'0' * 64}" in result.output
    assert "Engine: rdkit 2026.03.6" in result.output
    assert "Warnings:" in result.output
    assert "sd-name-property-absent" in result.output
    assert "Traceback" not in result.output


def test_ligand_promote_new_success_with_reused_entity_shows_reused_annotation(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_success(
        monkeypatch,
        attempt=_fake_promotion_attempt(
            result=_fake_promotion_result(entity_created=False, state_created=True)
        ),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 0
    assert "Entity: curcumin (reused)" in result.output
    assert "Initial state: curcumin-initial (created)" in result.output


def test_ligand_promote_idempotent_success_renders_frozen_fields(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_promote_success(
        monkeypatch,
        attempt=_fake_promotion_attempt(
            attempt_id="attempt-promo-retry",
            result=_fake_promotion_result(entity_created=False, state_created=False, reused_existing_promotion=True),
        ),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 0
    assert observed["calls"] == 1
    assert "LIGAND PROMOTION" in result.output
    assert "Attempt ID: attempt-promo-retry" in result.output
    assert "Result: existing promotion reused" in result.output
    assert "Entity: curcumin" in result.output
    assert "Initial state: curcumin-initial" in result.output
    # the idempotent template never claims a fresh (created) annotation.
    assert "(created)" not in result.output
    assert "Traceback" not in result.output


def test_ligand_promote_success_passes_frozen_policy_and_revision(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_promote_success(monkeypatch)

    result = runner.invoke(app, _promote_cli_args(revision="revision-1"))

    assert result.exit_code == 0
    assert observed["input_id"] == "curcumin-pubchem"
    assert observed["entity_id"] == "curcumin"
    assert observed["state_id"] == "curcumin-initial"
    assert observed["entity_name"] == "Curcumin"
    assert observed["intentional_revision"] == "revision-1"
    policy = observed["policy"]
    assert policy.sanitize is True
    assert policy.remove_hs is False
    assert policy.strict_parsing is True


def test_ligand_promote_success_omits_revision_when_not_given(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_promote_success(monkeypatch)

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 0
    assert observed["intentional_revision"] is None


# --- item 114: four banners and exit code, for the promotion workflow's
# own closed set of possible original_error types ---


@pytest.mark.parametrize(
    "original_error,expected_banner",
    [
        (FileNotFoundError("LigandInput not found"), "INPUT ERROR"),
        (UnsupportedPromotionFormatError("format is not sdf"), "INPUT ERROR"),
        (PromotionIdentityConflictError("ids diverge from the existing record"), "INPUT ERROR"),
        (PromotionRevisionRequiredError("a revision token is required"), "INPUT ERROR"),
        (InvalidPromotionSDFError("the SDF source could not be parsed"), "EXECUTION ERROR"),
        (PromotionRecordCountError(2), "EXECUTION ERROR"),
        (DisconnectedStructureError("more than one connected fragment"), "EXECUTION ERROR"),
        (RDKitUnavailableError("rdkit is not installed"), "CONFIGURATION ERROR"),
        (PromotionConsistencyError("lineage and filesystem diverge"), "CONFIGURATION ERROR"),
        (UnsafeRawArtifactPathError("unsafe raw artifact path"), "STORAGE ERROR"),
        (RawArtifactIdentityUnstableError("identity changed"), "STORAGE ERROR"),
        (RawArtifactSharingViolationError("the OS denied the operation"), "STORAGE ERROR"),
        (PromotionReservationConflictError("another attempt holds the reservation"), "STORAGE ERROR"),
        (PromotionPublicationUnavailableError("no exclusive publication primitive"), "STORAGE ERROR"),
    ],
    ids=[
        "input-not-found",
        "unsupported-format",
        "identity-conflict",
        "revision-required",
        "invalid-sdf",
        "record-count",
        "disconnected-structure",
        "rdkit-unavailable",
        "consistency-error",
        "unsafe-raw-path",
        "raw-identity-unstable",
        "raw-sharing-violation",
        "reservation-conflict",
        "promotion-primitive-unavailable",
    ],
)
def test_ligand_promote_failure_banners_cover_the_frozen_taxonomy(
    tmp_path, monkeypatch, original_error, expected_banner
):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_failure(monkeypatch, original_error, attempt_id="attempt-taxonomy")
    monkeypatch.setattr(
        cli_module,
        "load_promotion_attempt",
        lambda *a, **k: _fake_loaded_failed_attempt(message=str(original_error)),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert expected_banner in result.output
    assert "Traceback" not in result.output


# --- item 115: attempt ID/status/phase ---


def test_ligand_promote_failure_shows_attempt_id_status_phase(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_failure(
        monkeypatch,
        InvalidPromotionSDFError("the SDF source could not be parsed"),
        attempt_id="attempt-detail-001",
        status="execution_failed",
    )
    monkeypatch.setattr(
        cli_module,
        "load_promotion_attempt",
        lambda *a, **k: _fake_loaded_failed_attempt(phase="execution", message="the SDF source could not be parsed"),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "Attempt ID: attempt-detail-001" in result.output
    assert "Input: curcumin-pubchem" in result.output
    assert "Status: execution_failed" in result.output
    assert "Phase: execution" in result.output


# --- item 116: unconfirmed removals ---


def test_ligand_promote_failure_shows_unconfirmed_removals(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_failure(
        monkeypatch,
        PromotionConsistencyError("lineage and filesystem diverge"),
        attempt_id="attempt-residual-001",
    )
    monkeypatch.setattr(
        cli_module,
        "load_promotion_attempt",
        lambda *a, **k: _fake_loaded_failed_attempt(
            message="lineage and filesystem diverge",
            residual_entity_ids=("curcumin",),
            residual_state_ids=("curcumin-initial",),
            residual_reservation_keys=("semkey123",),
        ),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "Unconfirmed removals: 1 entity(ies), 1 state(s), 1 reservation(s)" in result.output
    assert "Entity 'curcumin'" in result.output
    assert "State 'curcumin-initial'" in result.output
    assert "Reservation 'semkey123'" in result.output


def test_ligand_promote_failure_omits_unconfirmed_removals_when_empty(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_failure(monkeypatch, FileNotFoundError("not found"), attempt_id="attempt-clean-001")
    monkeypatch.setattr(
        cli_module, "load_promotion_attempt", lambda *a, **k: _fake_loaded_failed_attempt()
    )

    result = runner.invoke(app, _promote_cli_args())

    assert "Unconfirmed removals" not in result.output


# --- item 117: no traceback, no absolute path ---


@pytest.mark.parametrize(
    "case_id,absolute_path,rendered_path",
    [
        ("windows-drive", r"C:\Users\Jane\private\ligand.sdf", r"C:\Users\Jane\private\ligand.sdf"),
        (
            "windows-drive-spaces",
            r"C:\Users\Jane Doe\private data\ligand.sdf",
            r"C:\Users\Jane Doe\private data\ligand.sdf",
        ),
        ("unc", r"\\server\share\patient\record.sdf", r"\\server\share\patient\record.sdf"),
        (
            "unc-spaces",
            r"\\server\clinical data\Jane Doe\record.sdf",
            r"\\server\clinical data\Jane Doe\record.sdf",
        ),
        ("posix", "/home/jane/private/ligand.sdf", "/home/jane/private/ligand.sdf"),
        (
            "posix-spaces",
            "/home/jane/private data/ligand sample.sdf",
            "/home/jane/private data/ligand sample.sdf",
        ),
        (
            "single-quoted",
            "/home/jane/private data/ligand sample.sdf",
            "'/home/jane/private data/ligand sample.sdf'",
        ),
        (
            "double-quoted",
            r"C:\Users\Jane Doe\private data\ligand.sdf",
            '"C:\\Users\\Jane Doe\\private data\\ligand.sdf"',
        ),
        (
            "repr-escaped",
            r"C:\Users\Jane Doe\private data\ligand.sdf",
            repr(r"C:\Users\Jane Doe\private data\ligand.sdf"),
        ),
        (
            "multiple-paths",
            "/home/jane/a.sdf",
            "copied /home/jane/a.sdf to /home/jane/b.sdf",
        ),
    ],
)
def test_ligand_promote_failure_never_exposes_absolute_path(
    tmp_path, monkeypatch, case_id, absolute_path, rendered_path
):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_failure(
        monkeypatch,
        InvalidPromotionSDFError(f"could not parse SDF at {rendered_path}"),
        attempt_id=f"attempt-path-{case_id}",
    )
    monkeypatch.setattr(
        cli_module,
        "load_promotion_attempt",
        lambda *a, **k: _fake_loaded_failed_attempt(message=f"could not parse SDF at {rendered_path}"),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "[redacted-path]" in result.output
    for stream in _result_streams(result):
        assert absolute_path not in stream


# --- item 118: persistence wrapper - no retry, no compensation from the CLI ---


def test_ligand_promote_persistence_wrapper_execution_failed(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_persistence_failure(
        monkeypatch,
        intended_status="execution_failed",
        attempt_id="attempt-persist-exec",
        phase="execution",
        original_error=RuntimeError("register_promotion failed"),
        persistence_error=OSError("disk full while marking execution_failed"),
        residual_entity_ids=("curcumin",),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output
    assert "Attempt ID: attempt-persist-exec" in result.output
    assert "Input: curcumin-pubchem" in result.output
    assert "Phase: execution" in result.output
    assert "Intended status: execution_failed" in result.output
    assert "Terminal status persisted: no" in result.output
    assert "register_promotion failed" in result.output
    assert "disk full while marking execution_failed" in result.output
    assert "Entity 'curcumin'" in result.output
    assert "Traceback" not in result.output
    # never reclassified as a false success/failure claim.
    assert "PROMOTION FAILED" not in result.output


def test_ligand_promote_persistence_wrapper_resolution_failed(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_persistence_failure(
        monkeypatch,
        intended_status="resolution_failed",
        attempt_id="attempt-persist-resolve",
        phase="resolution",
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output
    assert "Phase: resolution" in result.output
    assert "Intended status: resolution_failed" in result.output
    assert "Terminal status persisted: no" in result.output


def test_ligand_promote_calls_application_service_exactly_once_on_persistence_failure(
    tmp_path, monkeypatch
):
    _init_valid_experiment(tmp_path, monkeypatch)
    calls = {"n": 0}
    wrapper = LigandPromotionAttemptPersistenceError(
        attempt_id="attempt-once",
        input_id="curcumin-pubchem",
        phase="execution",
        intended_status="execution_failed",
        original_error=RuntimeError("x"),
        persistence_error=OSError("y"),
    )

    def _promote(*args, **kwargs):
        calls["n"] += 1
        raise wrapper

    monkeypatch.setattr(cli_module, "promote_ligand_input", _promote)

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert calls["n"] == 1


# --- explicit requirement: intended_status="succeeded" presented honestly ---


def test_ligand_promote_persistence_wrapper_succeeded_never_claims_failure(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_persistence_failure(
        monkeypatch,
        intended_status="succeeded",
        attempt_id="attempt-persist-succeeded",
        phase="execution",
        original_error=RuntimeError("mark_succeeded failed"),
        persistence_error=RuntimeError("mark_succeeded failed"),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output
    assert "Intended status: succeeded" in result.output
    assert "Scientific commit: may already be published" in result.output
    assert "Operational status persisted: no" in result.output
    # never claims the promotion itself failed.
    assert "PROMOTION FAILED" not in result.output
    assert "the promotion failed" not in result.output.lower()
    assert "Traceback" not in result.output


# --- fallback when the attempt cannot be reloaded ---


def test_ligand_promote_failure_fallback_when_attempt_cannot_be_reloaded(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    original_error = InvalidPromotionSDFError("the SDF source could not be parsed")
    _patch_promote_failure(monkeypatch, original_error, attempt_id="attempt-fallback-001")

    def _raise_on_reload(*args, **kwargs):
        raise FileNotFoundError("attempt metadata vanished")

    monkeypatch.setattr(cli_module, "load_promotion_attempt", _raise_on_reload)

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "Attempt ID: attempt-fallback-001" in result.output
    assert "Input: curcumin-pubchem" in result.output
    assert "Status: execution_failed" in result.output
    assert "the SDF source could not be parsed" in result.output
    assert "(additional attempt details could not be loaded)" in result.output
    assert "Traceback" not in result.output


# --- item 119: KeyboardInterrupt/SystemExit propagate, never absorbed ---


def test_ligand_promote_propagates_keyboard_interrupt(tmp_path, monkeypatch):
    context = SimpleNamespace(project_dir=tmp_path)
    interruption = KeyboardInterrupt()

    monkeypatch.setattr(cli_module, "_resolve_experiment_or_exit", lambda identifier, **kwargs: context)

    def _interrupt(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(cli_module, "promote_ligand_input", _interrupt)

    with pytest.raises(KeyboardInterrupt) as caught:
        cli_module.ligand_promote(
            "exp001",
            "curcumin-pubchem",
            entity="curcumin",
            state="curcumin-initial",
            name="Curcumin",
            revision=None,
        )

    assert caught.value is interruption


def test_ligand_promote_propagates_system_exit(tmp_path, monkeypatch):
    context = SimpleNamespace(project_dir=tmp_path)
    interruption = SystemExit(17)

    monkeypatch.setattr(cli_module, "_resolve_experiment_or_exit", lambda identifier, **kwargs: context)

    def _interrupt(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(cli_module, "promote_ligand_input", _interrupt)

    with pytest.raises(SystemExit) as caught:
        cli_module.ligand_promote(
            "exp001",
            "curcumin-pubchem",
            entity="curcumin",
            state="curcumin-initial",
            name="Curcumin",
            revision=None,
        )

    assert caught.value is interruption


# --- item 120: Application Service called exactly once (success path too) ---


def test_ligand_promote_calls_application_service_exactly_once_on_success(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_promote_success(monkeypatch)

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 0
    assert observed["calls"] == 1


# --- item 121: CLI never imports RDKit directly (static source check) ---


def test_cli_module_never_imports_rdkit_directly():
    source = inspect.getsource(cli_module)
    assert "import rdkit" not in source
    assert "from rdkit" not in source


# --- item 122: CLI never discovers attempts by listing/diffing ---


def test_cli_module_never_lists_or_diffs_promotions():
    source = inspect.getsource(cli_module)
    assert "list_promotions_by_input" not in source
    assert "list_promotions_by_state" not in source


def test_cli_module_never_acquires_or_releases_reservations_or_compensates():
    source = inspect.getsource(cli_module)
    for forbidden in (
        "acquire_semantic_reservation",
        "acquire_entity_claim",
        "acquire_state_claim",
        "release_semantic_reservation",
        "release_entity_claim",
        "release_state_claim",
        "compensate_partial_acquisition",
        "register_promotion",
        "compute_semantic_key",
        "build_semantic_key_payload",
    ):
        assert forbidden not in source


def test_cli_module_never_catches_bare_base_exception():
    source = inspect.getsource(cli_module)
    assert "except BaseException" not in source


# --- item 123: --revision valid/invalid ---


def test_ligand_promote_invalid_revision_rejected_before_calling_service(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_promote_success(monkeypatch)

    result = runner.invoke(app, _promote_cli_args(revision="not a valid id!"))

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert observed["calls"] == 0


def test_ligand_promote_valid_revision_is_forwarded(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_promote_success(monkeypatch)

    result = runner.invoke(app, _promote_cli_args(revision="revision-b"))

    assert result.exit_code == 0
    assert observed["intentional_revision"] == "revision-b"


def test_ligand_promote_invalid_entity_id_rejected_before_calling_service(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    observed = _patch_promote_success(monkeypatch)

    result = runner.invoke(app, _promote_cli_args(entity="../escape"))

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert observed["calls"] == 0
    assert "Traceback" not in result.output


# --- item 124: human identity conflict ---


def test_ligand_promote_identity_conflict_is_human_readable_input_error(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_failure(
        monkeypatch,
        PromotionIdentityConflictError(
            "a promotion record already exists for this semantic_key with "
            "different entity_id/state_id than requested"
        ),
        attempt_id="attempt-conflict-001",
    )
    monkeypatch.setattr(
        cli_module,
        "load_promotion_attempt",
        lambda *a, **k: _fake_loaded_failed_attempt(
            message="a promotion record already exists for this semantic_key with "
            "different entity_id/state_id than requested"
        ),
    )

    result = runner.invoke(app, _promote_cli_args(entity="different-entity", state="different-state"))

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "different entity_id/state_id" in result.output
    assert "Traceback" not in result.output


# --- item 125: STORAGE ERROR distinctly covers raw-permission-denied/
# raw-sharing-violation and promotion-primitive-unavailable ---


def test_ligand_promote_storage_error_covers_raw_sharing_violation(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_failure(
        monkeypatch,
        RawArtifactSharingViolationError("the operating system denied opening the RAW artifact"),
        attempt_id="attempt-storage-sharing",
    )
    monkeypatch.setattr(
        cli_module,
        "load_promotion_attempt",
        lambda *a, **k: _fake_loaded_failed_attempt(message="the operating system denied opening the RAW artifact"),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output


def test_ligand_promote_storage_error_covers_promotion_primitive_unavailable(tmp_path, monkeypatch):
    _init_valid_experiment(tmp_path, monkeypatch)
    _patch_promote_failure(
        monkeypatch,
        PromotionPublicationUnavailableError("no exclusive publication primitive available"),
        attempt_id="attempt-storage-primitive",
    )
    monkeypatch.setattr(
        cli_module,
        "load_promotion_attempt",
        lambda *a, **k: _fake_loaded_failed_attempt(message="no exclusive publication primitive available"),
    )

    result = runner.invoke(app, _promote_cli_args())

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output


# --- sanitizer reuse/generalization: _sanitize_cli_error_text ---


def test_sanitize_cli_error_text_is_an_alias_of_the_tautomer_sanitizer():
    assert cli_module._sanitize_cli_error_text is not cli_module._sanitize_tautomer_error_text
    sample = r"failure at C:\Users\Jane\ligand.sdf"
    assert cli_module._sanitize_cli_error_text(sample) == cli_module._sanitize_tautomer_error_text(sample)


@pytest.mark.parametrize(
    "message,redacted_fragment",
    [
        (r"C:\Users\Jane\private\ligand.sdf", "Jane"),
        (r"C:\Users\Jane Doe\private data\ligand.sdf", "Jane Doe"),
        (r"\\server\share\patient\record.sdf", "patient"),
        (r"\\server\clinical data\Jane Doe\record.sdf", "clinical data"),
        ("/home/jane/private/ligand.sdf", "private"),
        ("/home/jane/private data/ligand sample.sdf", "private data"),
        ("'/home/jane/private data/ligand sample.sdf'", "private data"),
        ('"C:\\Users\\Jane Doe\\private data\\ligand.sdf"', "Jane Doe"),
        (repr(r"C:\Users\Jane Doe\private data\ligand.sdf"), "Jane Doe"),
        ("/home/jane/private data/ligand sample.sdf,", "private data"),
    ],
    ids=[
        "windows-drive",
        "windows-drive-spaces",
        "unc",
        "unc-spaces",
        "posix",
        "posix-spaces",
        "single-quoted",
        "double-quoted",
        "repr-escaped",
        "trailing-punctuation",
    ],
)
def test_sanitize_cli_error_text_redacts_every_absolute_path_shape(message, redacted_fragment):
    sanitized = cli_module._sanitize_cli_error_text(message)

    assert "[redacted-path]" in sanitized
    assert redacted_fragment not in sanitized


def test_sanitize_cli_error_text_redacts_multiple_paths_in_one_message():
    sanitized = cli_module._sanitize_cli_error_text("copied /home/jane/a.sdf to /home/jane/b.sdf")

    assert sanitized.count("[redacted-path]") == 2
    assert "jane" not in sanitized


@pytest.mark.parametrize(
    "message,expected",
    [
        ("https://example.org/path/ligand.sdf", "https://example.org/path/ligand.sdf"),
        (
            "https://example.org/path/ligand.sdf?download=1",
            "https://example.org/path/ligand.sdf?download=1",
        ),
        ("https://example.org/path/ligand.sdf#record", "https://example.org/path/ligand.sdf#record"),
        ("http://example.org/a/b/file.sdf", "http://example.org/a/b/file.sdf"),
        (
            "https://example.org/path/ligand.sdf and /home/jane/a.sdf",
            "https://example.org/path/ligand.sdf and [redacted-path]",
        ),
    ],
    ids=["https", "https-query", "https-fragment", "http", "url-and-path"],
)
def test_sanitize_cli_error_text_preserves_urls_exactly(message, expected):
    assert cli_module._sanitize_cli_error_text(message) == expected


@pytest.mark.parametrize(
    "chemical_text",
    [
        "C(=O)/N=C/C",
        "H+/OH- equilibrium",
        r"N/C=C\C",
        "SMILES C/C=C/C",
        "ratio H2O/EtOH",
        "pH+/OH- equilibrium",
    ],
)
def test_sanitize_cli_error_text_preserves_chemical_slash_expressions(chemical_text):
    sanitized = cli_module._sanitize_cli_error_text(chemical_text)

    assert sanitized == chemical_text
    assert "[redacted-path]" not in sanitized


def test_sanitize_cli_error_text_leaves_pathless_messages_intact():
    message = "the RAW artifact's registered checksum changed since resolution"
    assert cli_module._sanitize_cli_error_text(message) == message


def test_ligand_promote_never_weakens_tautomer_command_behavior(tmp_path, monkeypatch):
    """Increment 017's own tautomer command must keep calling its own
    named sanitizer, completely unaffected by the generic alias added for
    Increment 018."""

    _init_valid_experiment(tmp_path, monkeypatch)
    absolute_path = r"C:\Users\Jane\private\ligand.sdf"
    _patch_tautomer_failure(
        monkeypatch,
        InvalidSDFError(f"could not parse SDF at {absolute_path}"),
        attempt_id="attempt-tautomer-unaffected",
    )

    result = runner.invoke(app, _tautomer_cli_args())

    assert result.exit_code == 1
    assert "EXECUTION ERROR" in result.output
    assert "[redacted-path]" in result.output
    assert "Jane" not in result.output


# --- end-to-end real RDKit flow ---


def test_ligand_promote_end_to_end_real_rdkit_flow(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    project = _init_valid_experiment(tmp_path, monkeypatch)

    raw_file = tmp_path / "curcumin.sdf"
    mol = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(raw_file))
    writer.write(mol)
    writer.close()

    add_result = runner.invoke(
        app,
        [
            "ligand",
            "add",
            "exp001",
            "curcumin-pubchem",
            "--name",
            "Curcumin",
            "--acquisition-method",
            "uploaded",
            "--raw-file",
            str(raw_file),
            "--format",
            "sdf",
        ],
    )
    assert add_result.exit_code == 0

    first = runner.invoke(app, _promote_cli_args())
    assert first.exit_code == 0, first.output
    assert "LIGAND PROMOTION" in first.output
    assert "Entity: curcumin (created)" in first.output
    assert "Initial state: curcumin-initial (created)" in first.output
    assert "Engine: rdkit" in first.output
    assert str(project) not in first.output

    second = runner.invoke(app, _promote_cli_args())
    assert second.exit_code == 0, second.output
    assert "Result: existing promotion reused" in second.output


# --- conditions add/revise/show/list (Increment 019, Checkpoint 3) ---
#
# Covers the CLI-layer scenarios from the normative matrix
# (INCREMENT_019_FINAL_SPECIFICATION.md, Section 25.2): T-070-T-090. Every
# `# T-NNN` tag below names the scenario the tagged test actually
# exercises; a test proving real, useful behavior with no scenario of its
# own in this range is marked "additional coverage" instead, never given
# a T-NNN it does not actually prove.
#
# declare_conditions already owns its own best-effort started/success/
# error activity logging (Section 22.6) - several tests below exist
# specifically to prove the CLI never adds a second, duplicate logging
# path of its own (Section 11 of the governing Checkpoint 3 task).


def _corrupt_conditions_record(project: Path, conditions_id: str) -> Path:
    """Write a structurally-corrupt YAML file directly at the path a real
    conditions declaration would occupy, simulating an already-persisted
    record that fails to parse."""

    conditions_dir = project / "scientific" / "experimental_conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)
    corrupted_file = conditions_dir / f"{conditions_id}.yaml"
    corrupted_file.write_text("schema_version: '1.0'\nid: [unterminated\n", encoding="utf-8")
    return corrupted_file


def test_conditions_add_with_point_ph_succeeds(tmp_path, monkeypatch):  # T-070
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code == 0, result.output
    assert "SCIENTIFIC CONDITIONS DECLARED" in result.output
    assert "Conditions ID: ph-001" in result.output
    assert "pH: 7.4" in result.output
    assert "Purpose: simulation_input" in result.output
    assert "Declared by: Dr. Jane Doe" in result.output


def test_conditions_add_with_range_ph_succeeds(tmp_path, monkeypatch):  # T-071
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "conditions", "add", "exp001", "ph-range-001",
            "--ph-min", "6.0", "--ph-max", "8.0",
            "--declared-by", "Dr. Jane Doe",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "SCIENTIFIC CONDITIONS DECLARED" in result.output
    assert "pH: 6.0 - 8.0" in result.output


def test_conditions_add_rejects_both_ph_forms(tmp_path, monkeypatch):  # T-072
    project = _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "conditions", "add", "exp001", "ph-001",
            "--ph", "7.0", "--ph-min", "6.0", "--ph-max", "8.0",
            "--declared-by", "Dr. Jane Doe",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert not (project / "scientific").exists()


def test_conditions_add_rejects_no_ph_form(tmp_path, monkeypatch):  # T-073
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--declared-by", "Dr. Jane Doe"]
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conditions_add_rejects_incomplete_range(tmp_path, monkeypatch):  # additional coverage
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["conditions", "add", "exp001", "ph-001", "--ph-min", "6.0", "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conditions_add_rejects_ph_over_64_chars_without_attempting_decimal_conversion(
    tmp_path, monkeypatch
):  # T-074
    _init_valid_experiment(tmp_path, monkeypatch)

    decimal_calls = []

    def _tracking_decimal(value):
        decimal_calls.append(value)
        return decimal.Decimal(value)

    monkeypatch.setattr(cli_module, "Decimal", _tracking_decimal)

    long_value = "1" * 65  # a syntactically valid Decimal literal, deliberately too long

    result = runner.invoke(
        app,
        ["conditions", "add", "exp001", "ph-001", "--ph", long_value, "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "64 characters" in result.output
    assert decimal_calls == []


@pytest.mark.parametrize(
    "ph_value",
    [
        "1000000",  # decimal_whole_digits
        "1.1234567",  # decimal_max_places
        "1234567.123456",  # decimal_max_digits
    ],
)
def test_conditions_add_rejects_ph_violating_decimal_constraints(
    tmp_path, monkeypatch, ph_value
):  # T-075
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["conditions", "add", "exp001", "ph-001", "--ph", ph_value, "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code == 1, result.output
    assert "INPUT ERROR" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_conditions_add_rejects_non_decimal_ph_string(tmp_path, monkeypatch):  # additional coverage
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["conditions", "add", "exp001", "ph-001", "--ph", "not-a-number", "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


@pytest.mark.parametrize("ph_value", ["NaN", "Infinity", "-Infinity"])
def test_conditions_add_rejects_non_finite_ph(tmp_path, monkeypatch, ph_value):  # additional coverage
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["conditions", "add", "exp001", "ph-001", "--ph", ph_value, "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conditions_add_rejects_inverted_ph_range(tmp_path, monkeypatch):  # additional coverage
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "conditions", "add", "exp001", "ph-001",
            "--ph-min", "8.0", "--ph-max", "6.0",
            "--declared-by", "Dr. Jane Doe",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conditions_add_rejects_label_over_200_chars(tmp_path, monkeypatch):  # T-076
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "conditions", "add", "exp001", "ph-001",
            "--ph", "7.4", "--declared-by", "Dr. Jane Doe",
            "--label", "x" * 201,
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conditions_add_rejects_empty_purpose(tmp_path, monkeypatch):  # T-077
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "conditions", "add", "exp001", "ph-001",
            "--ph", "7.4", "--declared-by", "Dr. Jane Doe",
            "--purpose", "",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output


def test_conditions_add_colliding_with_corrupted_record_reports_record_invalid(
    tmp_path, monkeypatch
):  # T-078
    project = _init_valid_experiment(tmp_path, monkeypatch)
    corrupted_file = _corrupt_conditions_record(project, "corrupt-001")
    original_bytes = corrupted_file.read_bytes()

    result = runner.invoke(
        app,
        ["conditions", "add", "exp001", "corrupt-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code == 1
    assert "RECORD INVALID" in result.output
    assert "SCIENTIFIC CONDITIONS DECLARED" not in result.output
    assert "CONFLICT" not in result.output
    assert "reused" not in result.output.lower()
    assert corrupted_file.read_bytes() == original_bytes

    log_file = project / "logs" / "activity.log"
    assert log_file.exists()
    assert "outcome=success" not in log_file.read_text(encoding="utf-8")


def test_conditions_add_reports_storage_error_on_forced_oserror(tmp_path, monkeypatch):  # T-079
    project = _init_valid_experiment(tmp_path, monkeypatch)

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("molecular.conditions_registry.atomic_write_text", _raise_storage_error)

    result = runner.invoke(
        app,
        ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output
    assert "SCIENTIFIC CONDITIONS DECLARED" not in result.output
    assert not (project / "scientific" / "experimental_conditions" / "ph-001.yaml").exists()
    assert "outcome=success" not in (project / "logs" / "activity.log").read_text(encoding="utf-8")


def test_conditions_revise_without_supersedes_is_rejected_by_typer(tmp_path, monkeypatch):  # T-080
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        ["conditions", "revise", "exp001", "ph-002", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"],
    )

    assert result.exit_code != 0
    assert "SCIENTIFIC CONDITIONS DECLARED" not in result.output


def test_conditions_revise_with_existing_predecessor_succeeds(tmp_path, monkeypatch):  # T-081
    _init_valid_experiment(tmp_path, monkeypatch)
    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )

    result = runner.invoke(
        app,
        [
            "conditions", "revise", "exp001", "ph-002", "--supersedes", "ph-001",
            "--ph", "9.0", "--declared-by", "Dr. Jane Doe",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "SCIENTIFIC CONDITIONS DECLARED" in result.output
    assert "Supersedes: ph-001" in result.output


def test_conditions_revise_with_nonexistent_predecessor_reports_predecessor_not_found(
    tmp_path, monkeypatch
):  # T-082
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "conditions", "revise", "exp001", "ph-002", "--supersedes", "does-not-exist",
            "--ph", "9.0", "--declared-by", "Dr. Jane Doe",
        ],
    )

    assert result.exit_code == 1
    assert "PREDECESSOR NOT FOUND" in result.output


def test_conditions_revise_with_corrupted_predecessor_reports_record_invalid(
    tmp_path, monkeypatch
):  # T-083
    project = _init_valid_experiment(tmp_path, monkeypatch)
    _corrupt_conditions_record(project, "ph-original-001")

    result = runner.invoke(
        app,
        [
            "conditions", "revise", "exp001", "ph-002", "--supersedes", "ph-original-001",
            "--ph", "9.0", "--declared-by", "Dr. Jane Doe",
        ],
    )

    assert result.exit_code == 1
    assert "RECORD INVALID" in result.output


def test_conditions_revise_colliding_with_corrupted_record_reports_record_invalid(
    tmp_path, monkeypatch
):  # T-084
    project = _init_valid_experiment(tmp_path, monkeypatch)
    runner.invoke(
        app,
        ["conditions", "add", "exp001", "ph-original-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"],
    )
    corrupted_file = _corrupt_conditions_record(project, "corrupt-001")
    original_bytes = corrupted_file.read_bytes()

    result = runner.invoke(
        app,
        [
            "conditions", "revise", "exp001", "corrupt-001", "--supersedes", "ph-original-001",
            "--ph", "9.0", "--declared-by", "Dr. Jane Doe",
        ],
    )

    assert result.exit_code == 1
    assert "RECORD INVALID" in result.output
    assert corrupted_file.read_bytes() == original_bytes


def test_conditions_revise_reports_storage_error_on_forced_oserror(tmp_path, monkeypatch):  # additional coverage
    project = _init_valid_experiment(tmp_path, monkeypatch)
    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )

    def _raise_storage_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("molecular.conditions_registry.atomic_write_text", _raise_storage_error)

    result = runner.invoke(
        app,
        [
            "conditions", "revise", "exp001", "ph-002", "--supersedes", "ph-001",
            "--ph", "9.0", "--declared-by", "Dr. Jane Doe",
        ],
    )

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output


def test_conditions_show_nonexistent_reports_not_found(tmp_path, monkeypatch):  # T-085
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["conditions", "show", "exp001", "does-not-exist"])

    assert result.exit_code == 1
    assert "NOT FOUND" in result.output


def test_conditions_show_corrupted_reports_record_invalid(tmp_path, monkeypatch):  # T-086
    project = _init_valid_experiment(tmp_path, monkeypatch)
    _corrupt_conditions_record(project, "corrupt-001")

    result = runner.invoke(app, ["conditions", "show", "exp001", "corrupt-001"])

    assert result.exit_code == 1
    assert "RECORD INVALID" in result.output


def test_conditions_show_succeeds_and_shows_full_record(tmp_path, monkeypatch):  # additional coverage
    _init_valid_experiment(tmp_path, monkeypatch)
    runner.invoke(
        app,
        [
            "conditions", "add", "exp001", "ph-001",
            "--ph", "7.4", "--declared-by", "Dr. Jane Doe",
            "--label", "baseline", "--rationale", "standard assay condition",
        ],
    )

    result = runner.invoke(app, ["conditions", "show", "exp001", "ph-001"])

    assert result.exit_code == 0
    assert "Conditions ID: ph-001" in result.output
    assert "Label: baseline" in result.output
    assert "Rationale: standard assay condition" in result.output


def test_conditions_list_output_order_is_stable_across_repeated_invocations(
    tmp_path, monkeypatch
):  # T-087
    _init_valid_experiment(tmp_path, monkeypatch)
    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-first-001", "--ph", "7.0", "--declared-by", "Dr. A"]
    )
    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-second-001", "--ph", "7.5", "--declared-by", "Dr. B"]
    )

    first = runner.invoke(app, ["conditions", "list", "exp001"])
    second = runner.invoke(app, ["conditions", "list", "exp001"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.output == second.output
    assert "ph-first-001" in first.output
    assert "ph-second-001" in first.output


def test_conditions_list_on_fresh_experiment_is_empty_and_exits_zero(
    tmp_path, monkeypatch
):  # additional coverage
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["conditions", "list", "exp001"])

    assert result.exit_code == 0
    assert "CONDITIONS" in result.output
    assert "No experimental conditions declared" in result.output


def test_conditions_list_with_corrupted_record_reports_record_invalid(
    tmp_path, monkeypatch
):  # additional coverage
    project = _init_valid_experiment(tmp_path, monkeypatch)
    _corrupt_conditions_record(project, "corrupt-001")

    result = runner.invoke(app, ["conditions", "list", "exp001"])

    assert result.exit_code == 1
    assert "RECORD INVALID" in result.output


def test_conditions_commands_never_prompt_interactively(tmp_path, monkeypatch):  # T-088
    _init_valid_experiment(tmp_path, monkeypatch)

    invocations = [
        ["conditions", "add", "exp001", "ph-001"],
        ["conditions", "revise", "exp001", "ph-002", "--supersedes", "ph-001"],
        ["conditions", "show", "exp001", "does-not-exist"],
        ["conditions", "list", "exp001"],
    ]

    for args in invocations:
        result = runner.invoke(app, args, input="")
        assert not isinstance(result.exception, EOFError)


def test_conditions_app_has_no_forbidden_subcommands(tmp_path, monkeypatch):  # T-089
    _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(app, ["conditions", "--help"])

    assert result.exit_code == 0
    for forbidden in ("set-default", "infer", "recommend", "apply-all"):
        assert forbidden not in result.output


def test_conditions_no_unhandled_exception_escapes_any_command(tmp_path, monkeypatch):  # T-090
    """Cross-command symmetry check against Section 22.0's coverage table:
    every (command, error-type) cell that can realistically occur produces
    a clean banner and a normal exit, never a raw Python traceback."""

    project = _init_valid_experiment(tmp_path, monkeypatch)
    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )
    runner.invoke(
        app,
        [
            "conditions", "revise", "exp001", "ph-rev-001", "--supersedes", "ph-001",
            "--ph", "8.0", "--declared-by", "Dr. Jane Doe",
        ],
    )
    _corrupt_conditions_record(project, "corrupt-001")

    invocations = [
        # add: ConditionsValidationError (malformed id)
        ["conditions", "add", "exp001", "not valid id!", "--ph", "7.4", "--declared-by", "Dr. X"],
        # add: ConditionsDeclarationConflictError (divergent retry)
        ["conditions", "add", "exp001", "ph-001", "--ph", "5.0", "--declared-by", "Dr. Jane Doe"],
        # add: ConditionsRecordInvalidError (colliding record corrupted)
        ["conditions", "add", "exp001", "corrupt-001", "--ph", "7.4", "--declared-by", "Dr. X"],
        # revise: ConditionsValidationError (self-supersession)
        [
            "conditions", "revise", "exp001", "ph-001", "--supersedes", "ph-001",
            "--ph", "7.4", "--declared-by", "Dr. X",
        ],
        # revise: ConditionsDeclarationConflictError (divergent retry of a revision)
        [
            "conditions", "revise", "exp001", "ph-rev-001", "--supersedes", "ph-001",
            "--ph", "1.0", "--declared-by", "Dr. Jane Doe",
        ],
        # revise: ConditionsPredecessorNotFoundError
        [
            "conditions", "revise", "exp001", "ph-rev-002", "--supersedes", "does-not-exist",
            "--ph", "7.4", "--declared-by", "Dr. X",
        ],
        # revise: ConditionsRecordInvalidError (predecessor corrupted)
        [
            "conditions", "revise", "exp001", "ph-rev-003", "--supersedes", "corrupt-001",
            "--ph", "7.4", "--declared-by", "Dr. X",
        ],
        # show: FileNotFoundError
        ["conditions", "show", "exp001", "does-not-exist"],
        # show: ConditionsRecordInvalidError
        ["conditions", "show", "exp001", "corrupt-001"],
        # list: ConditionsRecordInvalidError
        ["conditions", "list", "exp001"],
    ]

    for args in invocations:
        result = runner.invoke(app, args)
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            f"unhandled exception for {args!r}: {result.exception!r}"
        )
        assert result.exit_code in (0, 1), f"unexpected exit code for {args!r}"


# --- conditions: CLI -> Service -> activity log boundary ---
#
# declare_conditions is the sole owner of activity logging for this
# feature (Section 22.6). The CLI never calls record_activity itself -
# these tests prove the boundary holds under real filesystem integration,
# not mocks.


def test_conditions_add_produces_exactly_started_then_success_log_entries(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )

    lines = (project / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "command=conditions-declare" in lines[0]
    assert "outcome=started" in lines[0]
    assert "command=conditions-declare" in lines[1]
    assert "outcome=success" in lines[1]
    assert "reused=False" in lines[1]


def test_conditions_add_retry_via_cli_logs_reused_true(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )
    result = runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )

    assert result.exit_code == 0
    assert "Result: existing declaration reused" in result.output

    lines = (project / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert "reused=False" in lines[1]
    assert "reused=True" in lines[3]


def test_conditions_add_fresh_and_retry_log_lines_are_distinguishable(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )
    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )

    lines = (project / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()

    def _strip_timestamp(line: str) -> str:
        return line.split(" ", 1)[1]

    assert _strip_timestamp(lines[1]) != _strip_timestamp(lines[3])


def test_conditions_cli_never_logs_under_its_own_command_name(tmp_path, monkeypatch):
    """Only command=conditions-declare entries (declare_conditions's own)
    may ever appear - never command=conditions-add/-revise/-show/-list,
    proving the CLI never calls record_activity itself."""

    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )
    runner.invoke(
        app,
        [
            "conditions", "revise", "exp001", "ph-002", "--supersedes", "ph-001",
            "--ph", "9.0", "--declared-by", "Dr. Jane Doe",
        ],
    )
    runner.invoke(app, ["conditions", "show", "exp001", "ph-001"])
    runner.invoke(app, ["conditions", "list", "exp001"])
    runner.invoke(
        app, ["conditions", "add", "exp001", "bad id!", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )

    log_text = (project / "logs" / "activity.log").read_text(encoding="utf-8")
    assert "command=conditions-add" not in log_text
    assert "command=conditions-revise" not in log_text
    assert "command=conditions-show" not in log_text
    assert "command=conditions-list" not in log_text
    for line in log_text.splitlines():
        assert "command=conditions-declare" in line


def test_conditions_add_conflict_does_not_log_false_success(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )
    lines_before = (project / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()

    result = runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "5.0", "--declared-by", "Dr. Jane Doe"]
    )

    assert result.exit_code == 1
    assert "CONFLICT" in result.output

    lines_after = (project / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()
    new_lines = lines_after[len(lines_before):]
    assert len(new_lines) == 2
    assert "outcome=started" in new_lines[0]
    assert "outcome=error" in new_lines[1]
    assert all("outcome=success" not in line for line in new_lines)


def test_conditions_add_invalid_input_does_not_log_false_success(tmp_path, monkeypatch):
    project = _init_valid_experiment(tmp_path, monkeypatch)

    result = runner.invoke(
        app,
        [
            "conditions", "add", "exp001", "ph-001",
            "--ph", "7.4", "--declared-by", "Dr. Jane Doe", "--purpose", "",
        ],
    )

    assert result.exit_code == 1
    assert "INPUT ERROR" in result.output
    assert "outcome=success" not in (project / "logs" / "activity.log").read_text(encoding="utf-8")


def test_conditions_cli_error_banner_matches_service_level_error(tmp_path, monkeypatch):
    """The banner the CLI prints is exactly the message the typed
    exception declare_conditions itself raised - not a generic/guessed
    banner invented at the CLI layer."""

    _init_valid_experiment(tmp_path, monkeypatch)

    def _raise_persistence_error(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr("molecular.conditions_registry.atomic_write_text", _raise_persistence_error)

    result = runner.invoke(
        app, ["conditions", "add", "exp001", "ph-001", "--ph", "7.4", "--declared-by", "Dr. Jane Doe"]
    )

    assert result.exit_code == 1
    assert "STORAGE ERROR" in result.output
    assert "failed to persist conditions declaration 'ph-001'" in result.output
