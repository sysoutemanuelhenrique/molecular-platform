import tempfile
from pathlib import Path

import yaml
from typer.testing import CliRunner

import molecular.application as application_module
import molecular.cli as cli_module
from molecular.cli import app


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


def test_manual_bridge_from_ligand_to_state_duplicates_the_artifact(tmp_path, monkeypatch):
    """Demonstrates the manual, ergonomics-only bridge from ligand
    acquisition to the scientific model (Decision A): copying the ligand's
    discovered artifact path into a MolecularState creates a second,
    independently stored copy - nothing links them structurally, and
    provenance show never claims otherwise."""

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
    assert "inputs/ligands/curcumin-lig/curcumin.sdf" in show_result.output

    runner.invoke(app, ["entity", "add", "exp001", "curcumin", "--name", "Curcumin"])

    registered_ligand_path = project / "inputs" / "ligands" / "curcumin-lig" / "curcumin.sdf"

    state_result = runner.invoke(
        app,
        [
            "state",
            "add",
            "exp001",
            "curcumin-neutral",
            "--entity",
            "curcumin",
            "--source-file",
            str(registered_ligand_path),
            "--format",
            "sdf",
        ],
    )
    assert state_result.exit_code == 0

    state_artifact = (
        project / "scientific" / "artifacts" / "states" / "curcumin-neutral" / "curcumin.sdf"
    )
    assert state_artifact.exists()
    assert state_artifact != registered_ligand_path
    assert state_artifact.read_bytes() == registered_ligand_path.read_bytes()

    provenance_result = runner.invoke(
        app, ["provenance", "show", "exp001", "--state", "curcumin-neutral"]
    )
    assert "No registered state-transformation provenance exists" in provenance_result.output