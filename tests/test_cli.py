import tempfile
from pathlib import Path

import yaml
from typer.testing import CliRunner

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


def test_validate_rejects_invalid_experiment(tmp_path, monkeypatch):
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


def test_validate_records_activity_entry_on_failure(tmp_path, monkeypatch):
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

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert "command=validate" in lines[0]
    assert "outcome=error" in lines[0]

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


def test_validate_rejects_malformed_yaml(tmp_path, monkeypatch):
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
    assert "CONFIGURATION ERROR" in result.output
    assert "Experiment 'exp001' has malformed YAML." in result.output


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


def test_plan_rejects_invalid_experiment(tmp_path, monkeypatch):
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


def test_plan_records_activity_entry_on_failure(tmp_path, monkeypatch):
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

    log_file = project / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert "command=plan" in lines[0]
    assert "outcome=error" in lines[0]


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
    monkeypatch.setattr(cli_module, "fetch_pubchem_sdf_2d", fake_fetch)

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

    monkeypatch.setattr(cli_module, "fetch_pubchem_sdf_2d", _fail_if_called)

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

    monkeypatch.setattr(cli_module, "fetch_pubchem_sdf_2d", _raise_not_found)

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

    monkeypatch.setattr(cli_module, "fetch_pubchem_sdf_2d", _raise_acquisition_error)

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

    monkeypatch.setattr(cli_module, "fetch_pubchem_sdf_2d", _fail_if_called)

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
    monkeypatch.setattr(cli_module, "fetch_pubchem_sdf_2d", fake_fetch)

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


def test_plan_rejects_malformed_yaml(tmp_path, monkeypatch):
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
    assert "CONFIGURATION ERROR" in result.output
    assert "Experiment 'exp001' has malformed YAML." in result.output