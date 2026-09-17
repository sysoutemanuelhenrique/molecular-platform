from typer.testing import CliRunner

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