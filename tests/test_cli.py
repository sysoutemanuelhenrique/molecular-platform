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