import pytest
from pydantic import ValidationError

from molecular.config import load_experiment


def test_load_experiment_from_yaml(tmp_path):
    experiment_file = tmp_path / "experiment.yaml"

    experiment_file.write_text(
        'schema_version: "1.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    config = load_experiment(experiment_file)

    assert config.schema_version == "1.0"
    assert config.experiment_id == "exp001"


def test_load_experiment_legacy_1_0_without_metadata_fields(tmp_path):
    experiment_file = tmp_path / "experiment.yaml"

    experiment_file.write_text(
        'schema_version: "1.0"\n'
        'experiment_id: "exp001"\n',
        encoding="utf-8",
    )

    config = load_experiment(experiment_file)

    assert config.schema_version == "1.0"
    assert config.title is None
    assert config.responsible is None
    assert config.summary is None
    assert config.created_at is None


def test_load_experiment_1_1_with_full_metadata(tmp_path):
    experiment_file = tmp_path / "experiment.yaml"

    experiment_file.write_text(
        'schema_version: "1.1"\n'
        'experiment_id: "exp001"\n'
        'title: "Curcumin x Jack bean urease"\n'
        'responsible: "Emanuel Henrique"\n'
        'summary: "Docking study."\n'
        'created_at: "2026-01-01T00:00:00+00:00"\n',
        encoding="utf-8",
    )

    config = load_experiment(experiment_file)

    assert config.schema_version == "1.1"
    assert config.title == "Curcumin x Jack bean urease"
    assert config.responsible == "Emanuel Henrique"
    assert config.summary == "Docking study."
    assert config.created_at is not None


def test_load_experiment_rejects_invalid_yaml_data(tmp_path):
    experiment_file = tmp_path / "experiment.yaml"

    experiment_file.write_text(
        'schema_version: "1.0"\n'
        'unknown_field: "invalid"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_experiment(experiment_file)