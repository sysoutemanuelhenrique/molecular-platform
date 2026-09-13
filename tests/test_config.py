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


def test_load_experiment_rejects_invalid_yaml_data(tmp_path):
    experiment_file = tmp_path / "experiment.yaml"

    experiment_file.write_text(
        'schema_version: "1.0"\n'
        'unknown_field: "invalid"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_experiment(experiment_file)