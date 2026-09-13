import pytest
from pydantic import ValidationError

from molecular.models import ExperimentConfig


def test_experiment_config_accepts_valid_data():
    config = ExperimentConfig(
        schema_version="1.0",
        experiment_id="exp001",
    )

    assert config.schema_version == "1.0"
    assert config.experiment_id == "exp001"


def test_experiment_config_rejects_missing_experiment_id():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            schema_version="1.0",
        )


def test_experiment_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            schema_version="1.0",
            experiment_id="exp001",
            unknown_field="should-not-be-here",
        )
def test_experiment_config_rejects_unsupported_schema_version():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            schema_version="2.0",
            experiment_id="exp001",
        )


def test_experiment_config_rejects_empty_experiment_id():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            schema_version="1.0",
            experiment_id="",
        )