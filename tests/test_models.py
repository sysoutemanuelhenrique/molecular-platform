from datetime import datetime, timezone

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


def test_experiment_config_accepts_schema_version_1_1():
    config = ExperimentConfig(schema_version="1.1", experiment_id="exp001")

    assert config.schema_version == "1.1"


def test_experiment_config_defaults_metadata_fields_to_none():
    config = ExperimentConfig(schema_version="1.0", experiment_id="exp001")

    assert config.title is None
    assert config.responsible is None
    assert config.summary is None
    assert config.created_at is None


def test_experiment_config_accepts_full_metadata():
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    config = ExperimentConfig(
        schema_version="1.1",
        experiment_id="exp001",
        title="Curcumin x Jack bean urease",
        responsible="Emanuel Henrique",
        summary="Docking study.",
        created_at=created_at,
    )

    assert config.title == "Curcumin x Jack bean urease"
    assert config.responsible == "Emanuel Henrique"
    assert config.summary == "Docking study."
    assert config.created_at == created_at


@pytest.mark.parametrize("field", ["title", "responsible", "summary"])
def test_experiment_config_rejects_empty_metadata_string(field):
    with pytest.raises(ValidationError):
        ExperimentConfig(
            schema_version="1.1",
            experiment_id="exp001",
            **{field: ""},
        )


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