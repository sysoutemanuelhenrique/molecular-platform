from pathlib import Path

import yaml

from molecular.models import ExperimentConfig


def load_experiment(path: Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return ExperimentConfig.model_validate(data)