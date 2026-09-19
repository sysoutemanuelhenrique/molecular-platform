"""Experiment identity resolution.

An experiment's logical identity (`experiment_id`, declared inside its
experiment.yaml) is independent of where it happens to live on disk (its
directory). The filesystem is the source of truth for both facts, but only
one of them is ever used to *find* an experiment: its id. A directory name
is a location hint at most - it never satisfies a lookup by itself.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml
from pydantic import ValidationError

from molecular.config import load_experiment as _default_load_experiment
from molecular.models import ExperimentConfig

Loader = Callable[[Path], ExperimentConfig]


@dataclass(frozen=True)
class ExperimentContext:
    """The resolved identity and location of one experiment.

    `experiment_id` is the logical identity; `project_dir` is where it
    currently lives. Nothing should assume `project_dir.name` equals
    `experiment_id` - resolving one from the other is exactly what this
    module exists to do.
    """

    experiment_id: str
    project_dir: Path
    config: ExperimentConfig


class ExperimentNotFoundError(Exception):
    """Raised when no experiment declares the requested experiment_id."""

    def __init__(self, identifier: str):
        self.identifier = identifier
        super().__init__(f"Experiment '{identifier}' was not found.")


class AmbiguousExperimentError(Exception):
    """Raised when more than one experiment declares the same experiment_id."""

    def __init__(self, identifier: str, directories: list[str]):
        self.identifier = identifier
        self.directories = directories
        super().__init__(
            f"Multiple experiments declare experiment_id '{identifier}': "
            f"{', '.join(directories)}. Resolve the conflict before querying by id."
        )


def scan_experiments(
    base_dir: Path = Path("."), loader: Optional[Loader] = None
) -> tuple[list[ExperimentContext], list[str]]:
    """Scan immediate subdirectories of base_dir for experiments.

    Returns (valid, skipped_names). A subdirectory without experiment.yaml is
    not an experiment and is silently ignored. A subdirectory whose candidate
    fails for any reason - malformed/invalid YAML, or a filesystem error such
    as experiment.yaml being a directory, a permission error, or any other
    OSError while inspecting or reading it - is reported as skipped. A single
    broken candidate must never raise out of this function and take down the
    whole scan.

    This performs reads only: it never writes, creates, or mutates anything
    on disk.

    Raises OSError only if base_dir itself cannot be enumerated at all.
    """

    if loader is None:
        loader = _default_load_experiment

    valid: list[ExperimentContext] = []
    skipped: list[str] = []

    for entry in sorted(Path(base_dir).iterdir()):
        try:
            if not entry.is_dir():
                continue

            experiment_file = entry / "experiment.yaml"
            if not experiment_file.exists():
                continue

            config = loader(experiment_file)

        except (OSError, yaml.YAMLError, ValidationError):
            skipped.append(entry.name)
            continue

        valid.append(
            ExperimentContext(experiment_id=config.experiment_id, project_dir=entry, config=config)
        )

    return valid, skipped


def resolve_experiment(
    identifier: str, base_dir: Path = Path("."), loader: Optional[Loader] = None
) -> ExperimentContext:
    """Resolve an experiment by its logical experiment_id (exact match only).

    Directory names are never treated as identity - only config.experiment_id
    counts. A directory that merely shares a name with the identifier cannot
    satisfy the lookup on its own, and comparison is always exact (case
    sensitive), independent of the underlying filesystem's case sensitivity.

    A candidate that cannot be read as a valid experiment (malformed YAML,
    schema violation, filesystem error) never contributes an id to match
    against, so it cannot be resolved this way even if its directory name
    equals the identifier - see scan_experiments.
    """

    valid, _skipped = scan_experiments(base_dir, loader=loader)
    matches = [context for context in valid if context.experiment_id == identifier]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise ExperimentNotFoundError(identifier)

    raise AmbiguousExperimentError(identifier, [context.project_dir.name for context in matches])
