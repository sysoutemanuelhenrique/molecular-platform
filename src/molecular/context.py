"""Experiment identity resolution.

An experiment's logical identity (`experiment_id`, declared inside its
experiment.yaml) is independent of where it happens to live on disk (its
directory). The filesystem is the source of truth for both facts, but only
one of them is ever used to *find* an experiment: its id. A directory name
is a location hint at most - it never satisfies a lookup by itself.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import yaml
from pydantic import ValidationError

from molecular.config import load_experiment as _default_load_experiment
from molecular.models import ExperimentConfig

Loader = Callable[[Path], ExperimentConfig]


@dataclass(frozen=True)
class ExperimentContext:
    """The resolved location of one experiment, alongside its validated config.

    `experiment_id` is never stored here directly - `config.experiment_id`
    is the single authority for identity. It is exposed as a read-only
    property so the two can never diverge.
    """

    project_dir: Path
    config: ExperimentConfig

    @property
    def experiment_id(self) -> str:
        return self.config.experiment_id


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


class InvalidExperimentConfigError(Exception):
    """Raised when a candidate's experiment.yaml parses and declares the
    requested experiment_id, but the resulting ExperimentConfig fails
    schema validation.

    The identity here is confiably known - read directly from the parsed
    YAML's own experiment_id field, never guessed from the directory name -
    even though the experiment cannot be loaded as valid. This is distinct
    from a candidate with no recoverable identity at all (malformed YAML,
    a non-string experiment_id, or a filesystem error), which is simply not
    resolvable and surfaces as ExperimentNotFoundError instead.
    """

    def __init__(self, identifier: str, project_dir: Path, validation_error: ValidationError):
        self.identifier = identifier
        self.project_dir = project_dir
        self.validation_error = validation_error
        super().__init__(f"Experiment '{identifier}' is invalid.")


def _declared_experiment_id(experiment_file: Path) -> Optional[str]:
    """Best-effort read of the experiment_id declared in parseable YAML,
    without validating anything else about the document.

    Returns None whenever there is no confiably recoverable identity: the
    file can't be parsed, the parsed document isn't a mapping, or
    experiment_id isn't present as a non-empty string.
    """

    try:
        with experiment_file.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except (OSError, yaml.YAMLError):
        return None

    if not isinstance(raw, dict):
        return None

    value = raw.get("experiment_id")
    return value if isinstance(value, str) and value else None


def _scan_candidates(
    base_dir: Path, loader: Optional[Loader]
) -> Tuple[List[ExperimentContext], List[Tuple[Path, str, ValidationError]], List[str]]:
    """Scan immediate subdirectories of base_dir, classifying every candidate
    that has an experiment.yaml into exactly one of three buckets:

    - valid: loaded and validated successfully.
    - invalid_with_id: YAML parses and declares a usable experiment_id, but
      ExperimentConfig validation fails - identity is confiably known even
      though the config itself is broken.
    - skipped (by name): everything else - malformed YAML, a filesystem
      error, or a schema failure with no usable experiment_id declared. No
      confiable identity exists for these, so they can never be resolved by
      id, no matter what their directory happens to be named.

    A single broken candidate never raises out of this function and takes
    down the whole scan. Raises OSError only if base_dir itself cannot be
    enumerated at all.
    """

    if loader is None:
        loader = _default_load_experiment

    valid: List[ExperimentContext] = []
    invalid_with_id: List[Tuple[Path, str, ValidationError]] = []
    skipped: List[str] = []

    for entry in sorted(Path(base_dir).iterdir()):
        try:
            if not entry.is_dir():
                continue

            experiment_file = entry / "experiment.yaml"
            if not experiment_file.exists():
                continue
        except OSError:
            skipped.append(entry.name)
            continue

        try:
            config = loader(experiment_file)
        except ValidationError as error:
            declared_id = _declared_experiment_id(experiment_file)
            if declared_id is not None:
                invalid_with_id.append((entry, declared_id, error))
            else:
                skipped.append(entry.name)
            continue
        except (OSError, yaml.YAMLError):
            skipped.append(entry.name)
            continue

        valid.append(ExperimentContext(project_dir=entry, config=config))

    return valid, invalid_with_id, skipped


def scan_experiments(
    base_dir: Path = Path("."), loader: Optional[Loader] = None
) -> tuple[list[ExperimentContext], list[str]]:
    """Scan immediate subdirectories of base_dir for experiments.

    Returns (valid, skipped_names). A subdirectory without experiment.yaml is
    not an experiment and is silently ignored. Any candidate that isn't
    fully valid - malformed/invalid YAML, a schema violation (whether or not
    its experiment_id could be recovered), or a filesystem error - is
    reported as skipped, uniformly, regardless of which of those reasons
    applies. A single broken candidate must never raise out of this
    function and take down the whole scan.

    This performs reads only: it never writes, creates, or mutates anything
    on disk.

    Raises OSError only if base_dir itself cannot be enumerated at all.
    """

    valid, invalid_with_id, skipped = _scan_candidates(base_dir, loader)
    all_skipped = skipped + [entry.name for entry, _declared_id, _error in invalid_with_id]
    return valid, all_skipped


def resolve_experiment(
    identifier: str, base_dir: Path = Path("."), loader: Optional[Loader] = None
) -> ExperimentContext:
    """Resolve an experiment by its logical experiment_id (exact match only).

    Directory names are never treated as identity - only config.experiment_id
    (or, for an invalid candidate, its declared experiment_id field) counts.
    A directory that merely shares a name with the identifier cannot satisfy
    the lookup on its own, and comparison is always exact (case sensitive),
    independent of the underlying filesystem's case sensitivity.

    Ambiguity is checked across every candidate with a confiably known
    identity - valid ones and invalid-but-known ones alike - since a config
    being invalid doesn't make its declared identity any less real. Four
    outcomes are possible:
    - more than one candidate (valid, invalid-with-id, or a mix) declares
      the id: AmbiguousExperimentError, naming every conflicting directory.
    - exactly one candidate declares it: returned as an ExperimentContext if
      valid, or raised as InvalidExperimentConfigError if not.
    - otherwise: ExperimentNotFoundError - either genuinely absent, or only
      present in candidates with no recoverable identity at all (malformed
      YAML, filesystem errors, or a non-string experiment_id).
    """

    valid, invalid_with_id, _skipped = _scan_candidates(base_dir, loader)

    valid_matches = [context for context in valid if context.experiment_id == identifier]
    invalid_matches = [
        (project_dir, error)
        for project_dir, declared_id, error in invalid_with_id
        if declared_id == identifier
    ]

    if len(valid_matches) + len(invalid_matches) > 1:
        directories = sorted(
            [context.project_dir.name for context in valid_matches]
            + [project_dir.name for project_dir, _error in invalid_matches]
        )
        raise AmbiguousExperimentError(identifier, directories)

    if len(valid_matches) == 1:
        return valid_matches[0]

    if len(invalid_matches) == 1:
        project_dir, error = invalid_matches[0]
        raise InvalidExperimentConfigError(identifier, project_dir, error)

    raise ExperimentNotFoundError(identifier)
