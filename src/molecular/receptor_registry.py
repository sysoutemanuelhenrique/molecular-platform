"""Persistence for receptor inspection/selection/resolution records
(Increment 020).

Checkpoint 1 built the `receptor_inspections/` portion only. Checkpoint 2
(this revision) completes the module with `receptor_selections/` and
`receptor_selections/resolved/` (Section 24 of the frozen spec) - the
Application Service that orchestrates `inspect_receptor`/
`declare_receptor_selection`/`resolve_receptor_selection` (invoking these
functions, enforcing `selection_capability`, deciding retry-vs-conflict,
and resolving identifiers against a loaded inspection report) is
`receptor_service.py`'s responsibility (Checkpoint 3), never this
module's - every function here is a pure create-once write or a pure
read, with no cross-referencing of another registry's records.

Stores inspection reports under `project_dir/scientific/receptor_inspections/
<inspection_id>.yaml` - a first-class, create-once registry (Section 24.2
of the frozen spec), not an informal cache: guarded by the same shared
`check_storage_identity_conflict` primitive every other registry in this
codebase uses. Because the filename IS the content's own deterministic
hash (Section 15), a collision here can only mean the identical payload
was already persisted - `inspect_receptor` (a later checkpoint) treats
that as a create-once hit, never as an error, outside the residual TOCTOU
window every registry in this repository already accepts.

Stores selection policies under `project_dir/scientific/receptor_selections/
<selection_id>.yaml` (scientist-chosen identity, exactly like
`validate_input_id`-governed ids elsewhere in this codebase) and resolved
selections under `project_dir/scientific/receptor_selections/resolved/
<selection_id>.yaml` - both create-once, both guarded by the same shared
identity-conflict primitive. Recognizing a safe retry versus a genuine
content conflict under a colliding `selection_id` is
`receptor_service.py`'s responsibility, not this registry's - this module
only ever raises on any collision, exact or casefold, and never decides by
itself that a collision is a "hit".

This module owns translation of read-time I/O/parse/schema failures into
the three `*RecordInvalidError` classes, with `__cause__` always
preserved.
"""

from pathlib import Path
from typing import Tuple

import yaml
from pydantic import ValidationError

from molecular.atomic import atomic_write_text, check_storage_identity_conflict
from molecular.inputs import validate_input_id
from molecular.promotion import validate_semantic_key
from molecular.receptor_inspection import ReceptorInspectionReport
from molecular.receptor_resolution import ResolvedReceptorSelection
from molecular.receptor_selection import ReceptorSelectionPolicy


class ReceptorInspectionIdentityConflictError(FileExistsError):
    """Raised when `inspection_id` collides - exactly or by casefold - with
    an already-persisted report under `project_dir/scientific/
    receptor_inspections`. A create-once guard only; there is no mark_*
    transition in this module."""


class ReceptorInspectionRecordInvalidError(RuntimeError):
    """Raised when a persisted report exists on disk but cannot be read
    back: its YAML fails to parse, or its content fails schema validation
    against `ReceptorInspectionReport`. The original `yaml.YAMLError` or
    `pydantic.ValidationError` is always preserved as `__cause__`."""


class ReceptorInspectionNotFoundError(FileNotFoundError):
    """Raised when no report is registered under the given `inspection_id`."""


def _receptor_inspections_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "receptor_inspections"


def _persist(project_dir: Path, report: ReceptorInspectionReport) -> None:
    metadata_file = _receptor_inspections_dir(project_dir) / f"{report.inspection_id}.yaml"
    serialized_yaml = yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False)
    atomic_write_text(metadata_file, serialized_yaml)


def register_receptor_inspection(
    project_dir: Path, report: ReceptorInspectionReport
) -> ReceptorInspectionReport:
    """Create-once persist of a freshly-built `ReceptorInspectionReport`.

    Raises `ReceptorInspectionIdentityConflictError` on a collision -
    exact or casefold - against an already-persisted report's
    `inspection_id`. Never overwrites an existing file: recognizing a
    create-once hit versus a genuine conflict under the same id is the
    caller's (a later checkpoint's `inspect_receptor`) responsibility, not
    this registry's - though, because `inspection_id` is itself a
    deterministic hash of the report's content (Section 15), a real
    content conflict under the same id is structurally impossible here.
    """

    check_storage_identity_conflict(
        _receptor_inspections_dir(project_dir),
        report.inspection_id,
        conflict_error=ReceptorInspectionIdentityConflictError,
    )
    _persist(project_dir, report)
    return report


def load_receptor_inspection(project_dir: Path, inspection_id: str) -> ReceptorInspectionReport:
    """Load a previously persisted report by its `inspection_id`.

    Raises `ReceptorInspectionNotFoundError` if no such report is
    registered. Raises `ReceptorInspectionRecordInvalidError`, with
    `__cause__` preserved, if the file exists but its YAML fails to parse
    or its content fails schema validation.
    """

    validate_semantic_key(inspection_id)
    metadata_file = _receptor_inspections_dir(project_dir) / f"{inspection_id}.yaml"

    if not metadata_file.is_file():
        raise ReceptorInspectionNotFoundError(
            f"no receptor inspection report registered under '{inspection_id}'"
        )

    try:
        data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        return ReceptorInspectionReport.model_validate(data)
    except (yaml.YAMLError, ValidationError) as error:
        raise ReceptorInspectionRecordInvalidError(
            f"receptor inspection report '{inspection_id}' exists but is not readable/valid"
        ) from error


# ===========================================================================
# receptor_selections/ (Checkpoint 2) - ReceptorSelectionPolicy
# ===========================================================================


class ReceptorSelectionIdentityConflictError(FileExistsError):
    """Raised when `selection_id` collides - exactly or by casefold - with
    an already-persisted policy under `project_dir/scientific/
    receptor_selections`. A create-once guard only; recognizing a safe
    retry versus a genuine content conflict is `receptor_service.py`'s
    responsibility (Checkpoint 3), not this registry's."""


class ReceptorSelectionRecordInvalidError(RuntimeError):
    """Raised when a persisted policy exists on disk but cannot be read
    back: its YAML fails to parse, or its content fails schema validation
    against `ReceptorSelectionPolicy`. The original `yaml.YAMLError` or
    `pydantic.ValidationError` is always preserved as `__cause__`."""


class ReceptorSelectionNotFoundError(FileNotFoundError):
    """Raised when no policy is registered under the given `selection_id`."""


def _receptor_selections_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "receptor_selections"


def _resolved_receptor_selections_dir(project_dir: Path) -> Path:
    return _receptor_selections_dir(project_dir) / "resolved"


def register_receptor_selection(
    project_dir: Path, policy: ReceptorSelectionPolicy
) -> ReceptorSelectionPolicy:
    """Create-once persist of a freshly-constructed
    `ReceptorSelectionPolicy`.

    Raises `ReceptorSelectionIdentityConflictError` on a collision - exact
    or casefold - against an already-persisted policy's `selection_id`.
    Never overwrites an existing file: recognizing a create-once hit
    versus a genuine conflict under the same id is the caller's (a later
    checkpoint's `declare_receptor_selection`) responsibility, not this
    registry's. Never reads or validates `policy.inspection_id` against
    any persisted inspection report - that cross-check belongs to the
    Service (Section 25.3), not this write.
    """

    selections_dir = _receptor_selections_dir(project_dir)
    check_storage_identity_conflict(
        selections_dir,
        policy.selection_id,
        conflict_error=ReceptorSelectionIdentityConflictError,
        reserved_names=frozenset({"resolved"}),
    )

    metadata_file = selections_dir / f"{policy.selection_id}.yaml"
    serialized_yaml = yaml.safe_dump(policy.model_dump(mode="json"), sort_keys=False)
    atomic_write_text(metadata_file, serialized_yaml)
    return policy


def load_receptor_selection(project_dir: Path, selection_id: str) -> ReceptorSelectionPolicy:
    """Load a previously persisted policy by its `selection_id`.

    Raises `ReceptorSelectionNotFoundError` if no such policy is
    registered. Raises `ReceptorSelectionRecordInvalidError`, with
    `__cause__` preserved, if the file exists but its YAML fails to parse
    or its content fails schema validation.
    """

    validate_input_id(selection_id)
    metadata_file = _receptor_selections_dir(project_dir) / f"{selection_id}.yaml"

    if not metadata_file.is_file():
        raise ReceptorSelectionNotFoundError(
            f"no receptor selection policy registered under '{selection_id}'"
        )

    try:
        data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        return ReceptorSelectionPolicy.model_validate(data)
    except (yaml.YAMLError, ValidationError) as error:
        raise ReceptorSelectionRecordInvalidError(
            f"receptor selection policy '{selection_id}' exists but is not readable/valid"
        ) from error


def list_receptor_selections(project_dir: Path) -> Tuple[ReceptorSelectionPolicy, ...]:
    """List every `ReceptorSelectionPolicy` registered under `project_dir`,
    in a stable, deterministic `(declared_at, selection_id)` order.

    Returns an empty tuple - never raises - when `project_dir/scientific/
    receptor_selections` does not yet exist. This ordering carries no
    semantic meaning of "current" or "latest": no policy is ever selected
    by this function, only enumerated. The `resolved/` subdirectory is
    never mistaken for a selection's own metadata file (it is a directory,
    never a `<id>.yaml` entry directly under `receptor_selections/`).
    """

    selections_dir = _receptor_selections_dir(project_dir)
    if not selections_dir.is_dir():
        return ()

    policies = [
        load_receptor_selection(project_dir, path.stem)
        for path in sorted(selections_dir.glob("*.yaml"))
    ]
    return tuple(sorted(policies, key=lambda policy: (policy.declared_at, policy.selection_id)))


# ===========================================================================
# receptor_selections/resolved/ (Checkpoint 2) - ResolvedReceptorSelection
# ===========================================================================


class ReceptorResolutionIdentityConflictError(FileExistsError):
    """Raised when `selection_id` collides - exactly or by casefold - with
    an already-persisted resolution under `project_dir/scientific/
    receptor_selections/resolved`. A create-once guard only; recognizing a
    safe retry versus a genuine conflict is `receptor_service.py`'s
    responsibility (Checkpoint 3), not this registry's."""


class ReceptorResolutionRecordInvalidError(RuntimeError):
    """Raised when a persisted resolution exists on disk but cannot be
    read back: its YAML fails to parse, or its content fails schema
    validation against `ResolvedReceptorSelection`. The original
    `yaml.YAMLError` or `pydantic.ValidationError` is always preserved as
    `__cause__`."""


class ReceptorResolutionNotFoundError(FileNotFoundError):
    """Raised when no resolution is registered under the given
    `selection_id`."""


def register_resolved_receptor_selection(
    project_dir: Path, resolution: ResolvedReceptorSelection
) -> ResolvedReceptorSelection:
    """Create-once persist of a freshly-constructed
    `ResolvedReceptorSelection`.

    Raises `ReceptorResolutionIdentityConflictError` on a collision -
    exact or casefold - against an already-persisted resolution's
    `resolved_selection_id`. Never overwrites an existing file. Never
    reads or writes anything under `receptor_inspections/` - multiple
    resolutions referencing the same `inspection_id` never duplicate, or
    even touch, that already-persisted inspection report (Section 14.3,
    rule 3; T-130).
    """

    resolved_dir = _resolved_receptor_selections_dir(project_dir)
    check_storage_identity_conflict(
        resolved_dir,
        resolution.resolved_selection_id,
        conflict_error=ReceptorResolutionIdentityConflictError,
    )

    metadata_file = resolved_dir / f"{resolution.resolved_selection_id}.yaml"
    serialized_yaml = yaml.safe_dump(resolution.model_dump(mode="json"), sort_keys=False)
    atomic_write_text(metadata_file, serialized_yaml)
    return resolution


def load_resolved_receptor_selection(
    project_dir: Path, selection_id: str
) -> ResolvedReceptorSelection:
    """Load a previously persisted resolution by its `selection_id`.

    Raises `ReceptorResolutionNotFoundError` if no such resolution is
    registered. Raises `ReceptorResolutionRecordInvalidError`, with
    `__cause__` preserved, if the file exists but its YAML fails to parse
    or its content fails schema validation.
    """

    validate_input_id(selection_id)
    metadata_file = _resolved_receptor_selections_dir(project_dir) / f"{selection_id}.yaml"

    if not metadata_file.is_file():
        raise ReceptorResolutionNotFoundError(
            f"no resolved receptor selection registered under '{selection_id}'"
        )

    try:
        data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        return ResolvedReceptorSelection.model_validate(data)
    except (yaml.YAMLError, ValidationError) as error:
        raise ReceptorResolutionRecordInvalidError(
            f"resolved receptor selection '{selection_id}' exists but is not readable/valid"
        ) from error
