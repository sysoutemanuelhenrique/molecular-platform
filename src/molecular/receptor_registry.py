"""Persistence for ReceptorInspectionReport records (Increment 020,
Checkpoint 1 - the `receptor_inspections/` portion only; `receptor_selections/`
and `receptor_selections/resolved/` are Checkpoint 2's responsibility).

Stores inspection reports under `project_dir/scientific/receptor_inspections/
<inspection_id>.yaml` - a first-class, create-once registry (Section 24.2
of the frozen spec), not an informal cache: guarded by the same shared
`check_storage_identity_conflict` primitive every other registry in this
codebase uses. Because the filename IS the content's own deterministic
hash (Section 15), a collision here can only mean the identical payload
was already persisted - `inspect_receptor` (a later checkpoint) treats
that as a create-once hit, never as an error, outside the residual TOCTOU
window every registry in this repository already accepts.

This module owns translation of read-time I/O/parse/schema failures into
ReceptorInspectionRecordInvalidError, with `__cause__` always preserved.
"""

from pathlib import Path

import yaml
from pydantic import ValidationError

from molecular.atomic import atomic_write_text, check_storage_identity_conflict
from molecular.promotion import validate_semantic_key
from molecular.receptor_inspection import ReceptorInspectionReport


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
