"""Execution attempt persistence.

Stores ExecutionAttempts under project_dir/scientific/execution_attempts,
kept separate from project_dir/scientific/{entities,states,conformers,
transformations}. An ExecutionAttempt has no artifact of its own -
registration and every subsequent transition are single atomic metadata
writes, the same shape as molecular.scientific_registry.register_entity.

begin_execution_attempt never creates, requires, or checks for a
StateTransformationRecord at operation_id: an attempt starts before the
transformation it aims to produce exists, by design. Nothing in this
module writes to project_dir/scientific/transformations. Symmetrically,
mark_succeeded never verifies that a StateTransformationRecord now exists
at operation_id either - see its own docstring below, and
ExecutionAttempt's docstring in molecular.execution, for the obligation
this places on whatever future caller (Increment 016's orchestrating
Application Service) marks an attempt succeeded.

Every transition (mark_resolved, mark_resolution_failed,
mark_execution_failed, mark_succeeded) follows the same protocol: load the
current persisted snapshot, validate that the requested transition is
reachable from its current status, construct a brand new, fully-validated
ExecutionAttempt reflecting the transition, and atomically overwrite the
previous snapshot with it. There is never a partial/patched update - the
whole record is always rewritten as one unit, exactly like every other
registry in this domain.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from molecular.atomic import atomic_write_text, check_storage_identity_conflict
from molecular.execution import (
    ExecutionAttempt,
    ExecutionAttemptError,
    ResolvedExecutionSnapshot,
    TautomerEnumerationPolicy,
)
from molecular.inputs import validate_input_id

_SUPPORTED_OPERATION_TYPES = ("state-transformation",)


class ExecutionAttemptIdentityConflictError(FileExistsError):
    """Raised if a freshly generated attempt id (a uuid4 hex) collides
    with an already-persisted attempt - astronomically unlikely given
    uuid4's randomness, but begin_execution_attempt must never silently
    overwrite a pre-existing attempt file the way an unguarded
    atomic_write_text call would (os.replace() replaces its destination
    unconditionally, with no existence check of its own). This is a
    create-only guard: it applies solely to begin_execution_attempt's own
    initial write. The four mark_* transitions are expected, intended
    overwrites of that same attempt's own file and are not - and must
    not be - guarded by this check."""


def _execution_attempts_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "execution_attempts"


def _persist(project_dir: Path, attempt: ExecutionAttempt) -> None:
    metadata_file = _execution_attempts_dir(project_dir) / f"{attempt.id}.yaml"
    serialized_yaml = yaml.safe_dump(attempt.model_dump(mode="json"), sort_keys=False)
    atomic_write_text(metadata_file, serialized_yaml)


def begin_execution_attempt(
    project_dir: Path,
    *,
    operation_type: str,
    operation_id: str,
    requested: TautomerEnumerationPolicy,
) -> ExecutionAttempt:
    """Begin a new ExecutionAttempt in status 'started'.

    operation_id is validated only for filesystem-safe format - it is
    never checked against an existing StateTransformationRecord, and this
    function never creates one. This is intentional: the attempt exists
    before the transformation it targets does. Multiple attempts may
    share the same operation_id (independent retries); each gets its own
    generated id.
    """

    if operation_type not in _SUPPORTED_OPERATION_TYPES:
        raise ValueError(
            f"'{operation_type}' is not a supported operation_type "
            f"(supported: {', '.join(_SUPPORTED_OPERATION_TYPES)})"
        )

    validate_input_id(operation_id)

    attempts_dir = _execution_attempts_dir(project_dir)
    attempt_id = uuid.uuid4().hex

    # Guard the create path only: refuse to silently clobber a
    # pre-existing attempt file on the astronomically unlikely event of a
    # uuid4 collision. atomic_write_text's underlying os.replace() would
    # otherwise overwrite an existing destination unconditionally, with
    # no existence check of its own.
    check_storage_identity_conflict(
        attempts_dir, attempt_id, conflict_error=ExecutionAttemptIdentityConflictError
    )

    attempt = ExecutionAttempt(
        schema_version="1.0",
        id=attempt_id,
        operation_type=operation_type,
        operation_id=operation_id,
        status="started",
        requested=requested,
        resolved=None,
        error=None,
        created_at=datetime.now(timezone.utc),
        resolved_at=None,
        finished_at=None,
    )

    _persist(project_dir, attempt)

    return attempt


def load_execution_attempt(project_dir: Path, attempt_id: str) -> ExecutionAttempt:
    """Load a previously persisted ExecutionAttempt by id.

    Raises FileNotFoundError if no such attempt is registered - mirroring
    load_state/load_entity/load_conformer in molecular.scientific_registry.
    """

    validate_input_id(attempt_id)

    metadata_file = _execution_attempts_dir(project_dir) / f"{attempt_id}.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return ExecutionAttempt.model_validate(data)


def mark_resolved(
    project_dir: Path,
    attempt_id: str,
    *,
    resolved: ResolvedExecutionSnapshot,
) -> ExecutionAttempt:
    """Transition an attempt from 'started' to 'resolved'.

    Only a 'started' attempt may resolve. Any other current status -
    including every terminal status - raises ValueError; a failed attempt
    is never reopened, only retried via a brand new ExecutionAttempt.
    """

    current = load_execution_attempt(project_dir, attempt_id)

    if current.status != "started":
        raise ValueError(
            f"Cannot mark attempt '{attempt_id}' as resolved from status "
            f"'{current.status}' - only a 'started' attempt may resolve."
        )

    updated = ExecutionAttempt(
        schema_version=current.schema_version,
        id=current.id,
        operation_type=current.operation_type,
        operation_id=current.operation_id,
        status="resolved",
        requested=current.requested,
        resolved=resolved,
        error=None,
        created_at=current.created_at,
        resolved_at=datetime.now(timezone.utc),
        finished_at=None,
    )

    _persist(project_dir, updated)

    return updated


def mark_resolution_failed(
    project_dir: Path,
    attempt_id: str,
    *,
    error: ExecutionAttemptError,
) -> ExecutionAttempt:
    """Transition an attempt from 'started' to 'resolution_failed'.

    Only a 'started' attempt may fail resolution. `error.phase` must be
    'resolution' - enforced by ExecutionAttempt's own model validator, not
    duplicated here.
    """

    current = load_execution_attempt(project_dir, attempt_id)

    if current.status != "started":
        raise ValueError(
            f"Cannot mark attempt '{attempt_id}' as resolution_failed from "
            f"status '{current.status}' - only a 'started' attempt may."
        )

    updated = ExecutionAttempt(
        schema_version=current.schema_version,
        id=current.id,
        operation_type=current.operation_type,
        operation_id=current.operation_id,
        status="resolution_failed",
        requested=current.requested,
        resolved=None,
        error=error,
        created_at=current.created_at,
        resolved_at=None,
        finished_at=datetime.now(timezone.utc),
    )

    _persist(project_dir, updated)

    return updated


def mark_execution_failed(
    project_dir: Path,
    attempt_id: str,
    *,
    error: ExecutionAttemptError,
) -> ExecutionAttempt:
    """Transition an attempt from 'resolved' to 'execution_failed',
    preserving its resolved snapshot and resolved_at.

    Only a 'resolved' attempt may fail execution. `error.phase` must be
    'execution' - enforced by ExecutionAttempt's own model validator.
    """

    current = load_execution_attempt(project_dir, attempt_id)

    if current.status != "resolved":
        raise ValueError(
            f"Cannot mark attempt '{attempt_id}' as execution_failed from "
            f"status '{current.status}' - only a 'resolved' attempt may."
        )

    updated = ExecutionAttempt(
        schema_version=current.schema_version,
        id=current.id,
        operation_type=current.operation_type,
        operation_id=current.operation_id,
        status="execution_failed",
        requested=current.requested,
        resolved=current.resolved,
        error=error,
        created_at=current.created_at,
        resolved_at=current.resolved_at,
        finished_at=datetime.now(timezone.utc),
    )

    _persist(project_dir, updated)

    return updated


def mark_succeeded(project_dir: Path, attempt_id: str) -> ExecutionAttempt:
    """Transition an attempt from 'resolved' to 'succeeded', preserving
    its resolved snapshot and resolved_at.

    Only a 'resolved' attempt may succeed.

    IMPORTANT: this function does not check, and cannot check without
    coupling this module to a StateTransformationRecord producer that
    does not exist in this increment, whether operation_id actually
    resolves to a real StateTransformationRecord. Calling mark_succeeded
    only records the platform's own belief that its intended operation
    completed. Ensuring that belief is true - registering the
    StateTransformationRecord at operation_id (e.g. via
    register_state_transformation) *before* ever calling mark_succeeded,
    never after - is an obligation this increment places on whatever
    future caller orchestrates that ordering: in Increment 016, the
    Application Service that resolves an adapter, executes the engine,
    registers the resulting transformation, and only then marks the
    attempt succeeded.
    """

    current = load_execution_attempt(project_dir, attempt_id)

    if current.status != "resolved":
        raise ValueError(
            f"Cannot mark attempt '{attempt_id}' as succeeded from status "
            f"'{current.status}' - only a 'resolved' attempt may."
        )

    updated = ExecutionAttempt(
        schema_version=current.schema_version,
        id=current.id,
        operation_type=current.operation_type,
        operation_id=current.operation_id,
        status="succeeded",
        requested=current.requested,
        resolved=current.resolved,
        error=None,
        created_at=current.created_at,
        resolved_at=current.resolved_at,
        finished_at=datetime.now(timezone.utc),
    )

    _persist(project_dir, updated)

    return updated
