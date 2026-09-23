"""Execution attempt persistence.

Stores ExecutionAttempts under project_dir/scientific/execution_attempts,
kept separate from project_dir/scientific/{entities,states,conformers,
transformations}. An ExecutionAttempt has no artifact of its own -
registration and every subsequent transition are single atomic metadata
writes, the same shape as molecular.scientific_registry.register_entity.

begin_execution_attempt never creates, requires, or checks for a
StateTransformationRecord at any of operation_ids: an attempt starts
before the transformation(s) it aims to produce exist, by design. Nothing
in this module writes to project_dir/scientific/transformations.
Symmetrically, mark_succeeded never verifies that a StateTransformationRecord
now exists at any of operation_ids either - see its own docstring below,
and ExecutionAttempt's docstring in molecular.execution, for the
obligation this places on whatever future caller (Increment 016's
orchestrating Application Service) marks an attempt succeeded.

--- AMENDMENT TO D-015-11 (post-freeze) ---

Source: 08_INCREMENT_015_FINAL_SPEC.md, Section 12 ("Operation Reference")
and Decision D-015-11 ("operation_id é referência-alvo, garantida existir
só quando status=succeeded."). This amendment is recorded here, in the
code, because the source document lives outside this repository; anyone
applying it to that document should carry this text over verbatim.

Two problems were identified in the original wording, both real tensions
within the frozen spec itself, not implementation bugs:

1. CARDINALITY. Section 12 modeled the target as a single
   `operation_id: str`, presuming one ExecutionAttempt produces exactly
   one StateTransformationRecord. But TautomerEnumerationPolicy already
   allows a single enumeration to yield more than one resulting
   MolecularState (that is what `max_states` bounds), and
   StateTransformationRecord (molecular.transformation) is one-child-per-
   record - so one enumeration attempt may legitimately need to produce
   several StateTransformationRecords, not one. Splitting that single
   engine invocation across several independent ExecutionAttempts (one
   per resulting state) would misrepresent one actual execution event as
   several, duplicating the same resolved engine configuration across
   every one of them for no benefit.

   AMENDED: `operation_id` (singular) is renamed to `operation_ids` (a
   non-empty tuple of independently-validated, duplicate-free ids). One
   ExecutionAttempt still models exactly one engine invocation, one
   resolved configuration, and one success/failure outcome - but now
   names the complete set of StateTransformationRecord ids that single
   invocation intends to produce, which may be one or many.

2. GUARANTEE VS. CONTRACT. D-015-11 states the target becomes "guaranteed
   dereferenceable" once status=="succeeded" - read literally, an
   invariant of the persisted record itself. But this increment has no
   engine and, by its own scope decision, no orchestrating Application
   Service (see the "NO APPLICATION SERVICE" section of the frozen
   spec) - so nothing in this module can check, compute, or enforce that
   guarantee. Enforcing it here would require importing
   molecular.transformation_registry into this minimal state machine,
   coupling it to a producer this increment deliberately does not have.
   As implemented and tested, mark_succeeded permits status=="succeeded"
   with zero of the declared operation_ids resolving to anything.

   AMENDED: the guarantee is reclassified from a system-enforced
   invariant to a CALLER CONTRACT. Every id in operation_ids is *intended*
   to be dereferenceable once status=="succeeded", but ensuring that is
   the responsibility of whoever calls mark_succeeded - in Increment 016,
   the orchestrating Application Service, which MUST register every
   StateTransformationRecord named in operation_ids (e.g. via
   register_state_transformation) strictly BEFORE calling mark_succeeded,
   never after. This module neither verifies nor can verify that
   ordering. Until Increment 016's Application Service exists and is
   exercised, status=="succeeded" on a persisted ExecutionAttempt asserts
   only the platform's own belief that its intended operation completed,
   not a fact this repository has checked.

This amendment does not walk back the "no engine, no Application Service
in 015" scope decision - it makes explicit what that decision already
implied about what "succeeded" can and cannot mean here, and it does not
add any existence check, import molecular.transformation_registry, or
otherwise couple this module to a transformation producer. No filesystem
layout, YAML shape (beyond the operation_id -> operation_ids field
rename), or persisted schema_version changes as a result.

--- end amendment ---

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
    operation_ids: tuple[str, ...],
    requested: TautomerEnumerationPolicy,
) -> ExecutionAttempt:
    """Begin a new ExecutionAttempt in status 'started'.

    operation_ids names the complete set of StateTransformationRecord ids
    this one attempt intends to produce - one for a single-output
    operation, more than one for an operation whose engine invocation can
    fan out into several results (e.g. tautomer enumeration; see the
    D-015-11 amendment in this module's docstring). Each id is validated
    only for filesystem-safe format - none is ever checked against an
    existing StateTransformationRecord, and this function never creates
    one. This is intentional: the attempt exists before the
    transformation(s) it targets exist. Multiple attempts may share one
    or more operation_ids (independent retries); each attempt still gets
    its own generated id.
    """

    if operation_type not in _SUPPORTED_OPERATION_TYPES:
        raise ValueError(
            f"'{operation_type}' is not a supported operation_type "
            f"(supported: {', '.join(_SUPPORTED_OPERATION_TYPES)})"
        )

    if not operation_ids:
        raise ValueError(
            "operation_ids must name at least one intended StateTransformationRecord id"
        )

    for candidate in operation_ids:
        validate_input_id(candidate)

    if len(set(operation_ids)) != len(operation_ids):
        raise ValueError("operation_ids must not contain duplicate entries")

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
        operation_ids=tuple(operation_ids),
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
        operation_ids=current.operation_ids,
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
        operation_ids=current.operation_ids,
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
        operation_ids=current.operation_ids,
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

    IMPORTANT (see the D-015-11 amendment in this module's docstring):
    this function does not check, and cannot check without coupling this
    module to a StateTransformationRecord producer that does not exist in
    this increment, whether every id in operation_ids actually resolves
    to a real StateTransformationRecord. Calling mark_succeeded only
    records the platform's own belief that its intended operation
    completed. Ensuring that belief is true - registering every
    StateTransformationRecord named in operation_ids (e.g. via
    register_state_transformation) *before* ever calling mark_succeeded,
    never after - is a caller contract this increment places on whatever
    future caller orchestrates that ordering: in Increment 016, the
    Application Service that resolves an adapter, executes the engine,
    registers the resulting transformation(s), and only then marks the
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
        operation_ids=current.operation_ids,
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
