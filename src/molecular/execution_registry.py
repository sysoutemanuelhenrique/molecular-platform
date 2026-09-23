"""Execution attempt persistence.

Stores ExecutionAttempts under project_dir/scientific/execution_attempts,
kept separate from project_dir/scientific/{entities,states,conformers,
transformations}. An ExecutionAttempt has no artifact of its own -
registration and every subsequent transition are single atomic metadata
writes, the same shape as molecular.scientific_registry.register_entity.

begin_execution_attempt never creates, requires, or checks for a
StateTransformationRecord: an attempt starts before any transformation it
might eventually produce exists, by design, and before those results are
even knowable (see the amendment below). Nothing in this module writes to
project_dir/scientific/transformations. Symmetrically, mark_succeeded
never verifies that a StateTransformationRecord now exists at any id in
the operation_ids it is given either - see its own docstring below, and
ExecutionAttempt's docstring in molecular.execution, for the obligation
this places on whatever future caller (Increment 016's orchestrating
Application Service) marks an attempt succeeded.

--- AMENDMENT TO D-015-11 / D-015-12 (post-freeze) ---

Source: 08_INCREMENT_015_FINAL_SPEC.md, Section 12 ("Operation Reference")
and Decision D-015-11 ("operation_id é referência-alvo, garantida existir
só quando status=succeeded."). Recorded here because the source document
lives outside this repository; anyone applying it there should carry this
text over verbatim. This supersedes an earlier, incomplete revision of
D-015-11 that renamed the field to a tuple but still declared it at
begin_execution_attempt time.

Three problems were identified in the original wording, all real tensions
within the frozen spec itself, not implementation bugs:

1. CARDINALITY. Section 12 modeled the target as a single
   `operation_id: str`, presuming one ExecutionAttempt produces exactly
   one StateTransformationRecord. But TautomerEnumerationPolicy already
   allows a single enumeration to yield more than one resulting
   MolecularState (what `max_states` bounds), and StateTransformationRecord
   (molecular.transformation) is one-child-per-record - so one
   enumeration attempt may legitimately need to produce several
   StateTransformationRecords, not one.

2. TIMING. A first revision addressed (1) by making operation_ids a
   tuple declared at begin_execution_attempt time, as the attempt's
   stated *intent*. That still did not fit: which ids an engine
   invocation actually produces - indeed, how many - is not knowable
   until the engine has run and finished. Requiring the caller to
   pre-declare exact target ids before invoking the engine forces a
   commitment the real information timeline cannot support.

3. GUARANTEE VS. CONTRACT. D-015-11 states the target becomes "guaranteed
   dereferenceable" once status=="succeeded" - read literally, an
   invariant of the persisted record itself. But this increment has no
   engine and, by its own scope decision, no orchestrating Application
   Service (see the "NO APPLICATION SERVICE" section of the frozen
   spec) - so nothing in this module can check, compute, or enforce that
   guarantee without coupling this minimal state machine to
   molecular.transformation_registry, a producer this increment
   deliberately does not have.

AMENDED (D-015-12, superseding the tuple-at-begin revision of D-015-11):
operation_ids moves from begin-time declared intent to succeeded-time
declared result. begin_execution_attempt no longer accepts it at all - a
new attempt always starts with operation_ids=None, and it stays None
through every non-terminal-success status (started, resolved,
resolution_failed, execution_failed; ExecutionAttempt's own invariants
enforce this). Only mark_succeeded assigns it, as an explicit required
argument: a tuple, independently validated per entry and duplicate-free,
which MAY be empty (meaning the operation completed but produced no new
transformations). This matches the real timeline - the field exists only
once the information it carries actually exists - and, together with
D-015-11's original point 2, the guarantee it carries remains a CALLER
CONTRACT, not a system-enforced fact: every id mark_succeeded is given is
*intended* to be dereferenceable, but ensuring that is true - registering
every named StateTransformationRecord *before* calling mark_succeeded,
never after - is Increment 016's orchestrating Application Service's
responsibility, never verified here. A consequence of this timing fix:
retries are simply independent ExecutionAttempts, each with its own id
and its own None-until-succeeded operation_ids; there is no notion of two
attempts "sharing" a pre-declared target anymore, since no attempt
declares a target before it succeeds.

This amendment does not walk back the "no engine, no Application Service
in 015" scope decision - it makes explicit what that decision already
implied about when "succeeded" can mean anything at all, and it does not
add any existence check, import molecular.transformation_registry, or
otherwise couple this module to a transformation producer. No filesystem
layout or persisted schema_version changes as a result; the YAML shape's
operation_ids field simply reads null until an attempt succeeds.

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
    requested: TautomerEnumerationPolicy,
) -> ExecutionAttempt:
    """Begin a new ExecutionAttempt in status 'started'.

    Does not accept operation_ids: which StateTransformationRecords (if
    any) this attempt actually produces is only known once execution
    succeeds - see mark_succeeded and the D-015-11/D-015-12 amendment in
    this module's docstring. The new attempt always starts with
    operation_ids=None.
    """

    if operation_type not in _SUPPORTED_OPERATION_TYPES:
        raise ValueError(
            f"'{operation_type}' is not a supported operation_type "
            f"(supported: {', '.join(_SUPPORTED_OPERATION_TYPES)})"
        )

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
        operation_ids=None,
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
        operation_ids=None,
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
        operation_ids=None,
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
        operation_ids=None,
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


def mark_succeeded(
    project_dir: Path,
    attempt_id: str,
    *,
    operation_ids: tuple[str, ...],
) -> ExecutionAttempt:
    """Transition an attempt from 'resolved' to 'succeeded', preserving
    its resolved snapshot and resolved_at, and assigning operation_ids -
    the StateTransformationRecord ids this attempt actually produced.

    Only a 'resolved' attempt may succeed. operation_ids is required: it
    may be an empty tuple (the operation completed but produced no new
    transformations), but never omitted - this is the only point in the
    state machine where operation_ids is assigned, since which ids (if
    any) an engine invocation produces is only known once it has finished
    (see the D-015-12 amendment in this module's docstring). Each entry
    is independently validated for filesystem-safe format and duplicates
    are rejected, by ExecutionAttempt's own field validator.

    IMPORTANT: this function does not check, and cannot check without
    coupling this module to a StateTransformationRecord producer that
    does not exist in this increment, whether any id in operation_ids
    actually resolves to a real StateTransformationRecord. Calling
    mark_succeeded only records the platform's own belief that its
    intended operation completed with these results. Ensuring that
    belief is true - registering every StateTransformationRecord named in
    operation_ids (e.g. via register_state_transformation) *before* ever
    calling mark_succeeded, never after - is a caller contract this
    increment places on whatever future caller orchestrates that
    ordering: in Increment 016, the Application Service that resolves an
    adapter, executes the engine, determines the results, registers the
    resulting transformation(s), and only then marks the attempt
    succeeded with their ids.
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
        operation_ids=tuple(operation_ids),
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
