"""Application Service for scientist-declared experimental conditions.

declare_conditions is the sole creation entry point for
ScientificConditionsDeclaration - used identically for a fresh "add" and
for a "revise" (a revise is exactly a call with supersedes_id set; there is
no separate revise_conditions function, per Section 19.1 of the frozen
specification, INCREMENT_019_FINAL_SPECIFICATION.md).

This module owns:
  - translation of construction-time validation failures (an invalid
    caller-supplied conditions_id/supersedes_id, or any pydantic
    ValidationError raised while constructing the candidate record) into
    ConditionsValidationError, with __cause__ always preserved;
  - predecessor resolution (ConditionsPredecessorNotFoundError when a named
    supersedes_id has no persisted record at all - distinct from, and
    never conflated with, ConditionsRecordInvalidError, which the registry
    layer already raises for a predecessor that exists but is corrupted);
  - retry-equivalence recognition versus genuine content conflict under an
    already-existing id (Section 16.1): ph, purpose, declared_by, label,
    rationale, and supersedes_id are compared exactly; declared_at never
    participates;
  - translation of an unexpected filesystem failure during the write
    itself into ConditionsPersistenceError, with __cause__ preserved -
    distinct from, and never conflated with, an identity conflict (caught
    separately, and first, since ConditionsIdentityConflictError is itself
    a FileExistsError/OSError subtype);
  - best-effort activity logging (started/success/error), via the
    project's existing record_activity primitive - a logging failure can
    never change this function's own outcome or returned value.

declare_conditions never catches KeyboardInterrupt or SystemExit: every
except clause below names a specific Exception subtype, never a bare
`except:` or `except Exception:` at this function's own top level.

Domain (conditions.py) -> Registry (conditions_registry.py) -> this module
-> CLI (a later checkpoint). No chemistry-engine or RDKit import here, and
no reference to ExecutionAttempt, TransformationParameter, or any other
pre-existing scientific record: this module resolves only
ScientificConditionsDeclaration, resolved_conditions/EngineParameters/
ExecutionResult/Evidence remain untouched and unmodeled (Section 7.2 of
the frozen spec).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import ValidationError

from molecular.activity import record_activity
from molecular.conditions import PhDeclaration, ScientificConditionsDeclaration
from molecular.conditions_registry import (
    ConditionsIdentityConflictError,
    ConditionsRecordInvalidError,
    list_conditions_declarations,
    load_conditions_declaration,
    register_conditions_declaration,
)
from molecular.inputs import validate_input_id

_OPERATION = "conditions-declare"

_COMPARED_FIELDS: Tuple[str, ...] = (
    "ph",
    "purpose",
    "declared_by",
    "label",
    "rationale",
    "supersedes_id",
)


class ConditionsValidationError(ValueError):
    """Any input-shape problem: an invalid conditions_id/supersedes_id
    format, or a pydantic.ValidationError raised while constructing the
    candidate record (an invalid ph - including any of the three
    independent decimal_* constraint violations - an overlong label, a
    blank rationale, an invalid purpose, etc.). The original ValueError or
    ValidationError is always preserved as __cause__.
    """


class ConditionsDeclarationConflictError(RuntimeError):
    """A create/revise request under an already-existing conditions_id
    diverges from the persisted record on at least one of the compared
    fields (ph, purpose, declared_by, label, rationale, supersedes_id).
    Never raised when the content is equivalent - that case is a
    recognized, safe retry instead, returning the existing record
    unchanged.
    """


class ConditionsPredecessorNotFoundError(RuntimeError):
    """supersedes_id names an id with no persisted declaration at all.
    Distinct from ConditionsRecordInvalidError, which signals a
    predecessor that exists but is unreadable/invalid - absence and
    corruption are never conflated.
    """


class ConditionsPersistenceError(OSError):
    """An unexpected filesystem failure during the write itself, distinct
    from an identity conflict (caught separately, and first, since
    ConditionsIdentityConflictError is itself an OSError subtype). The
    original OSError is always preserved as __cause__.
    """


@dataclass(frozen=True)
class ConditionsActivityDetails:
    """Typed activity-log context for declare_conditions, mirroring
    DerivedStateActivityDetails/TautomerEnumerationActivityDetails in
    application.py: every field is a specific, known correlation value,
    never arbitrary metadata.
    """

    conditions_id: str
    supersedes_id: Optional[str]
    declared_by: str


def _record_activity_safely(
    project_dir: Path, *, outcome: str, details: ConditionsActivityDetails
) -> None:
    """Best-effort activity logging - a failure to write this line can
    never affect declare_conditions's own outcome or returned value.
    Catches Exception, not BaseException: a KeyboardInterrupt/SystemExit
    reaching this helper must still propagate, never be swallowed as if it
    were an ordinary logging failure.
    """

    try:
        record_activity(project_dir, command=_OPERATION, outcome=outcome, details=details)
    except Exception:
        pass


def _diverging_fields(
    existing: ScientificConditionsDeclaration, candidate: ScientificConditionsDeclaration
) -> Tuple[str, ...]:
    return tuple(
        field
        for field in _COMPARED_FIELDS
        if getattr(existing, field) != getattr(candidate, field)
    )


def declare_conditions(
    project_dir: Path,
    *,
    conditions_id: str,
    ph: PhDeclaration,
    purpose: Literal["simulation_input"],
    declared_by: str,
    label: Optional[str] = None,
    rationale: Optional[str] = None,
    supersedes_id: Optional[str] = None,
) -> ScientificConditionsDeclaration:
    """Declare a new ScientificConditionsDeclaration, or recognize a safe
    retry of an already-declared one. A "revise" is exactly a call with
    supersedes_id set - there is no separate revise_conditions function.

    declared_at is generated here, timezone-aware in UTC, before the
    identity-conflict check; discarded (never persisted) if the request
    turns out to be a recognized retry of an already-published record.
    """

    # Step 1: validate ids before any I/O, including before the
    # "started" activity entry - a malformed id never produces any
    # observable side effect at all.
    try:
        validate_input_id(conditions_id)
        if supersedes_id is not None:
            validate_input_id(supersedes_id)
            if supersedes_id == conditions_id:
                raise ValueError("a declaration must not supersede itself")
    except ValueError as error:
        raise ConditionsValidationError(str(error)) from error

    details = ConditionsActivityDetails(
        conditions_id=conditions_id,
        supersedes_id=supersedes_id,
        declared_by=declared_by,
    )

    # Step 2: best-effort log "started".
    _record_activity_safely(project_dir, outcome="started", details=details)

    # Step 3: if supersedes_id is given, the predecessor must already be
    # persisted and readable.
    if supersedes_id is not None:
        try:
            load_conditions_declaration(project_dir, supersedes_id)
        except FileNotFoundError as error:
            _record_activity_safely(project_dir, outcome="error", details=details)
            raise ConditionsPredecessorNotFoundError(
                f"supersedes_id '{supersedes_id}' names no persisted declaration"
            ) from error
        except ConditionsRecordInvalidError:
            _record_activity_safely(project_dir, outcome="error", details=details)
            raise

    # Step 4: construct the candidate with a freshly-generated,
    # timezone-aware UTC declared_at.
    declared_at = datetime.now(timezone.utc)
    try:
        candidate = ScientificConditionsDeclaration(
            schema_version="1.0",
            id=conditions_id,
            ph=ph,
            purpose=purpose,
            declared_by=declared_by,
            label=label,
            rationale=rationale,
            supersedes_id=supersedes_id,
            declared_at=declared_at,
        )
    except ValidationError as error:
        _record_activity_safely(project_dir, outcome="error", details=details)
        raise ConditionsValidationError(str(error)) from error

    # Step 5/6/7: attempt the registry's create-once write. On an identity
    # conflict, recognize a safe retry or raise a genuine content
    # conflict. On an unexpected OSError during the write itself (not an
    # identity conflict - caught separately, and first, since
    # ConditionsIdentityConflictError is itself an OSError subtype),
    # translate into ConditionsPersistenceError.
    try:
        register_conditions_declaration(project_dir, candidate)
    except ConditionsIdentityConflictError:
        try:
            existing = load_conditions_declaration(project_dir, conditions_id)
        except ConditionsRecordInvalidError:
            _record_activity_safely(project_dir, outcome="error", details=details)
            raise

        diverging = _diverging_fields(existing, candidate)
        if not diverging:
            _record_activity_safely(project_dir, outcome="success", details=details)
            return existing

        _record_activity_safely(project_dir, outcome="error", details=details)
        raise ConditionsDeclarationConflictError(
            f"conditions_id '{conditions_id}' already exists with divergent "
            f"{', '.join(diverging)}"
        )
    except OSError as error:
        _record_activity_safely(project_dir, outcome="error", details=details)
        raise ConditionsPersistenceError(
            f"failed to persist conditions declaration '{conditions_id}'"
        ) from error

    # Step 8: fresh successful write.
    _record_activity_safely(project_dir, outcome="success", details=details)
    return candidate


def load_conditions(project_dir: Path, conditions_id: str) -> ScientificConditionsDeclaration:
    """Thin wrapper over the registry's own load. Propagates
    FileNotFoundError/ConditionsRecordInvalidError unchanged. No activity
    logging - a read performs no state-changing operation to log.
    """

    return load_conditions_declaration(project_dir, conditions_id)


def list_conditions(project_dir: Path) -> Tuple[ScientificConditionsDeclaration, ...]:
    """Thin wrapper over the registry's own listing. No activity logging."""

    return list_conditions_declarations(project_dir)
