"""Application Service for scientist-declared experimental conditions.

declare_conditions is the sole creation entry point for
ScientificConditionsDeclaration - used identically for a fresh "add" and
for a "revise" (a revise is exactly a call with supersedes_id set; there is
no separate revise_conditions function, per Section 19.1 of the frozen
specification, INCREMENT_019_FINAL_SPECIFICATION.md).

This module owns:
  - translation of construction-time validation failures (an invalid
    caller-supplied conditions_id/supersedes_id, or any pydantic
    ValidationError raised while constructing the candidate record,
    including the ph field itself - see declare_conditions_from_values
    below) into ConditionsValidationError, with __cause__ always
    preserved;
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
    never change this function's own outcome or returned value. A
    recognized retry's success entry carries reused=True (Section 19.1
    step 6's "reused" annotation); every other logged event - fresh
    creation's success entry included - carries reused=False, so the two
    kinds of success are never byte-identical in the activity log;
  - producing the factual created/reused outcome itself
    (ConditionsDeclarationOutcome) for any caller that needs it - never
    inferred by a caller from a timestamp, from the activity log, or from
    any other side channel. declare_conditions and
    declare_conditions_from_values are two thin, non-duplicating entry
    points over the same private core (_declare_conditions_core); neither
    re-implements persistence, retry comparison, predecessor validation,
    or activity logging a second time.

declare_conditions never catches KeyboardInterrupt or SystemExit: every
except clause below names a specific Exception subtype, never a bare
`except:` or `except Exception:` at this function's own top level. The
same holds for declare_conditions_from_values and the shared core, since
all three share the exact same except clauses.

Domain (conditions.py) -> Registry (conditions_registry.py) -> this module
-> CLI (Checkpoint 3). No chemistry-engine or RDKit import here, and no
reference to ExecutionAttempt, TransformationParameter, or any other
pre-existing scientific record: this module resolves only
ScientificConditionsDeclaration, resolved_conditions/EngineParameters/
ExecutionResult/Evidence remain untouched and unmodeled (Section 7.2 of
the frozen spec).
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

from pydantic import ValidationError

from molecular.activity import record_activity
from molecular.conditions import (
    PhDeclaration,
    PhPoint,
    PhRange,
    ScientificConditionsDeclaration,
)
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

    reused distinguishes, in the activity log itself, a recognized-retry
    success (Section 19.1 step 6's "reused" annotation, reused=True) from
    every other logged event, including a fresh creation's own success
    (reused=False) - the default, and the only value ever used outside
    step 6's reuse branch.
    """

    conditions_id: str
    supersedes_id: Optional[str]
    declared_by: str
    reused: bool = False


@dataclass(frozen=True)
class ConditionsDeclarationOutcome:
    """The factual result of one declare_conditions_from_values call -
    never inferred by any caller from a timestamp, from the activity log,
    or from any other side channel.

    declaration is the resulting ScientificConditionsDeclaration (the
    freshly-persisted candidate, or the recognized-retry's existing
    record - byte-for-byte the same object identity the shared core
    itself used to decide reused, never recomputed independently).

    reused is True exactly when this call recognized a safe, idempotent
    retry of an already-persisted record (Section 16.1) and returned the
    existing record unchanged; False for a fresh, newly-persisted
    creation. This is the same fact already carried by the activity log's
    own `reused` field (ConditionsActivityDetails) - this dataclass simply
    makes that same, single, already-computed fact available to a
    programmatic caller (the CLI) without requiring it to read the log or
    re-derive the fact any other way.
    """

    declaration: ScientificConditionsDeclaration
    reused: bool


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


def _declare_conditions_core(
    project_dir: Path,
    *,
    conditions_id: str,
    ph_builder: Callable[[], PhDeclaration],
    purpose: Literal["simulation_input"],
    declared_by: str,
    label: Optional[str],
    rationale: Optional[str],
    supersedes_id: Optional[str],
) -> ConditionsDeclarationOutcome:
    """The single, private implementation behind both declare_conditions
    and declare_conditions_from_values - every step, every exception type,
    and every activity-log call below is shared, never duplicated between
    the two public entry points.

    ph_builder is called exactly once, at step 4, in the same try/except
    ValidationError block that already wraps the candidate's own
    construction - building ph (whatever its source) is itself part of
    "construct the candidate" (Section 19.1 step 4), not a step of its
    own. declare_conditions passes a trivial `lambda: ph` (an
    already-built, already-valid PhDeclaration - calling it can never
    raise); declare_conditions_from_values passes a closure that builds
    PhPoint/PhRange/PhDeclaration from raw Decimal values, so that any
    decimal_max_digits/decimal_max_places/decimal_whole_digits violation,
    or an inverted range, or a point-and-range/neither-provided
    exclusivity violation, is caught here - after "started" is logged,
    before any publish attempt - exactly like every other construction-
    time validation failure, never as a silent, unlogged CLI-side
    rejection.

    declared_at is generated here, timezone-aware in UTC, before the
    identity-conflict check; discarded (never persisted) if the request
    turns out to be a recognized retry of an already-published record.

    Returns the factual outcome (declaration + reused) directly - this is
    the one and only place that fact is ever decided; no caller needs to,
    or should, re-derive it from a timestamp or from the activity log.
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

    # Step 4: build ph (whatever its source) and construct the candidate
    # with a freshly-generated, timezone-aware UTC declared_at. Any
    # pydantic.ValidationError raised by either - ph_builder() or the
    # ScientificConditionsDeclaration(...) construction itself - is
    # caught identically; the caller cannot tell, and does not need to
    # tell, which of the two actually raised it.
    declared_at = datetime.now(timezone.utc)
    try:
        ph = ph_builder()
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
            _record_activity_safely(
                project_dir, outcome="success", details=replace(details, reused=True)
            )
            return ConditionsDeclarationOutcome(declaration=existing, reused=True)

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
    return ConditionsDeclarationOutcome(declaration=candidate, reused=False)


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

    Unchanged public contract for every existing caller: ph must already
    be a valid, fully-constructed PhDeclaration, and this function
    returns the bare ScientificConditionsDeclaration, exactly as before
    this module gained declare_conditions_from_values. A thin wrapper
    over _declare_conditions_core - see that function for the full,
    single implementation this delegates to.
    """

    outcome = _declare_conditions_core(
        project_dir,
        conditions_id=conditions_id,
        ph_builder=lambda: ph,
        purpose=purpose,
        declared_by=declared_by,
        label=label,
        rationale=rationale,
        supersedes_id=supersedes_id,
    )
    return outcome.declaration


def _build_ph_declaration(
    ph_point: Optional[Decimal], ph_min: Optional[Decimal], ph_max: Optional[Decimal]
) -> PhDeclaration:
    """Build a PhDeclaration from raw Decimal components.

    Deliberately delegates the point-xor-range exclusivity decision to
    PhDeclaration's own existing model_validator (Section 13.4) rather
    than re-implementing that rule here: both forms are constructed
    whenever their own inputs are present, and PhDeclaration itself
    raises pydantic.ValidationError if neither or both end up populated -
    the exact same validation path (and the exact same exception type)
    this function's own caller (_declare_conditions_core, step 4) already
    catches uniformly. This is not a second exclusivity check; it is the
    one, pre-existing domain-level check, simply reached from a new
    direction.
    """

    point = PhPoint(value=ph_point) if ph_point is not None else None
    range_ = (
        PhRange(minimum=ph_min, maximum=ph_max)
        if ph_min is not None and ph_max is not None
        else None
    )
    return PhDeclaration(point=point, range=range_)


def declare_conditions_from_values(
    project_dir: Path,
    *,
    conditions_id: str,
    ph_point: Optional[Decimal] = None,
    ph_min: Optional[Decimal] = None,
    ph_max: Optional[Decimal] = None,
    purpose: Literal["simulation_input"],
    declared_by: str,
    label: Optional[str] = None,
    rationale: Optional[str] = None,
    supersedes_id: Optional[str] = None,
) -> ConditionsDeclarationOutcome:
    """The Application Service's own text/value-facing entry point for a
    caller (the CLI) that has only raw pH values, not an already-built
    PhDeclaration.

    Exists to resolve two distinct problems with constructing
    PhPoint/PhRange/PhDeclaration one layer up, at the CLI:

    1. A decimal_max_digits/decimal_max_places/decimal_whole_digits
       violation (or an inverted range, or a point/range exclusivity
       violation) must surface as ConditionsValidationError, with
       __cause__ preserved, and must be preceded by a logged "started"
       and followed by a logged "error" entry (Section 19.1 steps 2/4,
       Section 23) - which is only possible if ph is built *inside* this
       module, after "started" is already logged, not one layer up
       before this module is ever called.
    2. The factual created/reused outcome (ConditionsDeclarationOutcome)
       is decided here, in the one place that already knows it for
       certain - never inferred by the CLI from a clock reading.

    Takes Decimal, not str: converting CLI option text into a Decimal
    (`Decimal(...)`, never float(...)) and rejecting a syntactically
    invalid string (decimal.InvalidOperation) or an inconsistent
    combination of CLI flags remain the CLI's own, narrower
    responsibility (Section 22.1 steps 1-3 of the frozen spec) - neither
    is a scientific/structural validation concern, and neither is
    duplicated here. Everything from "is this Decimal within the
    platform's representational limits" onward belongs to this module.

    Delegates to the exact same _declare_conditions_core as
    declare_conditions: no second persistence path, no second retry
    comparison, no second predecessor check, no second activity-logging
    owner.
    """

    return _declare_conditions_core(
        project_dir,
        conditions_id=conditions_id,
        ph_builder=lambda: _build_ph_declaration(ph_point, ph_min, ph_max),
        purpose=purpose,
        declared_by=declared_by,
        label=label,
        rationale=rationale,
        supersedes_id=supersedes_id,
    )


def load_conditions(project_dir: Path, conditions_id: str) -> ScientificConditionsDeclaration:
    """Thin wrapper over the registry's own load. Propagates
    FileNotFoundError/ConditionsRecordInvalidError unchanged. No activity
    logging - a read performs no state-changing operation to log.
    """

    return load_conditions_declaration(project_dir, conditions_id)


def list_conditions(project_dir: Path) -> Tuple[ScientificConditionsDeclaration, ...]:
    """Thin wrapper over the registry's own listing. No activity logging."""

    return list_conditions_declarations(project_dir)
