"""Application services: the seam between the CLI (and any future client)
and the existing domain and persistence modules.

register_ligand_from_pubchem orchestrates the `ligand fetch-pubchem` use
case. register_derived_state orchestrates creating a new MolecularState
from an already-registered parent, together with the
StateTransformationRecord proving that provenance, as one
application-level logical unit of work.

Neither service owns persistence or acquisition/chemistry logic of its
own, only ordering and outcome bookkeeping. CLI-agnostic by design: no
typer import, no console formatting, no default paths. Every domain/
persistence exception raised by the modules these call propagates
unchanged - callers decide how to present them.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple

from molecular.activity import record_activity
from molecular.config import load_experiment
from molecular.context import ExperimentContext, resolve_experiment
from molecular.inputs import LigandInput, validate_input_id
from molecular.pubchem import (
    build_pubchem_sdf_url,
    cleanup_temp_artifact,
    fetch_pubchem_sdf_2d,
)
from molecular.registry import register_ligand
from molecular.scientific import MolecularState
from molecular.scientific_registry import _remove_state_best_effort, load_state, register_state
from molecular.transformation import (
    StateTransformationRecord,
    TransformationParameter,
    TransformationProducer,
)
from molecular.transformation_registry import register_state_transformation

_OPERATION = "ligand-fetch-pubchem"
_DERIVED_STATE_OPERATION = "register-derived-state"


class InvalidPubChemCidError(ValueError):
    """Raised when a PubChem CID is not a positive integer."""


def _record_activity_safely(project_dir: Path, *, outcome: str) -> None:
    """Best-effort activity logging that must never affect the caller's own
    outcome. Catches Exception, not just OSError: any ordinary logging
    failure is swallowed, but a process interrupt (KeyboardInterrupt,
    SystemExit - both BaseException, not Exception) is deliberately never
    caught here, since this helper's only job is to swallow logging
    failures, never interrupts."""

    try:
        record_activity(project_dir, command=_OPERATION, outcome=outcome)
    except Exception:
        pass


def register_ligand_from_pubchem(
    experiment_id: str,
    input_id: str,
    cid: int,
    name: str,
    *,
    base_dir: Path,
) -> LigandInput:
    """Resolve `experiment_id` under `base_dir`, acquire ligand `input_id`
    from PubChem CID `cid`, and register it under the resolved experiment.

    Order is significant and preserved exactly: the experiment is resolved
    *before* the CID is validated, so an unresolvable experiment_id always
    surfaces as an experiment-resolution error (ExperimentNotFoundError,
    InvalidExperimentConfigError, AmbiguousExperimentError), even when the
    CID is also invalid.

    Once the experiment is resolved, every subsequent failure - an invalid
    CID (InvalidPubChemCidError), a PubChem acquisition failure
    (PubChemNotFoundError, PubChemAcquisitionError), or a registration
    failure (InputIdentityConflictError/FileExistsError, FileNotFoundError,
    pydantic ValidationError, OSError) - is recorded as exactly one
    outcome=error activity entry before the original exception propagates
    unchanged. A resolution failure records no activity entry at all, since
    there is no resolved project_dir to log against. On success, exactly
    one outcome=success entry is recorded. Activity logging is always
    best-effort: a failure to write the log entry can never mask the
    underlying success or error, and never substitutes a different
    exception for the one that actually occurred.

    The downloaded temporary SDF is always cleaned up once fetched,
    regardless of whether registration succeeds, conflicts, or raises.
    """

    context: ExperimentContext = resolve_experiment(
        experiment_id, base_dir=base_dir, loader=load_experiment
    )

    try:
        if cid <= 0:
            raise InvalidPubChemCidError(f"'{cid}' is not a valid PubChem CID.")

        temp_path = fetch_pubchem_sdf_2d(cid)

        try:
            ligand = register_ligand(
                context.project_dir,
                input_id=input_id,
                name=name,
                method="downloaded",
                raw_file=temp_path,
                external_id=f"CID {cid}",
                source_name="PubChem",
                source_identifier=f"CID {cid}",
                source_uri=build_pubchem_sdf_url(cid),
                file_format="sdf",
            )
        finally:
            cleanup_temp_artifact(temp_path)

    except Exception:
        _record_activity_safely(context.project_dir, outcome="error")
        raise

    _record_activity_safely(context.project_dir, outcome="success")
    return ligand


@dataclass(frozen=True)
class DerivedStateActivityDetails:
    """Typed activity-log context for register_derived_state.

    Deliberately not a dict[str, Any]: every field it carries is a
    specific, known correlation id, never arbitrary metadata. attempt_id
    is opaque and operational only - it never becomes a scientific
    identifier and is never derived from parent/child/transformation_id;
    it exists solely so that two separate attempts at the same
    transformation_id remain distinguishable in the log.
    """

    attempt_id: str
    parent_state_id: str
    child_state_id: str
    transformation_id: str


def _record_derived_state_activity_safely(
    project_dir: Path, *, outcome: str, details: DerivedStateActivityDetails
) -> None:
    """Best-effort activity logging for register_derived_state - mirrors
    _record_activity_safely above, but carries the typed correlation
    details so an interrupted attempt can later be identified. A failure
    to write this line must never affect the caller's own outcome.
    Catches Exception, not just OSError, for the same reason
    _record_activity_safely does - and, symmetrically, never catches
    BaseException: a KeyboardInterrupt/SystemExit reaching this helper
    must propagate, not be swallowed as if it were a logging failure."""

    try:
        record_activity(
            project_dir,
            command=_DERIVED_STATE_OPERATION,
            outcome=outcome,
            details=details,
        )
    except Exception:
        pass


def register_derived_state(
    project_dir: Path,
    *,
    parent_state_id: str,
    child_state_id: str,
    source_file: Path,
    format: str,
    transformation_id: str,
    operation: Literal["protonation", "tautomerization"],
    producer: Optional[TransformationProducer] = None,
    method_description: Optional[str] = None,
    parameters: Tuple[TransformationParameter, ...] = (),
    created_at: Optional[datetime] = None,
) -> Tuple[MolecularState, StateTransformationRecord]:
    """Register a new MolecularState derived from an already-registered
    parent state, together with the StateTransformationRecord that proves
    the provenance between them, as one application-level logical unit of
    work.

    This is not a database transaction - there is no distributed commit
    across state metadata, state artifact, transformation metadata, and
    the activity log. What exists is this ordering, plus best-effort
    compensation on failure:

    1. generate an opaque attempt_id (observability/audit correlation
       only - never a scientific identifier; it never enters
       MolecularState or StateTransformationRecord, and never derives
       transformation_id);
    2. best-effort log outcome="started", carrying attempt_id,
       parent_state_id, child_state_id, and transformation_id - before
       any write that could create the child state;
    3. load/validate the parent state - this also fixes the child's
       entity_id: a derived state always belongs to its parent's
       ScientificEntity, enforced by construction here rather than left
       to a later cross-check;
    4. register the child MolecularState;
    5. register the StateTransformationRecord connecting them;
    6. best-effort log outcome="success";
    7. return (child, transformation).

    On any exception once step 4 has created the child (i.e. from step 5
    onward, or if step 5 itself fails): best-effort compensating removal
    of that just-created child's metadata and artifact - never a
    preexisting state, since register_state already rejects a preexisting
    child_state_id outright - followed by a best-effort outcome="error"
    log, then the original exception is re-raised unchanged. A failure at
    or before step 3, or during step 4 itself, needs no compensation
    (nothing was created, or the failed registration already left nothing
    behind) and logs error the same way.

    Activity logging is always best-effort in both directions: a failure
    to write started/success/error can never block the operation, never
    turn success into failure, and never mask the exception that actually
    occurred. The scientific filesystem - not the activity log - remains
    the source of truth; a `started` entry with no matching `success`/
    `error` (e.g. after a process crash between steps 4 and 6) is only a
    candidate for investigation, never proof of an orphaned state on its
    own - that investigation must confirm against the filesystem via
    child_state_id and transformation_id.

    parent_state_id, child_state_id, and transformation_id are validated
    (filesystem-safe id format) before anything else happens - including
    before the started log - so a malformed id fails immediately, without
    generating an attempt_id, without any activity-log I/O, and without
    ever reaching the point where an unsafe id string could be written
    into a `field=value` log line.

    The compensation/error-logging boundary here is `except BaseException`,
    not `except Exception`: once the child has been created, a
    KeyboardInterrupt or SystemExit arriving before the transformation is
    registered must still trigger compensation and a best-effort error
    log, exactly like any other exception - this mirrors the boundary
    already used by molecular.registry and molecular.scientific_registry
    for the same reason (their own cleanup-on-failure paths use
    `except BaseException`, not `except Exception`). This is deliberately
    different from the activity-logging helpers' own boundary
    (`except Exception`, not `except BaseException`): a logging failure is
    always an ordinary error to swallow best-effort, while a process
    interrupt must never be caught and discarded by the logging helper
    itself - only the outer compensation boundary is allowed to observe an
    interrupt, and only to compensate and re-raise it unchanged, never to
    swallow it.
    """

    validate_input_id(parent_state_id)
    validate_input_id(child_state_id)
    validate_input_id(transformation_id)

    attempt_id = uuid.uuid4().hex

    details = DerivedStateActivityDetails(
        attempt_id=attempt_id,
        parent_state_id=parent_state_id,
        child_state_id=child_state_id,
        transformation_id=transformation_id,
    )

    _record_derived_state_activity_safely(project_dir, outcome="started", details=details)

    child_created = False

    try:
        parent = load_state(project_dir, parent_state_id)

        child = register_state(
            project_dir,
            state_id=child_state_id,
            entity_id=parent.entity_id,
            source_file=source_file,
            format=format,
        )
        child_created = True

        transformation = StateTransformationRecord(
            schema_version="1.0",
            id=transformation_id,
            parent_state_id=parent_state_id,
            child_state_id=child_state_id,
            operation=operation,
            producer=producer,
            method_description=method_description,
            parameters=tuple(parameters),
            created_at=created_at if created_at is not None else datetime.now(timezone.utc),
        )

        register_state_transformation(project_dir, transformation)

    except BaseException:
        if child_created:
            _remove_state_best_effort(project_dir, child_state_id)
        _record_derived_state_activity_safely(project_dir, outcome="error", details=details)
        raise

    _record_derived_state_activity_safely(project_dir, outcome="success", details=details)
    return child, transformation
