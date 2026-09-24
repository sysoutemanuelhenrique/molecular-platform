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

import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

from molecular.activity import record_activity
from molecular.atomic import compute_sha256
from molecular.config import load_experiment
from molecular.context import ExperimentContext, resolve_experiment
from molecular.execution import (
    ExecutionAttempt,
    ExecutionAttemptError,
    ResolvedExecutionSnapshot,
    TautomerEnumerationPolicy,
)
from molecular.execution_registry import (
    begin_execution_attempt,
    mark_execution_failed,
    mark_resolution_failed,
    mark_resolved,
    mark_succeeded,
)
from molecular.inputs import LigandInput, validate_input_id
from molecular.pubchem import (
    build_pubchem_sdf_url,
    cleanup_temp_artifact,
    fetch_pubchem_sdf_2d,
)
from molecular.rdkit_tautomer import (
    RDKitEngineResolution,
    RDKitTautomerEnumerationResult,
    enumerate_tautomers_rdkit,
    resolve_rdkit_engine,
)
from molecular.registry import register_ligand
from molecular.scientific import MolecularState
from molecular.scientific_registry import (
    _check_state_identity_conflict,
    _remove_state_best_effort,
    _states_dir,
    load_state,
    register_state,
)
from molecular.transformation import (
    StateTransformationRecord,
    TransformationParameter,
    TransformationProducer,
)
from molecular.transformation_registry import (
    _check_transformation_identity_conflict,
    _remove_state_transformation_best_effort,
    _transformations_dir,
    register_state_transformation,
)

_OPERATION = "ligand-fetch-pubchem"
_DERIVED_STATE_OPERATION = "register-derived-state"
_ENUMERATE_TAUTOMERS_OPERATION = "enumerate-state-tautomers"


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


@dataclass(frozen=True)
class TautomerEnumerationActivityDetails:
    """Typed activity-log context for enumerate_state_tautomers. Mirrors
    DerivedStateActivityDetails exactly in spirit: every field is a
    specific, known correlation id, never arbitrary metadata. attempt_id
    is opaque and operational only, exactly as in
    DerivedStateActivityDetails.

    published_state_ids/published_transformation_ids carry the SAME
    meaning on every outcome ("started", "success", or "error"): every id
    that this specific execution actually registered before it ended -
    never ids that were only staged/planned, and never the residual ids
    left behind by a failed best-effort compensation. On the error
    outcome this may therefore include ids that compensation went on to
    remove (successfully or not) after this snapshot of the local
    publication lists was taken - that is expected and intentional: this
    log line records what the execution itself did, not the post-
    compensation state of the filesystem. The authoritative record of
    which ids, if any, compensation could NOT confirm removed lives
    exclusively in ExecutionAttemptError.residual_state_ids/
    residual_operation_ids (Section 10.4 of the frozen spec) - this
    best-effort activity log is never the source of truth for that fact,
    and never redefines "published" as "residual"."""

    attempt_id: str
    input_state_id: str
    published_state_ids: Tuple[str, ...]
    published_transformation_ids: Tuple[str, ...]


def _record_enumerate_tautomers_activity_safely(
    project_dir: Path, *, outcome: str, details: TautomerEnumerationActivityDetails
) -> None:
    """Best-effort activity logging for enumerate_state_tautomers - mirrors
    _record_derived_state_activity_safely above. A failure to write this
    line must never affect the caller's own outcome. Catches Exception,
    not BaseException, for the same reason the other activity-logging
    helpers in this module do."""

    try:
        record_activity(
            project_dir,
            command=_ENUMERATE_TAUTOMERS_OPERATION,
            outcome=outcome,
            details=details,
        )
    except Exception:
        pass


def _describe_error(error: BaseException) -> str:
    """A concise, non-blank human message for ExecutionAttemptError.message -
    never a stack trace or arbitrary stdout (Section 18 of the frozen
    spec)."""

    text = str(error)
    return text if text else repr(error)


def _state_metadata_file(project_dir: Path, state_id: str) -> Path:
    return _states_dir(project_dir) / f"{state_id}.yaml"


def _transformation_metadata_file(project_dir: Path, transformation_id: str) -> Path:
    return _transformations_dir(project_dir) / f"{transformation_id}.yaml"


def _remove_state_confirmed(project_dir: Path, state_id: str) -> bool:
    """Best-effort remove a state this same call just created, then verify
    the removal actually took effect. _remove_state_best_effort never
    raises and never reports success/failure itself (its own OSErrors are
    logged and swallowed) - so confirmation here is a post-condition
    check against the filesystem, exactly what Section 17.3 of the frozen
    spec means by "não puder confirmar a remoção"."""

    _remove_state_best_effort(project_dir, state_id)
    return not _state_metadata_file(project_dir, state_id).exists()


def _remove_transformation_confirmed(project_dir: Path, transformation_id: str) -> bool:
    """Same confirmation contract as _remove_state_confirmed, for a
    StateTransformationRecord this same call just registered."""

    _remove_state_transformation_best_effort(project_dir, transformation_id)
    return not _transformation_metadata_file(project_dir, transformation_id).exists()


def _compensate_published_candidates(
    project_dir: Path,
    staged: List[dict],
    state_created_flags: List[bool],
    transformation_created_flags: List[bool],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Reverse-order best-effort compensation for candidates staged by
    THIS invocation of enumerate_state_tautomers only (Section 17.3).
    Covers both cases the frozen spec distinguishes: complete pairs
    published before the failure, and the asymmetric candidate whose
    state was created but whose transformation never was (or vice versa
    is impossible by construction - a transformation is only ever
    registered after its state). Iterating staged in strict reverse index
    order naturally removes the most-recently-written (in-flight)
    candidate first, exactly as required. Never touches a record from a
    previous call - only ids this call itself generated and flagged as
    created. Returns the ids that could not be confirmed removed, in the
    original deterministic publication order (never reordered).
    """

    residual_state_ids: List[str] = []
    residual_operation_ids: List[str] = []

    for index in range(len(staged) - 1, -1, -1):
        item = staged[index]

        if transformation_created_flags[index]:
            if not _remove_transformation_confirmed(project_dir, item["transformation_id"]):
                residual_operation_ids.append(item["transformation_id"])

        if state_created_flags[index]:
            if not _remove_state_confirmed(project_dir, item["state_id"]):
                residual_state_ids.append(item["state_id"])

    residual_state_ids.reverse()
    residual_operation_ids.reverse()
    return tuple(residual_state_ids), tuple(residual_operation_ids)


def enumerate_state_tautomers(
    project_dir: Path,
    *,
    input_state_id: str,
    policy: TautomerEnumerationPolicy,
    _resolve_engine: Callable[[], RDKitEngineResolution] = resolve_rdkit_engine,
    _enumerate: Callable[..., RDKitTautomerEnumerationResult] = enumerate_tautomers_rdkit,
) -> ExecutionAttempt:
    """Orchestrate one full tautomer-enumeration ExecutionAttempt end to
    end: resolve the input state and the RDKit engine, enumerate, apply
    policy, publish results, and conclude the attempt - exactly the
    sequence normatively defined in Section 15 of the Increment 016 frozen
    spec.

    _resolve_engine and _enumerate are private, keyword-only injection
    seams for tests only (Section 20.4) - never part of the public
    contract, never given a Protocol/ABC type of their own, and never
    documented to callers outside this module's own test suite. Every
    real caller uses the defaults.

    This is not a database transaction. What exists is the ordering in
    Section 15, plus best-effort compensation on failure (Section 17.3).
    input_state_id and every id inside `policy` are validated (Section
    15, step 1) before any I/O - including before begin_execution_attempt -
    so a malformed id fails immediately, without creating an attempt at
    all.

    Every domain/persistence exception raised by the modules this calls
    propagates unchanged to a corresponding mark_resolution_failed or
    mark_execution_failed call, mirroring register_derived_state's own
    error-handling convention in this module: this function never lets an
    exception escape without first recording it against the attempt it
    belongs to, except for the pre-attempt validation in step 1, which by
    construction has no attempt yet to record against.

    The compensation/error-logging boundary from resolve onward is
    `except BaseException`, not `except Exception`, mirroring
    register_derived_state's own justification in this module: once a
    write has occurred, a KeyboardInterrupt/SystemExit must still trigger
    compensation and a best-effort error log, exactly like any other
    exception.
    """

    validate_input_id(input_state_id)

    attempt = begin_execution_attempt(
        project_dir,
        operation_type="state-transformation",
        requested=policy,
        input_state_id=input_state_id,
    )

    started_details = TautomerEnumerationActivityDetails(
        attempt_id=attempt.id,
        input_state_id=input_state_id,
        published_state_ids=(),
        published_transformation_ids=(),
    )
    _record_enumerate_tautomers_activity_safely(project_dir, outcome="started", details=started_details)

    # --- resolution phase (Section 15, steps 1-5) ---
    try:
        input_state = load_state(project_dir, input_state_id)

        normalized_format = input_state.representation.format.strip().lower()
        if normalized_format != "sdf":
            raise ValueError(
                f"input state '{input_state_id}' has representation format "
                f"{input_state.representation.format!r}, expected 'sdf'"
            )

        source_path = project_dir / input_state.representation.path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"representation file for state '{input_state_id}' not found: {source_path}"
            )

        actual_checksum = compute_sha256(source_path)
        if actual_checksum != input_state.representation.checksum:
            raise ValueError(
                f"representation checksum mismatch for state '{input_state_id}': "
                f"expected {input_state.representation.checksum}, got {actual_checksum}"
            )

        engine_resolution = _resolve_engine()

    except BaseException as error:
        mark_resolution_failed(
            project_dir,
            attempt.id,
            error=ExecutionAttemptError(phase="resolution", message=_describe_error(error)),
        )
        _record_enumerate_tautomers_activity_safely(
            project_dir, outcome="error", details=started_details
        )
        raise

    resolved_snapshot = ResolvedExecutionSnapshot(
        schema_version="1.0",
        engine_name=engine_resolution.engine_name,
        engine_version=engine_resolution.engine_version,
        parameters=engine_resolution.parameters,
    )
    mark_resolved(project_dir, attempt.id, resolved=resolved_snapshot)

    # --- execution phase (Section 15, steps 6-23) ---
    published_state_ids: List[str] = []
    published_transformation_ids: List[str] = []
    staged: List[dict] = []
    state_created_flags: List[bool] = []
    transformation_created_flags: List[bool] = []

    try:
        enumeration_result = _enumerate(source_path=source_path, source_format="sdf")

        published_candidates = enumeration_result.candidates
        if policy.max_states is not None:
            published_candidates = published_candidates[: policy.max_states]

        if len(published_candidates) == 0:
            # Section 15.1: zero new states is a valid success, whatever
            # the cause - no staging, no preflight, no publication step.
            succeeded = mark_succeeded(project_dir, attempt.id, operation_ids=())
            _record_enumerate_tautomers_activity_safely(
                project_dir,
                outcome="success",
                details=TautomerEnumerationActivityDetails(
                    attempt_id=attempt.id,
                    input_state_id=input_state_id,
                    published_state_ids=(),
                    published_transformation_ids=(),
                ),
            )
            return succeeded

        staged = [
            {
                "state_id": uuid.uuid4().hex,
                "transformation_id": uuid.uuid4().hex,
                "candidate": candidate,
            }
            for candidate in published_candidates
        ]
        state_created_flags = [False] * len(staged)
        transformation_created_flags = [False] * len(staged)

        # Step 21: preflight every generated identity before any write.
        states_dir = _states_dir(project_dir)
        transformations_dir = _transformations_dir(project_dir)
        for item in staged:
            _check_state_identity_conflict(states_dir, item["state_id"])
            _check_transformation_identity_conflict(transformations_dir, item["transformation_id"])

        published_count = len(published_candidates)
        truncated_by_policy = published_count < enumeration_result.new_unique_count

        with tempfile.TemporaryDirectory(prefix="tautomer-enumeration-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)

            for index, item in enumerate(staged):
                state_id = item["state_id"]
                transformation_id = item["transformation_id"]
                smiles_text = item["candidate"].canonical_isomeric_smiles

                smiles_file = temp_dir / f"{state_id}.smi"
                smiles_file.write_text(smiles_text + "\n", encoding="utf-8")

                register_state(
                    project_dir,
                    state_id=state_id,
                    entity_id=input_state.entity_id,
                    source_file=smiles_file,
                    format="smiles",
                )
                state_created_flags[index] = True
                published_state_ids.append(state_id)

                transformation = StateTransformationRecord(
                    schema_version="1.0",
                    id=transformation_id,
                    parent_state_id=input_state_id,
                    child_state_id=state_id,
                    operation="tautomerization",
                    producer=TransformationProducer(
                        name="rdkit", version=engine_resolution.engine_version
                    ),
                    parameters=(
                        TransformationParameter(name="engine_name", value="rdkit"),
                        TransformationParameter(
                            name="engine_version", value=engine_resolution.engine_version
                        ),
                        TransformationParameter(name="candidate_rank", value=index),
                        TransformationParameter(
                            name="enumerated_unique_count",
                            value=enumeration_result.enumerated_unique_count,
                        ),
                        TransformationParameter(
                            name="new_unique_count", value=enumeration_result.new_unique_count
                        ),
                        TransformationParameter(name="published_count", value=published_count),
                        TransformationParameter(
                            name="truncated_by_policy", value=truncated_by_policy
                        ),
                        TransformationParameter(name="attempt_id", value=attempt.id),
                    ),
                    created_at=datetime.now(timezone.utc),
                )
                register_state_transformation(project_dir, transformation)
                transformation_created_flags[index] = True
                published_transformation_ids.append(transformation_id)

        succeeded = mark_succeeded(
            project_dir, attempt.id, operation_ids=tuple(published_transformation_ids)
        )
        _record_enumerate_tautomers_activity_safely(
            project_dir,
            outcome="success",
            details=TautomerEnumerationActivityDetails(
                attempt_id=attempt.id,
                input_state_id=input_state_id,
                published_state_ids=tuple(published_state_ids),
                published_transformation_ids=tuple(published_transformation_ids),
            ),
        )
        return succeeded

    except BaseException as error:
        residual_state_ids, residual_operation_ids = _compensate_published_candidates(
            project_dir, staged, state_created_flags, transformation_created_flags
        )
        mark_execution_failed(
            project_dir,
            attempt.id,
            error=ExecutionAttemptError(
                phase="execution",
                message=_describe_error(error),
                residual_state_ids=residual_state_ids,
                residual_operation_ids=residual_operation_ids,
            ),
        )
        _record_enumerate_tautomers_activity_safely(
            project_dir,
            outcome="error",
            details=TautomerEnumerationActivityDetails(
                attempt_id=attempt.id,
                input_state_id=input_state_id,
                published_state_ids=tuple(published_state_ids),
                published_transformation_ids=tuple(published_transformation_ids),
            ),
        )
        raise
