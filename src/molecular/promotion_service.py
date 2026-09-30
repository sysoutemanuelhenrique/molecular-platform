"""Application Service: promoting a LigandInput RAW artifact into an
auditable ScientificEntity/MolecularState pair (Section 14 of Increment
018's frozen final specification,
``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``,
SHA-256 ``347632175991b1d72b9f193312b9629ce71697fb0c3d0de2285e52b7299d8023``).

This is Checkpoint 4: the orchestration seam between an already-registered
LigandInput and the Checkpoint 1-3 primitives that already exist -
`molecular.promotion`/`molecular.promotion_registry` (domain models,
attempt lifecycle, semantic_key, cross-process reservation/claims,
promotion-record publication), `molecular.atomic.create_verified_raw_snapshot`
(Route B endurecida), `molecular.rdkit_tautomer.inspect_single_sdf_for_promotion`
(neutral SDF inspection), and `molecular.scientific_registry` (entity/state
registration and, new in this checkpoint, the narrow match-and-unreferenced
compensation primitives, Section 13.4). This module owns none of that logic
itself - only ordering, ownership bookkeeping, and outcome translation,
exactly like `molecular.application`'s existing services.

`promote_ligand_input`'s public signature (Section 14.2) never accepts a
caller-supplied snapshot path of any kind - the private, per-attempt
snapshot directory under `project_dir/scientific/promotion_snapshots/
<attempt_id>/` is entirely this module's own construction (`snapshot_root`,
the parameter this checkpoint added to `create_verified_raw_snapshot`,
MINOR-C1's correction), never influenced by a CLI, a user, or untrusted
metadata.

Algorithm (Section 12, literally):

- Section 12.1 (steps 1-5): validate every argument - input_id, entity_id,
  state_id, entity_name, intentional_revision - before any I/O at all, not
  even `begin_promotion_attempt`.
- Section 12.2 (steps 6-19, "resolution phase"): begin the attempt; load
  the LigandInput; require SDF format; snapshot the RAW via Route B
  endurecida; inspect it with RDKit; compute the semantic_key; persist
  `resolved`. Any failure here persists `resolution_failed` and removes
  the private snapshot directory - the RAW itself is never touched.
- Section 12.3 (steps 20-34, "execution phase"): recognize an existing
  promotion record as authoritative (idempotent reuse, no reservation ever
  taken for a pure reuse); otherwise require `intentional_revision` for a
  semantic revision of the same input_id; acquire semantic/entity/state
  reservations in that exact order; load-or-create the entity; register
  the state from the snapshot; publish the promotion record by exclusive
  creation; mark the attempt succeeded; release the reservations (best
  effort, never affecting the already-durable outcome); remove the private
  snapshot directory; log activity.
- Section 13 (compensation): any execution-phase failure BEFORE the
  promotion record is committed triggers reverse-order compensation
  (state, then entity, then reservations, matching only what this attempt
  itself created/acquired) and persists `execution_failed` with whatever
  residuals could not be confirmed removed/released. Once the promotion
  record is committed, NOTHING compensates it, its entity, or its state -
  Section 13.3's own rule, mirrored exactly. If `mark_succeeded` itself
  then fails to persist the attempt's own terminal status - the record,
  entity, and state are already durable - this is never reclassified as
  `execution_failed` (Checkpoint 4 correction, BLOCKER-01): Section 13.3's
  own last sentence calls this an operational persistence failure, so
  `LigandPromotionAttemptPersistenceError` is raised directly
  (`intended_status="succeeded"`), with no compensation and no attempt to
  persist any other terminal status - the attempt simply stays at its last
  honestly-persisted status ("resolved").

KeyboardInterrupt/SystemExit are never converted into a scientific error at
any point (Section 12.4): they still trigger the same best-effort
compensation/persistence a normal failure would at that phase, but the
interruption itself is always what ultimately propagates.
"""

import logging
import secrets
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple

from molecular.activity import record_activity
from molecular.atomic import (
    RawArtifactIdentityUnstableError,
    RawArtifactIntegrityError,
    RawArtifactNoFollowUnavailableError,
    RawArtifactSharingViolationError,
    RawArtifactTypeError,
    UnsafeRawArtifactPathError,
    create_verified_private_directory,
    create_verified_raw_snapshot,
)
from molecular.inputs import validate_input_id
from molecular.promotion import (
    AcquisitionProvenanceSnapshot,
    LigandPromotionAttempt,
    LigandPromotionError,
    LigandPromotionPolicy,
    PromotionPhase,
    PromotionResultSnapshot,
    ResolvedLigandPromotionSnapshot,
    ScientificInputPromotionRecord,
    build_semantic_key_payload,
    compute_semantic_key,
)
from molecular.promotion_registry import (
    EntityClaimConflictError,
    PromotionPublicationUnavailableError,
    PromotionRecordConflictError,
    SemanticReservationConflictError,
    StateClaimConflictError,
    acquire_entity_claim,
    acquire_semantic_reservation,
    acquire_state_claim,
    begin_promotion_attempt,
    compensate_partial_acquisition,
    list_promotions_by_input,
    load_promotion,
    mark_execution_failed,
    mark_resolution_failed,
    mark_resolved,
    mark_succeeded,
    register_promotion,
)
from molecular.rdkit_tautomer import (
    DisconnectedStructureError,
    InvalidPromotionSDFError,
    PromotionRecordCountError,
    RDKitUnavailableError,
    inspect_single_sdf_for_promotion,
)
from molecular.registry import load_ligand
from molecular.scientific import MolecularState, ScientificEntity
from molecular.scientific_registry import (
    load_entity,
    register_entity,
    register_state,
    remove_entity_if_matches_and_unreferenced,
    remove_state_if_matches_and_unreferenced,
)

logger = logging.getLogger(__name__)

_OPERATION = "ligand-promote"


class UnsupportedPromotionFormatError(ValueError):
    """`unsupported-format` (Section 17). Raised when the LigandInput's
    declared RAW format, case/whitespace-insensitively, is not `"sdf"` -
    Increment 018 promotes SDF only (Section 4.2)."""


class PromotionRevisionRequiredError(RuntimeError):
    """`revision-required` (Section 17). Raised when this `input_id` was
    already promoted under a different semantic_key (a different RAW
    checksum, engine, or policy) and the caller did not supply
    `intentional_revision` - a scientist must explicitly acknowledge a
    semantic revision of the same input, never have one happen implicitly
    (Section 10.7)."""


class PromotionIdentityConflictError(RuntimeError):
    """`identity-conflict` (Section 17). Raised when requested identity
    (entity_id/state_id, or an existing entity's name) is incompatible with
    what is already published or persisted for the relevant key - a retry
    with different ids against an already-promoted semantic_key (Section
    10.6), or an already-registered entity whose name does not match
    `requested_entity_name`."""


class PromotionReservationConflictError(RuntimeError):
    """`reservation-conflict` (Section 17). Raised when another attempt
    already holds the semantic reservation or an entity/state claim this
    attempt needs (Section 11.6's `BUSY_OR_INDETERMINATE`) - wraps
    `SemanticReservationConflictError`/`EntityClaimConflictError`/
    `StateClaimConflictError` from `molecular.promotion_registry` at this
    application boundary, preserving the original as `__cause__`."""


class PromotionConsistencyError(RuntimeError):
    """`consistency-error` (Section 17). Raised when the filesystem and
    the lineage this attempt just built disagree in a way that indicates a
    genuine anomaly rather than an ordinary, already-handled race - e.g. a
    promotion record appearing for this semantic_key while this attempt
    still exclusively holds the semantic reservation, which normal
    concurrent callers cannot produce (Section 11.5's acquisition order is
    the only path to publication); or a freshly-published representation
    whose checksum does not match the resolved snapshot it was built from."""


class LigandPromotionAttemptFailedError(RuntimeError):
    """A promotion attempt reached a persisted terminal failure status
    (Section 14.4 of the frozen spec)."""

    def __init__(
        self,
        *,
        attempt_id: str,
        input_id: str,
        status: str,
        original_error: Exception,
    ) -> None:
        super().__init__(
            f"promotion attempt '{attempt_id}' for input '{input_id}' failed "
            f"with status '{status}': {str(original_error) or repr(original_error)}"
        )
        self.attempt_id = attempt_id
        self.input_id = input_id
        self.status = status
        self.original_error = original_error


class LigandPromotionAttemptPersistenceError(RuntimeError):
    """A promotion attempt's terminal status could not itself be persisted
    (Section 14.4 of the frozen spec).

    `intended_status` widens the spec's own illustrative
    `Literal["resolution_failed", "execution_failed"]` (Section 14.4) to
    also include `"succeeded"` (Checkpoint 4 correction, BLOCKER-01): a
    `mark_succeeded` failure occurring AFTER the promotion record has
    already been committed (Section 13.3's own explicit rule - "falha ao
    gravar status do attempt produz erro de persistência operacional") is
    exactly this class of failure, and the spec's own literal simply never
    named the case where the intended status was success. Widening the type
    is the minimal change that lets this class honestly describe that
    scenario, without inventing a new status value or altering the frozen
    specification itself."""

    def __init__(
        self,
        *,
        attempt_id: str,
        input_id: str,
        phase: PromotionPhase,
        intended_status: Literal["resolution_failed", "execution_failed", "succeeded"],
        original_error: BaseException,
        persistence_error: Exception,
        residual_entity_ids: Tuple[str, ...] = (),
        residual_state_ids: Tuple[str, ...] = (),
        residual_reservation_keys: Tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            f"promotion attempt '{attempt_id}' for input '{input_id}' could not "
            f"persist intended status '{intended_status}': "
            f"{str(persistence_error) or repr(persistence_error)}"
        )
        self.attempt_id = attempt_id
        self.input_id = input_id
        self.phase = phase
        self.intended_status = intended_status
        self.original_error = original_error
        self.persistence_error = persistence_error
        self.residual_entity_ids = tuple(residual_entity_ids)
        self.residual_state_ids = tuple(residual_state_ids)
        self.residual_reservation_keys = tuple(residual_reservation_keys)


@dataclass(frozen=True)
class _PromotionActivityDetails:
    """Typed activity-log context - one specific correlation id per field,
    never arbitrary metadata, no path/URL/SMILES (Section 14.5)."""

    attempt_id: str
    input_id: str
    entity_id: str
    state_id: str


def _record_activity_safely(
    project_dir: Path, *, outcome: str, details: _PromotionActivityDetails
) -> None:
    """Best-effort activity logging - must never affect the caller's own
    outcome, and must never itself be mistaken for the source of truth
    (Section 14.5/32 of the frozen spec)."""

    try:
        record_activity(project_dir, command=_OPERATION, outcome=outcome, details=details)
    except Exception:
        pass


def _describe_error(error: BaseException) -> str:
    text = str(error)
    return text if text else repr(error)


def _snapshot_root(project_dir: Path, attempt_id: str) -> Path:
    return project_dir / "scientific" / "promotion_snapshots" / attempt_id


def _release_error_frames(error: BaseException) -> None:
    """Best-effort release of any resource (e.g. an RDKit `SDMolSupplier`'s
    open file handle inside `inspect_single_sdf_for_promotion`'s own,
    already-finished stack frame) still referenced only via this
    exception's own traceback. A live `except ... as error:` binding keeps
    every frame's locals along the whole call stack alive for as long as
    `error` remains in scope - even though those frames have already
    returned - which on Windows can keep a file handle open long enough to
    make an immediately-following cleanup (`_cleanup_snapshot_root`) fail
    with a sharing violation. `traceback.clear_frames` explicitly drops
    those locals; it is safe here because every frame in this traceback
    has already finished executing by the time this is called. Never
    raises."""

    try:
        traceback.clear_frames(error.__traceback__)
    except Exception:
        pass


def _cleanup_snapshot_root(snapshot_root: Path) -> None:
    """Best-effort removal of this attempt's own private snapshot
    directory - never raises. Safe unconditionally: the directory is named
    by this attempt's own id, under this module's own private area, never
    shared with any other attempt or with the RAW artifact's own tree."""

    try:
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
    except OSError:
        logger.warning(
            "Failed to clean up promotion snapshot directory at %s", snapshot_root, exc_info=True
        )


def _build_acquisition_snapshot(ligand) -> AcquisitionProvenanceSnapshot:
    acquisition = ligand.acquisition
    return AcquisitionProvenanceSnapshot(
        method=acquisition.method,
        source_name=acquisition.source_name,
        source_identifier=acquisition.source_identifier,
        source_uri=acquisition.source_uri,
        acquired_at=acquisition.acquired_at,
        raw_artifact_path=acquisition.raw_artifact.path,
        raw_artifact_format=acquisition.raw_artifact.format,
        raw_artifact_checksum=acquisition.raw_artifact.checksum,
        pubchem_cid=None,
        pubchem_record_type=None,
    )


def _resolve_snapshot(
    project_dir: Path,
    *,
    attempt_id: str,
    input_id: str,
    ligand,
    policy: LigandPromotionPolicy,
) -> ResolvedLigandPromotionSnapshot:
    """Section 12.2, steps 9-17: require SDF format, snapshot the RAW via
    Route B endurecida into this attempt's own private, internally
    constructed snapshot directory (MINOR-C1: never a caller-supplied
    path), inspect it with RDKit, and build the resolved snapshot."""

    declared_format = ligand.acquisition.raw_artifact.format.strip().lower()
    if declared_format != "sdf":
        raise UnsupportedPromotionFormatError(
            f"LigandInput '{input_id}' has RAW format {declared_format!r}, expected 'sdf'"
        )

    # Checkpoint 4 correction (MAJOR-02): the whole chain from project_dir
    # down to this attempt's own directory is created and validated one
    # component at a time - never a single Path.mkdir(parents=True,
    # exist_ok=True) that would silently cross an unvalidated (and
    # possibly redirected) `scientific` or `promotion_snapshots` ancestor.
    snapshot_root = create_verified_private_directory(
        project_dir, shared_parts=("scientific", "promotion_snapshots"), exclusive_part=attempt_id
    )
    snapshot_path = snapshot_root / "snapshot.sdf"

    raw_snapshot = create_verified_raw_snapshot(
        project_dir=project_dir,
        input_id=input_id,
        raw_artifact_path=ligand.acquisition.raw_artifact.path,
        raw_artifact_checksum=ligand.acquisition.raw_artifact.checksum,
        snapshot_root=snapshot_root,
        snapshot_path=snapshot_path,
    )

    inspection = inspect_single_sdf_for_promotion(
        source_path=raw_snapshot.snapshot_path, policy=policy
    )

    return ResolvedLigandPromotionSnapshot(
        engine_name=inspection.engine_name,
        engine_version=inspection.engine_version,
        input_format=inspection.input_format,
        raw_checksum_persisted=ligand.acquisition.raw_artifact.checksum,
        snapshot_checksum=raw_snapshot.snapshot_checksum,
        snapshot_size_bytes=raw_snapshot.snapshot_size_bytes,
        sanitize=inspection.sanitize,
        remove_hs=inspection.remove_hs,
        strict_parsing=inspection.strict_parsing,
        record_count=inspection.record_count,
        fragment_count=inspection.fragment_count,
        formal_charge=inspection.formal_charge,
        atom_count=inspection.atom_count,
        explicit_hydrogen_count=inspection.explicit_hydrogen_count,
        isotopic_hydrogen_count=inspection.isotopic_hydrogen_count,
        warnings=inspection.warnings,
    )


_RESOLUTION_ERROR_CODES = (
    (UnsupportedPromotionFormatError, "unsupported-format"),
    (UnsafeRawArtifactPathError, "unsafe-raw-path"),
    (RawArtifactTypeError, "raw-not-regular-file"),
    (RawArtifactIntegrityError, "raw-checksum-mismatch"),
    (RawArtifactNoFollowUnavailableError, "raw-nofollow-unavailable"),
    (RawArtifactIdentityUnstableError, "raw-identity-unstable"),
    (RawArtifactSharingViolationError, "raw-permission-denied"),
    (RDKitUnavailableError, "rdkit-unavailable"),
    (InvalidPromotionSDFError, "invalid-sdf"),
    (PromotionRecordCountError, "record-count"),
    (DisconnectedStructureError, "disconnected-structure"),
    (FileNotFoundError, "input-not-found"),
)

_EXECUTION_ERROR_CODES = (
    (PromotionRevisionRequiredError, "revision-required"),
    (PromotionIdentityConflictError, "identity-conflict"),
    (PromotionReservationConflictError, "reservation-conflict"),
    (PromotionPublicationUnavailableError, "promotion-primitive-unavailable"),
    (PromotionConsistencyError, "consistency-error"),
    (RawArtifactIntegrityError, "raw-checksum-mismatch"),
)


def _classify_error(error: Exception, table: Tuple[Tuple[type, str], ...], default: str) -> str:
    """Map `error` to the Section 17 taxonomy code of the first entry in
    `table` whose exception type it is an instance of, checked in
    declaration order (most specific first) - `isinstance` naturally
    matches subclasses, so this never needs an exhaustive exact-type list.
    Falls back to `default` only for an error this module's own typed
    contracts should never actually produce in practice."""

    for exc_type, code in table:
        if isinstance(error, exc_type):
            return code
    return default


def _resolution_error_code(error: Exception) -> str:
    return _classify_error(error, _RESOLUTION_ERROR_CODES, default="consistency-error")


def _execution_error_code(error: Exception) -> str:
    return _classify_error(error, _EXECUTION_ERROR_CODES, default="publication-failed")


def _load_consistent_promotion(
    project_dir: Path, *, semantic_key: str, entity_id: str, state_id: str
) -> Optional[ScientificInputPromotionRecord]:
    """Section 12.3, steps 20-21/24: look up an already-published promotion
    record for `semantic_key`. Returns None if none exists. If one exists
    and its entity_id/state_id match what THIS attempt requested, it is
    the recognized, authoritative outcome of an idempotent retry. If one
    exists with DIFFERENT ids, that is a genuine identity conflict (Section
    10.6: "retry equivalente com IDs diferentes falha sem publicar") -
    raised here rather than silently reused or silently ignored."""

    try:
        record = load_promotion(project_dir, semantic_key)
    except FileNotFoundError:
        return None

    if record.entity_id != entity_id or record.state_id != state_id:
        raise PromotionIdentityConflictError(
            "a promotion record already exists for this semantic_key with "
            "different entity_id/state_id than requested"
        )

    return record


def _require_no_conflicting_revision(
    project_dir: Path,
    *,
    input_id: str,
    semantic_key: str,
    intentional_revision: Optional[str],
) -> None:
    """Section 12.3, step 22: if `input_id` was already promoted under a
    DIFFERENT semantic_key (a real semantic revision - different RAW
    checksum, engine, or policy), an explicit `intentional_revision` is
    required to proceed (Section 10.7). Never triggered by this attempt's
    own semantic_key already having a record - that is idempotent reuse,
    handled entirely by `_load_consistent_promotion`, not a revision."""

    if intentional_revision is not None:
        return

    for prior in list_promotions_by_input(project_dir, input_id):
        if prior.semantic_key != semantic_key:
            raise PromotionRevisionRequiredError(
                f"input '{input_id}' was already promoted under a different "
                "semantic_key; intentional_revision is required for a new one"
            )


def _load_or_create_entity(
    project_dir: Path, *, entity_id: str, entity_name: str
) -> Tuple[ScientificEntity, bool]:
    """Section 12.3, step 25: load a compatible preexisting entity, or
    create a new one. "Compatible" means its persisted name matches
    `entity_name` exactly - a preexisting entity under this id with a
    DIFFERENT name is an identity conflict (Section 22 criterion,
    "entidade preexistente incompatível... são conflitos"), never silently
    accepted or silently renamed."""

    try:
        existing = load_entity(project_dir, entity_id)
    except FileNotFoundError:
        created = register_entity(
            project_dir, ScientificEntity(schema_version="1.0", id=entity_id, name=entity_name)
        )
        return created, True

    if existing.name != entity_name:
        raise PromotionIdentityConflictError(
            f"entity '{entity_id}' is already registered with a different name "
            "than requested"
        )

    return existing, False


def _release_reservations_best_effort(
    project_dir: Path,
    *,
    attempt_id: str,
    owner_token: str,
    semantic_key: Optional[str],
    entity_id: Optional[str],
    state_id: Optional[str],
) -> Tuple[str, ...]:
    """Best-effort release after a successful publication (Section 12.3,
    step 31) - never raises; any residual is logged (there is no field on
    a 'succeeded' LigandPromotionAttempt to carry it, Section 6.9's own
    matrix, since the promotion record is already the durable, authoritative
    outcome by this point)."""

    residuals = compensate_partial_acquisition(
        project_dir,
        attempt_id=attempt_id,
        owner_token=owner_token,
        semantic_key=semantic_key,
        entity_id=entity_id,
        state_id=state_id,
    )
    if residuals:
        logger.warning(
            "Promotion attempt '%s' could not confirm release of reservation(s): %s",
            attempt_id,
            residuals,
        )
    return residuals


@dataclass
class _AcquisitionProgress:
    """Mutable bookkeeping of exactly what THIS attempt has acquired/
    created so far, read by the compensation path on failure (Section
    13.2: "o attempt mantém flags locais... nenhum objeto é removido
    apenas porque seu ID existe numa variável")."""

    semantic_reserved: bool = False
    entity_claimed: bool = False
    state_claimed: bool = False
    entity_created: bool = False
    state_created: bool = False
    entity: Optional[ScientificEntity] = None
    state: Optional[MolecularState] = None


def _compensate_execution_failure(
    project_dir: Path,
    *,
    attempt_id: str,
    owner_token: str,
    semantic_key: str,
    entity_id: str,
    state_id: str,
    progress: _AcquisitionProgress,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Section 13.3's exact reverse order: state, then entity, then
    reservations - matching only what THIS attempt itself created/acquired
    (`progress`), never touching a preexisting object nor another owner's
    reservation."""

    residual_state_ids: Tuple[str, ...] = ()
    residual_entity_ids: Tuple[str, ...] = ()

    if progress.state_created and progress.state is not None:
        if not remove_state_if_matches_and_unreferenced(project_dir, expected=progress.state):
            residual_state_ids = (progress.state.id,)

    if progress.entity_created and progress.entity is not None:
        if not remove_entity_if_matches_and_unreferenced(project_dir, expected=progress.entity):
            residual_entity_ids = (progress.entity.id,)

    residual_reservation_keys = compensate_partial_acquisition(
        project_dir,
        attempt_id=attempt_id,
        owner_token=owner_token,
        semantic_key=semantic_key if progress.semantic_reserved else None,
        entity_id=entity_id if progress.entity_claimed else None,
        state_id=state_id if progress.state_claimed else None,
    )

    return residual_entity_ids, residual_state_ids, residual_reservation_keys


def _execute_new_promotion(
    project_dir: Path,
    *,
    attempt: LigandPromotionAttempt,
    input_id: str,
    entity_id: str,
    state_id: str,
    entity_name: str,
    policy: LigandPromotionPolicy,
    intentional_revision: Optional[str],
    semantic_key: str,
    resolved: ResolvedLigandPromotionSnapshot,
    ligand,
    owner_token: str,
    progress: _AcquisitionProgress,
) -> Optional[ScientificInputPromotionRecord]:
    """Section 12.3, steps 22-29: the non-reuse path. Returns the
    committed record on success, mutating `progress` as it goes so the
    caller's exception handler can compensate exactly what was acquired.
    Returns an already-existing, consistent record instead (without
    acquiring or creating anything) if one appears once this attempt wins
    its own exclusive semantic reservation (step 24's re-check) - the
    caller distinguishes the two cases only by whether `progress.
    state_created` ended up True."""

    _require_no_conflicting_revision(
        project_dir,
        input_id=input_id,
        semantic_key=semantic_key,
        intentional_revision=intentional_revision,
    )

    try:
        acquire_semantic_reservation(
            project_dir, semantic_key, attempt_id=attempt.id, owner_token=owner_token
        )
    except SemanticReservationConflictError as exc:
        raise PromotionReservationConflictError(
            "another attempt already holds the semantic reservation for this "
            "semantic_key"
        ) from exc
    progress.semantic_reserved = True

    try:
        acquire_entity_claim(project_dir, entity_id, attempt_id=attempt.id, owner_token=owner_token)
    except EntityClaimConflictError as exc:
        raise PromotionReservationConflictError(
            "another attempt already holds the claim for this entity_id"
        ) from exc
    progress.entity_claimed = True

    try:
        acquire_state_claim(project_dir, state_id, attempt_id=attempt.id, owner_token=owner_token)
    except StateClaimConflictError as exc:
        raise PromotionReservationConflictError(
            "another attempt already holds the claim for this state_id"
        ) from exc
    progress.state_claimed = True

    existing_record = _load_consistent_promotion(
        project_dir, semantic_key=semantic_key, entity_id=entity_id, state_id=state_id
    )
    if existing_record is not None:
        return existing_record

    fresh_ligand = load_ligand(project_dir, input_id)
    if fresh_ligand.acquisition.raw_artifact.checksum != resolved.raw_checksum_persisted:
        raise RawArtifactIntegrityError(
            "the RAW artifact's registered checksum changed since resolution"
        )

    entity, entity_created = _load_or_create_entity(
        project_dir, entity_id=entity_id, entity_name=entity_name
    )
    progress.entity = entity
    progress.entity_created = entity_created

    snapshot_file = _snapshot_root(project_dir, attempt.id) / "snapshot.sdf"
    state = register_state(
        project_dir, state_id=state_id, entity_id=entity_id, source_file=snapshot_file, format="sdf"
    )
    progress.state = state
    progress.state_created = True

    if state.representation.checksum != resolved.snapshot_checksum:
        raise PromotionConsistencyError(
            "published representation checksum does not match the resolved snapshot"
        )

    record = ScientificInputPromotionRecord(
        id=semantic_key,
        semantic_key=semantic_key,
        input_id=input_id,
        acquisition=_build_acquisition_snapshot(ligand),
        policy=policy,
        resolved=resolved,
        entity_id=entity.id,
        state_id=state.id,
        representation=state.representation,
        created_by_attempt_id=attempt.id,
        intentional_revision=intentional_revision,
        created_at=datetime.now(timezone.utc),
    )

    try:
        register_promotion(project_dir, record)
    except PromotionRecordConflictError as exc:
        raise PromotionConsistencyError(
            "a promotion record appeared for this semantic_key while this "
            "attempt exclusively held its reservation"
        ) from exc

    return record


def promote_ligand_input(
    project_dir: Path,
    *,
    input_id: str,
    entity_id: str,
    state_id: str,
    entity_name: str,
    policy: LigandPromotionPolicy,
    intentional_revision: Optional[str] = None,
) -> LigandPromotionAttempt:
    """Orchestrate one full LigandPromotionAttempt end to end (Section
    14.2/12 of the frozen spec). Always returns a `succeeded` attempt (a
    fresh promotion, or a recognized-idempotent reuse of an existing
    promotion record) - every other outcome is a raised exception.

    Section 12.1: input_id, entity_id, state_id, entity_name, and
    intentional_revision (when given) are all validated before any I/O -
    including before an attempt is even begun - so a malformed argument
    never creates an attempt record at all.
    """

    validate_input_id(input_id)
    validate_input_id(entity_id)
    validate_input_id(state_id)
    if not entity_name.strip():
        raise ValueError("entity_name must not be blank")
    if intentional_revision is not None:
        validate_input_id(intentional_revision)

    attempt = begin_promotion_attempt(
        project_dir,
        input_id=input_id,
        requested_entity_id=entity_id,
        requested_state_id=state_id,
        requested_entity_name=entity_name,
        requested=policy,
        intentional_revision=intentional_revision,
    )

    activity_details = _PromotionActivityDetails(
        attempt_id=attempt.id, input_id=input_id, entity_id=entity_id, state_id=state_id
    )
    _record_activity_safely(project_dir, outcome="started", details=activity_details)

    # ============================= resolution ===============================
    try:
        ligand = load_ligand(project_dir, input_id)
        resolved = _resolve_snapshot(
            project_dir, attempt_id=attempt.id, input_id=input_id, ligand=ligand, policy=policy
        )
        semantic_key = compute_semantic_key(
            build_semantic_key_payload(
                input_id=input_id,
                snapshot_checksum=resolved.snapshot_checksum,
                engine_name=resolved.engine_name,
                engine_version=resolved.engine_version,
                policy=policy,
            )
        )

    except (KeyboardInterrupt, SystemExit) as interrupt:
        _release_error_frames(interrupt)
        _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
        persistence_error: Optional[Exception] = None
        try:
            mark_resolution_failed(
                project_dir,
                attempt.id,
                error=LigandPromotionError(
                    phase="resolution",
                    code="consistency-error",
                    message=_describe_error(interrupt),
                ),
            )
        except Exception as error:
            persistence_error = error
        _record_activity_safely(project_dir, outcome="error", details=activity_details)
        if persistence_error is not None:
            raise interrupt from persistence_error
        raise

    except Exception as error:
        _release_error_frames(error)
        _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
        code = _resolution_error_code(error)
        try:
            mark_resolution_failed(
                project_dir,
                attempt.id,
                error=LigandPromotionError(phase="resolution", code=code, message=_describe_error(error)),
            )
        except Exception as persistence_error:
            _record_activity_safely(project_dir, outcome="error", details=activity_details)
            raise LigandPromotionAttemptPersistenceError(
                attempt_id=attempt.id,
                input_id=input_id,
                phase="resolution",
                intended_status="resolution_failed",
                original_error=error,
                persistence_error=persistence_error,
            ) from persistence_error

        _record_activity_safely(project_dir, outcome="error", details=activity_details)
        raise LigandPromotionAttemptFailedError(
            attempt_id=attempt.id, input_id=input_id, status="resolution_failed", original_error=error
        ) from error

    # Checkpoint 4 correction (MINOR-01): this call sits between the two
    # phase try/except blocks and must never be allowed to raise a bare,
    # untyped exception, abandon the private snapshot, or skip activity
    # logging - mirrored on the resolution phase's own handling above.
    try:
        mark_resolved(project_dir, attempt.id, semantic_key=semantic_key, resolved=resolved)
    except (KeyboardInterrupt, SystemExit) as interrupt:
        _release_error_frames(interrupt)
        _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
        persistence_error: Optional[Exception] = None
        try:
            mark_resolution_failed(
                project_dir,
                attempt.id,
                error=LigandPromotionError(
                    phase="resolution",
                    code="consistency-error",
                    message=_describe_error(interrupt),
                ),
            )
        except Exception as error:
            persistence_error = error
        _record_activity_safely(project_dir, outcome="error", details=activity_details)
        if persistence_error is not None:
            raise interrupt from persistence_error
        raise
    except Exception as error:
        _release_error_frames(error)
        _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
        try:
            mark_resolution_failed(
                project_dir,
                attempt.id,
                error=LigandPromotionError(
                    phase="resolution",
                    code="promotion-persistence-failed",
                    message=_describe_error(error),
                ),
            )
        except Exception as persistence_error:
            _record_activity_safely(project_dir, outcome="error", details=activity_details)
            raise LigandPromotionAttemptPersistenceError(
                attempt_id=attempt.id,
                input_id=input_id,
                phase="resolution",
                intended_status="resolution_failed",
                original_error=error,
                persistence_error=persistence_error,
            ) from persistence_error

        _record_activity_safely(project_dir, outcome="error", details=activity_details)
        raise LigandPromotionAttemptFailedError(
            attempt_id=attempt.id, input_id=input_id, status="resolution_failed", original_error=error
        ) from error

    # ============================= execution =================================
    owner_token = secrets.token_hex(32)
    progress = _AcquisitionProgress()
    record: Optional[ScientificInputPromotionRecord] = None
    record_committed = False

    try:
        record = _load_consistent_promotion(
            project_dir, semantic_key=semantic_key, entity_id=entity_id, state_id=state_id
        )
        if record is None:
            record = _execute_new_promotion(
                project_dir,
                attempt=attempt,
                input_id=input_id,
                entity_id=entity_id,
                state_id=state_id,
                entity_name=entity_name,
                policy=policy,
                intentional_revision=intentional_revision,
                semantic_key=semantic_key,
                resolved=resolved,
                ligand=ligand,
                owner_token=owner_token,
                progress=progress,
            )
        record_committed = True

        result = PromotionResultSnapshot(
            promotion_id=record.id,
            entity_id=record.entity_id,
            state_id=record.state_id,
            representation=record.representation,
            entity_created=progress.entity_created,
            state_created=progress.state_created,
            reused_existing_promotion=not progress.state_created,
        )

        try:
            succeeded = mark_succeeded(project_dir, attempt.id, result=result)
        except Exception as persist_error:
            # Checkpoint 4 correction (BLOCKER-01): `register_promotion`
            # already returned successfully above - the
            # ScientificInputPromotionRecord, its ScientificEntity, and its
            # MolecularState are ALL durably committed. Section 13.3's own
            # last sentence is explicit and unconditional: "depois que o
            # record foi comprometido, nenhuma compensação remove record,
            # state ou entity; falha ao gravar status do attempt produz
            # erro de persistência operacional." This is that exact
            # scenario - the only thing that failed is writing the
            # ATTEMPT's own bookkeeping status, never the science itself.
            # Reclassifying the attempt as `execution_failed` here would
            # assert a false, permanent audit claim that the promotion
            # failed when it did not - so this NEVER calls
            # mark_execution_failed (nor mark_resolution_failed), NEVER
            # compensates entity/state/reservations, and NEVER touches the
            # commit marker. The attempt simply stays at its last honestly-
            # persisted status ("resolved") - never a false terminal one -
            # while the caller is told directly, via a persistence error,
            # that only the bookkeeping write failed.
            _release_error_frames(persist_error)
            _release_reservations_best_effort(
                project_dir,
                attempt_id=attempt.id,
                owner_token=owner_token,
                semantic_key=semantic_key if progress.semantic_reserved else None,
                entity_id=entity_id if progress.entity_claimed else None,
                state_id=state_id if progress.state_claimed else None,
            )
            _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
            _record_activity_safely(project_dir, outcome="error", details=activity_details)
            raise LigandPromotionAttemptPersistenceError(
                attempt_id=attempt.id,
                input_id=input_id,
                phase="execution",
                intended_status="succeeded",
                original_error=persist_error,
                persistence_error=persist_error,
            ) from persist_error

    except (KeyboardInterrupt, SystemExit) as interrupt:
        _release_error_frames(interrupt)
        if record_committed:
            _release_reservations_best_effort(
                project_dir,
                attempt_id=attempt.id,
                owner_token=owner_token,
                semantic_key=semantic_key if progress.semantic_reserved else None,
                entity_id=entity_id if progress.entity_claimed else None,
                state_id=state_id if progress.state_claimed else None,
            )
            _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
            _record_activity_safely(project_dir, outcome="error", details=activity_details)
            raise

        residual_entity_ids, residual_state_ids, residual_reservation_keys = _compensate_execution_failure(
            project_dir,
            attempt_id=attempt.id,
            owner_token=owner_token,
            semantic_key=semantic_key,
            entity_id=entity_id,
            state_id=state_id,
            progress=progress,
        )
        _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
        persistence_error = None
        try:
            mark_execution_failed(
                project_dir,
                attempt.id,
                error=LigandPromotionError(
                    phase="execution",
                    code="publication-failed",
                    message=_describe_error(interrupt),
                    residual_entity_ids=residual_entity_ids,
                    residual_state_ids=residual_state_ids,
                    residual_reservation_keys=residual_reservation_keys,
                ),
            )
        except Exception as error:
            persistence_error = error
        _record_activity_safely(project_dir, outcome="error", details=activity_details)
        if persistence_error is not None:
            raise interrupt from persistence_error
        raise

    except Exception as error:
        if record_committed:
            raise

        _release_error_frames(error)
        residual_entity_ids, residual_state_ids, residual_reservation_keys = _compensate_execution_failure(
            project_dir,
            attempt_id=attempt.id,
            owner_token=owner_token,
            semantic_key=semantic_key,
            entity_id=entity_id,
            state_id=state_id,
            progress=progress,
        )
        _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
        code = _execution_error_code(error)
        try:
            mark_execution_failed(
                project_dir,
                attempt.id,
                error=LigandPromotionError(
                    phase="execution",
                    code=code,
                    message=_describe_error(error),
                    residual_entity_ids=residual_entity_ids,
                    residual_state_ids=residual_state_ids,
                    residual_reservation_keys=residual_reservation_keys,
                ),
            )
        except Exception as persistence_error:
            _record_activity_safely(project_dir, outcome="error", details=activity_details)
            raise LigandPromotionAttemptPersistenceError(
                attempt_id=attempt.id,
                input_id=input_id,
                phase="execution",
                intended_status="execution_failed",
                original_error=error,
                persistence_error=persistence_error,
                residual_entity_ids=residual_entity_ids,
                residual_state_ids=residual_state_ids,
                residual_reservation_keys=residual_reservation_keys,
            ) from persistence_error

        _record_activity_safely(project_dir, outcome="error", details=activity_details)
        raise LigandPromotionAttemptFailedError(
            attempt_id=attempt.id, input_id=input_id, status="execution_failed", original_error=error
        ) from error

    # ---- success: best-effort release + cleanup, never affecting outcome ----
    _release_reservations_best_effort(
        project_dir,
        attempt_id=attempt.id,
        owner_token=owner_token,
        semantic_key=semantic_key if progress.semantic_reserved else None,
        entity_id=entity_id if progress.entity_claimed else None,
        state_id=state_id if progress.state_claimed else None,
    )
    _cleanup_snapshot_root(_snapshot_root(project_dir, attempt.id))
    _record_activity_safely(project_dir, outcome="success", details=activity_details)

    return succeeded
