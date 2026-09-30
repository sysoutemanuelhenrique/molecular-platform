"""Ligand promotion attempt, promotion-record, and cross-process
reservation/claim persistence.

Stores LigandPromotionAttempts under
project_dir/scientific/promotion_attempts, ScientificInputPromotionRecords
under project_dir/scientific/promotions, semantic reservations under
project_dir/scientific/promotion_reservations/<semantic_key>, and
entity/state claims under
project_dir/scientific/promotion_reservations/by-{entity,state}/<claim hash>
(Section 21.3 / 11.2 / 11.4 of the frozen spec), kept entirely separate from
project_dir/scientific/{entities,states,conformers,execution_attempts}.

This module spans Checkpoint 1 (attempt lifecycle, Section 6.9's status
matrix; promotion record create-once commit-marker guarantee, Section
6.8/11.9) and Checkpoint 2 (cross-process semantic reservation and
entity/state claims, Section 11.2-11.10 - `os.mkdir`-based exclusive
reservation directories, `attempt_id`+`owner_token` ownership, casefold
claim normalization, owner-only release, reverse-order compensation of a
partial acquisition) of Increment 018's frozen final specification
(``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``,
SHA-256 ``347632175991b1d72b9f193312b9629ce71697fb0c3d0de2285e52b7299d8023``).

Checkpoint 2 implements exactly the *primitives* FS-018-02 proved sufficient
- `acquire_semantic_reservation`/`acquire_entity_claim`/`acquire_state_claim`
(exclusive `os.mkdir`, never check-then-create), `release_*` (owner-only,
exact `attempt_id`+`owner_token` match required), `describe_*` (read-only
inspection - never mutates, never reports a residue as free), and
`compensate_partial_acquisition` (reverse-order release of exactly what one
attempt already acquired, Section 11.10). It deliberately does NOT implement:
any TTL-based takeover (Section 11.7 - there is, by design, no code path
anywhere in this module that removes a reservation/claim because of its
age); the Application Service's full entity/state compensation (Section 13,
a later checkpoint); or the administrative reconciliation API named in
Section 11.14 (`inspect_promotion_reservation`/`PromotionReservationInspection`
- explicitly out of scope for the whole Increment, never built here under
that name or with removal authority).

Every attempt transition (mark_resolved, mark_resolution_failed,
mark_execution_failed, mark_succeeded) follows the same protocol as
execution_registry.py: load the current persisted snapshot, validate that
the requested transition is reachable from its current status, construct a
brand new fully-validated LigandPromotionAttempt, and atomically overwrite
the previous snapshot with it.

register_promotion publishes a ScientificInputPromotionRecord by the
mechanism Section 11.9 of the frozen spec confirms sufficient on the
Windows/NTFS local platform this Increment validates: the complete document
is written to a private staging file first, then linked into its final
`<semantic_key>.yaml` name via `os.link`, which fails with FileExistsError
if that name is already taken - there is no window in which partial content
is visible under the final name, and an already-published record is never
silently overwritten. If the platform cannot provide `os.link` at all, or
`os.link` fails for any reason other than an existing destination,
publication fails closed with `PromotionPublicationUnavailableError`
(Section 17's `promotion-primitive-unavailable`) rather than falling back to
a weaker, non-exclusive write.
"""

import contextlib
import hashlib
import os
import shutil
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Type

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from molecular.atomic import atomic_write_text, check_storage_identity_conflict
from molecular.inputs import validate_input_id
from molecular.promotion import (
    LigandPromotionAttempt,
    LigandPromotionError,
    LigandPromotionPolicy,
    PromotionResultSnapshot,
    ResolvedLigandPromotionSnapshot,
    ScientificInputPromotionRecord,
    validate_semantic_key,
)


class LigandPromotionAttemptIdentityConflictError(FileExistsError):
    """Raised if a freshly generated attempt id (a uuid4 hex) collides with
    an already-persisted attempt - astronomically unlikely, but
    begin_promotion_attempt must never silently overwrite a pre-existing
    attempt file. Mirrors ExecutionAttemptIdentityConflictError in
    molecular.execution_registry. Create-only: the four mark_* transitions
    are expected, intended overwrites of that same attempt's own file and
    are not guarded by this check."""


class PromotionRecordConflictError(FileExistsError):
    """Raised when a ScientificInputPromotionRecord already exists for the
    given semantic_key. The promotion record is the commit marker of a
    completed promotion (Section 6.8/11.9 of the frozen spec) and must
    never be silently overwritten - not on first publication, and not on
    any later retry."""


class PromotionPublicationUnavailableError(OSError):
    """promotion-primitive-unavailable (Section 17 of the frozen spec).
    Raised when the platform does not provide `os.link` at all, or `os.link`
    fails for any reason other than an already-existing destination
    (FileExistsError, handled separately as PromotionRecordConflictError).
    Exclusive publication cannot be proven in either case, so the promotion
    record is never published without that guarantee - no weaker,
    non-exclusive fallback write is ever attempted."""


class SemanticReservationConflictError(FileExistsError):
    """Raised when `os.mkdir` on a semantic reservation's slot directory
    fails because it already exists - Section 11.2 of the frozen spec.
    There is no check-then-create step: `os.mkdir` is the sole arbiter, and
    this exception is the only possible outcome of losing the race, whether
    the winner is still alive, has crashed, or has already published a
    promotion record - distinguishing those cases is the job of
    `describe_semantic_reservation`/`load_promotion`, never of the acquire
    call itself."""


class EntityClaimConflictError(FileExistsError):
    """Raised when `os.mkdir` on an entity claim's slot directory fails
    because it already exists - Section 11.4 of the frozen spec. The slot
    is named by the SHA-256 of the NFC-normalized, casefolded entity_id, so
    this also fires for a casefold-equivalent entity_id claimed by a
    different semantic_key."""


class StateClaimConflictError(FileExistsError):
    """Raised when `os.mkdir` on a state claim's slot directory fails
    because it already exists - Section 11.4 of the frozen spec. Mirrors
    EntityClaimConflictError for state_id claims."""


class ReservationOwnershipError(PermissionError):
    """Raised when a release is attempted with an `attempt_id`/`owner_token`
    pair that does not match the resource's recorded owner (Section 11.8 of
    the frozen spec). The resource is left completely untouched - a
    non-owner never removes another process's reservation or claim."""


class ReservationMetadataUnavailableError(LookupError):
    """Raised when a release cannot be confirmed because the resource's
    metadata is missing or unreadable at release time - ownership cannot be
    proven either way, so the resource is left untouched (Section 11.6/11.10
    of the frozen spec: absence or invalid metadata is BUSY/INDETERMINATE,
    never grounds for removal). The caller (Section 11.10's reverse-order
    compensation, or a future LigandPromotionError.residual_reservation_keys)
    must record the resource's raw key as a residual, never assume it was
    released."""


# The complete set of states a reservation/claim slot can report to a
# strictly read-only inspector (Section 11.6 of the frozen spec). A missing
# slot is the only state that means "free" - every other state, including
# both *_INDETERMINATE variants, means the slot must not be treated as
# available.
_ReservationState = Literal[
    "ABSENT_FREE",
    "RESERVED_KNOWN_OWNER",
    "RESERVED_METADATA_MISSING_INDETERMINATE",
    "RESERVED_METADATA_CORRUPT_INDETERMINATE",
]


class ReservationStatus(BaseModel):
    """The read-only status of one reservation/claim slot (Section 11.6 of
    the frozen spec). Never mutates anything. `owner_token` is deliberately
    never a field on this model - Section 11.3: it must never be exposed by
    any report. `age_seconds`, when present, is purely informational: no
    code anywhere in this module compares it to a limit to authorize
    takeover or removal (Section 11.7 - TTL never grants ownership)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: _ReservationState
    owner_attempt_id: Optional[str] = None
    phase: Optional[str] = None
    age_seconds: Optional[float] = None


class _ReservationMetadata(BaseModel):
    """Internal, unexported metadata shape written to a reservation/claim
    slot's `reservation.yaml` (Section 11.3 of the frozen spec). Not part
    of the frozen domain model (Section 6) - this is purely a persistence
    detail of the reservation/claim mechanism, never a public contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: str
    owner_token: str
    phase: str
    raw_key: str
    created_at: datetime


def _promotion_attempts_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "promotion_attempts"


def _promotions_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "promotions"


def _promotion_reservations_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "promotion_reservations"


def _entity_claims_dir(project_dir: Path) -> Path:
    return _promotion_reservations_dir(project_dir) / "by-entity"


def _state_claims_dir(project_dir: Path) -> Path:
    return _promotion_reservations_dir(project_dir) / "by-state"


def _normalize_claim_key(value: str) -> str:
    """Normalize a raw entity_id/state_id for claim hashing (Section 11.4
    of the frozen spec, in this exact order): Unicode NFC, then
    `str.casefold()`. `casefold()` is a fixed Unicode algorithm with no
    locale dependency whatsoever - unlike `str.lower()`/`str.upper()`,
    which can vary by the active OS locale (e.g. Turkish `I`/`i`)."""

    return unicodedata.normalize("NFC", value).casefold()


def _claim_hash(value: str) -> str:
    """SHA-256 hex digest of the NFC-normalized, casefolded `value`, encoded
    as UTF-8 - the sole identity of an entity/state claim slot (Section
    11.4). A fixed, explicit encoding is used throughout; nothing here
    depends on the platform's default text encoding."""

    normalized = _normalize_claim_key(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _acquire_slot(
    slot_dir: Path,
    *,
    attempt_id: str,
    owner_token: str,
    raw_key: str,
    phase: str,
    conflict_error: Type[FileExistsError],
) -> None:
    """Acquire one reservation/claim slot by exclusive directory creation -
    the sole arbiter, with no check-then-create step of any kind (Section
    11.2 of the frozen spec). `slot_dir`'s parent is created on demand
    (idempotently, `exist_ok=True` - safe under concurrent callers) before
    the exclusive `os.mkdir` on `slot_dir` itself, which is the only
    operation that can race. Metadata is written only after the exclusive
    create has already succeeded, via a temp file plus `os.replace` inside
    the same directory (molecular.atomic.atomic_write_text) - there is no
    window in which the directory exists with partially-written metadata
    visible to another process."""

    slot_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.mkdir(slot_dir)
    except FileExistsError as exc:
        raise conflict_error(
            f"A reservation/claim already exists at {slot_dir} "
            f"(phase={phase!r})."
        ) from exc

    metadata = _ReservationMetadata(
        attempt_id=attempt_id,
        owner_token=owner_token,
        phase=phase,
        raw_key=raw_key,
        created_at=datetime.now(timezone.utc),
    )
    metadata_file = slot_dir / "reservation.yaml"
    serialized_yaml = yaml.safe_dump(metadata.model_dump(mode="json"), sort_keys=False)
    atomic_write_text(metadata_file, serialized_yaml)


def _read_slot_metadata(slot_dir: Path) -> _ReservationMetadata:
    """Read and validate a slot's metadata, or raise
    ReservationMetadataUnavailableError - never returns a partial or
    best-guess result, and never lets a raw filesystem exception escape.
    Covers three distinct causes uniformly, all folded into the same typed
    "cannot confirm" outcome: the metadata file is missing
    (FileNotFoundError, an OSError subclass), a generic I/O failure while
    reading it (any other OSError/PermissionError - e.g. a transient
    permission or disk error, distinct from "file absent"), or the content
    being unreadable as valid metadata (yaml.YAMLError/ValidationError).
    Used by both `_release_slot` (Section 11.8) and `_inspect_slot`
    (Section 11.6), which then decide independently what to do with an
    unreadable result (refuse to release vs. report INDETERMINATE) -
    `_inspect_slot` further distinguishes "file absent" from "file present
    but unreadable" by checking existence itself, since this function
    intentionally reports both as the same exception type."""

    metadata_file = slot_dir / "reservation.yaml"
    try:
        data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        return _ReservationMetadata.model_validate(data)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ReservationMetadataUnavailableError(
            f"Metadata at {metadata_file} is missing or unreadable; "
            "ownership cannot be confirmed."
        ) from exc


def _release_slot(slot_dir: Path, *, attempt_id: str, owner_token: str) -> None:
    """Release one reservation/claim slot - Section 11.8 of the frozen
    spec. Requires an exact match of both `attempt_id` and `owner_token`
    against the slot's recorded metadata; any mismatch, or any failure to
    read the metadata at all, leaves the slot completely untouched and
    raises instead of guessing. Only `shutil.rmtree` (the actual removal)
    can still raise a genuine, unexpected OSError - that is a real
    infrastructure fault and is allowed to propagate as-is, never absorbed
    into one of the two typed "could not confirm" exceptions."""

    metadata = _read_slot_metadata(slot_dir)

    if metadata.attempt_id != attempt_id or metadata.owner_token != owner_token:
        raise ReservationOwnershipError(
            f"attempt_id/owner_token do not match the recorded owner of "
            f"{slot_dir}; the resource is left untouched."
        )

    shutil.rmtree(slot_dir)


def _inspect_slot(slot_dir: Path) -> ReservationStatus:
    """Strictly read-only status of one slot (Section 11.6 of the frozen
    spec). Never creates, deletes, or repairs anything. A slot directory
    that exists but whose metadata cannot be read is never reported as
    free - it is RESERVED_..._INDETERMINATE, distinct from both
    ABSENT_FREE and RESERVED_KNOWN_OWNER.

    ABSENT_FREE is reported only for a *confirmed* absence: `slot_dir.
    exists()` returning `False` normally. If the filesystem itself cannot
    answer that question (`slot_dir.exists()` raising `OSError`/
    `PermissionError`), this is conservatively folded into
    RESERVED_METADATA_CORRUPT_INDETERMINATE - the platform could not
    inspect the slot's identity, which is never grounds for treating it as
    free. The same conservative rule applies one level down: if metadata
    is unreadable and even `metadata_file.exists()` itself raises `OSError`
    (so it cannot be determined whether the metadata is merely missing or
    present-but-corrupt), the result is RESERVED_METADATA_CORRUPT_
    INDETERMINATE, never MISSING and never ABSENT_FREE. No exception from
    either existence check is ever allowed to escape this function."""

    try:
        slot_exists = slot_dir.exists()
    except OSError:
        return ReservationStatus(state="RESERVED_METADATA_CORRUPT_INDETERMINATE")

    if not slot_exists:
        return ReservationStatus(state="ABSENT_FREE")

    try:
        metadata = _read_slot_metadata(slot_dir)
    except ReservationMetadataUnavailableError:
        metadata_file = slot_dir / "reservation.yaml"
        try:
            metadata_exists = metadata_file.exists()
        except OSError:
            return ReservationStatus(state="RESERVED_METADATA_CORRUPT_INDETERMINATE")
        if not metadata_exists:
            return ReservationStatus(state="RESERVED_METADATA_MISSING_INDETERMINATE")
        return ReservationStatus(state="RESERVED_METADATA_CORRUPT_INDETERMINATE")

    age_seconds = (datetime.now(timezone.utc) - metadata.created_at).total_seconds()
    return ReservationStatus(
        state="RESERVED_KNOWN_OWNER",
        owner_attempt_id=metadata.attempt_id,
        phase=metadata.phase,
        age_seconds=age_seconds,
    )


def acquire_semantic_reservation(
    project_dir: Path, semantic_key: str, *, attempt_id: str, owner_token: str
) -> None:
    """Acquire the exclusive semantic reservation for `semantic_key`
    (Section 11.2 of the frozen spec) - the first, mandatory step of the
    semantic reservation -> entity claim -> state claim acquisition order
    (Section 11.5). Raises SemanticReservationConflictError if another
    attempt already holds it (alive, crashed, or already promoted - this
    function never distinguishes those; see describe_semantic_reservation
    and load_promotion for that)."""

    validate_semantic_key(semantic_key)
    slot_dir = _promotion_reservations_dir(project_dir) / semantic_key
    _acquire_slot(
        slot_dir,
        attempt_id=attempt_id,
        owner_token=owner_token,
        raw_key=semantic_key,
        phase="semantic_reserved",
        conflict_error=SemanticReservationConflictError,
    )


def release_semantic_reservation(
    project_dir: Path, semantic_key: str, *, attempt_id: str, owner_token: str
) -> None:
    """Release the semantic reservation for `semantic_key` - owner-only,
    Section 11.8 of the frozen spec."""

    validate_semantic_key(semantic_key)
    slot_dir = _promotion_reservations_dir(project_dir) / semantic_key
    _release_slot(slot_dir, attempt_id=attempt_id, owner_token=owner_token)


def describe_semantic_reservation(project_dir: Path, semantic_key: str) -> ReservationStatus:
    """Read-only status of the semantic reservation for `semantic_key` -
    Section 11.6 of the frozen spec. Never mutates."""

    validate_semantic_key(semantic_key)
    slot_dir = _promotion_reservations_dir(project_dir) / semantic_key
    return _inspect_slot(slot_dir)


def acquire_entity_claim(
    project_dir: Path, entity_id: str, *, attempt_id: str, owner_token: str
) -> None:
    """Acquire the exclusive claim for `entity_id`, normalized by Unicode
    NFC + casefold before hashing (Section 11.4 of the frozen spec). Raises
    EntityClaimConflictError if a casefold-equivalent entity_id is already
    claimed, regardless of which semantic_key claimed it."""

    validate_input_id(entity_id)
    slot_dir = _entity_claims_dir(project_dir) / _claim_hash(entity_id)
    _acquire_slot(
        slot_dir,
        attempt_id=attempt_id,
        owner_token=owner_token,
        raw_key=entity_id,
        phase="entity_claimed",
        conflict_error=EntityClaimConflictError,
    )


def release_entity_claim(
    project_dir: Path, entity_id: str, *, attempt_id: str, owner_token: str
) -> None:
    """Release the claim for `entity_id` - owner-only, Section 11.8 of the
    frozen spec."""

    validate_input_id(entity_id)
    slot_dir = _entity_claims_dir(project_dir) / _claim_hash(entity_id)
    _release_slot(slot_dir, attempt_id=attempt_id, owner_token=owner_token)


def describe_entity_claim(project_dir: Path, entity_id: str) -> ReservationStatus:
    """Read-only status of the claim for `entity_id` - Section 11.6 of the
    frozen spec. Never mutates."""

    validate_input_id(entity_id)
    slot_dir = _entity_claims_dir(project_dir) / _claim_hash(entity_id)
    return _inspect_slot(slot_dir)


def acquire_state_claim(
    project_dir: Path, state_id: str, *, attempt_id: str, owner_token: str
) -> None:
    """Acquire the exclusive claim for `state_id`, normalized by Unicode
    NFC + casefold before hashing (Section 11.4 of the frozen spec). Raises
    StateClaimConflictError if a casefold-equivalent state_id is already
    claimed, regardless of which semantic_key claimed it."""

    validate_input_id(state_id)
    slot_dir = _state_claims_dir(project_dir) / _claim_hash(state_id)
    _acquire_slot(
        slot_dir,
        attempt_id=attempt_id,
        owner_token=owner_token,
        raw_key=state_id,
        phase="state_claimed",
        conflict_error=StateClaimConflictError,
    )


def release_state_claim(
    project_dir: Path, state_id: str, *, attempt_id: str, owner_token: str
) -> None:
    """Release the claim for `state_id` - owner-only, Section 11.8 of the
    frozen spec."""

    validate_input_id(state_id)
    slot_dir = _state_claims_dir(project_dir) / _claim_hash(state_id)
    _release_slot(slot_dir, attempt_id=attempt_id, owner_token=owner_token)


def describe_state_claim(project_dir: Path, state_id: str) -> ReservationStatus:
    """Read-only status of the claim for `state_id` - Section 11.6 of the
    frozen spec. Never mutates."""

    validate_input_id(state_id)
    slot_dir = _state_claims_dir(project_dir) / _claim_hash(state_id)
    return _inspect_slot(slot_dir)


def compensate_partial_acquisition(
    project_dir: Path,
    *,
    attempt_id: str,
    owner_token: str,
    semantic_key: Optional[str] = None,
    entity_id: Optional[str] = None,
    state_id: Optional[str] = None,
) -> Tuple[str, ...]:
    """Release, in reverse order of acquisition, exactly the resources this
    one attempt already holds (Section 11.10 of the frozen spec): state
    claim, then entity claim, then semantic reservation. Pass only the ids
    actually acquired (`None` for anything never attempted or already lost
    at a claim stage) - each of the three is attempted **independently**:
    a failure releasing one never skips the attempt to release the others,
    in either direction (an earlier failure does not stop a later one from
    being tried, and vice versa).

    Never raises: a release that cannot be confirmed contributes its raw
    key (the same shape LigandPromotionError.residual_reservation_keys
    validates) to the returned tuple instead of propagating. Three distinct
    causes are folded into the same "could not confirm, treat as residual"
    outcome here - and only here, never inside `_release_slot`/`release_*`
    themselves, whose direct callers still see a real infrastructure
    failure as the `OSError` it is:

    1. ownership mismatch (`ReservationOwnershipError`);
    2. metadata missing or unreadable at release time
       (`ReservationMetadataUnavailableError`);
    3. a genuine infrastructure fault during the removal itself - e.g.
       `shutil.rmtree` failing with `OSError`/`PermissionError` because
       another process holds an open handle, a transient disk/I-O error,
       or a filesystem permission problem. Ownership was already confirmed
       in this case; what failed is the removal, so the caller cannot
       assume the resource was released, nor that it still fully belongs
       to this attempt either (`rmtree` may have partially removed it) -
       treating it as a residual, exactly like the other two causes, is
       the only outcome that neither overclaims success nor aborts the
       compensation of the other, independent resources.

    This function never removes a resource belonging to another owner
    (`ReservationOwnershipError` is exactly the guard against that, in
    every one of the three blocks below) and never claims a resource was
    released when it could not confirm that it was - an empty return means
    full compensation; any entry means that resource is now a residual the
    caller must record, never silently retry or assume gone.
    """

    residuals: List[str] = []

    if state_id is not None:
        try:
            release_state_claim(
                project_dir, state_id, attempt_id=attempt_id, owner_token=owner_token
            )
        except (ReservationOwnershipError, ReservationMetadataUnavailableError, OSError):
            residuals.append(state_id)

    if entity_id is not None:
        try:
            release_entity_claim(
                project_dir, entity_id, attempt_id=attempt_id, owner_token=owner_token
            )
        except (ReservationOwnershipError, ReservationMetadataUnavailableError, OSError):
            residuals.append(entity_id)

    if semantic_key is not None:
        try:
            release_semantic_reservation(
                project_dir, semantic_key, attempt_id=attempt_id, owner_token=owner_token
            )
        except (ReservationOwnershipError, ReservationMetadataUnavailableError, OSError):
            residuals.append(semantic_key)

    return tuple(residuals)


def _persist_attempt(project_dir: Path, attempt: LigandPromotionAttempt) -> None:
    metadata_file = _promotion_attempts_dir(project_dir) / f"{attempt.id}.yaml"
    serialized_yaml = yaml.safe_dump(attempt.model_dump(mode="json"), sort_keys=False)
    atomic_write_text(metadata_file, serialized_yaml)


def begin_promotion_attempt(
    project_dir: Path,
    *,
    input_id: str,
    requested_entity_id: str,
    requested_state_id: str,
    requested_entity_name: str,
    requested: LigandPromotionPolicy,
    intentional_revision: Optional[str] = None,
) -> LigandPromotionAttempt:
    """Begin a new LigandPromotionAttempt in status 'started'.

    Validates input_id/requested_entity_id/requested_state_id/
    intentional_revision before any I/O, exactly like
    begin_execution_attempt validates input_state_id in
    molecular.execution_registry. Never checks that input_id resolves to an
    actually-registered LigandInput: that check belongs to the Application
    Service's resolution phase (Section 12.2, step 8 of the frozen spec),
    which runs strictly after this function returns - this function only
    ever records the platform's *intent* to promote.
    """

    validate_input_id(input_id)
    validate_input_id(requested_entity_id)
    validate_input_id(requested_state_id)
    if intentional_revision is not None:
        validate_input_id(intentional_revision)

    attempts_dir = _promotion_attempts_dir(project_dir)
    attempt_id = uuid.uuid4().hex

    # Guard the create path only: refuse to silently clobber a pre-existing
    # attempt file on the astronomically unlikely event of a uuid4
    # collision. atomic_write_text's underlying os.replace() would
    # otherwise overwrite an existing destination unconditionally.
    check_storage_identity_conflict(
        attempts_dir,
        attempt_id,
        conflict_error=LigandPromotionAttemptIdentityConflictError,
    )

    attempt = LigandPromotionAttempt(
        id=attempt_id,
        input_id=input_id,
        requested_entity_id=requested_entity_id,
        requested_state_id=requested_state_id,
        requested_entity_name=requested_entity_name,
        intentional_revision=intentional_revision,
        requested=requested,
        status="started",
        semantic_key=None,
        resolved=None,
        result=None,
        error=None,
        created_at=datetime.now(timezone.utc),
        resolved_at=None,
        finished_at=None,
    )

    _persist_attempt(project_dir, attempt)

    return attempt


def load_promotion_attempt(project_dir: Path, attempt_id: str) -> LigandPromotionAttempt:
    """Load a previously persisted LigandPromotionAttempt by id.

    Raises FileNotFoundError if no such attempt is registered - mirroring
    load_execution_attempt in molecular.execution_registry. Propagates
    pydantic.ValidationError for an internally inconsistent record and
    yaml.YAMLError for corrupted YAML, in both cases unmodified - this
    function never repairs or reinterprets what it reads.
    """

    validate_input_id(attempt_id)

    metadata_file = _promotion_attempts_dir(project_dir) / f"{attempt_id}.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return LigandPromotionAttempt.model_validate(data)


def mark_resolved(
    project_dir: Path,
    attempt_id: str,
    *,
    semantic_key: str,
    resolved: ResolvedLigandPromotionSnapshot,
) -> LigandPromotionAttempt:
    """Transition an attempt from 'started' to 'resolved', assigning its
    semantic_key (Section 10.2) and resolved snapshot together - the
    matrix (Section 6.9) requires both or neither.

    Only a 'started' attempt may resolve. Any other current status raises
    ValueError; a failed attempt is never reopened, only retried via a
    brand new LigandPromotionAttempt.
    """

    current = load_promotion_attempt(project_dir, attempt_id)

    if current.status != "started":
        raise ValueError(
            f"Cannot mark attempt '{attempt_id}' as resolved from status "
            f"'{current.status}' - only a 'started' attempt may resolve."
        )

    updated = LigandPromotionAttempt(
        schema_version=current.schema_version,
        id=current.id,
        input_id=current.input_id,
        requested_entity_id=current.requested_entity_id,
        requested_state_id=current.requested_state_id,
        requested_entity_name=current.requested_entity_name,
        intentional_revision=current.intentional_revision,
        requested=current.requested,
        status="resolved",
        semantic_key=semantic_key,
        resolved=resolved,
        result=None,
        error=None,
        created_at=current.created_at,
        resolved_at=datetime.now(timezone.utc),
        finished_at=None,
    )

    _persist_attempt(project_dir, updated)

    return updated


def mark_resolution_failed(
    project_dir: Path,
    attempt_id: str,
    *,
    error: LigandPromotionError,
) -> LigandPromotionAttempt:
    """Transition an attempt from 'started' to 'resolution_failed'.

    Only a 'started' attempt may fail resolution. `error.phase` must be
    'resolution' - enforced by LigandPromotionAttempt's own model
    validator, not duplicated here. No promotion record is ever created by
    this transition.
    """

    current = load_promotion_attempt(project_dir, attempt_id)

    if current.status != "started":
        raise ValueError(
            f"Cannot mark attempt '{attempt_id}' as resolution_failed from "
            f"status '{current.status}' - only a 'started' attempt may."
        )

    updated = LigandPromotionAttempt(
        schema_version=current.schema_version,
        id=current.id,
        input_id=current.input_id,
        requested_entity_id=current.requested_entity_id,
        requested_state_id=current.requested_state_id,
        requested_entity_name=current.requested_entity_name,
        intentional_revision=current.intentional_revision,
        requested=current.requested,
        status="resolution_failed",
        semantic_key=None,
        resolved=None,
        result=None,
        error=error,
        created_at=current.created_at,
        resolved_at=None,
        finished_at=datetime.now(timezone.utc),
    )

    _persist_attempt(project_dir, updated)

    return updated


def mark_execution_failed(
    project_dir: Path,
    attempt_id: str,
    *,
    error: LigandPromotionError,
) -> LigandPromotionAttempt:
    """Transition an attempt from 'resolved' to 'execution_failed',
    preserving its semantic_key, resolved snapshot, and resolved_at.

    Only a 'resolved' attempt may fail execution. `error.phase` must be
    'execution' - enforced by LigandPromotionAttempt's own model
    validator. No promotion record is ever created by this transition.
    """

    current = load_promotion_attempt(project_dir, attempt_id)

    if current.status != "resolved":
        raise ValueError(
            f"Cannot mark attempt '{attempt_id}' as execution_failed from "
            f"status '{current.status}' - only a 'resolved' attempt may."
        )

    updated = LigandPromotionAttempt(
        schema_version=current.schema_version,
        id=current.id,
        input_id=current.input_id,
        requested_entity_id=current.requested_entity_id,
        requested_state_id=current.requested_state_id,
        requested_entity_name=current.requested_entity_name,
        intentional_revision=current.intentional_revision,
        requested=current.requested,
        status="execution_failed",
        semantic_key=current.semantic_key,
        resolved=current.resolved,
        result=None,
        error=error,
        created_at=current.created_at,
        resolved_at=current.resolved_at,
        finished_at=datetime.now(timezone.utc),
    )

    _persist_attempt(project_dir, updated)

    return updated


def mark_succeeded(
    project_dir: Path,
    attempt_id: str,
    *,
    result: PromotionResultSnapshot,
) -> LigandPromotionAttempt:
    """Transition an attempt from 'resolved' to 'succeeded', preserving its
    semantic_key, resolved snapshot, and resolved_at, and assigning
    `result`.

    Only a 'resolved' attempt may succeed. This function does not itself
    publish a ScientificInputPromotionRecord and does not check that one
    exists for this attempt's semantic_key - registering the promotion
    record (register_promotion, below) is always a separate, prior step
    performed by the caller (the future Application Service), exactly like
    mark_succeeded in molecular.execution_registry never verifies its own
    operation_ids resolve to a real StateTransformationRecord.
    """

    current = load_promotion_attempt(project_dir, attempt_id)

    if current.status != "resolved":
        raise ValueError(
            f"Cannot mark attempt '{attempt_id}' as succeeded from status "
            f"'{current.status}' - only a 'resolved' attempt may."
        )

    updated = LigandPromotionAttempt(
        schema_version=current.schema_version,
        id=current.id,
        input_id=current.input_id,
        requested_entity_id=current.requested_entity_id,
        requested_state_id=current.requested_state_id,
        requested_entity_name=current.requested_entity_name,
        intentional_revision=current.intentional_revision,
        requested=current.requested,
        status="succeeded",
        semantic_key=current.semantic_key,
        resolved=current.resolved,
        result=result,
        error=None,
        created_at=current.created_at,
        resolved_at=current.resolved_at,
        finished_at=datetime.now(timezone.utc),
    )

    _persist_attempt(project_dir, updated)

    return updated


def register_promotion(
    project_dir: Path, record: ScientificInputPromotionRecord
) -> ScientificInputPromotionRecord:
    """Publish a ScientificInputPromotionRecord as the commit marker for
    `record.semantic_key`, refusing to overwrite an existing one.

    The complete YAML document is written to a private staging file first
    (never to the final name directly), then linked into place via
    `os.link` - which fails atomically with FileExistsError if the final
    name already exists. There is no window in which a partially-written
    document is visible under the final name: the staging file is fully
    written, flushed, and closed before the link is attempted.

    If the platform does not provide `os.link` at all, or `os.link` fails
    for any reason other than an existing destination, this raises
    PromotionPublicationUnavailableError (Section 17's
    `promotion-primitive-unavailable`) rather than falling back to a
    weaker, non-exclusive write - the exclusivity guarantee is never
    downgraded silently.
    """

    if not hasattr(os, "link"):
        raise PromotionPublicationUnavailableError(
            "This platform does not provide os.link; exclusive promotion "
            "record publication cannot be proven, so publication must fail "
            "closed rather than use a weaker, non-exclusive write."
        )

    promotions_dir = _promotions_dir(project_dir)
    promotions_dir.mkdir(parents=True, exist_ok=True)

    final_path = promotions_dir / f"{record.semantic_key}.yaml"
    if final_path.exists():
        raise PromotionRecordConflictError(
            f"A promotion record already exists for semantic_key "
            f"'{record.semantic_key}' at {final_path}."
        )

    staging_path = promotions_dir / f".{record.semantic_key}.{uuid.uuid4().hex}.tmp"
    serialized_yaml = yaml.safe_dump(record.model_dump(mode="json"), sort_keys=False)

    try:
        staging_path.write_text(serialized_yaml, encoding="utf-8")
        os.link(staging_path, final_path)
    except FileExistsError as exc:
        raise PromotionRecordConflictError(
            f"A promotion record already exists for semantic_key "
            f"'{record.semantic_key}' at {final_path}."
        ) from exc
    except OSError as exc:
        raise PromotionPublicationUnavailableError(
            f"os.link failed for a reason other than an existing "
            f"destination ({exc.__class__.__name__}: {exc}); exclusive "
            "publication cannot be proven, so publication must fail closed."
        ) from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            staging_path.unlink()

    return record


def load_promotion(project_dir: Path, semantic_key: str) -> ScientificInputPromotionRecord:
    """Load a previously published ScientificInputPromotionRecord by its
    semantic_key. Raises FileNotFoundError if none is published - mirroring
    load_state/load_entity in molecular.scientific_registry."""

    validate_semantic_key(semantic_key)

    metadata_file = _promotions_dir(project_dir) / f"{semantic_key}.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return ScientificInputPromotionRecord.model_validate(data)


def list_promotions_by_input(
    project_dir: Path, input_id: str
) -> List[ScientificInputPromotionRecord]:
    """List every published ScientificInputPromotionRecord whose input_id
    matches `input_id`, sorted by semantic_key. Returns an empty list if no
    promotions directory exists yet, or none match - a read-only scan,
    mirroring list_entity_states in molecular.scientific_registry."""

    promotions_dir = _promotions_dir(project_dir)

    if not promotions_dir.is_dir():
        return []

    matches = []

    for entry in sorted(promotions_dir.iterdir()):
        if not (entry.is_file() and entry.suffix == ".yaml"):
            continue

        data = yaml.safe_load(entry.read_text(encoding="utf-8"))
        record = ScientificInputPromotionRecord.model_validate(data)

        if record.input_id == input_id:
            matches.append(record)

    return matches


def list_promotions_by_state(
    project_dir: Path, state_id: str
) -> List[ScientificInputPromotionRecord]:
    """List every published ScientificInputPromotionRecord whose state_id
    matches `state_id`, sorted by semantic_key. Returns an empty list if no
    promotions directory exists yet, or none match - a read-only scan,
    mirroring list_state_conformers in molecular.scientific_registry."""

    promotions_dir = _promotions_dir(project_dir)

    if not promotions_dir.is_dir():
        return []

    matches = []

    for entry in sorted(promotions_dir.iterdir()):
        if not (entry.is_file() and entry.suffix == ".yaml"):
            continue

        data = yaml.safe_load(entry.read_text(encoding="utf-8"))
        record = ScientificInputPromotionRecord.model_validate(data)

        if record.state_id == state_id:
            matches.append(record)

    return matches
