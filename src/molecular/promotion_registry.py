"""Ligand promotion attempt and promotion-record persistence.

Stores LigandPromotionAttempts under
project_dir/scientific/promotion_attempts and ScientificInputPromotionRecords
under project_dir/scientific/promotions (Section 21.3 of the frozen spec),
kept entirely separate from project_dir/scientific/{entities,states,
conformers,execution_attempts}.

This module is Checkpoint 1 of Increment 018's frozen final specification
(``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``,
SHA-256 ``347632175991b1d72b9f193312b9629ce71697fb0c3d0de2285e52b7299d8023``).
It implements the attempt lifecycle (Section 6.9's status matrix) and the
promotion record's create-once commit-marker guarantee (Section 6.8/11.9) -
NOT the cross-process semantic reservation or entity/state claims of
Section 11.2-11.14 (`os.mkdir`-based reservation directories,
`attempt_id`+`owner_token` ownership, TTL/BUSY semantics). Those require
FS-018-02's full reservation protocol and are a later, dedicated checkpoint;
nothing in this module attempts cross-process mutual exclusion for
*starting* a promotion, only single-writer-at-a-time correctness for the
persisted records it does own (an attempt never mutates in place - each
transition constructs a whole new validated snapshot and atomically
overwrites the previous one, exactly like execution_registry.py).

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
silently overwritten. If the platform cannot provide `os.link` on the
directory in question, publication fails closed (`OSError` propagates)
rather than falling back to a weaker, non-exclusive write.
"""

import contextlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

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


def _promotion_attempts_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "promotion_attempts"


def _promotions_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "promotions"


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
    """

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
