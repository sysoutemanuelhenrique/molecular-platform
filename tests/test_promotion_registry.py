"""Persistence tests for Increment 018 Checkpoints 1 and 2.

Checkpoint 1 (items 25-33, 37-38 of Section 19.2 of the frozen final spec):
LigandPromotionAttempt lifecycle (begin/load/transitions/atomic overwrite)
and ScientificInputPromotionRecord create-once publication and queries.

Checkpoint 2 (items 34-36 of Section 19.2, plus all of Section 19.9, items
151-169): exclusive `os.mkdir` semantic reservation, casefold-normalized
entity/state claims, `attempt_id`+`owner_token` ownership, owner-only
release, reverse-order compensation of a partial acquisition, TTL/takeover
prohibition, and the cross-process guarantees FS-018-02 proved sufficient -
exercised here with real OS processes (`multiprocessing`, Windows-spawn
compatible), never threads, per Section 11.1's own methodology.

``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``.
"""

import json
import multiprocessing
import os
import secrets
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from molecular.promotion import (
    AcquisitionProvenanceSnapshot,
    LigandPromotionError,
    LigandPromotionPolicy,
    PromotionResultSnapshot,
    ResolvedLigandPromotionSnapshot,
    ScientificInputPromotionRecord,
)
from molecular.promotion_registry import (
    EntityClaimConflictError,
    LigandPromotionAttemptIdentityConflictError,
    PromotionPublicationUnavailableError,
    PromotionRecordConflictError,
    ReservationMetadataUnavailableError,
    ReservationOwnershipError,
    SemanticReservationConflictError,
    StateClaimConflictError,
    acquire_entity_claim,
    acquire_semantic_reservation,
    acquire_state_claim,
    begin_promotion_attempt,
    compensate_partial_acquisition,
    describe_entity_claim,
    describe_semantic_reservation,
    describe_state_claim,
    list_promotions_by_input,
    list_promotions_by_state,
    load_promotion,
    load_promotion_attempt,
    mark_execution_failed,
    mark_resolution_failed,
    mark_resolved,
    mark_succeeded,
    register_promotion,
    release_entity_claim,
    release_semantic_reservation,
    release_state_claim,
)
import molecular.promotion_registry as promotion_registry_module
from molecular.scientific import MolecularRepresentationRef

# A real SHA-256 digest (64 hex chars), distinct from the frozen spec's own
# illustrative example checksum in Section 6.10/10.3, which is 63 characters
# - a pre-existing documentation typo carried unchanged through every
# revision. Fixtures here need a value that actually satisfies
# ResolvedLigandPromotionSnapshot's checksum-format validator.
_CHECKSUM = "0e4e4f1767e11f8529e663ae2ba078094cf05bc09b7b17eb96691f07096014bb"
_SEMANTIC_KEY = "1bd810f669ad078fa744fc88b6248eeca8a5a2730d56493e518419dc316577f4"
_OTHER_SEMANTIC_KEY = "e3dfa4f5f16f664981b511e2e5abea0596fbe618cf0524d9ef75ff24a630d996"


# --- builders ---------------------------------------------------------------


def _begin(tmp_path, **overrides):
    fields = {
        "input_id": "curcumin-pubchem",
        "requested_entity_id": "curcumin",
        "requested_state_id": "curcumin-pubchem-initial",
        "requested_entity_name": "Curcumin",
        "requested": LigandPromotionPolicy(),
    }
    fields.update(overrides)
    return begin_promotion_attempt(tmp_path, **fields)


def _resolved(**overrides):
    fields = {
        "engine_name": "rdkit",
        "engine_version": "2026.03.6",
        "input_format": "sdf",
        "raw_checksum_persisted": _CHECKSUM,
        "snapshot_checksum": _CHECKSUM,
        "snapshot_size_bytes": 4096,
        "sanitize": True,
        "remove_hs": False,
        "strict_parsing": True,
        "record_count": 1,
        "fragment_count": 1,
        "formal_charge": 0,
        "atom_count": 27,
        "explicit_hydrogen_count": 0,
        "isotopic_hydrogen_count": 0,
    }
    fields.update(overrides)
    return ResolvedLigandPromotionSnapshot(**fields)


def _representation(**overrides):
    fields = {
        "path": "scientific/artifacts/states/curcumin-pubchem-initial/source.sdf",
        "format": "sdf",
        "checksum": _CHECKSUM,
    }
    fields.update(overrides)
    return MolecularRepresentationRef(**fields)


def _result(**overrides):
    fields = {
        "promotion_id": _SEMANTIC_KEY,
        "entity_id": "curcumin",
        "state_id": "curcumin-pubchem-initial",
        "representation": _representation(),
        "entity_created": True,
        "state_created": True,
        "reused_existing_promotion": False,
    }
    fields.update(overrides)
    return PromotionResultSnapshot(**fields)


def _error(**overrides):
    fields = {"phase": "resolution", "code": "invalid-sdf", "message": "no valid record found"}
    fields.update(overrides)
    return LigandPromotionError(**fields)


def _acquisition(**overrides):
    fields = {
        "method": "downloaded",
        "source_name": "PubChem",
        "source_identifier": "curcumin-pubchem",
        "acquired_at": datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc),
        "raw_artifact_path": "inputs/ligands/curcumin-pubchem/raw.sdf",
        "raw_artifact_format": "sdf",
        "raw_artifact_checksum": _CHECKSUM,
    }
    fields.update(overrides)
    return AcquisitionProvenanceSnapshot(**fields)


def _promotion_record(**overrides):
    fields = {
        "id": _SEMANTIC_KEY,
        "semantic_key": _SEMANTIC_KEY,
        "input_id": "curcumin-pubchem",
        "acquisition": _acquisition(),
        "policy": LigandPromotionPolicy(),
        "resolved": _resolved(),
        "entity_id": "curcumin",
        "state_id": "curcumin-pubchem-initial",
        "representation": _representation(),
        "created_by_attempt_id": uuid.uuid4().hex,
        "created_at": datetime(2026, 9, 25, 15, 0, 2, tzinfo=timezone.utc),
    }
    fields.update(overrides)
    return ScientificInputPromotionRecord(**fields)


# --- 25: begin / load attempt ---


def test_begin_creates_started_attempt(tmp_path):
    attempt = _begin(tmp_path)

    assert attempt.status == "started"
    assert attempt.semantic_key is None
    assert attempt.resolved is None
    assert attempt.result is None
    assert attempt.error is None


def test_begin_id_is_a_uuid4_hex_string(tmp_path):
    attempt = _begin(tmp_path)

    parsed = uuid.UUID(attempt.id)
    assert parsed.version == 4
    assert attempt.id == parsed.hex


def test_begin_persists_to_the_expected_path(tmp_path):
    attempt = _begin(tmp_path)

    metadata_file = tmp_path / "scientific" / "promotion_attempts" / f"{attempt.id}.yaml"
    assert metadata_file.exists()


def test_load_round_trips(tmp_path):
    attempt = _begin(tmp_path)

    loaded = load_promotion_attempt(tmp_path, attempt.id)

    assert loaded == attempt


def test_load_raises_for_unknown_attempt(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_promotion_attempt(tmp_path, uuid.uuid4().hex)


def test_begin_rejects_malformed_ids(tmp_path):
    with pytest.raises(ValueError):
        _begin(tmp_path, input_id="not a valid id")


# --- 26: transições permitidas ---


def test_started_to_resolved(tmp_path):
    attempt = _begin(tmp_path)

    resolved = mark_resolved(tmp_path, attempt.id, semantic_key=_SEMANTIC_KEY, resolved=_resolved())

    assert resolved.status == "resolved"
    assert resolved.semantic_key == _SEMANTIC_KEY
    assert resolved.resolved_at is not None


def test_started_to_resolution_failed(tmp_path):
    attempt = _begin(tmp_path)

    failed = mark_resolution_failed(tmp_path, attempt.id, error=_error())

    assert failed.status == "resolution_failed"
    assert failed.finished_at is not None


def test_resolved_to_execution_failed(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, semantic_key=_SEMANTIC_KEY, resolved=_resolved())

    failed = mark_execution_failed(
        tmp_path, attempt.id, error=_error(phase="execution", code="publication-failed")
    )

    assert failed.status == "execution_failed"
    assert failed.semantic_key == _SEMANTIC_KEY
    assert failed.resolved is not None


def test_resolved_to_succeeded(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, semantic_key=_SEMANTIC_KEY, resolved=_resolved())

    succeeded = mark_succeeded(tmp_path, attempt.id, result=_result())

    assert succeeded.status == "succeeded"
    assert succeeded.result is not None


# --- 27: transições proibidas/terminais ---


def test_cannot_resolve_a_non_started_attempt(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolution_failed(tmp_path, attempt.id, error=_error())

    with pytest.raises(ValueError):
        mark_resolved(tmp_path, attempt.id, semantic_key=_SEMANTIC_KEY, resolved=_resolved())


def test_cannot_reopen_a_terminal_attempt(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, semantic_key=_SEMANTIC_KEY, resolved=_resolved())
    mark_succeeded(tmp_path, attempt.id, result=_result())

    with pytest.raises(ValueError):
        mark_succeeded(tmp_path, attempt.id, result=_result())

    with pytest.raises(ValueError):
        mark_execution_failed(tmp_path, attempt.id, error=_error(phase="execution", code="publication-failed"))


def test_cannot_succeed_a_started_attempt_directly(tmp_path):
    attempt = _begin(tmp_path)

    with pytest.raises(ValueError):
        mark_succeeded(tmp_path, attempt.id, result=_result())


# --- 28: atomic overwrite de attempt ---


def test_transition_overwrites_the_single_persisted_file(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, semantic_key=_SEMANTIC_KEY, resolved=_resolved())

    metadata_dir = tmp_path / "scientific" / "promotion_attempts"
    assert sorted(p.name for p in metadata_dir.iterdir()) == [f"{attempt.id}.yaml"]

    reloaded = load_promotion_attempt(tmp_path, attempt.id)
    assert reloaded.status == "resolved"


def test_begin_never_clobbers_an_existing_attempt_file(tmp_path, monkeypatch):
    attempt = _begin(tmp_path)

    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(attempt.id))

    with pytest.raises(LigandPromotionAttemptIdentityConflictError):
        _begin(tmp_path)


# --- 29: record de sucesso create-once (criação exclusiva) ---


def test_register_promotion_persists_to_expected_path(tmp_path):
    record = _promotion_record()

    register_promotion(tmp_path, record)

    metadata_file = tmp_path / "scientific" / "promotions" / f"{record.semantic_key}.yaml"
    assert metadata_file.exists()


def test_register_promotion_never_overwrites_an_existing_record(tmp_path):
    record = _promotion_record()
    register_promotion(tmp_path, record)

    with pytest.raises(PromotionRecordConflictError):
        register_promotion(tmp_path, record)


def test_register_promotion_leaves_no_staging_file_behind(tmp_path):
    record = _promotion_record()

    register_promotion(tmp_path, record)

    promotions_dir = tmp_path / "scientific" / "promotions"
    assert [p.name for p in promotions_dir.iterdir()] == [f"{record.semantic_key}.yaml"]


# --- 30: conflito exato ---


def test_register_promotion_rejects_second_record_under_same_semantic_key(tmp_path):
    register_promotion(tmp_path, _promotion_record())

    different_content = _promotion_record(entity_id="different-entity")

    with pytest.raises(PromotionRecordConflictError):
        register_promotion(tmp_path, different_content)

    # the original record is untouched by the rejected attempt
    assert load_promotion(tmp_path, _SEMANTIC_KEY).entity_id == "curcumin"


def test_register_promotion_allows_distinct_semantic_keys(tmp_path):
    register_promotion(tmp_path, _promotion_record())
    register_promotion(
        tmp_path,
        _promotion_record(
            id=_OTHER_SEMANTIC_KEY,
            semantic_key=_OTHER_SEMANTIC_KEY,
            input_id="curcumin-pubchem-2",
        ),
    )

    assert load_promotion(tmp_path, _SEMANTIC_KEY).input_id == "curcumin-pubchem"
    assert load_promotion(tmp_path, _OTHER_SEMANTIC_KEY).input_id == "curcumin-pubchem-2"


# --- 31: consultas por semantic key/input/state ---


def test_load_promotion_by_semantic_key(tmp_path):
    register_promotion(tmp_path, _promotion_record())

    loaded = load_promotion(tmp_path, _SEMANTIC_KEY)

    assert loaded.semantic_key == _SEMANTIC_KEY


def test_query_promotions_by_input_id(tmp_path):
    register_promotion(tmp_path, _promotion_record())
    register_promotion(
        tmp_path,
        _promotion_record(
            id=_OTHER_SEMANTIC_KEY, semantic_key=_OTHER_SEMANTIC_KEY, input_id="other-input"
        ),
    )

    matches = list_promotions_by_input(tmp_path, "curcumin-pubchem")

    assert [record.semantic_key for record in matches] == [_SEMANTIC_KEY]


def test_query_promotions_by_state_id(tmp_path):
    register_promotion(tmp_path, _promotion_record())
    register_promotion(
        tmp_path,
        _promotion_record(
            id=_OTHER_SEMANTIC_KEY,
            semantic_key=_OTHER_SEMANTIC_KEY,
            input_id="curcumin-pubchem-2",
            state_id="curcumin-pubchem-2-initial",
        ),
    )

    matches = list_promotions_by_state(tmp_path, "curcumin-pubchem-initial")

    assert [record.semantic_key for record in matches] == [_SEMANTIC_KEY]


def test_query_by_input_returns_empty_list_when_nothing_registered(tmp_path):
    assert list_promotions_by_input(tmp_path, "curcumin-pubchem") == []
    assert list_promotions_by_state(tmp_path, "curcumin-pubchem-initial") == []


# --- 32: record inconsistente rejeitado ---


def test_load_promotion_rejects_internally_inconsistent_record(tmp_path):
    promotions_dir = tmp_path / "scientific" / "promotions"
    promotions_dir.mkdir(parents=True)

    inconsistent = _promotion_record().model_dump(mode="json")
    inconsistent["id"] = _OTHER_SEMANTIC_KEY  # id != semantic_key, invalid

    (promotions_dir / f"{_SEMANTIC_KEY}.yaml").write_text(
        yaml.safe_dump(inconsistent), encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_promotion(tmp_path, _SEMANTIC_KEY)


# --- 33: YAML corrompido propaga validation/config error ---


def test_load_promotion_propagates_corrupted_yaml(tmp_path):
    promotions_dir = tmp_path / "scientific" / "promotions"
    promotions_dir.mkdir(parents=True)
    (promotions_dir / f"{_SEMANTIC_KEY}.yaml").write_text(
        "not: valid: yaml: [", encoding="utf-8"
    )

    with pytest.raises(yaml.YAMLError):
        load_promotion(tmp_path, _SEMANTIC_KEY)


def test_load_attempt_propagates_corrupted_yaml(tmp_path):
    attempt = _begin(tmp_path)
    metadata_file = tmp_path / "scientific" / "promotion_attempts" / f"{attempt.id}.yaml"
    metadata_file.write_text("not: valid: yaml: [", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_promotion_attempt(tmp_path, attempt.id)


# --- 37: semantic key minúscula/64 hex ---


def test_load_promotion_rejects_non_lowercase_semantic_key(tmp_path):
    register_promotion(tmp_path, _promotion_record())

    with pytest.raises(ValueError):
        load_promotion(tmp_path, _SEMANTIC_KEY.upper())


def test_load_promotion_rejects_wrong_length_semantic_key(tmp_path):
    with pytest.raises(ValueError):
        load_promotion(tmp_path, "abc123")


# --- 38: attempt failure não cria promotion record ---


def test_resolution_failed_attempt_creates_no_promotion_record(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolution_failed(tmp_path, attempt.id, error=_error())

    assert list_promotions_by_input(tmp_path, "curcumin-pubchem") == []
    with pytest.raises(FileNotFoundError):
        load_promotion(tmp_path, _SEMANTIC_KEY)


def test_execution_failed_attempt_creates_no_promotion_record(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, semantic_key=_SEMANTIC_KEY, resolved=_resolved())
    mark_execution_failed(
        tmp_path, attempt.id, error=_error(phase="execution", code="publication-failed")
    )

    assert list_promotions_by_input(tmp_path, "curcumin-pubchem") == []
    with pytest.raises(FileNotFoundError):
        load_promotion(tmp_path, _SEMANTIC_KEY)


# =============================================================================
# Checkpoint 2: cross-process semantic reservation and entity/state claims
# (Section 11.2-11.10 of the frozen spec). Items 34-36 of Section 19.2, and
# all of Section 19.9 (items 151-169).
# =============================================================================


# --- 34: reservation create exclusive ---


def test_acquire_semantic_reservation_persists_metadata(tmp_path):
    acquire_semantic_reservation(
        tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
    )

    slot_dir = tmp_path / "scientific" / "promotion_reservations" / _SEMANTIC_KEY
    assert slot_dir.is_dir()
    assert (slot_dir / "reservation.yaml").exists()


def test_acquire_semantic_reservation_is_exclusive(tmp_path):
    acquire_semantic_reservation(
        tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
    )

    with pytest.raises(SemanticReservationConflictError):
        acquire_semantic_reservation(
            tmp_path, _SEMANTIC_KEY, attempt_id="c" * 32, owner_token="d" * 64
        )


def test_acquire_entity_claim_is_exclusive(tmp_path):
    acquire_entity_claim(tmp_path, "curcumin", attempt_id="a" * 32, owner_token="b" * 64)

    with pytest.raises(EntityClaimConflictError):
        acquire_entity_claim(tmp_path, "curcumin", attempt_id="c" * 32, owner_token="d" * 64)


def test_acquire_state_claim_is_exclusive(tmp_path):
    acquire_state_claim(
        tmp_path, "curcumin-pubchem-initial", attempt_id="a" * 32, owner_token="b" * 64
    )

    with pytest.raises(StateClaimConflictError):
        acquire_state_claim(
            tmp_path, "curcumin-pubchem-initial", attempt_id="c" * 32, owner_token="d" * 64
        )


def test_acquire_entity_claim_rejects_casefold_collision_single_process(tmp_path):
    acquire_entity_claim(tmp_path, "Entity-A", attempt_id="a" * 32, owner_token="b" * 64)

    with pytest.raises(EntityClaimConflictError):
        acquire_entity_claim(tmp_path, "entity-a", attempt_id="c" * 32, owner_token="d" * 64)


def test_acquire_state_claim_rejects_casefold_collision_single_process(tmp_path):
    acquire_state_claim(tmp_path, "State-A", attempt_id="a" * 32, owner_token="b" * 64)

    with pytest.raises(StateClaimConflictError):
        acquire_state_claim(tmp_path, "state-a", attempt_id="c" * 32, owner_token="d" * 64)


# --- 166 (see also the two describe_* tests below, same item) ---


def test_acquire_never_treats_an_existing_slot_without_metadata_as_free():
    """§19.9-166: a slot directory created (e.g. by a crash mid-write,
    before metadata was ever persisted) but lacking a readable
    reservation.yaml must still refuse a second acquire - exclusivity is
    decided by directory existence alone (`os.mkdir`), never by whether
    metadata can be read."""

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp)
        slot_dir = project_dir / "scientific" / "promotion_reservations" / _SEMANTIC_KEY
        slot_dir.mkdir(parents=True)  # directory exists, but no reservation.yaml at all

        status = describe_semantic_reservation(project_dir, _SEMANTIC_KEY)
        assert status.state == "RESERVED_METADATA_MISSING_INDETERMINATE"

        with pytest.raises(SemanticReservationConflictError):
            acquire_semantic_reservation(
                project_dir, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
            )


# --- 35: reservation release somente por owner ---


def test_release_semantic_reservation_requires_exact_owner(tmp_path):
    acquire_semantic_reservation(
        tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
    )

    with pytest.raises(ReservationOwnershipError):
        release_semantic_reservation(
            tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="wrong-token"
        )
    with pytest.raises(ReservationOwnershipError):
        release_semantic_reservation(
            tmp_path, _SEMANTIC_KEY, attempt_id="wrong-attempt", owner_token="b" * 64
        )

    # the resource is untouched by either rejected attempt
    assert describe_semantic_reservation(tmp_path, _SEMANTIC_KEY).state == "RESERVED_KNOWN_OWNER"

    release_semantic_reservation(tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64)

    assert describe_semantic_reservation(tmp_path, _SEMANTIC_KEY).state == "ABSENT_FREE"


def test_release_entity_claim_requires_exact_owner(tmp_path):
    acquire_entity_claim(tmp_path, "curcumin", attempt_id="a" * 32, owner_token="b" * 64)

    with pytest.raises(ReservationOwnershipError):
        release_entity_claim(tmp_path, "curcumin", attempt_id="a" * 32, owner_token="wrong")

    release_entity_claim(tmp_path, "curcumin", attempt_id="a" * 32, owner_token="b" * 64)
    assert describe_entity_claim(tmp_path, "curcumin").state == "ABSENT_FREE"


def test_release_state_claim_requires_exact_owner(tmp_path):
    acquire_state_claim(tmp_path, "curcumin-pubchem-initial", attempt_id="a" * 32, owner_token="b" * 64)

    with pytest.raises(ReservationOwnershipError):
        release_state_claim(
            tmp_path, "curcumin-pubchem-initial", attempt_id="a" * 32, owner_token="wrong"
        )

    release_state_claim(
        tmp_path, "curcumin-pubchem-initial", attempt_id="a" * 32, owner_token="b" * 64
    )
    assert describe_state_claim(tmp_path, "curcumin-pubchem-initial").state == "ABSENT_FREE"


def test_release_nonexistent_slot_raises_metadata_unavailable(tmp_path):
    with pytest.raises(ReservationMetadataUnavailableError):
        release_semantic_reservation(
            tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
        )


def test_repeated_release_after_success_raises_metadata_unavailable(tmp_path):
    """Release semantics are exact-match-or-refuse; a second release of an
    already-released resource finds nothing and raises the same typed
    'cannot confirm' error as a resource that never existed - it never
    silently succeeds twice."""

    acquire_semantic_reservation(
        tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
    )
    release_semantic_reservation(tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64)

    with pytest.raises(ReservationMetadataUnavailableError):
        release_semantic_reservation(
            tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
        )


# --- 36: claims em ordem e cleanup parcial ---


def test_compensate_partial_acquisition_releases_in_reverse_order(tmp_path):
    attempt_id, owner_token = "a" * 32, "b" * 64
    acquire_semantic_reservation(tmp_path, _SEMANTIC_KEY, attempt_id=attempt_id, owner_token=owner_token)
    acquire_entity_claim(tmp_path, "curcumin", attempt_id=attempt_id, owner_token=owner_token)
    # state claim lost to someone else - never acquired by this attempt

    residuals = compensate_partial_acquisition(
        tmp_path,
        attempt_id=attempt_id,
        owner_token=owner_token,
        semantic_key=_SEMANTIC_KEY,
        entity_id="curcumin",
        state_id=None,
    )

    assert residuals == ()
    assert describe_semantic_reservation(tmp_path, _SEMANTIC_KEY).state == "ABSENT_FREE"
    assert describe_entity_claim(tmp_path, "curcumin").state == "ABSENT_FREE"


def test_compensate_partial_acquisition_never_touches_another_owners_resource(tmp_path):
    other_attempt, other_token = "c" * 32, "d" * 64
    acquire_semantic_reservation(tmp_path, _SEMANTIC_KEY, attempt_id=other_attempt, owner_token=other_token)

    my_attempt, my_token = "a" * 32, "b" * 64
    residuals = compensate_partial_acquisition(
        tmp_path, attempt_id=my_attempt, owner_token=my_token, semantic_key=_SEMANTIC_KEY
    )

    # this attempt never owned it, so the release could not be confirmed -
    # it becomes a residual, and the other owner's reservation survives.
    assert residuals == (_SEMANTIC_KEY,)
    status = describe_semantic_reservation(tmp_path, _SEMANTIC_KEY)
    assert status.state == "RESERVED_KNOWN_OWNER"
    assert status.owner_attempt_id == other_attempt


# --- 37/163: falha de release/compensação produz resíduo tipado ---


def test_compensation_failure_produces_typed_residual_when_metadata_corrupted(tmp_path):
    attempt_id, owner_token = "a" * 32, "b" * 64
    acquire_semantic_reservation(tmp_path, _SEMANTIC_KEY, attempt_id=attempt_id, owner_token=owner_token)
    acquire_entity_claim(tmp_path, "curcumin", attempt_id=attempt_id, owner_token=owner_token)

    entity_slot = promotion_registry_module._entity_claims_dir(tmp_path) / promotion_registry_module._claim_hash(
        "curcumin"
    )
    (entity_slot / "reservation.yaml").write_text("not: valid: yaml: [", encoding="utf-8")

    residuals = compensate_partial_acquisition(
        tmp_path,
        attempt_id=attempt_id,
        owner_token=owner_token,
        semantic_key=_SEMANTIC_KEY,
        entity_id="curcumin",
    )

    assert residuals == ("curcumin",)
    # the semantic reservation (released after the entity claim, in
    # reverse order) was confirmed and removed regardless of the entity
    # claim's failure - each resource is compensated independently.
    assert describe_semantic_reservation(tmp_path, _SEMANTIC_KEY).state == "ABSENT_FREE"
    # the entity claim directory itself is left completely untouched.
    assert entity_slot.is_dir()


# --- Checkpoint 2 correction, MAJOR-1: a genuine infrastructure failure
# during removal (shutil.rmtree raising OSError/PermissionError, e.g. a
# file locked by another process or a transient disk error) must become a
# typed residual too - exactly like ownership mismatch or unreadable
# metadata - and must never stop the independent compensation of the other
# two, unrelated resources. Section 11.10 / acceptance criterion 29. ---


@pytest.mark.parametrize("failing_resource", ["state", "entity", "semantic"])
def test_compensate_partial_acquisition_continues_after_infrastructure_failure(
    tmp_path, monkeypatch, failing_resource
):
    attempt_id, owner_token = "a" * 32, "b" * 64
    acquire_semantic_reservation(tmp_path, _SEMANTIC_KEY, attempt_id=attempt_id, owner_token=owner_token)
    acquire_entity_claim(tmp_path, "curcumin", attempt_id=attempt_id, owner_token=owner_token)
    acquire_state_claim(
        tmp_path, "curcumin-pubchem-initial", attempt_id=attempt_id, owner_token=owner_token
    )

    semantic_slot = promotion_registry_module._promotion_reservations_dir(tmp_path) / _SEMANTIC_KEY
    entity_slot = promotion_registry_module._entity_claims_dir(
        tmp_path
    ) / promotion_registry_module._claim_hash("curcumin")
    state_slot = promotion_registry_module._state_claims_dir(
        tmp_path
    ) / promotion_registry_module._claim_hash("curcumin-pubchem-initial")
    slots = {"semantic": semantic_slot, "entity": entity_slot, "state": state_slot}
    raw_keys = {"semantic": _SEMANTIC_KEY, "entity": "curcumin", "state": "curcumin-pubchem-initial"}

    failing_slot = slots[failing_resource]
    real_rmtree = shutil.rmtree

    def _flaky_rmtree(path, *args, **kwargs):
        # Real shutil.rmtree is used for every path except the one under
        # test - the other two resources must be genuinely, independently
        # attempted and removed, never mocked away wholesale.
        if Path(path) == failing_slot:
            raise OSError("simulated infrastructure failure removing directory")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(promotion_registry_module.shutil, "rmtree", _flaky_rmtree)

    residuals = compensate_partial_acquisition(
        tmp_path,
        attempt_id=attempt_id,
        owner_token=owner_token,
        semantic_key=_SEMANTIC_KEY,
        entity_id="curcumin",
        state_id="curcumin-pubchem-initial",
    )

    # no exception escaped (the call above already proves that), only the
    # failing resource's raw key is reported as a residual...
    assert residuals == (raw_keys[failing_resource],)
    # ...the failing directory is left exactly as it was, never declared removed...
    assert failing_slot.is_dir()
    # ...and the other two, unrelated resources were still independently
    # attempted and genuinely removed, in both directions relative to the
    # failing one (order is always state -> entity -> semantic).
    for resource, slot in slots.items():
        if resource != failing_resource:
            assert not slot.exists(), f"{resource} should have been removed independently"


# --- describe_*: read-only inspection never mutates, never reports a
# residue as free (Section 11.6) ---


def test_describe_reports_absent_free_when_nothing_exists(tmp_path):
    assert describe_semantic_reservation(tmp_path, _SEMANTIC_KEY).state == "ABSENT_FREE"
    assert describe_entity_claim(tmp_path, "curcumin").state == "ABSENT_FREE"
    assert describe_state_claim(tmp_path, "curcumin-pubchem-initial").state == "ABSENT_FREE"


def test_describe_reports_corrupt_metadata_as_indeterminate_not_free(tmp_path):
    """§19.9-166 (second angle): metadata present but unparseable is also
    never reported as free."""

    acquire_semantic_reservation(
        tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
    )
    slot_dir = tmp_path / "scientific" / "promotion_reservations" / _SEMANTIC_KEY
    (slot_dir / "reservation.yaml").write_text("not: valid: yaml: [", encoding="utf-8")

    status = describe_semantic_reservation(tmp_path, _SEMANTIC_KEY)

    assert status.state == "RESERVED_METADATA_CORRUPT_INDETERMINATE"
    assert status.owner_attempt_id is None


# --- Checkpoint 2 correction, MINOR-1: a generic filesystem read failure
# (PermissionError/OSError) while reading an existing reservation.yaml -
# distinct from the file being absent, and distinct from it being
# syntactically corrupt YAML - must still be folded into the same
# INDETERMINATE contract, never let a raw, untyped exception escape a
# strictly read-only describe_* call, and never be reported as
# RESERVED_METADATA_MISSING (the file *does* exist - "missing" would be
# inaccurate) nor as ABSENT_FREE. Section 11.6. ---


@pytest.mark.parametrize(
    "kind, acquire, describe, dir_helper",
    [
        (
            "semantic",
            lambda p, aid, tok: acquire_semantic_reservation(p, _SEMANTIC_KEY, attempt_id=aid, owner_token=tok),
            describe_semantic_reservation,
            lambda p: promotion_registry_module._promotion_reservations_dir(p) / _SEMANTIC_KEY,
        ),
        (
            "entity",
            lambda p, aid, tok: acquire_entity_claim(p, "curcumin", attempt_id=aid, owner_token=tok),
            describe_entity_claim,
            lambda p: promotion_registry_module._entity_claims_dir(p)
            / promotion_registry_module._claim_hash("curcumin"),
        ),
        (
            "state",
            lambda p, aid, tok: acquire_state_claim(
                p, "curcumin-pubchem-initial", attempt_id=aid, owner_token=tok
            ),
            describe_state_claim,
            lambda p: promotion_registry_module._state_claims_dir(p)
            / promotion_registry_module._claim_hash("curcumin-pubchem-initial"),
        ),
    ],
    ids=["semantic", "entity", "state"],
)
def test_describe_reports_indeterminate_on_generic_read_failure(
    tmp_path, monkeypatch, kind, acquire, describe, dir_helper
):
    raw_key = {"semantic": _SEMANTIC_KEY, "entity": "curcumin", "state": "curcumin-pubchem-initial"}[kind]
    acquire(tmp_path, "a" * 32, "b" * 64)

    slot_dir = dir_helper(tmp_path)
    metadata_file = slot_dir / "reservation.yaml"
    real_read_text = Path.read_text

    def _flaky_read_text(self, *args, **kwargs):
        if self == metadata_file:
            raise PermissionError("simulated permission failure reading metadata")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _flaky_read_text)

    status = describe(tmp_path, raw_key)

    assert status.state == "RESERVED_METADATA_CORRUPT_INDETERMINATE"
    assert status.state != "ABSENT_FREE"
    assert status.owner_attempt_id is None
    # the resource itself, and its (unreadable, but present) metadata file,
    # are both left completely untouched by a read-only inspection.
    assert slot_dir.is_dir()
    assert metadata_file.exists()


# --- Checkpoint 2 final correction, MINOR-2: a filesystem failure while
# checking a slot's own existence - or, one level down, while checking
# whether its unreadable metadata file exists at all - must never be
# mistaken for a confirmed absence. Only a *normal* `False` return from
# `Path.exists()` means ABSENT_FREE; an `OSError`/`PermissionError` raised
# by `exists()` itself means the platform could not even prove the
# resource is gone, which is conservatively folded into
# RESERVED_METADATA_CORRUPT_INDETERMINATE - never ABSENT_FREE, never an
# escaped exception, never any attempt to create/repair/remove anything.
# Section 11.6. ---


@pytest.mark.parametrize(
    "kind, acquire, describe, dir_helper",
    [
        (
            "semantic",
            lambda p, aid, tok: acquire_semantic_reservation(p, _SEMANTIC_KEY, attempt_id=aid, owner_token=tok),
            describe_semantic_reservation,
            lambda p: promotion_registry_module._promotion_reservations_dir(p) / _SEMANTIC_KEY,
        ),
        (
            "entity",
            lambda p, aid, tok: acquire_entity_claim(p, "curcumin", attempt_id=aid, owner_token=tok),
            describe_entity_claim,
            lambda p: promotion_registry_module._entity_claims_dir(p)
            / promotion_registry_module._claim_hash("curcumin"),
        ),
        (
            "state",
            lambda p, aid, tok: acquire_state_claim(
                p, "curcumin-pubchem-initial", attempt_id=aid, owner_token=tok
            ),
            describe_state_claim,
            lambda p: promotion_registry_module._state_claims_dir(p)
            / promotion_registry_module._claim_hash("curcumin-pubchem-initial"),
        ),
    ],
    ids=["semantic", "entity", "state"],
)
def test_describe_reports_indeterminate_when_slot_exists_check_fails(
    tmp_path, monkeypatch, kind, acquire, describe, dir_helper
):
    """Test A: `slot_dir.exists()` itself raises PermissionError. The slot
    exists physically the whole time - only the ability to *confirm* that
    via `exists()` is disrupted."""

    raw_key = {"semantic": _SEMANTIC_KEY, "entity": "curcumin", "state": "curcumin-pubchem-initial"}[kind]
    acquire(tmp_path, "a" * 32, "b" * 64)

    slot_dir = dir_helper(tmp_path)
    metadata_file = slot_dir / "reservation.yaml"
    real_exists = Path.exists
    # preconditions, checked with the real, not-yet-patched exists()
    assert real_exists(slot_dir)
    assert real_exists(metadata_file)

    def _flaky_exists(self, *args, **kwargs):
        if self == slot_dir:
            raise PermissionError("simulated permission failure checking slot existence")
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _flaky_exists)

    status = describe(tmp_path, raw_key)

    assert status.state == "RESERVED_METADATA_CORRUPT_INDETERMINATE"
    assert status.state != "ABSENT_FREE"
    assert status.owner_attempt_id is None
    # the slot and its metadata are left exactly as they were - confirmed
    # via the real exists() captured above, never the patched one.
    assert real_exists(slot_dir)
    assert real_exists(metadata_file)


@pytest.mark.parametrize(
    "kind, acquire, describe, dir_helper",
    [
        (
            "semantic",
            lambda p, aid, tok: acquire_semantic_reservation(p, _SEMANTIC_KEY, attempt_id=aid, owner_token=tok),
            describe_semantic_reservation,
            lambda p: promotion_registry_module._promotion_reservations_dir(p) / _SEMANTIC_KEY,
        ),
        (
            "entity",
            lambda p, aid, tok: acquire_entity_claim(p, "curcumin", attempt_id=aid, owner_token=tok),
            describe_entity_claim,
            lambda p: promotion_registry_module._entity_claims_dir(p)
            / promotion_registry_module._claim_hash("curcumin"),
        ),
        (
            "state",
            lambda p, aid, tok: acquire_state_claim(
                p, "curcumin-pubchem-initial", attempt_id=aid, owner_token=tok
            ),
            describe_state_claim,
            lambda p: promotion_registry_module._state_claims_dir(p)
            / promotion_registry_module._claim_hash("curcumin-pubchem-initial"),
        ),
    ],
    ids=["semantic", "entity", "state"],
)
def test_describe_reports_indeterminate_when_metadata_exists_check_also_fails(
    tmp_path, monkeypatch, kind, acquire, describe, dir_helper
):
    """Test B: double failure. `Path.read_text()` fails for
    `reservation.yaml` (so `_read_slot_metadata` raises
    ReservationMetadataUnavailableError), and the fallback
    `metadata_file.exists()` check *also* raises - so it cannot even be
    determined whether the metadata is missing or merely corrupt. `slot_dir.
    exists()` itself is left working normally (only the metadata file's own
    existence check is disrupted), so this exercises the second failure
    point in isolation from the first (Test A above)."""

    raw_key = {"semantic": _SEMANTIC_KEY, "entity": "curcumin", "state": "curcumin-pubchem-initial"}[kind]
    acquire(tmp_path, "a" * 32, "b" * 64)

    slot_dir = dir_helper(tmp_path)
    metadata_file = slot_dir / "reservation.yaml"
    real_exists = Path.exists
    real_read_text = Path.read_text
    # preconditions, checked with the real, not-yet-patched functions
    assert real_exists(slot_dir)
    assert real_exists(metadata_file)

    def _flaky_read_text(self, *args, **kwargs):
        if self == metadata_file:
            raise PermissionError("simulated permission failure reading metadata")
        return real_read_text(self, *args, **kwargs)

    def _flaky_exists(self, *args, **kwargs):
        if self == metadata_file:
            raise PermissionError("simulated permission failure checking metadata existence")
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _flaky_read_text)
    monkeypatch.setattr(Path, "exists", _flaky_exists)

    status = describe(tmp_path, raw_key)

    assert status.state == "RESERVED_METADATA_CORRUPT_INDETERMINATE"
    assert status.state != "ABSENT_FREE"
    assert status.owner_attempt_id is None
    # the slot and its metadata are left exactly as they were - confirmed
    # via the real, captured-before-patching functions.
    assert real_exists(slot_dir)
    assert real_exists(metadata_file)


def test_describe_never_exposes_owner_token(tmp_path):
    acquire_semantic_reservation(
        tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="super-secret-token"
    )

    status = describe_semantic_reservation(tmp_path, _SEMANTIC_KEY)

    assert not hasattr(status, "owner_token")
    assert "super-secret-token" not in status.model_dump_json()


# --- 161: TTL nunca autoriza takeover ---


def test_backdated_reservation_still_refuses_reacquisition(tmp_path):
    acquire_semantic_reservation(
        tmp_path, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64
    )

    slot_dir = tmp_path / "scientific" / "promotion_reservations" / _SEMANTIC_KEY
    metadata_file = slot_dir / "reservation.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    ancient = datetime.now(timezone.utc) - timedelta(days=3650)
    data["created_at"] = ancient.isoformat()
    metadata_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    status = describe_semantic_reservation(tmp_path, _SEMANTIC_KEY)
    assert status.age_seconds > 3600 * 24 * 300  # informational only, ~10 years old

    with pytest.raises(SemanticReservationConflictError):
        acquire_semantic_reservation(
            tmp_path, _SEMANTIC_KEY, attempt_id="c" * 32, owner_token="d" * 64
        )


def test_no_code_path_removes_a_reservation_by_age():
    """Structural proof mirroring FS-018-02's own evidence (Section 11.7):
    the module source contains no comparison of age/created_at against a
    limit anywhere outside informational reporting."""

    import inspect

    source = inspect.getsource(promotion_registry_module)
    forbidden_patterns = ("age_seconds >", "age_seconds >=", "> _TTL", ">= _TTL", "ttl_seconds")
    for pattern in forbidden_patterns:
        assert pattern not in source


# --- 167: normalização Unicode NFC + casefold ---


def test_claim_hash_normalizes_nfc_and_casefold():
    composed = "Café"
    decomposed = "Café"
    assert composed != decomposed

    assert promotion_registry_module._claim_hash(composed) == promotion_registry_module._claim_hash(
        decomposed
    )
    assert promotion_registry_module._claim_hash("Entity-A") == promotion_registry_module._claim_hash(
        "entity-a"
    )


def test_same_raw_value_as_entity_and_state_id_does_not_collide():
    """Checkpoint 2 correction, cheap traceability (NOTE-3 of the
    micro-review): the same raw string used as entity_id in one claim and
    as state_id in another must not collide, even though `_claim_hash`
    produces the exact same SHA-256 digest for both - `by-entity/` and
    `by-state/` are distinct sibling namespaces (Section 11.4), so the
    identical hash lands under two different parent directories."""

    shared_value = "curcumin"
    assert promotion_registry_module._claim_hash(shared_value) == promotion_registry_module._claim_hash(
        shared_value
    )

    entity_slot = promotion_registry_module._entity_claims_dir(Path("x")) / promotion_registry_module._claim_hash(
        shared_value
    )
    state_slot = promotion_registry_module._state_claims_dir(Path("x")) / promotion_registry_module._claim_hash(
        shared_value
    )
    assert entity_slot != state_slot
    assert entity_slot.parent != state_slot.parent


def test_same_raw_value_as_entity_and_state_id_can_both_be_acquired(tmp_path):
    shared_value = "curcumin"

    acquire_entity_claim(tmp_path, shared_value, attempt_id="a" * 32, owner_token="b" * 64)
    acquire_state_claim(tmp_path, shared_value, attempt_id="c" * 32, owner_token="d" * 64)

    assert describe_entity_claim(tmp_path, shared_value).owner_attempt_id == "a" * 32
    assert describe_state_claim(tmp_path, shared_value).owner_attempt_id == "c" * 32


# --- project_dir isolation ---


def test_reservations_are_isolated_per_project_dir(tmp_path):
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    acquire_semantic_reservation(project_a, _SEMANTIC_KEY, attempt_id="a" * 32, owner_token="b" * 64)
    # the same semantic_key in a different project_dir never collides
    acquire_semantic_reservation(project_b, _SEMANTIC_KEY, attempt_id="c" * 32, owner_token="d" * 64)

    assert describe_semantic_reservation(project_a, _SEMANTIC_KEY).owner_attempt_id == "a" * 32
    assert describe_semantic_reservation(project_b, _SEMANTIC_KEY).owner_attempt_id == "c" * 32


# --- 169: promotion-primitive-unavailable ---


def test_register_promotion_fails_closed_when_os_link_raises_other_oserror(tmp_path, monkeypatch):
    def _broken_link(_src, _dst):
        raise OSError("simulated cross-device link failure")

    monkeypatch.setattr(promotion_registry_module.os, "link", _broken_link)

    with pytest.raises(PromotionPublicationUnavailableError):
        register_promotion(tmp_path, _promotion_record())

    # no half-published record and no leftover staging file
    promotions_dir = tmp_path / "scientific" / "promotions"
    assert not promotions_dir.exists() or list(promotions_dir.iterdir()) == []


def test_register_promotion_fails_closed_when_os_link_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.delattr(promotion_registry_module.os, "link")

    with pytest.raises(PromotionPublicationUnavailableError):
        register_promotion(tmp_path, _promotion_record())


# =============================================================================
# Real cross-process concurrency (Section 19.9, items 151-160, 162, 164-165,
# 168). multiprocessing.Process + multiprocessing.Barrier, Windows-spawn
# compatible - never threads. Every worker writes its own JSON result file
# under a per-test results directory; a simulated crash (os._exit) writes no
# result file at all, exactly like a real crash leaves no clean exit trail.
# """
# =============================================================================


def _write_worker_result(result_dir: Path, tag: str, data: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    tmp = result_dir / f".{tag}.tmp"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, result_dir / f"{tag}.json")


def _promotion_chain_worker(barrier, project_dir_str, job, result_dir_str):
    """Real-process worker: acquires semantic reservation -> entity claim
    (if requested) -> state claim (if requested) -> optionally publishes a
    promotion record -> optionally releases everything it holds - stopping
    at the first failure. `job` is a plain, picklable dict; see call sites
    below for its shape."""

    project_dir = Path(project_dir_str)
    result_dir = Path(result_dir_str)
    tag = job["tag"]
    attempt_id = uuid.uuid4().hex
    owner_token = secrets.token_hex(32)
    result = {
        "tag": tag,
        "attempt_id": attempt_id,
        "semantic_acquired": False,
        "entity_claimed": False,
        "state_claimed": False,
        "published": False,
        "outcome": None,
    }

    barrier.wait()

    semantic_key = job["semantic_key"]
    entity_id = job.get("entity_id")
    state_id = job.get("state_id")
    crash_after = job.get("crash_after")

    try:
        acquire_semantic_reservation(
            project_dir, semantic_key, attempt_id=attempt_id, owner_token=owner_token
        )
    except SemanticReservationConflictError:
        result["outcome"] = "LOST_SEMANTIC_BUSY"
        _write_worker_result(result_dir, tag, result)
        return
    result["semantic_acquired"] = True

    if crash_after == "semantic":
        os._exit(37)

    if entity_id is not None:
        try:
            acquire_entity_claim(
                project_dir, entity_id, attempt_id=attempt_id, owner_token=owner_token
            )
        except EntityClaimConflictError:
            result["outcome"] = "LOST_ENTITY_CLAIM_BUSY"
            _write_worker_result(result_dir, tag, result)
            return
        result["entity_claimed"] = True

    if state_id is not None:
        try:
            acquire_state_claim(
                project_dir, state_id, attempt_id=attempt_id, owner_token=owner_token
            )
        except StateClaimConflictError:
            result["outcome"] = "LOST_STATE_CLAIM_BUSY"
            _write_worker_result(result_dir, tag, result)
            return
        result["state_claimed"] = True

    if crash_after == "claims":
        os._exit(38)

    if job.get("publish"):
        record = _promotion_record(
            id=semantic_key,
            semantic_key=semantic_key,
            input_id=job.get("input_id", "curcumin-pubchem"),
            entity_id=entity_id,
            state_id=state_id,
            created_by_attempt_id=attempt_id,
        )
        try:
            register_promotion(project_dir, record)
            result["published"] = True
        except PromotionRecordConflictError:
            result["published"] = False
            result["outcome"] = "LOST_ALREADY_PROMOTED"
            _write_worker_result(result_dir, tag, result)
            return

    if crash_after == "after_promotion":
        os._exit(39)

    if job.get("release", True):
        if state_id is not None:
            release_state_claim(project_dir, state_id, attempt_id=attempt_id, owner_token=owner_token)
        if entity_id is not None:
            release_entity_claim(project_dir, entity_id, attempt_id=attempt_id, owner_token=owner_token)
        release_semantic_reservation(
            project_dir, semantic_key, attempt_id=attempt_id, owner_token=owner_token
        )

    result["outcome"] = "WON"
    _write_worker_result(result_dir, tag, result)


def _run_processes(jobs, project_dir: Path, result_dir: Path, timeout: float = 30):
    """Spawn one real OS process per job, release them simultaneously via a
    Barrier (never sleep), join with a deadlock-guard timeout only, and
    collect each worker's JSON result. Asserts no process is left alive -
    every process is joined or, on timeout, terminated and joined again,
    and the test fails loudly rather than leaking an orphan."""

    barrier = multiprocessing.Barrier(len(jobs))
    procs = [
        multiprocessing.Process(
            target=_promotion_chain_worker,
            args=(barrier, str(project_dir), job, str(result_dir)),
        )
        for job in jobs
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=timeout)

    orphans = [proc for proc in procs if proc.is_alive()]
    for proc in orphans:
        proc.terminate()
        proc.join(timeout=5)
    assert not orphans, f"{len(orphans)} worker process(es) did not finish within {timeout}s"

    results = {}
    if result_dir.exists():
        for entry in result_dir.glob("*.json"):
            results[entry.stem] = json.loads(entry.read_text(encoding="utf-8"))
    return results


# --- 151 ---


def test_eight_processes_same_key_same_ids_exactly_one_winner(tmp_path):
    jobs = [
        {"tag": f"p{i}", "semantic_key": _SEMANTIC_KEY, "entity_id": "curcumin", "state_id": "curcumin-pubchem-initial"}
        for i in range(8)
    ]
    results = _run_processes(jobs, tmp_path, tmp_path / "results")

    winners = [r for r in results.values() if r["outcome"] == "WON"]
    losers = [r for r in results.values() if r["outcome"] == "LOST_SEMANTIC_BUSY"]

    assert len(results) == 8
    assert len(winners) == 1
    assert len(losers) == 7
    assert describe_semantic_reservation(tmp_path, _SEMANTIC_KEY).state == "ABSENT_FREE"


# --- 152 / 168 ---


def test_eight_processes_same_key_different_ids_no_partial_claims(tmp_path):
    jobs = [
        {
            "tag": f"p{i}",
            "semantic_key": _SEMANTIC_KEY,
            "entity_id": f"entity-{i}",
            "state_id": f"state-{i}",
            "release": False,
        }
        for i in range(8)
    ]
    results = _run_processes(jobs, tmp_path, tmp_path / "results")

    winners = [r for r in results.values() if r["outcome"] == "WON"]
    losers = [r for r in results.values() if r["outcome"] != "WON"]

    assert len(winners) == 1
    assert len(losers) == 7
    # item 168: every loser lost at the semantic-reservation stage and
    # never attempted (let alone acquired) any claim at all.
    assert all(r["outcome"] == "LOST_SEMANTIC_BUSY" for r in losers)
    assert all(not r["entity_claimed"] and not r["state_claimed"] for r in losers)

    entity_claims_dir = tmp_path / "scientific" / "promotion_reservations" / "by-entity"
    state_claims_dir = tmp_path / "scientific" / "promotion_reservations" / "by-state"
    assert len(list(entity_claims_dir.iterdir())) == 1
    assert len(list(state_claims_dir.iterdir())) == 1


# --- 153 ---


def test_entity_claim_casefold_collision_across_processes(tmp_path):
    jobs = [
        {"tag": "A", "semantic_key": _SEMANTIC_KEY, "entity_id": "Entity-A", "release": False},
        {"tag": "B", "semantic_key": _OTHER_SEMANTIC_KEY, "entity_id": "entity-a", "release": False},
    ]
    results = _run_processes(jobs, tmp_path, tmp_path / "results")

    winners = [r for r in results.values() if r["outcome"] == "WON"]
    losers = [r for r in results.values() if r["outcome"] == "LOST_ENTITY_CLAIM_BUSY"]

    assert len(winners) == 1
    assert len(losers) == 1
    # both processes independently acquired their own, distinct semantic
    # reservation - the collision is exclusively on the entity claim.
    assert all(r["semantic_acquired"] for r in results.values())


# --- 154 ---


def test_state_claim_casefold_collision_across_processes(tmp_path):
    jobs = [
        {"tag": "A", "semantic_key": _SEMANTIC_KEY, "state_id": "State-A", "release": False},
        {"tag": "B", "semantic_key": _OTHER_SEMANTIC_KEY, "state_id": "state-a", "release": False},
    ]
    results = _run_processes(jobs, tmp_path, tmp_path / "results")

    winners = [r for r in results.values() if r["outcome"] == "WON"]
    losers = [r for r in results.values() if r["outcome"] == "LOST_STATE_CLAIM_BUSY"]

    assert len(winners) == 1
    assert len(losers) == 1


# --- 155 ---


def test_independent_transactions_both_win(tmp_path):
    jobs = [
        {"tag": "A", "semantic_key": _SEMANTIC_KEY, "entity_id": "entity-a", "state_id": "state-a"},
        {
            "tag": "B",
            "semantic_key": _OTHER_SEMANTIC_KEY,
            "entity_id": "entity-b",
            "state_id": "state-b",
        },
    ]
    results = _run_processes(jobs, tmp_path, tmp_path / "results")

    assert all(r["outcome"] == "WON" for r in results.values())


# --- 156 ---


def test_crash_after_semantic_reservation_leaves_detectable_residue(tmp_path):
    crash_job = [{"tag": "crasher", "semantic_key": _SEMANTIC_KEY, "crash_after": "semantic"}]
    crashed = _run_processes(crash_job, tmp_path, tmp_path / "results-1")
    assert crashed == {}  # the crashed worker never wrote a result file

    status_before = describe_semantic_reservation(tmp_path, _SEMANTIC_KEY)
    assert status_before.state == "RESERVED_KNOWN_OWNER"
    assert status_before.owner_attempt_id is not None

    contender_job = [{"tag": "contender", "semantic_key": _SEMANTIC_KEY, "entity_id": "curcumin"}]
    results = _run_processes(contender_job, tmp_path, tmp_path / "results-2")

    assert results["contender"]["outcome"] == "LOST_SEMANTIC_BUSY"
    assert not results["contender"]["entity_claimed"]
    # the residue is not stolen or mutated by the contender's failed attempt
    status_after = describe_semantic_reservation(tmp_path, _SEMANTIC_KEY)
    assert status_after.owner_attempt_id == status_before.owner_attempt_id


# --- 157 ---


def test_crash_after_claims_leaves_residue_no_duplication(tmp_path):
    crash_job = [
        {
            "tag": "crasher",
            "semantic_key": _SEMANTIC_KEY,
            "entity_id": "curcumin",
            "state_id": "curcumin-pubchem-initial",
            "crash_after": "claims",
        }
    ]
    _run_processes(crash_job, tmp_path, tmp_path / "results-1")

    semantic_status = describe_semantic_reservation(tmp_path, _SEMANTIC_KEY)
    entity_status = describe_entity_claim(tmp_path, "curcumin")
    state_status = describe_state_claim(tmp_path, "curcumin-pubchem-initial")
    assert semantic_status.state == "RESERVED_KNOWN_OWNER"
    assert entity_status.state == "RESERVED_KNOWN_OWNER"
    assert state_status.state == "RESERVED_KNOWN_OWNER"
    with pytest.raises(FileNotFoundError):
        load_promotion(tmp_path, _SEMANTIC_KEY)

    contender_job = [
        {"tag": "contender", "semantic_key": _SEMANTIC_KEY, "entity_id": "curcumin-2"}
    ]
    results = _run_processes(contender_job, tmp_path, tmp_path / "results-2")

    assert results["contender"]["outcome"] == "LOST_SEMANTIC_BUSY"
    # no duplication: still exactly one entity claim on disk (the crashed
    # owner's), the contender's alternate id was never claimed.
    entity_claims_dir = tmp_path / "scientific" / "promotion_reservations" / "by-entity"
    assert len(list(entity_claims_dir.iterdir())) == 1


# --- 158 ---


def test_crash_after_promotion_record_concurrent_recognizes_success(tmp_path):
    crash_job = [
        {
            "tag": "publisher",
            "semantic_key": _SEMANTIC_KEY,
            "entity_id": "curcumin",
            "state_id": "curcumin-pubchem-initial",
            "publish": True,
            "crash_after": "after_promotion",
        }
    ]
    _run_processes(crash_job, tmp_path, tmp_path / "results-1")

    published = load_promotion(tmp_path, _SEMANTIC_KEY)
    assert published.semantic_key == _SEMANTIC_KEY

    contender_job = [{"tag": "contender", "semantic_key": _SEMANTIC_KEY, "entity_id": "curcumin-2"}]
    results = _run_processes(contender_job, tmp_path, tmp_path / "results-2")

    # the contender never gets far enough to attempt publication at all -
    # the semantic reservation is still held by the crashed publisher.
    assert results["contender"]["outcome"] == "LOST_SEMANTIC_BUSY"
    assert not results["contender"]["published"]
    # the existing promotion record is unaffected
    assert load_promotion(tmp_path, _SEMANTIC_KEY).semantic_key == _SEMANTIC_KEY


# --- 159 / 160 ---


def test_owner_only_release_after_crash(tmp_path):
    crash_job = [{"tag": "crasher", "semantic_key": _SEMANTIC_KEY, "crash_after": "semantic"}]
    _run_processes(crash_job, tmp_path, tmp_path / "results")

    # A legitimate recovery process recovers the real credentials by
    # reading the persisted metadata directly - exactly as FS-018-02's own
    # S08 methodology did - never via a "log" or "report" API, which must
    # never expose owner_token (see test_describe_never_exposes_owner_token).
    slot_dir = tmp_path / "scientific" / "promotion_reservations" / _SEMANTIC_KEY
    real_metadata = yaml.safe_load((slot_dir / "reservation.yaml").read_text(encoding="utf-8"))
    real_attempt_id = real_metadata["attempt_id"]
    real_owner_token = real_metadata["owner_token"]

    with pytest.raises(ReservationOwnershipError):
        release_semantic_reservation(
            tmp_path, _SEMANTIC_KEY, attempt_id=real_attempt_id, owner_token="wrong-token"
        )
    assert slot_dir.is_dir()

    release_semantic_reservation(
        tmp_path, _SEMANTIC_KEY, attempt_id=real_attempt_id, owner_token=real_owner_token
    )
    assert not slot_dir.exists()


# --- 162 ---


def test_partial_acquisition_compensated_in_reverse_order_after_losing_state_claim(tmp_path):
    # Pre-occupy the state claim, simulating another owner already holding
    # it - the contending attempt below must lose specifically at the
    # state-claim stage after already acquiring semantic + entity.
    acquire_state_claim(
        tmp_path, "curcumin-pubchem-initial", attempt_id="z" * 32, owner_token="z" * 64
    )

    jobs = [
        {
            "tag": "contender",
            "semantic_key": _SEMANTIC_KEY,
            "entity_id": "curcumin",
            "state_id": "curcumin-pubchem-initial",
        }
    ]
    results = _run_processes(jobs, tmp_path, tmp_path / "results")

    contender = results["contender"]
    assert contender["outcome"] == "LOST_STATE_CLAIM_BUSY"
    assert contender["semantic_acquired"] is True
    assert contender["entity_claimed"] is True

    residuals = compensate_partial_acquisition(
        tmp_path,
        attempt_id=contender["attempt_id"],
        owner_token=_read_owner_token(tmp_path, "entity", "curcumin"),
        semantic_key=_SEMANTIC_KEY,
        entity_id="curcumin",
    )

    assert residuals == ()
    assert describe_semantic_reservation(tmp_path, _SEMANTIC_KEY).state == "ABSENT_FREE"
    assert describe_entity_claim(tmp_path, "curcumin").state == "ABSENT_FREE"
    # the pre-existing state claim, owned by someone else, is untouched
    assert describe_state_claim(tmp_path, "curcumin-pubchem-initial").state == "RESERVED_KNOWN_OWNER"


def _read_owner_token(project_dir: Path, kind: str, raw_key: str) -> str:
    """Test-only helper: read the real owner_token directly off disk for a
    resource this same test just had a real subprocess acquire - mirrors
    how a legitimate same-process continuation (or FS-018-02-style
    recovery) would recover its own credentials, never exposed by any
    describe_*/report function."""

    if kind == "semantic":
        slot_dir = promotion_registry_module._promotion_reservations_dir(project_dir) / raw_key
    elif kind == "entity":
        slot_dir = promotion_registry_module._entity_claims_dir(project_dir) / promotion_registry_module._claim_hash(
            raw_key
        )
    else:
        slot_dir = promotion_registry_module._state_claims_dir(project_dir) / promotion_registry_module._claim_hash(
            raw_key
        )
    data = yaml.safe_load((slot_dir / "reservation.yaml").read_text(encoding="utf-8"))
    return data["owner_token"]


# --- 164 ---


def test_promotion_record_publication_refuses_overwrite_across_processes(tmp_path):
    jobs = [
        {
            "tag": "A",
            "semantic_key": _SEMANTIC_KEY,
            "entity_id": "curcumin",
            "state_id": "curcumin-pubchem-initial",
            "publish": True,
        },
        {
            "tag": "B",
            "semantic_key": _SEMANTIC_KEY,
            "entity_id": "curcumin",
            "state_id": "curcumin-pubchem-initial",
            "publish": True,
        },
    ]
    results = _run_processes(jobs, tmp_path, tmp_path / "results")

    winners = [r for r in results.values() if r["outcome"] == "WON"]
    assert len(winners) == 1
    assert winners[0]["published"] is True

    promotions_dir = tmp_path / "scientific" / "promotions"
    assert len(list(promotions_dir.glob("*.yaml"))) == 1


# --- 165 ---


def test_promotion_record_recognized_as_commit_marker_not_removed_by_release(tmp_path):
    jobs = [
        {
            "tag": "publisher",
            "semantic_key": _SEMANTIC_KEY,
            "entity_id": "curcumin",
            "state_id": "curcumin-pubchem-initial",
            "publish": True,
            "release": True,
        }
    ]
    _run_processes(jobs, tmp_path, tmp_path / "results")

    # release cleans up the reservation/claims, but never the promotion
    # record itself - there is no release_promotion function at all.
    assert describe_semantic_reservation(tmp_path, _SEMANTIC_KEY).state == "ABSENT_FREE"
    assert load_promotion(tmp_path, _SEMANTIC_KEY).semantic_key == _SEMANTIC_KEY


# --- 164 (supplementary): the os.link exclusivity itself, isolated from the
# semantic-reservation gate. The chain-based test above can never actually
# race two processes at register_promotion for the same key (the semantic
# reservation already serializes them earlier) - this test calls
# register_promotion directly and simultaneously from real processes,
# stressing exactly the mechanism item 164 names ("criação exclusiva via
# hardlink"), independent of any other primitive in this module.


def _register_promotion_worker(barrier, project_dir_str, tag, semantic_key, entity_id, result_dir_str):
    project_dir = Path(project_dir_str)
    result_dir = Path(result_dir_str)
    record = _promotion_record(
        id=semantic_key,
        semantic_key=semantic_key,
        entity_id=entity_id,
        created_by_attempt_id=uuid.uuid4().hex,
    )

    barrier.wait()

    result = {"tag": tag}
    try:
        register_promotion(project_dir, record)
        result["outcome"] = "WON"
    except PromotionRecordConflictError:
        result["outcome"] = "LOST"
    _write_worker_result(result_dir, tag, result)


def test_register_promotion_exclusive_hardlink_under_real_concurrency(tmp_path):
    n = 8
    result_dir = tmp_path / "results"
    barrier = multiprocessing.Barrier(n)
    procs = [
        multiprocessing.Process(
            target=_register_promotion_worker,
            args=(barrier, str(tmp_path), f"p{i}", _SEMANTIC_KEY, f"entity-{i}", str(result_dir)),
        )
        for i in range(n)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
    orphans = [proc for proc in procs if proc.is_alive()]
    for proc in orphans:
        proc.terminate()
        proc.join(timeout=5)
    assert not orphans

    results = {entry.stem: json.loads(entry.read_text(encoding="utf-8")) for entry in result_dir.glob("*.json")}
    winners = [r for r in results.values() if r["outcome"] == "WON"]
    losers = [r for r in results.values() if r["outcome"] == "LOST"]

    assert len(results) == n
    assert len(winners) == 1
    assert len(losers) == n - 1
    promotions_dir = tmp_path / "scientific" / "promotions"
    assert len(list(promotions_dir.glob("*.yaml"))) == 1
    # no staging file survives from any of the n-1 losing attempts
    assert [p.name for p in promotions_dir.iterdir()] == [f"{_SEMANTIC_KEY}.yaml"]
