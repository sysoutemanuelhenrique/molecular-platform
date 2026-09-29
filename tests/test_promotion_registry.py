"""Persistence tests for Increment 018 Checkpoint 1: LigandPromotionAttempt
lifecycle (begin/load/transitions/atomic overwrite) and
ScientificInputPromotionRecord create-once publication and queries. Covers
the non-concurrency normative scenario lines of Section 19.2 of the frozen
final specification (items 25-33, 37-38) -
``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``.

Items 34-36 (exclusive `os.mkdir` semantic reservation, owner-only release,
casefold entity/state claims - Section 11.2-11.8) belong to the dedicated
cross-process-reservation checkpoint and are deliberately NOT exercised
here - this module has no reservation directories at all yet.
"""

import uuid
from datetime import datetime, timezone

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
    LigandPromotionAttemptIdentityConflictError,
    PromotionRecordConflictError,
    begin_promotion_attempt,
    list_promotions_by_input,
    list_promotions_by_state,
    load_promotion,
    load_promotion_attempt,
    mark_execution_failed,
    mark_resolution_failed,
    mark_resolved,
    mark_succeeded,
    register_promotion,
)
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
