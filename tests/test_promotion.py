"""Domain tests for Increment 018 Checkpoint 1: LigandPromotionPolicy and
the rest of the promotion domain models (Section 6 of the frozen spec), the
semantic_key canonicalization algorithm (Section 10), and the golden test
vector (Section 10.3). Covers the 24 normative scenario lines of Section
19.1 of the frozen final specification
(``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``).
"""

import math
import uuid
from datetime import datetime, timezone

import pytest
import yaml
from pydantic import ValidationError

from molecular.promotion import (
    AcquisitionProvenanceSnapshot,
    LigandPromotionAttempt,
    LigandPromotionError,
    LigandPromotionPolicy,
    PromotionResultSnapshot,
    ResolvedLigandPromotionSnapshot,
    ScientificInputPromotionRecord,
    build_semantic_key_payload,
    canonicalize_payload,
    compute_semantic_key,
)
from molecular.scientific import MolecularRepresentationRef

_GOLDEN_DIGEST = "1bd810f669ad078fa744fc88b6248eeca8a5a2730d56493e518419dc316577f4"
_GOLDEN_DIGEST_SECOND_INPUT = (
    "e3dfa4f5f16f664981b511e2e5abea0596fbe618cf0524d9ef75ff24a630d996"
)
_GOLDEN_CANONICAL_JSON = (
    '{"engine_name":"rdkit","engine_version":"2026.03.6","input_id":"curcumin-pubchem",'
    '"policy":{"charge_policy":"preserve","component_policy":"reject-disconnected",'
    '"record_policy":"exactly-one","remove_hs":false,'
    '"representation_policy":"verified-source-bytes","sanitize":true,'
    '"stereo_policy":"preserve","strict_parsing":true},'
    '"semantic_key_algorithm_version":"1",'
    '"snapshot_checksum":"3f786850e387550fdab836ed7e6dc881de23001b0eb99f37a6e6dfb47a54a08"}'
)
_GOLDEN_SNAPSHOT_CHECKSUM = "3f786850e387550fdab836ed7e6dc881de23001b0eb99f37a6e6dfb47a54a08"
# NOTE: the frozen spec's own illustrative checksum (_GOLDEN_SNAPSHOT_CHECKSUM
# above, Section 6.10/10.3) is 63 hex characters, one short of a real
# SHA-256 digest - a pre-existing typo carried unchanged through every
# revision. It is used exactly as-is, verbatim, wherever the golden vector's
# own payload/canonical-JSON/digest values are being reproduced (those never
# pass through a Pydantic checksum-format validator). Everywhere a fixture
# instead needs a *valid-format* 64-character checksum to satisfy
# ResolvedLigandPromotionSnapshot's own format validator, _TEST_CHECKSUM
# (a real SHA-256 digest) is used instead.
_TEST_CHECKSUM = "0e4e4f1767e11f8529e663ae2ba078094cf05bc09b7b17eb96691f07096014bb"

_EIGHT_COMBINATIONS = [
    (sanitize, remove_hs, strict_parsing)
    for sanitize in (True, False)
    for remove_hs in (True, False)
    for strict_parsing in (True, False)
]


# --- builders -------------------------------------------------------------


def _policy(**overrides):
    fields = {}
    fields.update(overrides)
    return LigandPromotionPolicy(**fields)


def _resolved(**overrides):
    fields = {
        "engine_name": "rdkit",
        "engine_version": "2026.03.6",
        "input_format": "sdf",
        "raw_checksum_persisted": _TEST_CHECKSUM,
        "snapshot_checksum": _TEST_CHECKSUM,
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
        "warnings": (),
    }
    fields.update(overrides)
    return ResolvedLigandPromotionSnapshot(**fields)


def _acquisition(**overrides):
    fields = {
        "method": "downloaded",
        "source_name": "PubChem",
        "source_identifier": "curcumin-pubchem",
        "source_uri": "https://pubchem.ncbi.nlm.nih.gov/compound/969516",
        "acquired_at": datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc),
        "raw_artifact_path": "inputs/ligands/curcumin-pubchem/raw.sdf",
        "raw_artifact_format": "sdf",
        "raw_artifact_checksum": _TEST_CHECKSUM,
        "pubchem_cid": 969516,
        "pubchem_record_type": "2d",
    }
    fields.update(overrides)
    return AcquisitionProvenanceSnapshot(**fields)


def _representation(**overrides):
    fields = {
        "path": "scientific/artifacts/states/curcumin-pubchem-initial/source.sdf",
        "format": "sdf",
        "checksum": _TEST_CHECKSUM,
    }
    fields.update(overrides)
    return MolecularRepresentationRef(**fields)


def _result(**overrides):
    fields = {
        "promotion_id": _GOLDEN_DIGEST,
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


def _attempt(**overrides):
    fields = {
        "id": uuid.uuid4().hex,
        "input_id": "curcumin-pubchem",
        "requested_entity_id": "curcumin",
        "requested_state_id": "curcumin-pubchem-initial",
        "requested_entity_name": "Curcumin",
        "requested": _policy(),
        "status": "started",
        "created_at": datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc),
    }
    fields.update(overrides)
    return LigandPromotionAttempt(**fields)


def _promotion_record(**overrides):
    fields = {
        "id": _GOLDEN_DIGEST,
        "semantic_key": _GOLDEN_DIGEST,
        "input_id": "curcumin-pubchem",
        "acquisition": _acquisition(),
        "policy": _policy(),
        "resolved": _resolved(),
        "entity_id": "curcumin",
        "state_id": "curcumin-pubchem-initial",
        "representation": _representation(),
        "created_by_attempt_id": uuid.uuid4().hex,
        "created_at": datetime(2026, 9, 25, 15, 0, 2, tzinfo=timezone.utc),
    }
    fields.update(overrides)
    return ScientificInputPromotionRecord(**fields)


# --- 1: policy válida e frozen ---


def test_policy_defaults_are_the_production_candidate():
    policy = _policy()

    assert policy.sanitize is True
    assert policy.remove_hs is False
    assert policy.strict_parsing is True


def test_policy_is_frozen():
    policy = _policy()

    with pytest.raises(ValidationError):
        policy.sanitize = False


# --- 2: extra proibido ---


def test_policy_rejects_extra_fields():
    with pytest.raises(ValidationError):
        LigandPromotionPolicy(unexpected_field="value")


# --- 3: tipos bool permitem representar as 8 linhas da matriz ---


def test_policy_type_permits_all_eight_boolean_combinations():
    for sanitize, remove_hs, strict_parsing in _EIGHT_COMBINATIONS:
        policy = _policy(sanitize=sanitize, remove_hs=remove_hs, strict_parsing=strict_parsing)
        assert policy.sanitize is sanitize
        assert policy.remove_hs is remove_hs
        assert policy.strict_parsing is strict_parsing


# --- 4: resolved checksum válido/inválido ---


def test_resolved_snapshot_accepts_matching_valid_checksums():
    snapshot = _resolved()
    assert snapshot.snapshot_checksum == snapshot.raw_checksum_persisted


def test_resolved_snapshot_rejects_malformed_checksum():
    with pytest.raises(ValidationError):
        _resolved(raw_checksum_persisted="not-a-checksum", snapshot_checksum="not-a-checksum")


def test_resolved_snapshot_rejects_mismatched_checksums():
    other_checksum = "0" * 64
    with pytest.raises(ValidationError):
        _resolved(raw_checksum_persisted=other_checksum)


# --- 5: contagens não negativas ---


def test_resolved_snapshot_rejects_negative_counts():
    with pytest.raises(ValidationError):
        _resolved(atom_count=0)

    with pytest.raises(ValidationError):
        _resolved(explicit_hydrogen_count=-1)

    with pytest.raises(ValidationError):
        _resolved(isotopic_hydrogen_count=-1)


# --- 6: result reusado proíbe flags created ---


def test_result_reused_forbids_entity_or_state_created():
    with pytest.raises(ValidationError):
        _result(reused_existing_promotion=True, entity_created=True, state_created=False)

    with pytest.raises(ValidationError):
        _result(reused_existing_promotion=True, entity_created=False, state_created=True)

    reused = _result(reused_existing_promotion=True, entity_created=False, state_created=False)
    assert reused.reused_existing_promotion is True


# --- 7: error resolution proíbe resíduos ---


def test_error_resolution_phase_forbids_residuals():
    with pytest.raises(ValidationError):
        _error(phase="resolution", residual_entity_ids=("curcumin",))

    with pytest.raises(ValidationError):
        _error(phase="resolution", residual_reservation_keys=("a" * 64,))

    execution_error = _error(
        phase="execution", code="publication-failed", residual_entity_ids=("curcumin",)
    )
    assert execution_error.residual_entity_ids == ("curcumin",)


# --- 8: resíduos validam IDs e unicidade ---


def test_error_residuals_validate_id_format():
    with pytest.raises(ValidationError):
        _error(phase="execution", code="publication-failed", residual_entity_ids=("../bad",))


def test_error_residuals_reject_duplicates():
    with pytest.raises(ValidationError):
        _error(
            phase="execution",
            code="publication-failed",
            residual_state_ids=("state-1", "state-1"),
        )


# --- 9: cinco status válidos ---


def test_five_statuses_are_valid():
    assert _attempt(status="started").status == "started"
    assert _attempt(
        status="resolved", semantic_key=_GOLDEN_DIGEST, resolved=_resolved(),
        resolved_at=datetime.now(timezone.utc),
    ).status == "resolved"
    assert _attempt(
        status="resolution_failed", error=_error(), finished_at=datetime.now(timezone.utc)
    ).status == "resolution_failed"
    assert _attempt(
        status="execution_failed",
        semantic_key=_GOLDEN_DIGEST,
        resolved=_resolved(),
        error=_error(phase="execution", code="publication-failed"),
        resolved_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    ).status == "execution_failed"
    assert _attempt(
        status="succeeded",
        semantic_key=_GOLDEN_DIGEST,
        resolved=_resolved(),
        result=_result(),
        resolved_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    ).status == "succeeded"


def test_invalid_status_is_rejected():
    with pytest.raises(ValidationError):
        _attempt(status="not-a-real-status")


# --- 10: matriz completa status/campos ---


def test_started_attempt_forbids_all_terminal_fields():
    with pytest.raises(ValidationError):
        _attempt(status="started", semantic_key=_GOLDEN_DIGEST)

    with pytest.raises(ValidationError):
        _attempt(status="started", resolved=_resolved())

    with pytest.raises(ValidationError):
        _attempt(status="started", result=_result())

    with pytest.raises(ValidationError):
        _attempt(status="started", error=_error())

    with pytest.raises(ValidationError):
        _attempt(status="started", resolved_at=datetime.now(timezone.utc))

    with pytest.raises(ValidationError):
        _attempt(status="started", finished_at=datetime.now(timezone.utc))


def test_resolved_attempt_requires_semantic_key_and_resolved():
    with pytest.raises(ValidationError):
        _attempt(status="resolved", resolved_at=datetime.now(timezone.utc))

    with pytest.raises(ValidationError):
        _attempt(
            status="resolved", semantic_key=_GOLDEN_DIGEST, resolved_at=datetime.now(timezone.utc)
        )

    with pytest.raises(ValidationError):
        _attempt(
            status="resolved",
            semantic_key=_GOLDEN_DIGEST,
            resolved=_resolved(),
            result=_result(),
            resolved_at=datetime.now(timezone.utc),
        )


def test_resolution_failed_attempt_requires_matching_error_phase():
    with pytest.raises(ValidationError):
        _attempt(status="resolution_failed", finished_at=datetime.now(timezone.utc))

    with pytest.raises(ValidationError):
        _attempt(
            status="resolution_failed",
            error=_error(phase="execution", code="publication-failed"),
            finished_at=datetime.now(timezone.utc),
        )


def test_execution_failed_attempt_requires_resolved_and_matching_error_phase():
    with pytest.raises(ValidationError):
        _attempt(
            status="execution_failed",
            semantic_key=_GOLDEN_DIGEST,
            error=_error(phase="execution", code="publication-failed"),
            resolved_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    with pytest.raises(ValidationError):
        _attempt(
            status="execution_failed",
            semantic_key=_GOLDEN_DIGEST,
            resolved=_resolved(),
            error=_error(),
            resolved_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )


def test_succeeded_attempt_requires_result_and_forbids_error():
    with pytest.raises(ValidationError):
        _attempt(
            status="succeeded",
            semantic_key=_GOLDEN_DIGEST,
            resolved=_resolved(),
            resolved_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    with pytest.raises(ValidationError):
        _attempt(
            status="succeeded",
            semantic_key=_GOLDEN_DIGEST,
            resolved=_resolved(),
            result=_result(),
            error=_error(phase="execution", code="publication-failed"),
            resolved_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )


# --- 11: promotion record exige id == semantic_key ---


def test_promotion_record_requires_id_equal_semantic_key():
    with pytest.raises(ValidationError):
        _promotion_record(id="a" * 64)


# --- 12: representation checksum igual ao snapshot ---


def test_promotion_record_requires_representation_checksum_match():
    with pytest.raises(ValidationError):
        _promotion_record(representation=_representation(checksum="b" * 64))


# --- 13: intentional revision valida ID ---


def test_intentional_revision_validates_id_format():
    attempt = _attempt(intentional_revision="curcumin-revision-2")
    assert attempt.intentional_revision == "curcumin-revision-2"

    with pytest.raises(ValidationError):
        _attempt(intentional_revision="not a valid id")

    assert _attempt(intentional_revision=None).intentional_revision is None


# --- 14: YAML round-trip dos modelos ---


def test_attempt_round_trips_through_yaml():
    attempt = _attempt(
        status="succeeded",
        semantic_key=_GOLDEN_DIGEST,
        resolved=_resolved(),
        result=_result(),
        resolved_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )

    dumped = yaml.safe_dump(attempt.model_dump(mode="json"), sort_keys=False)
    loaded = LigandPromotionAttempt.model_validate(yaml.safe_load(dumped))

    assert loaded == attempt


def test_promotion_record_round_trips_through_yaml():
    record = _promotion_record()

    dumped = yaml.safe_dump(record.model_dump(mode="json"), sort_keys=False)
    loaded = ScientificInputPromotionRecord.model_validate(yaml.safe_load(dumped))

    assert loaded == record


# --- 15: vetor de teste golden da Seção 10.3 ---


def test_golden_vector_canonical_json_and_digest():
    payload = build_semantic_key_payload(
        input_id="curcumin-pubchem",
        snapshot_checksum=_GOLDEN_SNAPSHOT_CHECKSUM,
        engine_name="rdkit",
        engine_version="2026.03.6",
        policy=_policy(),
    )

    assert canonicalize_payload(payload) == _GOLDEN_CANONICAL_JSON
    assert compute_semantic_key(payload) == _GOLDEN_DIGEST


# --- 16: independência da ordem de chaves no nível superior ---


def test_top_level_key_order_does_not_affect_digest():
    payload_a = {
        "semantic_key_algorithm_version": "1",
        "input_id": "curcumin-pubchem",
        "snapshot_checksum": _GOLDEN_SNAPSHOT_CHECKSUM,
        "engine_name": "rdkit",
        "engine_version": "2026.03.6",
        "policy": {
            "sanitize": True,
            "remove_hs": False,
            "strict_parsing": True,
            "record_policy": "exactly-one",
            "component_policy": "reject-disconnected",
            "charge_policy": "preserve",
            "stereo_policy": "preserve",
            "representation_policy": "verified-source-bytes",
        },
    }
    payload_b = {
        "engine_version": "2026.03.6",
        "engine_name": "rdkit",
        "snapshot_checksum": _GOLDEN_SNAPSHOT_CHECKSUM,
        "input_id": "curcumin-pubchem",
        "semantic_key_algorithm_version": "1",
        "policy": payload_a["policy"],
    }

    assert compute_semantic_key(payload_a) == compute_semantic_key(payload_b) == _GOLDEN_DIGEST


# --- 17: independência da ordem de chaves em policy (aninhado) ---


def test_nested_policy_key_order_does_not_affect_digest():
    payload = build_semantic_key_payload(
        input_id="curcumin-pubchem",
        snapshot_checksum=_GOLDEN_SNAPSHOT_CHECKSUM,
        engine_name="rdkit",
        engine_version="2026.03.6",
        policy=_policy(),
    )
    reordered_policy = dict(reversed(list(payload["policy"].items())))
    reordered_payload = {**payload, "policy": reordered_policy}

    assert compute_semantic_key(reordered_payload) == _GOLDEN_DIGEST


# --- 18: preservação da ordem de listas ---


def test_canonicalization_preserves_list_order():
    payload = {"items": ["b", "a", "c"]}

    canonical = canonicalize_payload(payload)

    assert canonical == '{"items":["b","a","c"]}'


# --- 19: normalização NFC de strings Unicode equivalentes ---


def test_canonicalization_normalizes_unicode_to_nfc():
    composed = "Café"
    decomposed = "Café"
    assert composed != decomposed

    canonical_composed = canonicalize_payload({"name": composed})
    canonical_decomposed = canonicalize_payload({"name": decomposed})

    assert canonical_composed == canonical_decomposed


# --- 20: distinção entre valores semanticamente diferentes ---


def test_different_input_id_produces_different_digest():
    payload = build_semantic_key_payload(
        input_id="curcumin-pubchem-2",
        snapshot_checksum=_GOLDEN_SNAPSHOT_CHECKSUM,
        engine_name="rdkit",
        engine_version="2026.03.6",
        policy=_policy(),
    )

    assert compute_semantic_key(payload) == _GOLDEN_DIGEST_SECOND_INPUT
    assert _GOLDEN_DIGEST_SECOND_INPUT != _GOLDEN_DIGEST


# --- 21: rejeição de NaN/Infinity/-Infinity ---


def test_canonicalization_rejects_non_finite_floats():
    with pytest.raises(ValueError):
        canonicalize_payload({"value": math.nan})

    with pytest.raises(ValueError):
        canonicalize_payload({"value": math.inf})

    with pytest.raises(ValueError):
        canonicalize_payload({"value": -math.inf})


# --- 22: estabilidade após serialização e leitura YAML/JSON ---


def test_digest_stable_after_yaml_round_trip():
    payload = build_semantic_key_payload(
        input_id="curcumin-pubchem",
        snapshot_checksum=_GOLDEN_SNAPSHOT_CHECKSUM,
        engine_name="rdkit",
        engine_version="2026.03.6",
        policy=_policy(),
    )

    round_tripped = yaml.safe_load(yaml.safe_dump(payload))

    assert compute_semantic_key(round_tripped) == _GOLDEN_DIGEST


# --- 23: ausência de IDs, timestamps e caminhos incidentais no payload ---


def test_payload_contains_only_the_specified_fields():
    payload = build_semantic_key_payload(
        input_id="curcumin-pubchem",
        snapshot_checksum=_GOLDEN_SNAPSHOT_CHECKSUM,
        engine_name="rdkit",
        engine_version="2026.03.6",
        policy=_policy(),
    )

    assert set(payload.keys()) == {
        "semantic_key_algorithm_version",
        "input_id",
        "snapshot_checksum",
        "engine_name",
        "engine_version",
        "policy",
    }
    assert set(payload["policy"].keys()) == {
        "sanitize",
        "remove_hs",
        "strict_parsing",
        "record_policy",
        "component_policy",
        "charge_policy",
        "stereo_policy",
        "representation_policy",
    }


# --- 24: LigandPromotionPolicy aceita, sem erro, as 8 combinações ---


def test_policy_constructs_all_eight_combinations_without_error():
    for sanitize, remove_hs, strict_parsing in _EIGHT_COMBINATIONS:
        LigandPromotionPolicy(sanitize=sanitize, remove_hs=remove_hs, strict_parsing=strict_parsing)
