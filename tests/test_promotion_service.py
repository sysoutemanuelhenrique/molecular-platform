"""Tests for the Application Service that orchestrates a full ligand RAW
promotion end to end (src/molecular/promotion_service.py, Checkpoint 4 of
Increment 018's frozen final specification). Requires RDKit - protected by
pytest.importorskip at module scope, since the Application Service itself
always calls into the RDKit inspection boundary during resolution."""

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

import molecular.promotion_service as promotion_service_module
from molecular.atomic import compute_sha256
from molecular.promotion import LigandPromotionPolicy
from molecular.promotion_registry import (
    describe_entity_claim,
    describe_semantic_reservation,
    describe_state_claim,
    list_promotions_by_input,
    load_promotion,
)
from molecular.promotion_service import (
    LigandPromotionAttemptFailedError,
    LigandPromotionAttemptPersistenceError,
    PromotionConsistencyError,
    PromotionIdentityConflictError,
    PromotionRevisionRequiredError,
    UnsupportedPromotionFormatError,
    promote_ligand_input,
)
from molecular.rdkit_tautomer import enumerate_tautomers_rdkit
from molecular.registry import register_ligand
from molecular.scientific import MolecularState, ScientificEntity
from molecular.scientific_registry import load_entity, load_state, register_state


def _write_sdf(tmp_path, filename, smiles="CCO"):
    mol = Chem.MolFromSmiles(smiles)
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def _register_ligand(
    project_dir, tmp_path, *, input_id="curcumin-pubchem", smiles="CCO", filename="raw.sdf"
):
    raw_file = _write_sdf(tmp_path / "sources", filename, smiles)
    return register_ligand(
        project_dir,
        input_id=input_id,
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
        file_format="sdf",
    )


def _promote(
    project_dir,
    *,
    input_id="curcumin-pubchem",
    entity_id="curcumin",
    state_id="curcumin-initial",
    entity_name="Curcumin",
    policy=None,
    intentional_revision=None,
):
    return promote_ligand_input(
        project_dir,
        input_id=input_id,
        entity_id=entity_id,
        state_id=state_id,
        entity_name=entity_name,
        policy=policy if policy is not None else LigandPromotionPolicy(),
        intentional_revision=intentional_revision,
    )


def _new_project(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return project_dir


# --- item 89: success with a new entity ---


def test_promote_success_creates_new_entity_and_state(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    attempt = _promote(project_dir)

    assert attempt.status == "succeeded"
    assert attempt.result.entity_created is True
    assert attempt.result.state_created is True
    assert attempt.result.reused_existing_promotion is False
    assert load_entity(project_dir, "curcumin").name == "Curcumin"
    assert load_state(project_dir, "curcumin-initial").entity_id == "curcumin"
    record = load_promotion(project_dir, attempt.semantic_key)
    assert record.entity_id == "curcumin"
    assert record.state_id == "curcumin-initial"


# --- item 90: success with an exactly-compatible preexisting entity ---


def test_promote_success_reuses_compatible_preexisting_entity(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)
    from molecular.scientific_registry import register_entity

    register_entity(project_dir, ScientificEntity(schema_version="1.0", id="curcumin", name="Curcumin"))

    attempt = _promote(project_dir)

    assert attempt.status == "succeeded"
    assert attempt.result.entity_created is False
    assert attempt.result.state_created is True


# --- item 91: incompatible preexisting entity / preexisting state without
# a promotion record are both conflicts ---


def test_promote_fails_on_incompatible_preexisting_entity(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)
    from molecular.scientific_registry import register_entity

    register_entity(
        project_dir, ScientificEntity(schema_version="1.0", id="curcumin", name="A Different Name")
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert isinstance(exc_info.value.original_error, PromotionIdentityConflictError)
    assert exc_info.value.status == "execution_failed"
    # the preexisting entity is left completely untouched
    assert load_entity(project_dir, "curcumin").name == "A Different Name"


def test_promote_fails_on_preexisting_state_without_promotion_record(tmp_path):
    project_dir = _new_project(tmp_path)
    ligand = _register_ligand(project_dir, tmp_path)
    from molecular.scientific_registry import register_entity

    register_entity(project_dir, ScientificEntity(schema_version="1.0", id="curcumin", name="Curcumin"))
    unrelated_source = _write_sdf(tmp_path, "unrelated.sdf", "CC")
    register_state(
        project_dir,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=unrelated_source,
        format="sdf",
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "execution_failed"
    # the preexisting, unrelated state is untouched - never removed, since
    # this attempt never created it.
    assert load_state(project_dir, "curcumin-initial").representation.path.endswith("unrelated.sdf")
    # the entity was NOT created by this attempt (it preexisted), so it must
    # survive too.
    assert load_entity(project_dir, "curcumin").name == "Curcumin"


# --- item 92: retry with the same ids returns the existing record ---


def test_promote_retry_same_ids_reuses_record(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    first = _promote(project_dir)
    second = _promote(project_dir)

    assert second.semantic_key == first.semantic_key
    assert second.result.reused_existing_promotion is True
    assert second.result.entity_created is False
    assert second.result.state_created is False
    assert second.result.promotion_id == first.result.promotion_id


# --- item 93: retry with different ids fails without publishing ---


def test_promote_retry_different_ids_fails(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)
    first = _promote(project_dir)

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir, entity_id="different-entity", state_id="different-state")

    assert isinstance(exc_info.value.original_error, PromotionIdentityConflictError)
    # the original record is untouched, and no second record was created.
    record = load_promotion(project_dir, first.semantic_key)
    assert record.entity_id == "curcumin"
    with pytest.raises(FileNotFoundError):
        load_entity(project_dir, "different-entity")


# --- item 94: semantic key changed without a revision fails ---


def test_promote_semantic_key_changed_without_revision_fails(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path, smiles="CCO")
    _promote(project_dir)

    # A different policy produces a different semantic_key for the SAME
    # input_id - re-registering under a new checksum is unnecessary; a
    # different policy alone is enough to change the key (Section 10.1).
    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir, policy=LigandPromotionPolicy(strict_parsing=False))

    assert isinstance(exc_info.value.original_error, PromotionRevisionRequiredError)


def test_promote_semantic_key_changed_with_revision_succeeds(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path, smiles="CCO")
    first = _promote(project_dir)

    second = _promote(
        project_dir,
        state_id="curcumin-revised",
        policy=LigandPromotionPolicy(strict_parsing=False),
        intentional_revision="rev-1",
    )

    assert second.semantic_key != first.semantic_key
    assert second.status == "succeeded"
    assert second.result.reused_existing_promotion is False
    records = list_promotions_by_input(project_dir, "curcumin-pubchem")
    assert len(records) == 2


# --- item 96: two distinct input_ids with identical RAW bytes produce two
# independent promotions ---


def test_two_distinct_inputs_with_identical_raw_bytes_produce_independent_promotions(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path, input_id="input-a", smiles="CCO", filename="a.sdf")
    _register_ligand(project_dir, tmp_path, input_id="input-b", smiles="CCO", filename="b.sdf")

    attempt_a = _promote(
        project_dir, input_id="input-a", entity_id="entity-a", state_id="state-a", entity_name="A"
    )
    attempt_b = _promote(
        project_dir, input_id="input-b", entity_id="entity-b", state_id="state-b", entity_name="B"
    )

    assert attempt_a.semantic_key != attempt_b.semantic_key
    assert load_promotion(project_dir, attempt_a.semantic_key).input_id == "input-a"
    assert load_promotion(project_dir, attempt_b.semantic_key).input_id == "input-b"


# --- item 97: checksum divergent before scientific write ---


def test_promote_fails_when_raw_checksum_diverges_before_scientific_write(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    ligand = _register_ligand(project_dir, tmp_path)

    real_load_ligand = promotion_service_module.load_ligand
    call_count = {"n": 0}

    def _fake_load_ligand(project_dir_arg, input_id_arg):
        real = real_load_ligand(project_dir_arg, input_id_arg)
        call_count["n"] += 1
        if call_count["n"] == 2:
            tampered_raw_artifact = real.acquisition.raw_artifact.model_copy(
                update={"checksum": "0" * 64}
            )
            tampered_acquisition = real.acquisition.model_copy(
                update={"raw_artifact": tampered_raw_artifact}
            )
            return real.model_copy(update={"acquisition": tampered_acquisition})
        return real

    monkeypatch.setattr(promotion_service_module, "load_ligand", _fake_load_ligand)

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "execution_failed"
    with pytest.raises(FileNotFoundError):
        load_entity(project_dir, "curcumin")


# --- item 98: each resolution failure persists status ---


def test_promote_resolution_failure_input_not_found_persists_status(tmp_path):
    project_dir = _new_project(tmp_path)

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "resolution_failed"
    assert isinstance(exc_info.value.original_error, FileNotFoundError)


def test_promote_resolution_failure_unsupported_format_persists_status(tmp_path):
    project_dir = _new_project(tmp_path)
    raw_file = tmp_path / "raw.mol2"
    raw_file.write_text("not an sdf", encoding="utf-8")
    register_ligand(
        project_dir,
        input_id="curcumin-pubchem",
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
        file_format="mol2",
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "resolution_failed"
    assert isinstance(exc_info.value.original_error, UnsupportedPromotionFormatError)


def test_promote_resolution_failure_invalid_sdf_persists_status_and_removes_snapshot(tmp_path):
    project_dir = _new_project(tmp_path)
    raw_file = tmp_path / "raw.sdf"
    raw_file.write_bytes(b"")
    register_ligand(
        project_dir,
        input_id="curcumin-pubchem",
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
        file_format="sdf",
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "resolution_failed"
    snapshot_dir = project_dir / "scientific" / "promotion_snapshots" / exc_info.value.attempt_id
    assert not snapshot_dir.exists()


# --- item 100: terminal persistence failure uses the wrapper ---


def test_promote_resolution_persistence_failure_uses_wrapper(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)

    def _poisoned_mark_resolution_failed(*args, **kwargs):
        raise OSError("simulated disk failure persisting resolution_failed")

    monkeypatch.setattr(
        promotion_service_module, "mark_resolution_failed", _poisoned_mark_resolution_failed
    )

    with pytest.raises(LigandPromotionAttemptPersistenceError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.intended_status == "resolution_failed"
    assert isinstance(exc_info.value.original_error, FileNotFoundError)
    assert isinstance(exc_info.value.persistence_error, OSError)


def test_promote_execution_persistence_failure_uses_wrapper(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    def _poisoned_mark_execution_failed(*args, **kwargs):
        raise OSError("simulated disk failure persisting execution_failed")

    monkeypatch.setattr(
        promotion_service_module, "mark_execution_failed", _poisoned_mark_execution_failed
    )
    monkeypatch.setattr(
        promotion_service_module,
        "register_promotion",
        lambda *a, **k: (_ for _ in ()).throw(OSError("simulated publish failure")),
    )

    with pytest.raises(LigandPromotionAttemptPersistenceError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.intended_status == "execution_failed"
    assert exc_info.value.phase == "execution"


# --- MINOR-01: mark_resolved itself failing, between the two phases ---


def test_promote_mark_resolved_failure_falls_back_to_resolution_failed(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    poisoned_error = RuntimeError("simulated mark_resolved failure")

    def _poisoned_mark_resolved(*args, **kwargs):
        raise poisoned_error

    monkeypatch.setattr(promotion_service_module, "mark_resolved", _poisoned_mark_resolved)

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "resolution_failed"
    assert exc_info.value.original_error is poisoned_error
    assert exc_info.value.__cause__ is poisoned_error
    assert exc_info.value.input_id == "curcumin-pubchem"

    # the private snapshot was cleaned up.
    snapshot_dir = project_dir / "scientific" / "promotion_snapshots" / exc_info.value.attempt_id
    assert not snapshot_dir.exists()

    # no reservation was ever acquired (the execution phase is never
    # reached at all), no scientific object published.
    assert not (project_dir / "scientific" / "promotion_reservations").exists()
    with pytest.raises(FileNotFoundError):
        load_entity(project_dir, "curcumin")
    with pytest.raises(FileNotFoundError):
        load_state(project_dir, "curcumin-initial")
    assert list_promotions_by_input(project_dir, "curcumin-pubchem") == []

    log_file = project_dir / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert any("outcome=error" in line for line in lines)


def test_promote_mark_resolved_and_fallback_both_fail_raises_persistence_error(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    mark_resolved_error = RuntimeError("simulated mark_resolved failure")
    mark_resolution_failed_error = OSError("simulated fallback persistence failure")

    def _poisoned_mark_resolved(*args, **kwargs):
        raise mark_resolved_error

    def _poisoned_mark_resolution_failed(*args, **kwargs):
        raise mark_resolution_failed_error

    monkeypatch.setattr(promotion_service_module, "mark_resolved", _poisoned_mark_resolved)
    monkeypatch.setattr(
        promotion_service_module, "mark_resolution_failed", _poisoned_mark_resolution_failed
    )

    with pytest.raises(LigandPromotionAttemptPersistenceError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.intended_status == "resolution_failed"
    assert exc_info.value.phase == "resolution"
    assert exc_info.value.original_error is mark_resolved_error
    assert exc_info.value.persistence_error is mark_resolution_failed_error
    assert exc_info.value.__cause__ is mark_resolution_failed_error
    assert exc_info.value.attempt_id
    assert exc_info.value.input_id == "curcumin-pubchem"

    with pytest.raises(FileNotFoundError):
        load_entity(project_dir, "curcumin")
    assert list_promotions_by_input(project_dir, "curcumin-pubchem") == []


def test_promote_mark_resolved_failure_leaves_raw_untouched(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    ligand = _register_ligand(project_dir, tmp_path)
    raw_path = project_dir / ligand.acquisition.raw_artifact.path
    raw_before = raw_path.read_bytes()

    monkeypatch.setattr(
        promotion_service_module,
        "mark_resolved",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated mark_resolved failure")),
    )

    with pytest.raises(LigandPromotionAttemptFailedError):
        _promote(project_dir)

    assert raw_path.read_bytes() == raw_before


# --- item 101/102: KeyboardInterrupt/SystemExit ---


def test_promote_keyboard_interrupt_during_resolution_propagates(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    def _raising_inspect(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(promotion_service_module, "inspect_single_sdf_for_promotion", _raising_inspect)

    with pytest.raises(KeyboardInterrupt):
        _promote(project_dir)


def test_promote_keyboard_interrupt_after_publication_propagates_without_compensation(
    tmp_path, monkeypatch
):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    def _raising_mark_succeeded(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(promotion_service_module, "mark_succeeded", _raising_mark_succeeded)

    with pytest.raises(KeyboardInterrupt):
        _promote(project_dir)

    # the record, entity, and state all survive - post-commit interrupt
    # never compensates.
    records = list_promotions_by_input(project_dir, "curcumin-pubchem")
    assert len(records) == 1
    assert load_entity(project_dir, "curcumin").name == "Curcumin"
    assert load_state(project_dir, "curcumin-initial") is not None


# --- item 103: entity created, state registration fails -> entity is
# compensated, reservations released ---


def test_promote_entity_created_state_registration_fails_compensates_entity(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    # pre-register a DIFFERENT entity owning the target state_id, so that
    # register_state's own identity-conflict check fires deterministically.
    from molecular.scientific_registry import register_entity

    register_entity(project_dir, ScientificEntity(schema_version="1.0", id="other-owner", name="Other"))
    unrelated_source = _write_sdf(tmp_path, "unrelated2.sdf", "CC")
    register_state(
        project_dir,
        state_id="curcumin-initial",
        entity_id="other-owner",
        source_file=unrelated_source,
        format="sdf",
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "execution_failed"
    # the freshly-created entity was rolled back.
    with pytest.raises(FileNotFoundError):
        load_entity(project_dir, "curcumin")
    # reservations were released (never left dangling for a failed attempt).
    assert describe_entity_claim(project_dir, "curcumin").state == "ABSENT_FREE"
    assert describe_state_claim(project_dir, "curcumin-initial").state == "ABSENT_FREE"


# --- item 104: state created, lineage (promotion record) publication
# fails -> both entity and state are compensated ---


def test_promote_state_created_publication_fails_compensates_entity_and_state(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    monkeypatch.setattr(
        promotion_service_module,
        "register_promotion",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated publish failure")),
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "execution_failed"
    with pytest.raises(FileNotFoundError):
        load_entity(project_dir, "curcumin")
    with pytest.raises(FileNotFoundError):
        load_state(project_dir, "curcumin-initial")
    assert list_promotions_by_input(project_dir, "curcumin-pubchem") == []


def test_promote_publication_record_conflict_while_holding_reservation_is_consistency_error(
    tmp_path, monkeypatch
):
    """A `PromotionRecordConflictError` from `register_promotion` while this
    attempt still exclusively holds the semantic reservation indicates a
    genuine anomaly (no normal concurrent caller can produce it, Section
    11.5) - translated to `PromotionConsistencyError`, distinct from an
    ordinary publication failure."""

    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    from molecular.promotion_registry import PromotionRecordConflictError

    monkeypatch.setattr(
        promotion_service_module,
        "register_promotion",
        lambda *a, **k: (_ for _ in ()).throw(PromotionRecordConflictError("simulated anomaly")),
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert isinstance(exc_info.value.original_error, PromotionConsistencyError)


# --- item 105: lineage committed, status persistence fails -> promotion
# is never removed ---


def test_promote_status_persistence_fails_after_commit_raises_persistence_error_not_execution_failed(
    tmp_path, monkeypatch
):
    """BLOCKER-01 correction: once `register_promotion` has committed the
    record, a `mark_succeeded` failure must NEVER be reclassified as
    `execution_failed` (Section 13.3's own literal rule: "falha ao gravar
    status do attempt produz erro de persistência operacional"). It must
    raise `LigandPromotionAttemptPersistenceError` directly, with
    `intended_status="succeeded"`, never attempting `mark_execution_failed`
    at all - proven here by making `mark_execution_failed` itself explode
    if called."""

    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    real_mark_succeeded = promotion_service_module.mark_succeeded
    call_count = {"n": 0}
    poisoned_error = RuntimeError("simulated status persistence failure")

    def _flaky_mark_succeeded(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise poisoned_error
        return real_mark_succeeded(*args, **kwargs)

    def _forbidden_mark_execution_failed(*args, **kwargs):
        raise AssertionError(
            "mark_execution_failed must never be called once the promotion "
            "record is already committed"
        )

    monkeypatch.setattr(promotion_service_module, "mark_succeeded", _flaky_mark_succeeded)
    monkeypatch.setattr(
        promotion_service_module, "mark_execution_failed", _forbidden_mark_execution_failed
    )

    with pytest.raises(LigandPromotionAttemptPersistenceError) as exc_info:
        _promote(project_dir)

    failed_attempt_id = exc_info.value.attempt_id
    assert exc_info.value.input_id == "curcumin-pubchem"
    assert exc_info.value.phase == "execution"
    assert exc_info.value.intended_status == "succeeded"
    assert exc_info.value.persistence_error is poisoned_error
    assert exc_info.value.original_error is poisoned_error
    assert exc_info.value.__cause__ is poisoned_error

    # the promotion record, entity, and state are ALL intact - a status
    # persistence failure after commit never undoes the commit, and never
    # compensates anything.
    records = list_promotions_by_input(project_dir, "curcumin-pubchem")
    assert len(records) == 1
    assert load_entity(project_dir, "curcumin").name == "Curcumin"
    assert load_state(project_dir, "curcumin-initial") is not None

    # the ORIGINAL attempt is never persisted with a false terminal status -
    # it simply stays at its last honestly-persisted status ("resolved").
    from molecular.promotion_registry import load_promotion_attempt

    original_attempt = load_promotion_attempt(project_dir, failed_attempt_id)
    assert original_attempt.status == "resolved"
    assert original_attempt.error is None

    # a subsequent retry is idempotent: it finds the commit marker and
    # returns success, without duplicating entity, state, or record.
    retry = _promote(project_dir)
    assert retry.status == "succeeded"
    assert retry.id != failed_attempt_id
    assert retry.result.reused_existing_promotion is True
    assert retry.result.entity_created is False
    assert retry.result.state_created is False
    assert list_promotions_by_input(project_dir, "curcumin-pubchem") == records


# --- item 106: residuals survive a persistence failure; the promotion
# record is never itself a residual ---


def test_promote_residuals_recorded_when_compensation_cannot_confirm_removal(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    monkeypatch.setattr(
        promotion_service_module,
        "register_promotion",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated publish failure")),
    )
    monkeypatch.setattr(
        promotion_service_module, "remove_state_if_matches_and_unreferenced", lambda *a, **k: False
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    from molecular.promotion_registry import load_promotion_attempt

    persisted = load_promotion_attempt(project_dir, exc_info.value.attempt_id)
    assert "curcumin-initial" in persisted.error.residual_state_ids
    # no field on LigandPromotionError even exists to name a residual
    # promotion id - structurally impossible for the record to appear as one.
    assert not hasattr(persisted.error, "residual_promotion_ids")


# --- item 108: RAW remains byte-for-byte intact across every failure mode
# exercised above ---


def test_promote_raw_artifact_untouched_across_failures(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    ligand = _register_ligand(project_dir, tmp_path)
    raw_path = project_dir / ligand.acquisition.raw_artifact.path
    raw_before = raw_path.read_bytes()
    checksum_before = compute_sha256(raw_path)

    monkeypatch.setattr(
        promotion_service_module,
        "register_promotion",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated publish failure")),
    )
    with pytest.raises(LigandPromotionAttemptFailedError):
        _promote(project_dir)

    assert raw_path.read_bytes() == raw_before
    assert compute_sha256(raw_path) == checksum_before


# --- item 109/110: the resulting state is valid and accepted by
# enumerate_tautomers_rdkit ---


def test_promote_resulting_state_accepted_by_enumerate_tautomers(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path, smiles="CC(=O)CC(C)=O")

    attempt = _promote(project_dir)

    state = load_state(project_dir, attempt.result.state_id)
    assert isinstance(state, MolecularState)
    source_path = project_dir / state.representation.path

    result = enumerate_tautomers_rdkit(source_path=source_path, source_format="sdf")
    assert result.enumerated_unique_count >= 1


# --- item 107: activity started/success/error, and best-effort failure ---


def test_promote_success_records_started_and_success_activity(tmp_path):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    _promote(project_dir)

    log_file = project_dir / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert any("command=ligand-promote" in line and "outcome=started" in line for line in lines)
    assert any("command=ligand-promote" in line and "outcome=success" in line for line in lines)


def test_promote_failure_records_started_and_error_activity(tmp_path):
    project_dir = _new_project(tmp_path)

    with pytest.raises(LigandPromotionAttemptFailedError):
        _promote(project_dir)

    log_file = project_dir / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert any("command=ligand-promote" in line and "outcome=started" in line for line in lines)
    assert any("command=ligand-promote" in line and "outcome=error" in line for line in lines)


def test_promote_activity_logging_failure_never_breaks_a_successful_promotion(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    def _poisoned_record_activity(*args, **kwargs):
        raise OSError("simulated activity log failure")

    monkeypatch.setattr(promotion_service_module, "record_activity", _poisoned_record_activity)

    attempt = _promote(project_dir)

    assert attempt.status == "succeeded"


# --- item 162: reverse-order compensation of a partial acquisition (lost
# the state claim after already acquiring semantic + entity) ---


def test_partial_acquisition_compensated_in_reverse_order_on_state_claim_loss(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    def _poisoned_state_claim(*args, **kwargs):
        raise promotion_service_module.StateClaimConflictError("simulated: another attempt owns this state claim")

    monkeypatch.setattr(promotion_service_module, "acquire_state_claim", _poisoned_state_claim)

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    assert exc_info.value.status == "execution_failed"
    from molecular.promotion_registry import load_promotion_attempt

    persisted = load_promotion_attempt(project_dir, exc_info.value.attempt_id)
    assert describe_semantic_reservation(project_dir, persisted.semantic_key).state == "ABSENT_FREE"
    assert describe_entity_claim(project_dir, "curcumin").state == "ABSENT_FREE"
    assert describe_state_claim(project_dir, "curcumin-initial").state == "ABSENT_FREE"
    with pytest.raises(FileNotFoundError):
        load_entity(project_dir, "curcumin")


# --- item 163: failure to confirm a release produces a typed residual,
# never a silent claim of full compensation ---


def test_release_failure_produces_typed_residual_not_silent_success(tmp_path, monkeypatch):
    project_dir = _new_project(tmp_path)
    _register_ligand(project_dir, tmp_path)

    monkeypatch.setattr(
        promotion_service_module,
        "register_promotion",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated publish failure")),
    )

    real_release_entity_claim = promotion_service_module.compensate_partial_acquisition

    def _partially_failing_compensate(*args, **kwargs):
        # Simulate an entity-claim release that cannot be confirmed by
        # pre-removing its metadata so ownership can't be proven, while
        # still exercising the real function for the other resources.
        from molecular.promotion_registry import _entity_claims_dir, _claim_hash

        slot_dir = _entity_claims_dir(project_dir) / _claim_hash("curcumin")
        metadata_file = slot_dir / "reservation.yaml"
        if metadata_file.exists():
            metadata_file.unlink()
        return real_release_entity_claim(*args, **kwargs)

    monkeypatch.setattr(
        promotion_service_module, "compensate_partial_acquisition", _partially_failing_compensate
    )

    with pytest.raises(LigandPromotionAttemptFailedError) as exc_info:
        _promote(project_dir)

    from molecular.promotion_registry import load_promotion_attempt

    persisted = load_promotion_attempt(project_dir, exc_info.value.attempt_id)
    assert "curcumin" in persisted.error.residual_reservation_keys


# --- MINOR-C1 item 7: the public API never accepts a caller-supplied
# snapshot path of any kind ---


def test_promote_ligand_input_signature_has_no_snapshot_path_parameter():
    import inspect

    parameters = inspect.signature(promote_ligand_input).parameters
    assert "snapshot_path" not in parameters
    assert "snapshot_root" not in parameters
    assert set(parameters) == {
        "project_dir",
        "input_id",
        "entity_id",
        "state_id",
        "entity_name",
        "policy",
        "intentional_revision",
    }
