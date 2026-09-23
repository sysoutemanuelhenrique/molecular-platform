"""Persistence and state-machine tests for execution attempts: begin/load,
every valid transition, every rejected transition, and the central
semantic guarantee that an attempt never creates or requires a
StateTransformationRecord."""

import uuid

import pytest
import yaml
from pydantic import ValidationError

from molecular.execution import (
    ExecutionAttemptError,
    ResolvedExecutionSnapshot,
    TautomerEnumerationPolicy,
)
from molecular.execution_registry import (
    ExecutionAttemptIdentityConflictError,
    begin_execution_attempt,
    load_execution_attempt,
    mark_execution_failed,
    mark_resolution_failed,
    mark_resolved,
    mark_succeeded,
)


def _policy(**overrides):
    fields = {"schema_version": "1.0", "max_states": None, "source": "scientist"}
    fields.update(overrides)
    return TautomerEnumerationPolicy(**fields)


def _snapshot(**overrides):
    fields = {
        "schema_version": "1.0",
        "engine_name": "rdkit",
        "engine_version": "2024.03",
        "parameters": {"max_states": 5},
    }
    fields.update(overrides)
    return ResolvedExecutionSnapshot(**fields)


def _error(**overrides):
    fields = {"phase": "resolution", "message": "no engine available"}
    fields.update(overrides)
    return ExecutionAttemptError(**fields)


def _begin(tmp_path, operation_id="t-future-001", **overrides):
    """`operation_id` is a convenience for the common single-target case -
    it is wrapped into a 1-tuple for the real (plural) operation_ids
    parameter. Pass operation_ids=(...) directly in overrides to exercise
    a multi-target attempt instead."""

    fields = {
        "operation_type": "state-transformation",
        "operation_ids": (operation_id,),
        "requested": _policy(),
    }
    fields.update(overrides)
    return begin_execution_attempt(tmp_path, **fields)


# --- begin / load ---


def test_begin_creates_started_attempt(tmp_path):
    attempt = _begin(tmp_path)

    assert attempt.status == "started"
    assert attempt.resolved is None
    assert attempt.error is None
    assert attempt.resolved_at is None
    assert attempt.finished_at is None
    assert attempt.operation_type == "state-transformation"
    assert attempt.operation_ids == ("t-future-001",)


def test_begin_id_is_a_uuid4_hex_string(tmp_path):
    attempt = _begin(tmp_path)

    parsed = uuid.UUID(attempt.id)
    assert parsed.version == 4
    assert attempt.id == parsed.hex


def test_begin_persists_to_the_expected_path(tmp_path):
    attempt = _begin(tmp_path)

    metadata_file = tmp_path / "scientific" / "execution_attempts" / f"{attempt.id}.yaml"
    assert metadata_file.exists()


def test_begin_rejects_unsupported_operation_type(tmp_path):
    with pytest.raises(ValueError):
        _begin(tmp_path, operation_type="docking-run")


def test_begin_rejects_malformed_operation_id(tmp_path):
    with pytest.raises(ValueError):
        _begin(tmp_path, operation_id="bad id")


def test_load_round_trips(tmp_path):
    attempt = _begin(tmp_path)

    loaded = load_execution_attempt(tmp_path, attempt.id)

    assert loaded == attempt


def test_load_missing_attempt_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_execution_attempt(tmp_path, "does-not-exist")


def test_multiple_attempts_may_share_the_same_operation_id(tmp_path):
    first = _begin(tmp_path, operation_id="t-shared")
    second = _begin(tmp_path, operation_id="t-shared")

    assert first.id != second.id
    assert first.operation_ids == second.operation_ids == ("t-shared",)

    assert load_execution_attempt(tmp_path, first.id).id == first.id
    assert load_execution_attempt(tmp_path, second.id).id == second.id


def test_begin_supports_multiple_operation_ids_for_a_single_attempt(tmp_path):
    """One attempt models one engine invocation, which may fan out into
    several resulting StateTransformationRecords (e.g. tautomer
    enumeration producing several new states from one parent) - it does
    not force an artificial 1:1 split across several attempts for what
    is, physically, a single execution event."""

    attempt = begin_execution_attempt(
        tmp_path,
        operation_type="state-transformation",
        operation_ids=("t1", "t2", "t3"),
        requested=_policy(),
    )

    assert attempt.operation_ids == ("t1", "t2", "t3")


def test_begin_rejects_empty_operation_ids(tmp_path):
    with pytest.raises(ValueError):
        begin_execution_attempt(
            tmp_path,
            operation_type="state-transformation",
            operation_ids=(),
            requested=_policy(),
        )


def test_begin_rejects_duplicate_operation_ids(tmp_path):
    with pytest.raises(ValueError):
        begin_execution_attempt(
            tmp_path,
            operation_type="state-transformation",
            operation_ids=("t1", "t1"),
            requested=_policy(),
        )


def test_lifecycle_with_multiple_operation_ids_never_creates_any_transformation(tmp_path):
    """Extends test_lifecycle_never_creates_a_state_transformation_record
    to the multi-target case: a full begin -> resolve -> succeed
    lifecycle for an attempt declaring several operation_ids must still
    never write any of them into scientific/transformations."""

    attempt = _begin(tmp_path, operation_ids=("t-a", "t-b", "t-c"))
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())
    mark_succeeded(tmp_path, attempt.id)

    transformations_dir = tmp_path / "scientific" / "transformations"
    assert not transformations_dir.exists()
    for operation_id in ("t-a", "t-b", "t-c"):
        assert not (transformations_dir / f"{operation_id}.yaml").exists()


# --- mark_resolved ---


def test_mark_resolved_transitions_started_to_resolved(tmp_path):
    attempt = _begin(tmp_path)
    snapshot = _snapshot()

    resolved = mark_resolved(tmp_path, attempt.id, resolved=snapshot)

    assert resolved.status == "resolved"
    assert resolved.resolved == snapshot
    assert resolved.error is None
    assert resolved.finished_at is None


def test_mark_resolved_sets_resolved_at(tmp_path):
    attempt = _begin(tmp_path)
    assert attempt.resolved_at is None

    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    assert resolved.resolved_at is not None


# --- mark_resolution_failed ---


def test_mark_resolution_failed_transitions_started_to_resolution_failed(tmp_path):
    attempt = _begin(tmp_path)

    failed = mark_resolution_failed(
        tmp_path, attempt.id, error=_error(phase="resolution", message="no engine")
    )

    assert failed.status == "resolution_failed"
    assert failed.resolved is None
    assert failed.error.phase == "resolution"
    assert failed.resolved_at is None
    assert failed.finished_at is not None


def test_mark_resolution_failed_rejects_execution_phase_error(tmp_path):
    attempt = _begin(tmp_path)

    with pytest.raises(ValidationError):
        mark_resolution_failed(tmp_path, attempt.id, error=_error(phase="execution"))


# --- mark_execution_failed ---


def test_mark_execution_failed_transitions_resolved_to_execution_failed(tmp_path):
    attempt = _begin(tmp_path)
    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    failed = mark_execution_failed(
        tmp_path, attempt.id, error=_error(phase="execution", message="engine crashed")
    )

    assert failed.status == "execution_failed"
    assert failed.resolved == resolved.resolved
    assert failed.resolved_at == resolved.resolved_at
    assert failed.error.phase == "execution"
    assert failed.finished_at is not None


def test_mark_execution_failed_rejects_resolution_phase_error(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(ValidationError):
        mark_execution_failed(tmp_path, attempt.id, error=_error(phase="resolution"))


# --- mark_succeeded ---


def test_mark_succeeded_transitions_resolved_to_succeeded(tmp_path):
    attempt = _begin(tmp_path)
    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id)

    assert succeeded.status == "succeeded"
    assert succeeded.error is None
    assert succeeded.resolved == resolved.resolved
    assert succeeded.resolved_at == resolved.resolved_at


def test_mark_succeeded_sets_finished_at(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id)

    assert succeeded.finished_at is not None


# --- rejected transitions ---


def test_started_to_succeeded_is_rejected(tmp_path):
    attempt = _begin(tmp_path)

    with pytest.raises(ValueError):
        mark_succeeded(tmp_path, attempt.id)


def test_started_to_execution_failed_is_rejected(tmp_path):
    attempt = _begin(tmp_path)

    with pytest.raises(ValueError):
        mark_execution_failed(tmp_path, attempt.id, error=_error(phase="execution"))


def test_resolved_to_resolution_failed_is_rejected(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(ValueError):
        mark_resolution_failed(tmp_path, attempt.id, error=_error(phase="resolution"))


@pytest.mark.parametrize(
    "terminal_transition",
    [
        lambda tmp_path, attempt_id: mark_resolution_failed(
            tmp_path, attempt_id, error=_error(phase="resolution")
        ),
        lambda tmp_path, attempt_id: (
            mark_resolved(tmp_path, attempt_id, resolved=_snapshot()),
            mark_succeeded(tmp_path, attempt_id),
        ),
    ],
    ids=["resolution_failed", "succeeded"],
)
def test_terminal_attempt_rejects_mark_resolved(tmp_path, terminal_transition):
    attempt = _begin(tmp_path)
    terminal_transition(tmp_path, attempt.id)

    with pytest.raises(ValueError):
        mark_resolved(tmp_path, attempt.id, resolved=_snapshot())


@pytest.mark.parametrize(
    "terminal_transition",
    [
        lambda tmp_path, attempt_id: mark_resolution_failed(
            tmp_path, attempt_id, error=_error(phase="resolution")
        ),
        lambda tmp_path, attempt_id: (
            mark_resolved(tmp_path, attempt_id, resolved=_snapshot()),
            mark_succeeded(tmp_path, attempt_id),
        ),
    ],
    ids=["resolution_failed", "succeeded"],
)
def test_terminal_attempt_rejects_mark_succeeded(tmp_path, terminal_transition):
    attempt = _begin(tmp_path)
    terminal_transition(tmp_path, attempt.id)

    with pytest.raises(ValueError):
        mark_succeeded(tmp_path, attempt.id)


def test_resolved_to_resolved_again_is_rejected(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(ValueError):
        mark_resolved(tmp_path, attempt.id, resolved=_snapshot(engine_name="other"))


def test_unknown_attempt_transition_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        mark_resolved(tmp_path, "does-not-exist", resolved=_snapshot())

    with pytest.raises(FileNotFoundError):
        mark_succeeded(tmp_path, "does-not-exist")


# --- transition preserves identity/requested/atomicity ---


def test_transition_preserves_immutable_identity_fields(tmp_path):
    attempt = _begin(tmp_path)

    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())
    succeeded = mark_succeeded(tmp_path, attempt.id)

    for later in (resolved, succeeded):
        assert later.id == attempt.id
        assert later.operation_type == attempt.operation_type
        assert later.operation_ids == attempt.operation_ids
        assert later.schema_version == attempt.schema_version
        assert later.created_at == attempt.created_at


def test_transition_preserves_requested_policy(tmp_path):
    policy = _policy(max_states=7)
    attempt = _begin(tmp_path, requested=policy)

    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    assert resolved.requested == policy


def test_transition_overwrites_the_same_file_atomically(tmp_path):
    attempt = _begin(tmp_path)
    metadata_file = tmp_path / "scientific" / "execution_attempts" / f"{attempt.id}.yaml"

    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    # Same single file, no second file for the new status - a transition
    # rewrites the one record in place, it never appends or forks it.
    execution_dir = tmp_path / "scientific" / "execution_attempts"
    assert [entry.name for entry in execution_dir.iterdir()] == [metadata_file.name]

    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    assert data["status"] == "resolved"


# --- central semantic guarantee: attempts never touch transformations ---


def test_begin_does_not_require_an_existing_state_transformation_record(tmp_path):
    """The whole point of an attempt: it may target an operation_id that
    no StateTransformationRecord has ever been registered under."""

    transformations_dir = tmp_path / "scientific" / "transformations"
    assert not transformations_dir.exists()

    attempt = _begin(tmp_path, operation_id="t-does-not-exist-yet")

    assert attempt.operation_ids == ("t-does-not-exist-yet",)
    assert not (transformations_dir / "t-does-not-exist-yet.yaml").exists()


def test_lifecycle_never_creates_a_state_transformation_record(tmp_path):
    """Neither begin nor any transition ever writes into
    scientific/transformations - Increment 015 registers intent and a
    state machine only, never a transformation."""

    attempt = _begin(tmp_path, operation_id="t-never-created")
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())
    mark_succeeded(tmp_path, attempt.id)

    transformations_dir = tmp_path / "scientific" / "transformations"
    assert not transformations_dir.exists()
    assert not (transformations_dir / "t-never-created.yaml").exists()


def test_mark_succeeded_does_not_verify_any_declared_operation_id_resolves_to_a_transformation(
    tmp_path,
):
    """mark_succeeded records only the platform's own belief that its
    intended operation completed - it never checks that a
    StateTransformationRecord actually exists at any of operation_ids,
    since no producer of one exists in this increment. This is
    deliberately exercised with *several* declared operation_ids, not
    just one: D-015-11's original wording read as a per-record guarantee
    tied to a single target, but the guarantee (or rather, its absence
    here) is identical no matter how many ids one attempt declares.
    Reconciling this belief with real StateTransformationRecords
    (registering every one of them *before* ever calling mark_succeeded)
    is a caller contract this increment deliberately places on Increment
    016's future orchestrating Application Service - see the D-015-11
    amendment in execution_registry.py's module docstring - not
    something this registry enforces or could enforce without coupling
    it to a producer that does not yet exist."""

    attempt = _begin(
        tmp_path, operation_ids=("t-never-registered-1", "t-never-registered-2", "t-never-registered-3")
    )
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id)

    assert succeeded.status == "succeeded"
    assert succeeded.operation_ids == (
        "t-never-registered-1",
        "t-never-registered-2",
        "t-never-registered-3",
    )

    transformations_dir = tmp_path / "scientific" / "transformations"
    for operation_id in succeeded.operation_ids:
        assert not (transformations_dir / f"{operation_id}.yaml").exists()


def test_begin_rejects_a_colliding_generated_attempt_id(tmp_path, monkeypatch):
    """A uuid4 collision is astronomically unlikely in practice, but
    begin_execution_attempt must never silently overwrite a pre-existing
    attempt file if it happens: atomic_write_text's underlying
    os.replace() replaces an existing destination unconditionally, with
    no existence check of its own, so this registry must guard its own
    create path explicitly."""

    fixed_uuid = uuid.uuid4()
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)

    first = _begin(tmp_path, operation_id="t-one")

    with pytest.raises(ExecutionAttemptIdentityConflictError):
        _begin(tmp_path, operation_id="t-two")

    # the original attempt must be completely unaffected by the rejected
    # collision - not partially overwritten, not gone.
    reloaded = load_execution_attempt(tmp_path, first.id)
    assert reloaded == first
    assert reloaded.operation_ids == ("t-one",)


def test_mark_resolved_still_overwrites_the_same_attempts_own_file_after_collision_guard(
    tmp_path,
):
    """The create-only collision guard in begin_execution_attempt must
    never interfere with a transition's own, expected overwrite of that
    same attempt's file."""

    attempt = _begin(tmp_path)

    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    assert resolved.id == attempt.id
    assert resolved.status == "resolved"
