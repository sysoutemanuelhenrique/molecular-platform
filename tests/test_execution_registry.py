"""Persistence and state-machine tests for execution attempts: begin/load,
every valid transition, every rejected transition, and the central
semantic guarantee that an attempt never creates or requires a
StateTransformationRecord.

operation_ids timing (D-015-12): a new attempt always starts with
operation_ids=None and stays None through every non-terminal-success
status; only mark_succeeded assigns it, as a required argument that may
be an empty tuple. Tests below exercise this explicitly - there is no
notion of "declaring" or "sharing" operation_ids at begin time anymore.
"""

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


def _begin(tmp_path, **overrides):
    fields = {
        "operation_type": "state-transformation",
        "requested": _policy(),
        "input_state_id": "state0001",
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
    assert attempt.operation_ids is None


def test_begin_does_not_accept_operation_ids(tmp_path):
    """begin_execution_attempt has no operation_ids parameter at all -
    which ids (if any) an attempt produces is only knowable once it
    succeeds, so declaring them at begin time is not a valid call."""

    with pytest.raises(TypeError):
        begin_execution_attempt(
            tmp_path,
            operation_type="state-transformation",
            operation_ids=("t1",),
            requested=_policy(),
        )


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


def test_load_round_trips(tmp_path):
    attempt = _begin(tmp_path)

    loaded = load_execution_attempt(tmp_path, attempt.id)

    assert loaded == attempt


def test_load_missing_attempt_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_execution_attempt(tmp_path, "does-not-exist")


def test_multiple_independent_attempts_each_start_with_no_operation_ids(tmp_path):
    """Retries are independent ExecutionAttempts, each with its own
    generated id - there is no notion of two attempts "sharing" a
    pre-declared operation_ids anymore, since no attempt declares one
    before it succeeds."""

    first = _begin(tmp_path)
    second = _begin(tmp_path)

    assert first.id != second.id
    assert first.operation_ids is None
    assert second.operation_ids is None

    assert load_execution_attempt(tmp_path, first.id).id == first.id
    assert load_execution_attempt(tmp_path, second.id).id == second.id


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


def test_mark_resolved_preserves_operation_ids_as_none(tmp_path):
    attempt = _begin(tmp_path)

    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    assert resolved.operation_ids is None


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


def test_mark_resolution_failed_ends_with_operation_ids_none(tmp_path):
    attempt = _begin(tmp_path)

    failed = mark_resolution_failed(tmp_path, attempt.id, error=_error(phase="resolution"))

    assert failed.operation_ids is None


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


def test_mark_execution_failed_ends_with_operation_ids_none(tmp_path):
    """This increment declares no partial results on execution failure -
    operation_ids stays None even though the attempt did reach
    'resolved' along the way."""

    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    failed = mark_execution_failed(tmp_path, attempt.id, error=_error(phase="execution"))

    assert failed.operation_ids is None


def test_mark_execution_failed_rejects_resolution_phase_error(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(ValidationError):
        mark_execution_failed(tmp_path, attempt.id, error=_error(phase="resolution"))


# --- mark_succeeded ---


def test_mark_succeeded_requires_operation_ids_argument(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(TypeError):
        mark_succeeded(tmp_path, attempt.id)


def test_mark_succeeded_rejects_none_operation_ids(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(TypeError):
        mark_succeeded(tmp_path, attempt.id, operation_ids=None)


def test_mark_succeeded_accepts_a_single_operation_id(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id, operation_ids=("t1",))

    assert succeeded.operation_ids == ("t1",)


def test_mark_succeeded_accepts_multiple_operation_ids(tmp_path):
    """One attempt (one engine invocation) may report more than one
    resulting StateTransformationRecord id - e.g. tautomer enumeration
    fanning out into several new states from one parent."""

    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id, operation_ids=("t1", "t2", "t3"))

    assert succeeded.operation_ids == ("t1", "t2", "t3")


def test_mark_succeeded_accepts_an_empty_tuple(tmp_path):
    """An empty tuple is a valid, meaningful result: the operation
    completed but produced no new transformations."""

    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id, operation_ids=())

    assert succeeded.operation_ids == ()


def test_mark_succeeded_sets_finished_at(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id, operation_ids=())

    assert succeeded.finished_at is not None


def test_mark_succeeded_preserves_resolved_snapshot_and_resolved_at(tmp_path):
    attempt = _begin(tmp_path)
    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id, operation_ids=())

    assert succeeded.error is None
    assert succeeded.resolved == resolved.resolved
    assert succeeded.resolved_at == resolved.resolved_at


def test_mark_succeeded_rejects_malformed_operation_id(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(ValidationError):
        mark_succeeded(tmp_path, attempt.id, operation_ids=("bad id",))


def test_mark_succeeded_rejects_duplicate_operation_ids(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(ValidationError):
        mark_succeeded(tmp_path, attempt.id, operation_ids=("t1", "t1"))


def test_mark_succeeded_validation_failure_does_not_alter_the_previous_yaml(tmp_path):
    """A rejected mark_succeeded call must leave the attempt's persisted
    file exactly as it was: ExecutionAttempt construction (which raises
    on a malformed/duplicate id) always happens before _persist, so a
    validation failure here can never have written anything."""

    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    metadata_file = tmp_path / "scientific" / "execution_attempts" / f"{attempt.id}.yaml"
    before = metadata_file.read_text(encoding="utf-8")

    with pytest.raises(ValidationError):
        mark_succeeded(tmp_path, attempt.id, operation_ids=("bad id",))

    after = metadata_file.read_text(encoding="utf-8")
    assert after == before

    # the attempt itself must still load as 'resolved', unaffected
    reloaded = load_execution_attempt(tmp_path, attempt.id)
    assert reloaded.status == "resolved"


# --- rejected transitions ---


def test_started_to_succeeded_is_rejected(tmp_path):
    attempt = _begin(tmp_path)

    with pytest.raises(ValueError):
        mark_succeeded(tmp_path, attempt.id, operation_ids=())


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
            mark_succeeded(tmp_path, attempt_id, operation_ids=()),
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
            mark_succeeded(tmp_path, attempt_id, operation_ids=()),
        ),
    ],
    ids=["resolution_failed", "succeeded"],
)
def test_terminal_attempt_rejects_mark_succeeded(tmp_path, terminal_transition):
    attempt = _begin(tmp_path)
    terminal_transition(tmp_path, attempt.id)

    with pytest.raises(ValueError):
        mark_succeeded(tmp_path, attempt.id, operation_ids=())


def test_resolved_to_resolved_again_is_rejected(tmp_path):
    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    with pytest.raises(ValueError):
        mark_resolved(tmp_path, attempt.id, resolved=_snapshot(engine_name="other"))


def test_unknown_attempt_transition_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        mark_resolved(tmp_path, "does-not-exist", resolved=_snapshot())

    with pytest.raises(FileNotFoundError):
        mark_succeeded(tmp_path, "does-not-exist", operation_ids=())


# --- transition preserves identity/requested/atomicity ---


def test_transition_preserves_immutable_identity_fields(tmp_path):
    attempt = _begin(tmp_path)

    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())
    succeeded = mark_succeeded(tmp_path, attempt.id, operation_ids=("t1",))

    for later in (resolved, succeeded):
        assert later.id == attempt.id
        assert later.operation_type == attempt.operation_type
        assert later.schema_version == attempt.schema_version
        assert later.created_at == attempt.created_at

    # operation_ids itself is the one field that legitimately changes -
    # None through 'started'/'resolved', only assigned at 'succeeded'.
    assert attempt.operation_ids is None
    assert resolved.operation_ids is None
    assert succeeded.operation_ids == ("t1",)


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


def test_serialization_distinguishes_none_from_empty_tuple(tmp_path):
    """None (not yet known) and an empty tuple (known: no results) must
    round-trip as distinguishable YAML values, not collapse into the
    same on-disk representation."""

    attempt = _begin(tmp_path)
    started_file = tmp_path / "scientific" / "execution_attempts" / f"{attempt.id}.yaml"
    started_data = yaml.safe_load(started_file.read_text(encoding="utf-8"))
    assert started_data["operation_ids"] is None

    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())
    succeeded = mark_succeeded(tmp_path, attempt.id, operation_ids=())

    succeeded_data = yaml.safe_load(started_file.read_text(encoding="utf-8"))
    assert succeeded_data["operation_ids"] == []

    reloaded = load_execution_attempt(tmp_path, attempt.id)
    assert reloaded.operation_ids == ()
    assert reloaded.operation_ids is not None
    assert reloaded == succeeded


# --- central semantic guarantee: attempts never touch transformations ---


def test_begin_does_not_require_or_create_any_state_transformation_record(tmp_path):
    """The whole point of an attempt: it targets no id at all until it
    succeeds, and even then, none of those ids need to pre-exist as a
    StateTransformationRecord."""

    transformations_dir = tmp_path / "scientific" / "transformations"
    assert not transformations_dir.exists()

    attempt = _begin(tmp_path)

    assert attempt.operation_ids is None
    assert not transformations_dir.exists()


def test_lifecycle_never_creates_a_state_transformation_record(tmp_path):
    """Neither begin nor any transition ever writes into
    scientific/transformations - Increment 015 registers intent and a
    state machine only, never a transformation."""

    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())
    mark_succeeded(tmp_path, attempt.id, operation_ids=("t-never-created",))

    transformations_dir = tmp_path / "scientific" / "transformations"
    assert not transformations_dir.exists()
    assert not (transformations_dir / "t-never-created.yaml").exists()


def test_lifecycle_with_multiple_operation_ids_never_creates_any_transformation(tmp_path):
    """Extends the single-id lifecycle test to the multi-target case: a
    full begin -> resolve -> succeed lifecycle reporting several
    operation_ids must still never write any of them into
    scientific/transformations."""

    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())
    mark_succeeded(tmp_path, attempt.id, operation_ids=("t-a", "t-b", "t-c"))

    transformations_dir = tmp_path / "scientific" / "transformations"
    assert not transformations_dir.exists()
    for operation_id in ("t-a", "t-b", "t-c"):
        assert not (transformations_dir / f"{operation_id}.yaml").exists()


def test_mark_succeeded_does_not_verify_any_declared_operation_id_resolves_to_a_transformation(
    tmp_path,
):
    """mark_succeeded records only the platform's own belief that its
    intended operation completed - it never checks that a
    StateTransformationRecord actually exists at any of operation_ids,
    since no producer of one exists in this increment. This is
    deliberately exercised with *several* ids, not just one: the
    guarantee (or rather, its absence here) is identical no matter how
    many ids one attempt reports. Reconciling this belief with real
    StateTransformationRecords (registering every one of them *before*
    ever calling mark_succeeded) is a caller contract this increment
    deliberately places on Increment 016's future orchestrating
    Application Service - see the D-015-11/D-015-12 amendment in
    execution_registry.py's module docstring - not something this
    registry enforces or could enforce without coupling it to a producer
    that does not yet exist."""

    attempt = _begin(tmp_path)
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(
        tmp_path,
        attempt.id,
        operation_ids=("t-never-registered-1", "t-never-registered-2", "t-never-registered-3"),
    )

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

    first = _begin(tmp_path)

    with pytest.raises(ExecutionAttemptIdentityConflictError):
        _begin(tmp_path)

    # the original attempt must be completely unaffected by the rejected
    # collision - not partially overwritten, not gone.
    reloaded = load_execution_attempt(tmp_path, first.id)
    assert reloaded == first
    assert reloaded.operation_ids is None


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


# --- Increment 016: input_state_id / schema_version 1.1 ---


def test_begin_requires_input_state_id_argument(tmp_path):
    with pytest.raises(TypeError):
        begin_execution_attempt(
            tmp_path, operation_type="state-transformation", requested=_policy()
        )


def test_begin_always_produces_schema_version_1_1(tmp_path):
    attempt = _begin(tmp_path)
    assert attempt.schema_version == "1.1"


def test_begin_never_constructs_schema_version_1_0(tmp_path):
    for _ in range(5):
        attempt = _begin(tmp_path, input_state_id=f"state{uuid.uuid4().hex[:8]}")
        assert attempt.schema_version == "1.1"


def test_begin_sets_input_state_id(tmp_path):
    attempt = _begin(tmp_path, input_state_id="state0042")
    assert attempt.input_state_id == "state0042"


def test_begin_rejects_malformed_input_state_id_before_any_io(tmp_path):
    attempts_dir = tmp_path / "scientific" / "execution_attempts"

    with pytest.raises(ValueError):
        _begin(tmp_path, input_state_id="bad id")

    assert not attempts_dir.exists()


def test_mark_resolved_preserves_input_state_id_and_schema_version(tmp_path):
    attempt = _begin(tmp_path, input_state_id="state0042")

    resolved = mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    assert resolved.input_state_id == "state0042"
    assert resolved.schema_version == "1.1"


def test_mark_resolution_failed_preserves_input_state_id_and_schema_version(tmp_path):
    attempt = _begin(tmp_path, input_state_id="state0042")

    failed = mark_resolution_failed(tmp_path, attempt.id, error=_error(phase="resolution"))

    assert failed.input_state_id == "state0042"
    assert failed.schema_version == "1.1"


def test_mark_execution_failed_preserves_input_state_id_and_schema_version(tmp_path):
    attempt = _begin(tmp_path, input_state_id="state0042")
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    failed = mark_execution_failed(tmp_path, attempt.id, error=_error(phase="execution"))

    assert failed.input_state_id == "state0042"
    assert failed.schema_version == "1.1"


def test_mark_succeeded_preserves_input_state_id_and_schema_version(tmp_path):
    attempt = _begin(tmp_path, input_state_id="state0042")
    mark_resolved(tmp_path, attempt.id, resolved=_snapshot())

    succeeded = mark_succeeded(tmp_path, attempt.id, operation_ids=())

    assert succeeded.input_state_id == "state0042"
    assert succeeded.schema_version == "1.1"


def test_load_legacy_schema_1_0_attempt_without_input_state_id(tmp_path):
    """A real 1.0-shaped YAML document (no input_state_id key at all, as
    Increment 015 would have written it) must load without error and
    without rewriting the file - no migration on read."""

    attempts_dir = tmp_path / "scientific" / "execution_attempts"
    attempts_dir.mkdir(parents=True)
    legacy_yaml = (
        "schema_version: '1.0'\n"
        "id: legacyattempt1\n"
        "operation_type: state-transformation\n"
        "operation_ids: null\n"
        "status: started\n"
        "requested:\n"
        "  schema_version: '1.0'\n"
        "  max_states: null\n"
        "  source: scientist\n"
        "resolved: null\n"
        "error: null\n"
        "created_at: '2024-01-01T00:00:00+00:00'\n"
        "resolved_at: null\n"
        "finished_at: null\n"
    )
    metadata_file = attempts_dir / "legacyattempt1.yaml"
    metadata_file.write_text(legacy_yaml, encoding="utf-8")
    before = metadata_file.read_text(encoding="utf-8")

    loaded = load_execution_attempt(tmp_path, "legacyattempt1")

    assert loaded.schema_version == "1.0"
    assert loaded.input_state_id is None

    after = metadata_file.read_text(encoding="utf-8")
    assert after == before
