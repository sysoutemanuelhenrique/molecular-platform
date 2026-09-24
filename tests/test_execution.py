"""Model-level tests for the execution attempt foundation: field
validation, immutability, and the per-status invariants ExecutionAttempt
enforces on itself. Storage/state-machine behavior belongs to
test_execution_registry.py, not here."""

import pytest
from pydantic import ValidationError

from molecular.execution import (
    ExecutionAttempt,
    ExecutionAttemptError,
    ResolvedExecutionSnapshot,
    TautomerEnumerationPolicy,
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


def _started_attempt(**overrides):
    from datetime import datetime, timezone

    fields = {
        "schema_version": "1.0",
        "id": "attempt0001",
        "operation_type": "state-transformation",
        "operation_ids": None,
        "status": "started",
        "requested": _policy(),
        "resolved": None,
        "error": None,
        "created_at": datetime.now(timezone.utc),
        "resolved_at": None,
        "finished_at": None,
    }
    fields.update(overrides)
    return ExecutionAttempt(**fields)


# --- TautomerEnumerationPolicy ---


def test_tautomer_enumeration_policy_accepts_valid_max_states():
    policy = _policy(max_states=10)
    assert policy.max_states == 10


def test_tautomer_enumeration_policy_accepts_none_max_states():
    policy = _policy(max_states=None)
    assert policy.max_states is None


def test_tautomer_enumeration_policy_rejects_zero_max_states():
    with pytest.raises(ValidationError):
        _policy(max_states=0)


def test_tautomer_enumeration_policy_rejects_negative_max_states():
    with pytest.raises(ValidationError):
        _policy(max_states=-1)


def test_tautomer_enumeration_policy_rejects_invalid_source():
    with pytest.raises(ValidationError):
        _policy(source="preset")


def test_tautomer_enumeration_policy_rejects_extra_field():
    with pytest.raises(ValidationError):
        TautomerEnumerationPolicy(
            schema_version="1.0", max_states=5, source="scientist", enumerate=True
        )


def test_tautomer_enumeration_policy_is_immutable():
    policy = _policy(max_states=5)
    with pytest.raises(ValidationError):
        policy.max_states = 10


# --- ResolvedExecutionSnapshot ---


def test_resolved_execution_snapshot_valid_construction():
    snapshot = _snapshot(parameters={"count": 5, "flag": True, "label": "x", "ratio": 1.5})
    assert snapshot.engine_name == "rdkit"
    assert snapshot.parameters["ratio"] == 1.5


def test_resolved_execution_snapshot_rejects_nested_non_scalar_parameter():
    with pytest.raises(ValidationError):
        _snapshot(parameters={"bad": [1, 2, 3]})


def test_resolved_execution_snapshot_rejects_none_parameter_value():
    with pytest.raises(ValidationError):
        _snapshot(parameters={"bad": None})


def test_resolved_execution_snapshot_rejects_extra_field():
    with pytest.raises(ValidationError):
        ResolvedExecutionSnapshot(
            schema_version="1.0",
            engine_name="rdkit",
            engine_version="2024.03",
            parameters={},
            extra_field="not allowed",
        )


def test_resolved_execution_snapshot_is_immutable():
    snapshot = _snapshot()
    with pytest.raises(ValidationError):
        snapshot.engine_name = "other"


def test_resolved_execution_snapshot_parameters_cannot_be_mutated_in_place():
    """model_config frozen=True only blocks reassigning the `parameters`
    field itself (snapshot.parameters = {...}); it does nothing to stop
    mutating a mutable dict already sitting in that field. `parameters`
    must be a read-only MappingProxyType precisely to close that gap -
    this is a distinct guarantee from test_..._is_immutable above, which
    only proves top-level field reassignment is blocked."""

    snapshot = _snapshot(parameters={"count": 5})

    with pytest.raises(TypeError):
        snapshot.parameters["count"] = 999

    assert snapshot.parameters["count"] == 5


def test_resolved_execution_snapshot_default_parameters_are_also_frozen():
    """The empty default (no parameters passed at all) must be frozen
    too, not just an explicitly-supplied dict."""

    snapshot = ResolvedExecutionSnapshot(
        schema_version="1.0", engine_name="rdkit", engine_version="2024.03"
    )

    with pytest.raises(TypeError):
        snapshot.parameters["x"] = 1


def test_resolved_execution_snapshot_parameters_round_trip_via_model_dump():
    """Freezing parameters into a MappingProxyType must not change the
    persisted shape: model_dump still produces a plain JSON-safe dict,
    and re-validating it reconstructs an equal, still-frozen snapshot."""

    snapshot = _snapshot(parameters={"count": 5, "flag": True, "label": "x", "ratio": 1.5})

    dumped = snapshot.model_dump(mode="json")
    assert dumped["parameters"] == {"count": 5, "flag": True, "label": "x", "ratio": 1.5}
    assert isinstance(dumped["parameters"], dict)

    reloaded = ResolvedExecutionSnapshot.model_validate(dumped)

    assert reloaded == snapshot
    assert dict(reloaded.parameters) == dict(snapshot.parameters)
    with pytest.raises(TypeError):
        reloaded.parameters["count"] = 0


# --- ExecutionAttemptError ---


def test_execution_attempt_error_valid_resolution_phase():
    error = _error(phase="resolution", message="no engine")
    assert error.phase == "resolution"


def test_execution_attempt_error_valid_execution_phase():
    error = _error(phase="execution", message="engine crashed")
    assert error.phase == "execution"


def test_execution_attempt_error_rejects_invalid_phase():
    with pytest.raises(ValidationError):
        _error(phase="planning")


def test_execution_attempt_error_rejects_blank_message():
    with pytest.raises(ValidationError):
        _error(message="   ")


def test_execution_attempt_error_is_immutable():
    error = _error()
    with pytest.raises(ValidationError):
        error.phase = "execution"


# --- ExecutionAttempt: construction ---


def test_execution_attempt_valid_started():
    attempt = _started_attempt()
    assert attempt.status == "started"
    assert attempt.resolved is None
    assert attempt.error is None
    assert attempt.operation_ids is None


def test_execution_attempt_rejects_malformed_id():
    with pytest.raises(ValidationError):
        _started_attempt(id="bad id")


def test_execution_attempt_rejects_invalid_operation_type():
    with pytest.raises(ValidationError):
        _started_attempt(operation_type="docking-run")


def test_execution_attempt_rejects_extra_field():
    from datetime import datetime, timezone

    with pytest.raises(ValidationError):
        ExecutionAttempt(
            schema_version="1.0",
            id="attempt0001",
            operation_type="state-transformation",
            status="started",
            requested=_policy(),
            created_at=datetime.now(timezone.utc),
            engine="rdkit",
        )


def test_execution_attempt_is_immutable():
    attempt = _started_attempt()
    with pytest.raises(ValidationError):
        attempt.status = "resolved"


# --- ExecutionAttempt: per-status invariants ---


def test_started_invariants_enforced():
    from datetime import datetime, timezone

    # valid started construction already covered above; here we prove the
    # invariant is actively enforced, not merely satisfied by default.
    with pytest.raises(ValidationError):
        _started_attempt(resolved=_snapshot())

    with pytest.raises(ValidationError):
        _started_attempt(error=_error())

    with pytest.raises(ValidationError):
        _started_attempt(resolved_at=datetime.now(timezone.utc))

    with pytest.raises(ValidationError):
        _started_attempt(finished_at=datetime.now(timezone.utc))

    # operation_ids is only ever assigned by mark_succeeded - a 'started'
    # attempt declaring it is a timing violation, not a valid state.
    with pytest.raises(ValidationError):
        _started_attempt(operation_ids=("t1",))


def test_resolution_failed_invariants_enforced():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    valid = _started_attempt(
        status="resolution_failed",
        error=_error(phase="resolution"),
        finished_at=now,
    )
    assert valid.status == "resolution_failed"
    assert valid.operation_ids is None

    # resolved must be None
    with pytest.raises(ValidationError):
        _started_attempt(
            status="resolution_failed",
            resolved=_snapshot(),
            error=_error(phase="resolution"),
            finished_at=now,
        )

    # error is required
    with pytest.raises(ValidationError):
        _started_attempt(status="resolution_failed", finished_at=now)

    # error.phase must be "resolution"
    with pytest.raises(ValidationError):
        _started_attempt(
            status="resolution_failed",
            error=_error(phase="execution"),
            finished_at=now,
        )

    # resolved_at must remain None
    with pytest.raises(ValidationError):
        _started_attempt(
            status="resolution_failed",
            error=_error(phase="resolution"),
            resolved_at=now,
            finished_at=now,
        )

    # finished_at is required
    with pytest.raises(ValidationError):
        _started_attempt(status="resolution_failed", error=_error(phase="resolution"))

    # operation_ids never applies to a failure that never resolved
    with pytest.raises(ValidationError):
        _started_attempt(
            status="resolution_failed",
            error=_error(phase="resolution"),
            finished_at=now,
            operation_ids=("t1",),
        )


def test_resolved_invariants_enforced():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    valid = _started_attempt(status="resolved", resolved=_snapshot(), resolved_at=now)
    assert valid.status == "resolved"
    assert valid.operation_ids is None

    # resolved is required
    with pytest.raises(ValidationError):
        _started_attempt(status="resolved", resolved_at=now)

    # error must be None
    with pytest.raises(ValidationError):
        _started_attempt(
            status="resolved", resolved=_snapshot(), resolved_at=now, error=_error()
        )

    # resolved_at is required
    with pytest.raises(ValidationError):
        _started_attempt(status="resolved", resolved=_snapshot())

    # finished_at must remain None
    with pytest.raises(ValidationError):
        _started_attempt(
            status="resolved", resolved=_snapshot(), resolved_at=now, finished_at=now
        )

    # operation_ids is not yet knowable at 'resolved' - execution has not
    # run yet, only its configuration has been determined.
    with pytest.raises(ValidationError):
        _started_attempt(
            status="resolved", resolved=_snapshot(), resolved_at=now, operation_ids=("t1",)
        )


def test_execution_failed_invariants_enforced():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    valid = _started_attempt(
        status="execution_failed",
        resolved=_snapshot(),
        error=_error(phase="execution"),
        resolved_at=now,
        finished_at=now,
    )
    assert valid.status == "execution_failed"
    assert valid.operation_ids is None

    # resolved is required
    with pytest.raises(ValidationError):
        _started_attempt(
            status="execution_failed",
            error=_error(phase="execution"),
            resolved_at=now,
            finished_at=now,
        )

    # error is required
    with pytest.raises(ValidationError):
        _started_attempt(
            status="execution_failed", resolved=_snapshot(), resolved_at=now, finished_at=now
        )

    # error.phase must be "execution"
    with pytest.raises(ValidationError):
        _started_attempt(
            status="execution_failed",
            resolved=_snapshot(),
            error=_error(phase="resolution"),
            resolved_at=now,
            finished_at=now,
        )

    # resolved_at is required
    with pytest.raises(ValidationError):
        _started_attempt(
            status="execution_failed",
            resolved=_snapshot(),
            error=_error(phase="execution"),
            finished_at=now,
        )

    # finished_at is required
    with pytest.raises(ValidationError):
        _started_attempt(
            status="execution_failed",
            resolved=_snapshot(),
            error=_error(phase="execution"),
            resolved_at=now,
        )

    # this increment declares no partial results on execution failure
    with pytest.raises(ValidationError):
        _started_attempt(
            status="execution_failed",
            resolved=_snapshot(),
            error=_error(phase="execution"),
            resolved_at=now,
            finished_at=now,
            operation_ids=("t1",),
        )


def test_succeeded_invariants_enforced():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    valid = _started_attempt(
        status="succeeded",
        resolved=_snapshot(),
        resolved_at=now,
        finished_at=now,
        operation_ids=(),
    )
    assert valid.status == "succeeded"
    assert valid.operation_ids == ()

    # resolved is required
    with pytest.raises(ValidationError):
        _started_attempt(
            status="succeeded", resolved_at=now, finished_at=now, operation_ids=()
        )

    # error must be None
    with pytest.raises(ValidationError):
        _started_attempt(
            status="succeeded",
            resolved=_snapshot(),
            resolved_at=now,
            finished_at=now,
            operation_ids=(),
            error=_error(),
        )

    # resolved_at is required
    with pytest.raises(ValidationError):
        _started_attempt(
            status="succeeded", resolved=_snapshot(), finished_at=now, operation_ids=()
        )

    # finished_at is required
    with pytest.raises(ValidationError):
        _started_attempt(
            status="succeeded", resolved=_snapshot(), resolved_at=now, operation_ids=()
        )

    # operation_ids is required - this is the one status where it must be
    # set, precisely because it is the one status where the result is
    # actually known.
    with pytest.raises(ValidationError):
        _started_attempt(status="succeeded", resolved=_snapshot(), resolved_at=now, finished_at=now)


# --- ExecutionAttempt: operation_ids only exists once succeeded ---


def test_succeeded_accepts_empty_operation_ids():
    """An empty tuple is a valid, meaningful result: the operation
    completed but produced no new transformations."""

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    attempt = _started_attempt(
        status="succeeded",
        resolved=_snapshot(),
        resolved_at=now,
        finished_at=now,
        operation_ids=(),
    )
    assert attempt.operation_ids == ()


def test_succeeded_accepts_single_operation_id():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    attempt = _started_attempt(
        status="succeeded",
        resolved=_snapshot(),
        resolved_at=now,
        finished_at=now,
        operation_ids=("t1",),
    )
    assert attempt.operation_ids == ("t1",)


def test_succeeded_accepts_multiple_operation_ids():
    """One attempt (one engine invocation) may report more than one
    resulting StateTransformationRecord id - e.g. tautomer enumeration
    fanning out into several new states from one parent."""

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    attempt = _started_attempt(
        status="succeeded",
        resolved=_snapshot(),
        resolved_at=now,
        finished_at=now,
        operation_ids=("t1", "t2", "t3"),
    )
    assert attempt.operation_ids == ("t1", "t2", "t3")


def test_succeeded_rejects_malformed_operation_ids_entry():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        _started_attempt(
            status="succeeded",
            resolved=_snapshot(),
            resolved_at=now,
            finished_at=now,
            operation_ids=("bad id",),
        )


def test_succeeded_rejects_duplicate_operation_ids():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        _started_attempt(
            status="succeeded",
            resolved=_snapshot(),
            resolved_at=now,
            finished_at=now,
            operation_ids=("t1", "t1"),
        )


# --- Increment 016: schema_version 1.1, input_state_id, residuals ---


def _started_attempt_11(**overrides):
    """Same shape as _started_attempt, but schema_version="1.1" with a
    valid input_state_id - the shape every new attempt actually uses."""

    fields = {"schema_version": "1.1", "input_state_id": "state0001"}
    fields.update(overrides)
    return _started_attempt(**fields)


def _error_with_residuals(**overrides):
    fields = {
        "phase": "execution",
        "message": "compensation could not confirm removal",
        "residual_state_ids": ("state0002",),
        "residual_operation_ids": ("transform0002",),
    }
    fields.update(overrides)
    return ExecutionAttemptError(**fields)


# --- ExecutionAttemptError: residuals (Section 10.4/10.5) ---


def test_execution_attempt_error_resolution_phase_accepts_empty_residuals():
    error = _error(phase="resolution")
    assert error.residual_state_ids == ()
    assert error.residual_operation_ids == ()


def test_execution_attempt_error_resolution_phase_rejects_residual_state_ids():
    with pytest.raises(ValidationError):
        ExecutionAttemptError(
            phase="resolution", message="x", residual_state_ids=("state0001",)
        )


def test_execution_attempt_error_resolution_phase_rejects_residual_operation_ids():
    with pytest.raises(ValidationError):
        ExecutionAttemptError(
            phase="resolution", message="x", residual_operation_ids=("t1",)
        )


def test_execution_attempt_error_execution_phase_accepts_empty_residuals():
    error = _error(phase="execution")
    assert error.residual_state_ids == ()
    assert error.residual_operation_ids == ()


def test_execution_attempt_error_execution_phase_accepts_nonempty_residuals():
    error = _error_with_residuals()
    assert error.residual_state_ids == ("state0002",)
    assert error.residual_operation_ids == ("transform0002",)


def test_execution_attempt_error_rejects_malformed_residual_state_id():
    with pytest.raises(ValidationError):
        _error_with_residuals(residual_state_ids=("bad id",))


def test_execution_attempt_error_rejects_malformed_residual_operation_id():
    with pytest.raises(ValidationError):
        _error_with_residuals(residual_operation_ids=("bad id",))


def test_execution_attempt_error_rejects_duplicate_residual_state_ids():
    with pytest.raises(ValidationError):
        _error_with_residuals(residual_state_ids=("state0002", "state0002"))


def test_execution_attempt_error_rejects_duplicate_residual_operation_ids():
    with pytest.raises(ValidationError):
        _error_with_residuals(residual_operation_ids=("transform0002", "transform0002"))


def test_execution_attempt_error_legacy_dict_without_residual_fields_defaults_empty():
    """A dict shaped like a pre-016 error (no residual fields at all) must
    still validate, with both tuples defaulting empty - no migration or
    reshaping is required to read a legacy record."""

    error = ExecutionAttemptError.model_validate(
        {"phase": "resolution", "message": "no engine available"}
    )
    assert error.residual_state_ids == ()
    assert error.residual_operation_ids == ()


# --- schema_version x status x input_state_id matrix (Section 10.3) ---


@pytest.mark.parametrize("status", ["started", "resolution_failed", "resolved", "execution_failed", "succeeded"])
def test_schema_1_0_accepts_none_input_state_id_in_every_status(status):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    kwargs = {"schema_version": "1.0", "status": status}
    if status in ("resolution_failed",):
        kwargs.update(error=_error(phase="resolution"), finished_at=now)
    elif status == "resolved":
        kwargs.update(resolved=_snapshot(), resolved_at=now)
    elif status == "execution_failed":
        kwargs.update(
            resolved=_snapshot(), error=_error(phase="execution"), resolved_at=now, finished_at=now
        )
    elif status == "succeeded":
        kwargs.update(resolved=_snapshot(), resolved_at=now, finished_at=now, operation_ids=())

    attempt = _started_attempt(**kwargs)
    assert attempt.input_state_id is None


@pytest.mark.parametrize("status", ["started", "resolution_failed", "resolved", "execution_failed", "succeeded"])
def test_schema_1_0_rejects_non_none_input_state_id_in_every_status(status):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    kwargs = {"schema_version": "1.0", "status": status, "input_state_id": "state0001"}
    if status in ("resolution_failed",):
        kwargs.update(error=_error(phase="resolution"), finished_at=now)
    elif status == "resolved":
        kwargs.update(resolved=_snapshot(), resolved_at=now)
    elif status == "execution_failed":
        kwargs.update(
            resolved=_snapshot(), error=_error(phase="execution"), resolved_at=now, finished_at=now
        )
    elif status == "succeeded":
        kwargs.update(resolved=_snapshot(), resolved_at=now, finished_at=now, operation_ids=())

    with pytest.raises(ValidationError):
        _started_attempt(**kwargs)


@pytest.mark.parametrize("status", ["started", "resolution_failed", "resolved", "execution_failed", "succeeded"])
def test_schema_1_1_requires_input_state_id_in_every_status(status):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    kwargs = {"schema_version": "1.1", "status": status, "input_state_id": "state0001"}
    if status in ("resolution_failed",):
        kwargs.update(error=_error(phase="resolution"), finished_at=now)
    elif status == "resolved":
        kwargs.update(resolved=_snapshot(), resolved_at=now)
    elif status == "execution_failed":
        kwargs.update(
            resolved=_snapshot(), error=_error(phase="execution"), resolved_at=now, finished_at=now
        )
    elif status == "succeeded":
        kwargs.update(resolved=_snapshot(), resolved_at=now, finished_at=now, operation_ids=())

    attempt = _started_attempt(**kwargs)
    assert attempt.input_state_id == "state0001"


@pytest.mark.parametrize("status", ["started", "resolution_failed", "resolved", "execution_failed", "succeeded"])
def test_schema_1_1_rejects_none_input_state_id_in_every_status(status):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    kwargs = {"schema_version": "1.1", "status": status, "input_state_id": None}
    if status in ("resolution_failed",):
        kwargs.update(error=_error(phase="resolution"), finished_at=now)
    elif status == "resolved":
        kwargs.update(resolved=_snapshot(), resolved_at=now)
    elif status == "execution_failed":
        kwargs.update(
            resolved=_snapshot(), error=_error(phase="execution"), resolved_at=now, finished_at=now
        )
    elif status == "succeeded":
        kwargs.update(resolved=_snapshot(), resolved_at=now, finished_at=now, operation_ids=())

    with pytest.raises(ValidationError):
        _started_attempt(**kwargs)


def test_input_state_id_rejects_malformed_value():
    with pytest.raises(ValidationError):
        _started_attempt_11(input_state_id="bad id")


# --- schema_version x residuals matrix (Section 12) ---


def test_schema_1_0_execution_failed_rejects_residuals_on_error():
    """schema_version='1.0' predates residuals entirely - even though
    ExecutionAttemptError itself would accept them for an execution-phase
    error in isolation, ExecutionAttempt._validate_status_invariants must
    reject them when the enclosing attempt is schema '1.0'."""

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        _started_attempt(
            schema_version="1.0",
            status="execution_failed",
            resolved=_snapshot(),
            error=_error_with_residuals(),
            resolved_at=now,
            finished_at=now,
        )


def test_schema_1_1_execution_failed_accepts_empty_residuals():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    attempt = _started_attempt_11(
        status="execution_failed",
        resolved=_snapshot(),
        error=_error(phase="execution"),
        resolved_at=now,
        finished_at=now,
    )
    assert attempt.error.residual_state_ids == ()
    assert attempt.error.residual_operation_ids == ()


def test_schema_1_1_execution_failed_accepts_nonempty_residuals():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    attempt = _started_attempt_11(
        status="execution_failed",
        resolved=_snapshot(),
        error=_error_with_residuals(),
        resolved_at=now,
        finished_at=now,
    )
    assert attempt.error.residual_state_ids == ("state0002",)
    assert attempt.error.residual_operation_ids == ("transform0002",)


@pytest.mark.parametrize("status", ["started", "resolved", "succeeded"])
def test_error_free_statuses_reject_any_error_including_with_residuals(status):
    """started/resolved/succeeded reject any error at all - including one
    that would otherwise be a valid execution-phase error with residuals -
    because these statuses require error=None outright."""

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    kwargs = {"status": status}
    if status == "resolved":
        kwargs.update(resolved=_snapshot(), resolved_at=now)
    elif status == "succeeded":
        kwargs.update(resolved=_snapshot(), resolved_at=now, finished_at=now, operation_ids=())

    with pytest.raises(ValidationError):
        _started_attempt_11(error=_error_with_residuals(), **kwargs)


def test_succeeded_never_carries_residuals_because_error_is_none():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    attempt = _started_attempt_11(
        status="succeeded", resolved=_snapshot(), resolved_at=now, finished_at=now, operation_ids=()
    )
    assert attempt.error is None


# --- round-trip and regression guards ---


def test_round_trip_preserves_schema_version_input_state_id_and_residuals():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    attempt = _started_attempt_11(
        status="execution_failed",
        resolved=_snapshot(),
        error=_error_with_residuals(
            residual_state_ids=("state0003", "state0002"),
            residual_operation_ids=("transform0003", "transform0002"),
        ),
        resolved_at=now,
        finished_at=now,
    )

    dumped = attempt.model_dump(mode="json")
    reloaded = ExecutionAttempt.model_validate(dumped)

    assert reloaded == attempt
    assert reloaded.schema_version == "1.1"
    assert reloaded.input_state_id == "state0001"
    assert reloaded.error.residual_state_ids == ("state0003", "state0002")
    assert reloaded.error.residual_operation_ids == ("transform0003", "transform0002")


def test_load_legacy_1_0_attempt_without_input_state_id():
    """A dict shaped like a pre-016 attempt (no input_state_id key at
    all) must still validate as schema_version='1.0' with
    input_state_id=None - no migration or reshaping required to read a
    legacy record."""

    from datetime import datetime, timezone

    data = {
        "schema_version": "1.0",
        "id": "attempt0001",
        "operation_type": "state-transformation",
        "operation_ids": None,
        "status": "started",
        "requested": {"schema_version": "1.0", "max_states": None, "source": "scientist"},
        "resolved": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_at": None,
        "finished_at": None,
    }
    attempt = ExecutionAttempt.model_validate(data)
    assert attempt.schema_version == "1.0"
    assert attempt.input_state_id is None


def test_tautomer_enumeration_policy_rejects_schema_version_1_1():
    """Regression guard for D-016-DRAFT-01/Section 10.1: only
    ExecutionAttempt is versioned by Increment 016."""

    with pytest.raises(ValidationError):
        TautomerEnumerationPolicy(schema_version="1.1", max_states=None, source="scientist")


def test_resolved_execution_snapshot_rejects_schema_version_1_1():
    """Regression guard for D-016-DRAFT-01/Section 10.1: only
    ExecutionAttempt is versioned by Increment 016."""

    with pytest.raises(ValidationError):
        ResolvedExecutionSnapshot(
            schema_version="1.1", engine_name="rdkit", engine_version="2026.03.6", parameters={}
        )
