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
        "operation_id": "t-future-001",
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


def test_execution_attempt_rejects_malformed_operation_id():
    with pytest.raises(ValidationError):
        _started_attempt(operation_id="bad id")


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
            operation_id="t-future-001",
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


def test_resolution_failed_invariants_enforced():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    valid = _started_attempt(
        status="resolution_failed",
        error=_error(phase="resolution"),
        finished_at=now,
    )
    assert valid.status == "resolution_failed"

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


def test_resolved_invariants_enforced():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    valid = _started_attempt(status="resolved", resolved=_snapshot(), resolved_at=now)
    assert valid.status == "resolved"

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


def test_succeeded_invariants_enforced():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    valid = _started_attempt(
        status="succeeded", resolved=_snapshot(), resolved_at=now, finished_at=now
    )
    assert valid.status == "succeeded"

    # resolved is required
    with pytest.raises(ValidationError):
        _started_attempt(status="succeeded", resolved_at=now, finished_at=now)

    # error must be None
    with pytest.raises(ValidationError):
        _started_attempt(
            status="succeeded",
            resolved=_snapshot(),
            resolved_at=now,
            finished_at=now,
            error=_error(),
        )

    # resolved_at is required
    with pytest.raises(ValidationError):
        _started_attempt(status="succeeded", resolved=_snapshot(), finished_at=now)

    # finished_at is required
    with pytest.raises(ValidationError):
        _started_attempt(status="succeeded", resolved=_snapshot(), resolved_at=now)
