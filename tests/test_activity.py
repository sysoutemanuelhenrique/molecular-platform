import re
from dataclasses import dataclass

import pytest

from molecular.activity import record_activity

LINE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\S* command=(\S+) outcome=(\S+)$"
)


@dataclass(frozen=True)
class _FakeDerivedStateDetails:
    """A stand-in for application.DerivedStateActivityDetails - these tests
    exercise record_activity's generic details-formatting contract in
    isolation, independent of any specific scientific shape."""

    attempt_id: str
    parent_state_id: str
    child_state_id: str
    transformation_id: str


def test_record_activity_creates_log_file_on_first_call(tmp_path):
    log_file = tmp_path / "logs" / "activity.log"

    assert not log_file.exists()

    record_activity(tmp_path, command="validate", outcome="success")

    assert log_file.exists()


def test_record_activity_appends_across_multiple_calls(tmp_path):
    record_activity(tmp_path, command="validate", outcome="success")
    record_activity(tmp_path, command="plan", outcome="success")

    log_file = tmp_path / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2


def test_record_activity_writes_correct_command_and_outcome(tmp_path):
    record_activity(tmp_path, command="validate", outcome="success")

    log_file = tmp_path / "logs" / "activity.log"
    line = log_file.read_text(encoding="utf-8").splitlines()[0]

    match = LINE_PATTERN.match(line)

    assert match is not None
    assert match.group(1) == "validate"
    assert match.group(2) == "success"


def test_record_activity_does_not_create_missing_project_directory(tmp_path):
    project_dir = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        record_activity(project_dir, command="validate", outcome="error")

    assert not project_dir.exists()


# --- backward-compatible `details` extension ---


def _details():
    return _FakeDerivedStateDetails(
        attempt_id="attempt-1",
        parent_state_id="state-a",
        child_state_id="state-b",
        transformation_id="transformation-1",
    )


def test_record_activity_legacy_call_preserves_old_line_shape(tmp_path):
    """A call with no `details` (every call site that existed before this
    parameter did) still produces exactly the historical line shape."""

    record_activity(tmp_path, command="validate", outcome="success")

    log_file = tmp_path / "logs" / "activity.log"
    line = log_file.read_text(encoding="utf-8").splitlines()[0]

    assert LINE_PATTERN.match(line) is not None


def test_record_activity_detailed_call_writes_all_fields(tmp_path):
    record_activity(
        tmp_path, command="register-derived-state", outcome="started", details=_details()
    )

    log_file = tmp_path / "logs" / "activity.log"
    line = log_file.read_text(encoding="utf-8").splitlines()[0]

    assert "command=register-derived-state" in line
    assert "outcome=started" in line
    assert "attempt_id=attempt-1" in line
    assert "parent_state_id=state-a" in line
    assert "child_state_id=state-b" in line
    assert "transformation_id=transformation-1" in line


def test_record_activity_detailed_line_is_parseable_deterministically(tmp_path):
    record_activity(
        tmp_path, command="register-derived-state", outcome="started", details=_details()
    )

    log_file = tmp_path / "logs" / "activity.log"
    line = log_file.read_text(encoding="utf-8").splitlines()[0]

    tokens = line.split(" ")
    fields = dict(token.split("=", 1) for token in tokens[1:])

    assert fields["command"] == "register-derived-state"
    assert fields["outcome"] == "started"
    assert fields["attempt_id"] == "attempt-1"
    assert fields["parent_state_id"] == "state-a"
    assert fields["child_state_id"] == "state-b"
    assert fields["transformation_id"] == "transformation-1"


def test_record_activity_detailed_call_appends_no_arbitrary_metadata(tmp_path):
    """Only the dataclass's declared fields appear on the line - nothing
    else can sneak in."""

    record_activity(
        tmp_path, command="register-derived-state", outcome="started", details=_details()
    )

    log_file = tmp_path / "logs" / "activity.log"
    line = log_file.read_text(encoding="utf-8").splitlines()[0]

    # timestamp + command=... + outcome=... + 4 detail fields = 7 tokens
    assert len(line.split(" ")) == 7


def test_record_activity_log_append_semantics_preserved_with_details(tmp_path):
    record_activity(tmp_path, command="validate", outcome="success")
    record_activity(
        tmp_path, command="register-derived-state", outcome="started", details=_details()
    )
    record_activity(
        tmp_path, command="register-derived-state", outcome="success", details=_details()
    )

    log_file = tmp_path / "logs" / "activity.log"
    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert LINE_PATTERN.match(lines[0]) is not None
    assert "outcome=started" in lines[1]
    assert "outcome=success" in lines[2]
