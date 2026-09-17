import re

from molecular.activity import record_activity

LINE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\S* command=(\S+) outcome=(\S+)$"
)


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
