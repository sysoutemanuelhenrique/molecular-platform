import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def record_activity(
    project_dir: Path,
    command: str,
    outcome: str,
    *,
    details: Optional[object] = None,
) -> None:
    """Append one line describing a CLI command's outcome to the experiment's activity log.

    project_dir must already exist; this never creates the experiment directory itself.

    `details`, when given, must be a frozen dataclass instance - its fields
    are appended to the line, in declaration order, as `name=value` tokens
    after `outcome`. This module makes no assumption about which dataclass
    shape a caller uses; it only knows how to format one generically via
    `dataclasses.fields`.

    Legacy calls (details=None, the default - every call site before this
    parameter existed) produce exactly the historical line shape,
    unchanged:

        timestamp command=<...> outcome=<...>

    so no existing activity log line, and no existing caller, is affected
    by this extension. This is an additive, backward-compatible extension
    of the log format, not a new schema - there is still exactly one
    append-only text file per experiment, one line per event.
    """

    if not project_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {project_dir}")

    log_file = project_dir / "logs" / "activity.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{timestamp} command={command} outcome={outcome}"

    if details is not None:
        for field in dataclasses.fields(details):
            line += f" {field.name}={getattr(details, field.name)}"

    with log_file.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
