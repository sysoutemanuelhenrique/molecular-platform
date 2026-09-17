from datetime import datetime, timezone
from pathlib import Path


def record_activity(project_dir: Path, command: str, outcome: str) -> None:
    """Append one line describing a CLI command's outcome to the experiment's activity log.

    project_dir must already exist; this never creates the experiment directory itself.
    """

    if not project_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {project_dir}")

    log_file = project_dir / "logs" / "activity.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"{timestamp} command={command} outcome={outcome}\n")
