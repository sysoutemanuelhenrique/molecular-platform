"""Minimal atomic text-file write primitive."""

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write `content` to `path` atomically.

    A temp file is created in the same directory as `path` (so the final
    rename is on the same filesystem), written, flushed, and fsynced, then
    swapped into place with an atomic replace. If anything fails before the
    replace, the temp file is removed on a best-effort basis and the
    original `path` - if it existed - is left untouched.
    """

    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)

    try:
        with os.fdopen(fd, "w", encoding=encoding) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)

    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
