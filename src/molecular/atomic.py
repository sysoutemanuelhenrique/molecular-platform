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


def publish_staged_directory(staging_dir: Path, final_dir: Path) -> None:
    """Publish a fully-prepared staging directory to its final location.

    This is a single filesystem rename, not a transaction manager: it only
    guarantees that `final_dir` is not deliberately overwritten by this
    call, and that the move itself (`os.rename`) is a single atomic
    operation on the same filesystem. It never touches metadata and never
    commits anything else - the caller decides what "published" means.

    Raises FileExistsError if `final_dir` already exists, checked
    explicitly beforehand: `os.rename`'s behavior on an existing directory
    destination differs across platforms (POSIX silently replaces an empty
    directory; Windows always raises), so relying on it alone would make
    overwrite-or-not filesystem-dependent. This pre-check narrows but does
    not eliminate a TOCTOU race - another process could create `final_dir`
    between the check and the rename, in which case the OSError raised by
    `os.rename` itself is the last guard and is left to propagate. This
    same-destination guard is not a general-purpose mutex - it does not
    serialize two callers whose destination names merely casefold-collide
    rather than match exactly; see the identity-conflict check in
    registry.py for the fuller discussion of that residual window.
    """

    staging_dir = Path(staging_dir)
    final_dir = Path(final_dir)

    if final_dir.exists():
        raise FileExistsError(f"Destination already exists: {final_dir}")

    os.rename(staging_dir, final_dir)
