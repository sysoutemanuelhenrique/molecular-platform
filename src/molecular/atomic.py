"""Minimal atomic text-file write primitive."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import FrozenSet, Type


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 checksum of a file's contents, streamed in
    fixed-size chunks so the whole file never has to fit in memory at once."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


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


def check_storage_identity_conflict(
    directory: Path,
    candidate_id: str,
    *,
    conflict_error: Type[FileExistsError],
    reserved_names: FrozenSet[str] = frozenset(),
) -> None:
    """Raise `conflict_error` if `candidate_id` collides - exactly or by
    casefold - with an existing entry directly under `directory`: a
    committed metadata file (`<id>.yaml`) or a published subdirectory
    (`<id>/`). Entries named in `reserved_names` (e.g. a `.staging`
    namespace) are never candidates.

    Shared by registry.py, scientific_registry.py, and
    transformation_registry.py - each passes its own `conflict_error`
    subclass so callers keep catching the specific exception type they
    already catch today. This is a storage-layer check only: identity is
    compared case-insensitively (via .casefold()) no matter how the
    underlying filesystem itself treats case, because two distinct logical
    ids that only differ by case cannot both be given a directory or
    metadata file in the same `directory` portably across case-sensitive
    and case-insensitive filesystems.

    Detection is pure Python string comparison over a directory listing; it
    never depends on - and is never masked by - whatever case sensitivity
    the underlying filesystem happens to have.

    This is NOT equivalent to multi-process/multi-thread serialization. Two
    concurrent callers can each run this check - and each see no conflict -
    before either one has published its final entry: there is a residual
    TOCTOU window between this check and the actual filesystem write. When
    two candidate ids are exactly equal, that window happens to be closed
    as a side effect of the eventual write itself (an atomic replace, or
    publish_staged_directory's destination-exists guard) - but that is not
    a general-purpose mutex, and it does not help when two ids merely
    casefold-collide. This window is knowingly accepted: every caller of
    this function is used as a single process today, multi-process
    locking/mutex is explicitly out of scope, and none of these protocols
    promise concurrent-writer safety.
    """

    if not directory.is_dir():
        return

    folded = candidate_id.casefold()

    for entry in directory.iterdir():
        if entry.name in reserved_names:
            continue

        if entry.is_dir():
            existing_id = entry.name
        elif entry.is_file() and entry.suffix == ".yaml":
            existing_id = entry.stem
        else:
            continue

        if existing_id.casefold() == folded:
            raise conflict_error(
                f"'{candidate_id}' is already registered under {directory} "
                f"(conflicts with existing entry '{existing_id}'; "
                "storage identity is case-insensitive)."
            )


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
    rather than match exactly; see check_storage_identity_conflict above
    for the fuller discussion of that residual window.
    """

    staging_dir = Path(staging_dir)
    final_dir = Path(final_dir)

    if final_dir.exists():
        raise FileExistsError(f"Destination already exists: {final_dir}")

    os.rename(staging_dir, final_dir)
