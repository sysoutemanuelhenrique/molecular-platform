"""Minimal atomic text-file write primitive, plus the hardened Route B
mechanism for safely opening and snapshotting a LigandInput RAW artifact
(Section 8 of Increment 018's frozen final specification,
``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``,
SHA-256 ``347632175991b1d72b9f193312b9629ce71697fb0c3d0de2285e52b7299d8023``).

Route B endurecida is the ONLY route implemented (Section 8.3.0): FS-018-01
confirmed `os.supports_dir_fd` is empty for `os.stat`/`os.open` and
`os.O_NOFOLLOW` does not exist on this Windows/CPython combination, so
there is no `openat`-style descriptor-relative technique available to
implement Route A's stronger, race-closing guarantee. Every identity token
below is captured via `os.lstat` (never `os.stat`, never following a
link), using `st_dev`/`st_ino` for identity and `st_file_attributes`/
`st_reparse_tag` (both directly exposed by `os.lstat` on this Python/
Windows combination - confirmed empirically, no `ctypes` needed for
identity) to detect ANY reparse point (symlink, junction, or otherwise) -
Section 8.4's requirement to reject "qualquer reparse point detectável"
does not distinguish by reparse tag; only the `FILE_ATTRIBUTE_REPARSE_
POINT` bit matters.

This module's own containment check uses exclusively `Path.is_relative_to`
- never `str(...).startswith(...)`, in any form, at any point, including
after `resolve()` (Section 8.2, FS-018-01's central finding: a sibling
directory whose name is a textual prefix of the authorized root,
e.g. `allowed_root_EVIL` vs `allowed_root`, defeats `startswith()` even
post-`resolve()`, but never defeats `is_relative_to()`).

Only Windows 11 with a local NTFS volume is validated (Section 8.7); any
other platform or filesystem fails closed with
RawArtifactNoFollowUnavailableError before the RAW file is ever touched -
detected via `os.name` plus `GetVolumeInformationW`/`GetDriveTypeW`
(`ctypes`, the same technique FS-018-02's own spike script used), the only
place in this module `ctypes` is needed at all.
"""

import contextlib
import ctypes
import hashlib
import os
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Optional, Type


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


# ===========================================================================
# Route B endurecida (Section 8 of the frozen spec) - RAW artifact safe
# open, identity verification, and snapshot creation.
# ===========================================================================


class UnsafeRawArtifactPathError(ValueError):
    """`unsafe-raw-path` (Section 17). Raised for any lexical or real
    containment failure: absolute path (POSIX/Windows/UNC/drive-relative),
    `.`/`..` component, backslash in a POSIX-canonical path, empty path,
    wrong prefix, a reparse point at any level (final or ancestral), or a
    resolved path outside the authorized root - including the
    `allowed_root_EVIL` sibling-prefix-collision case (Section 8.2). Never
    carries the absolute path in its message (Section 8.2, item 7)."""


class RawArtifactTypeError(OSError):
    """`raw-not-regular-file` (Section 17). Raised when the final path is
    absent or is not a regular file (e.g. a directory)."""


class RawArtifactIdentityUnstableError(RuntimeError):
    """`raw-identity-unstable` (Section 17). Raised when the identity of
    the final file or of any ancestor diverges between two of the four
    normative capture moments (Section 8.3.1, items 8-10; Section
    8.3.2.1) - substitution, rename, reparse-point swap, redirection
    outside the root, or removal-and-recreation with the same name."""


class RawArtifactNoFollowUnavailableError(RuntimeError):
    """`raw-nofollow-unavailable` (Section 17). Raised before the RAW file
    is ever touched when the current platform/filesystem is not the one
    Route B endurecida was validated against (Section 8.7-8.8): anything
    other than Windows with a local NTFS volume. Never claims support;
    never degrades to a weaker check."""


class RawArtifactIntegrityError(ValueError):
    """`raw-checksum-mismatch` (Section 17). Raised when the SHA-256
    computed over the bytes actually read does not equal the
    `RawArtifact.checksum` already persisted for this input (Section
    8.3.2.1, step 6) - including when a same-size, same-inode in-place
    content mutation defeated identity-based detection entirely (Section
    8.6, item 1): this is the only mechanism that catches that class."""


class RawArtifactSharingViolationError(OSError):
    """`raw-permission-denied` (Section 17). Raised when the operating
    system itself blocks an operation (e.g. a rename attempted by a third
    party while this module holds an open handle) - an empirical
    observation of this platform/configuration, never relied upon as a
    security guarantee of the contract (Section 8.6, item 2)."""


_AUTHORIZED_ROOT_TEMPLATE = "inputs/ligands/{input_id}"


@dataclass(frozen=True)
class RawSnapshot:
    """The verified result of `create_verified_raw_snapshot`: a private
    copy of the RAW artifact's bytes, at `snapshot_path`, together with the
    SHA-256 and size computed exclusively from the bytes actually read
    (Section 8.3.3) - never from a fresh read of `snapshot_path` itself,
    never from filesystem metadata alone. The caller (a future Application
    Service) owns `snapshot_path` from this point on, including its
    eventual removal (Section 12.2, step 32) - this function never removes
    it on success."""

    snapshot_path: Path
    snapshot_checksum: str
    snapshot_size_bytes: int


@dataclass(frozen=True)
class _IdentityToken:
    """One `os.lstat`-based identity capture - never `os.stat`, never
    following a link (Section 8.3.1, items 2-6)."""

    st_dev: int
    st_ino: int
    is_reparse_point: bool
    file_attributes: Optional[int]
    reparse_tag: Optional[int]
    is_regular_file: bool
    is_directory: bool
    size: int


def _capture_identity(path: Path) -> _IdentityToken:
    """`lstat` a single path component and extract everything Section
    8.3.1 requires: `st_dev`/`st_ino` for identity, and
    `st_file_attributes`/`st_reparse_tag` (both exposed directly by
    `os.lstat` on this Windows/Python combination - confirmed empirically,
    no `ctypes` needed) to detect any reparse point, of any tag."""

    result = os.lstat(path)
    file_attributes = getattr(result, "st_file_attributes", None)
    is_reparse_point = bool(
        file_attributes is not None and file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )
    reparse_tag = getattr(result, "st_reparse_tag", None) if is_reparse_point else None

    return _IdentityToken(
        st_dev=result.st_dev,
        st_ino=result.st_ino,
        is_reparse_point=is_reparse_point,
        file_attributes=file_attributes,
        reparse_tag=reparse_tag,
        is_regular_file=stat.S_ISREG(result.st_mode),
        is_directory=stat.S_ISDIR(result.st_mode),
        size=result.st_size,
    )


def _identity_matches(a: _IdentityToken, b: _IdentityToken) -> bool:
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _reject_reparse_point(token: _IdentityToken, description: str) -> None:
    if token.is_reparse_point:
        raise UnsafeRawArtifactPathError(
            f"{description} is a symlink, junction, or other reparse point "
            "(reparse tag "
            f"{token.reparse_tag!r}); never followed, never accepted, at "
            "any level of the path."
        )


def _validate_raw_path_lexical(raw_artifact_path: str, *, input_id: str) -> Path:
    """Section 8.1: purely lexical checks, no filesystem I/O at all.
    Raises UnsafeRawArtifactPathError on any violation. Returns the
    validated path as a relative `Path` (POSIX-style), never resolved,
    never touched on disk."""

    if not raw_artifact_path:
        raise UnsafeRawArtifactPathError("raw artifact path must not be empty")

    if "\\" in raw_artifact_path:
        raise UnsafeRawArtifactPathError(
            "raw artifact path must not contain a backslash in its canonical POSIX form"
        )

    # Windows absolute (drive letter or bare-slash-relative-to-drive) and
    # UNC forms.
    if raw_artifact_path.startswith("\\\\") or raw_artifact_path.startswith("//"):
        raise UnsafeRawArtifactPathError("raw artifact path must not be a UNC path")
    if len(raw_artifact_path) >= 2 and raw_artifact_path[1] == ":":
        raise UnsafeRawArtifactPathError(
            "raw artifact path must not be a Windows drive-absolute or drive-relative path"
        )
    if raw_artifact_path.startswith("/"):
        raise UnsafeRawArtifactPathError("raw artifact path must not be POSIX-absolute")

    components = raw_artifact_path.split("/")
    if any(component in (".", "..") for component in components):
        raise UnsafeRawArtifactPathError(
            "raw artifact path must not contain '.' or '..' components"
        )
    if any(component == "" for component in components):
        raise UnsafeRawArtifactPathError("raw artifact path must not contain empty components")

    expected_prefix = _AUTHORIZED_ROOT_TEMPLATE.format(input_id=input_id) + "/"
    if not raw_artifact_path.startswith(expected_prefix):
        # This is a lexical prefix check against a value THIS module
        # derives from `input_id`, not a containment proof against the
        # real filesystem - Section 8.2's ban on `startswith()` is about
        # proving REAL containment after resolution, never about this
        # narrower, purely textual pre-check of the declared path shape.
        raise UnsafeRawArtifactPathError(
            f"raw artifact path must start with '{expected_prefix}'"
        )
    if len(components) <= len(expected_prefix.rstrip("/").split("/")):
        raise UnsafeRawArtifactPathError(
            "raw artifact path must name a file strictly inside the authorized input directory"
        )

    return Path(raw_artifact_path)


def _get_volume_filesystem(path: Path) -> str:
    drive = os.path.splitdrive(str(path.resolve()))[0] + "\\"
    volume_name_buf = ctypes.create_unicode_buffer(261)
    fs_name_buf = ctypes.create_unicode_buffer(261)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(drive), volume_name_buf, 261, None, None, None, fs_name_buf, 261
    )
    return fs_name_buf.value if ok else ""


_DRIVE_FIXED = 3


def _get_drive_type(path: Path) -> int:
    drive = os.path.splitdrive(str(path.resolve()))[0] + "\\"
    return ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(drive))


def _require_validated_platform(path: Path) -> None:
    """Section 8.7-8.8: fail closed, before touching the RAW file, unless
    running on Windows with a local (non-network, non-cloud-mounted) NTFS
    volume - the only platform Route B endurecida was validated against.
    Detected via `GetVolumeInformationW`/`GetDriveTypeW` (`ctypes`), the
    same technique FS-018-02's own spike script used - the only place in
    this module `ctypes` is needed at all."""

    if os.name != "nt":
        raise RawArtifactNoFollowUnavailableError(
            "This platform is not Windows; Route B endurecida is validated "
            "only for Windows 11 with a local NTFS volume (Section 8.7). "
            "Failing closed rather than claiming support."
        )

    try:
        filesystem = _get_volume_filesystem(path)
        drive_type = _get_drive_type(path)
    except OSError as exc:
        raise RawArtifactNoFollowUnavailableError(
            "Could not determine the volume's filesystem/drive type; "
            "failing closed rather than assuming support."
        ) from exc

    if filesystem != "NTFS":
        raise RawArtifactNoFollowUnavailableError(
            f"The volume's filesystem ({filesystem!r}) is not NTFS; Route B "
            "endurecida is validated only for local NTFS (Section 8.7)."
        )
    if drive_type != _DRIVE_FIXED:
        raise RawArtifactNoFollowUnavailableError(
            "The volume is not a local fixed drive (network share or "
            "cloud-mounted filesystem are not validated, Section 8.7)."
        )


def _build_ancestor_chain(root: Path, final_path: Path) -> List[Path]:
    """Every directory component from `root` (inclusive) down to
    `final_path`'s immediate parent (inclusive), in that order (Section
    8.3.1, item 1). For the common case of a RAW artifact stored directly
    under the input's root, this is exactly `[root]`."""

    relative = final_path.relative_to(root)
    ancestors = [root]
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        ancestors.append(current)
    return ancestors


def _require_relative_to_root(candidate: Path, resolved_root: Path) -> Path:
    """Section 8.2's central, empirically-confirmed requirement:
    `Path.is_relative_to()` on RESOLVED paths - never
    `str(...).startswith(...)`, in any form, at any point. Defeats the
    `allowed_root_EVIL` sibling-prefix-collision case by construction,
    since containment is decided by path COMPONENTS, never by substring."""

    resolved_candidate = candidate.resolve(strict=True)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise UnsafeRawArtifactPathError(
            "resolved path is not contained within the authorized root "
            "(is_relative_to() check failed)"
        )
    return resolved_candidate


def _capture_and_validate_ancestor_chain(
    ancestor_paths: List[Path], resolved_root: Path
) -> List[_IdentityToken]:
    """Section 8.3.1, items 1-7 / Section 8.3.2, steps 1-3: lstat every
    ancestor, reject any reparse point at any level, and validate
    containment via `is_relative_to()` for every level of the chain, not
    only the deepest one."""

    tokens = []
    for ancestor in ancestor_paths:
        try:
            token = _capture_identity(ancestor)
        except OSError as exc:
            raise UnsafeRawArtifactPathError(
                f"an ancestor directory could not be inspected: {exc.__class__.__name__}"
            ) from exc
        _reject_reparse_point(token, "an ancestor directory")
        if not token.is_directory:
            raise UnsafeRawArtifactPathError("an ancestor component is not a directory")
        _require_relative_to_root(ancestor, resolved_root)
        tokens.append(token)
    return tokens


def _reverify_ancestor_chain(
    ancestor_paths: List[Path], previous_tokens: List[_IdentityToken]
) -> List[_IdentityToken]:
    """Recapture every ancestor's identity and compare against the
    previous capture - Section 8.3.1, items 8-10. Any divergence,
    including a reparse point appearing where there was none, is
    `raw-identity-unstable`."""

    fresh_tokens = []
    for ancestor, previous in zip(ancestor_paths, previous_tokens):
        try:
            fresh = _capture_identity(ancestor)
        except OSError as exc:
            raise RawArtifactIdentityUnstableError(
                f"an ancestor directory could not be re-inspected: {exc.__class__.__name__}"
            ) from exc
        _reject_reparse_point(fresh, "an ancestor directory")
        if not _identity_matches(previous, fresh):
            raise RawArtifactIdentityUnstableError(
                "an ancestor directory's identity changed since the previous capture "
                "(substituted, renamed, or removed and recreated)"
            )
        fresh_tokens.append(fresh)
    return fresh_tokens


def create_verified_raw_snapshot(
    *,
    project_dir: Path,
    input_id: str,
    raw_artifact_path: str,
    raw_artifact_checksum: str,
    snapshot_path: Path,
) -> RawSnapshot:
    """Validate, open, read, and verify a LigandInput RAW artifact via the
    hardened Route B (Section 8 of the frozen spec), writing a private
    snapshot of its bytes to `snapshot_path` (caller-supplied - this
    function never invents its own staging convention). Read-only against
    the RAW artifact itself: never rewritten, sanitized in place,
    converted, its timestamp touched, moved, copied back onto, or removed
    (Section 8, "RAW imutável"). On any failure, `snapshot_path` is removed
    before the typed error is raised - no partial snapshot is ever left
    for a caller to mistake for a complete one.

    Order of operations (Section 8.3.2 / 8.3.2.1, literally):

    1. fail closed if the platform/filesystem is not the validated one;
    2. validate the path lexically (Section 8.1) and resolve+contain it
       against the authorized root `inputs/ligands/<input_id>/`
       (Section 8.2), rejecting any reparse point at any level;
    3. capture the initial identity of the full ancestor chain and of the
       final file (all via `lstat`, never following);
    4. reverify the ancestor chain immediately before opening;
    5. open the final file by its validated path, `fstat` the descriptor,
       and compare its identity against the pre-open `lstat` - divergence
       is `raw-identity-unstable`;
    6. reverify the ancestor chain again, at this same moment;
    7. read the file entirely through the same open descriptor, computing
       SHA-256 and size incrementally over the bytes actually read, while
       writing those same bytes to `snapshot_path`;
    8. mandatory post-read verification: reverify the final file's
       identity and the full ancestor chain once more, compare the size
       actually read against the pre-open `lstat`'s `st_size`, then
       compare the SHA-256 against `raw_artifact_checksum`;
    9. only then return the `RawSnapshot`.
    """

    _require_validated_platform(project_dir)

    validated_relative_path = _validate_raw_path_lexical(raw_artifact_path, input_id=input_id)
    root = project_dir / _AUTHORIZED_ROOT_TEMPLATE.format(input_id=input_id)

    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise RawArtifactTypeError(
            f"the input's authorized root does not exist: {exc.__class__.__name__}"
        ) from exc

    final_path = root / validated_relative_path.relative_to(
        _AUTHORIZED_ROOT_TEMPLATE.format(input_id=input_id)
    )
    ancestor_paths = _build_ancestor_chain(root, final_path)

    initial_ancestor_tokens = _capture_and_validate_ancestor_chain(ancestor_paths, resolved_root)

    try:
        pre_open_final_token = _capture_identity(final_path)
    except OSError as exc:
        raise RawArtifactTypeError(
            f"the RAW artifact is absent or could not be inspected: {exc.__class__.__name__}"
        ) from exc
    _reject_reparse_point(pre_open_final_token, "the RAW artifact")
    if not pre_open_final_token.is_regular_file:
        raise RawArtifactTypeError("the RAW artifact path does not name a regular file")
    _require_relative_to_root(final_path, resolved_root)

    pre_open_ancestor_tokens = _reverify_ancestor_chain(ancestor_paths, initial_ancestor_tokens)

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_written = False

    try:
        try:
            file = open(final_path, "rb")
        except PermissionError as exc:
            raise RawArtifactSharingViolationError(
                f"the operating system denied opening the RAW artifact: {exc}"
            ) from exc
        except OSError as exc:
            raise RawArtifactTypeError(
                f"the RAW artifact could not be opened: {exc.__class__.__name__}"
            ) from exc

        with file:
            post_open_stat = os.fstat(file.fileno())
            post_open_token = _IdentityToken(
                st_dev=post_open_stat.st_dev,
                st_ino=post_open_stat.st_ino,
                is_reparse_point=False,
                file_attributes=getattr(post_open_stat, "st_file_attributes", None),
                reparse_tag=None,
                is_regular_file=stat.S_ISREG(post_open_stat.st_mode),
                is_directory=stat.S_ISDIR(post_open_stat.st_mode),
                size=post_open_stat.st_size,
            )
            if not _identity_matches(pre_open_final_token, post_open_token):
                raise RawArtifactIdentityUnstableError(
                    "the RAW artifact's identity changed between validation and opening"
                )

            _reverify_ancestor_chain(ancestor_paths, pre_open_ancestor_tokens)

            digest = hashlib.sha256()
            bytes_read = 0
            with open(snapshot_path, "wb") as snapshot_file:
                for chunk in iter(lambda: file.read(65536), b""):
                    digest.update(chunk)
                    snapshot_file.write(chunk)
                    bytes_read += len(chunk)
                snapshot_file.flush()
                os.fsync(snapshot_file.fileno())
            snapshot_written = True

        # --- mandatory post-read verification (Section 8.3.2.1) ---
        try:
            post_read_final_token = _capture_identity(final_path)
        except OSError as exc:
            raise RawArtifactIdentityUnstableError(
                f"the RAW artifact could not be re-inspected after reading: "
                f"{exc.__class__.__name__}"
            ) from exc
        _reject_reparse_point(post_read_final_token, "the RAW artifact")
        if not _identity_matches(pre_open_final_token, post_read_final_token):
            raise RawArtifactIdentityUnstableError(
                "the RAW artifact's identity changed during or after reading "
                "(e.g. removed, renamed, or replaced)"
            )

        _reverify_ancestor_chain(ancestor_paths, pre_open_ancestor_tokens)

        if bytes_read != pre_open_final_token.size:
            raise RawArtifactIdentityUnstableError(
                "the number of bytes actually read does not match the size "
                "observed before opening the RAW artifact"
            )

        snapshot_checksum = digest.hexdigest()
        if snapshot_checksum != raw_artifact_checksum:
            raise RawArtifactIntegrityError(
                "the SHA-256 computed over the bytes actually read does not "
                "match the checksum persisted for this input"
            )

        return RawSnapshot(
            snapshot_path=snapshot_path,
            snapshot_checksum=snapshot_checksum,
            snapshot_size_bytes=bytes_read,
        )

    except BaseException:
        if snapshot_written or snapshot_path.exists():
            with contextlib.suppress(OSError):
                snapshot_path.unlink()
        raise
