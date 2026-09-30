import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import molecular.atomic as atomic_module
from molecular.atomic import (
    RawArtifactIdentityUnstableError,
    RawArtifactIntegrityError,
    RawArtifactNoFollowUnavailableError,
    RawArtifactSharingViolationError,
    RawArtifactTypeError,
    UnsafeRawArtifactPathError,
    atomic_write_text,
    create_verified_raw_snapshot,
    publish_staged_directory,
)


def test_atomic_write_creates_new_file(tmp_path):
    target = tmp_path / "file.txt"

    atomic_write_text(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"


def test_atomic_write_replaces_existing_target_on_success(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("old content", encoding="utf-8")

    atomic_write_text(target, "new content")

    assert target.read_text(encoding="utf-8") == "new content"


def test_atomic_write_leaves_no_temp_file_behind_on_success(tmp_path):
    target = tmp_path / "file.txt"

    atomic_write_text(target, "hello")

    leftovers = [path for path in tmp_path.iterdir() if path != target]
    assert leftovers == []


def test_atomic_write_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "nested" / "dir" / "file.txt"

    atomic_write_text(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"


def test_atomic_write_preserves_target_when_failure_happens_before_replace(tmp_path, monkeypatch):
    target = tmp_path / "file.txt"
    target.write_text("original content", encoding="utf-8")

    def _raise_during_write(*args, **kwargs):
        raise OSError("simulated disk failure during write")

    monkeypatch.setattr(os, "fsync", _raise_during_write)

    with pytest.raises(OSError):
        atomic_write_text(target, "new content")

    assert target.read_text(encoding="utf-8") == "original content"


def test_atomic_write_cleans_up_temp_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "file.txt"
    target.write_text("original content", encoding="utf-8")

    def _raise_during_write(*args, **kwargs):
        raise OSError("simulated disk failure during write")

    monkeypatch.setattr(os, "fsync", _raise_during_write)

    with pytest.raises(OSError):
        atomic_write_text(target, "new content")

    leftovers = [path for path in tmp_path.iterdir() if path != target]
    assert leftovers == []


def test_atomic_write_uses_same_directory_as_target(tmp_path, monkeypatch):
    target = tmp_path / "file.txt"
    seen_dirs = []
    real_mkstemp = tempfile.mkstemp

    def _tracking_mkstemp(*args, **kwargs):
        seen_dirs.append(kwargs.get("dir"))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", _tracking_mkstemp)

    atomic_write_text(target, "hello")

    assert seen_dirs == [target.parent]


# --- publish_staged_directory ---


def test_publish_staged_directory_publishes_staging_as_final(tmp_path):
    staging_dir = tmp_path / "staging_attempt"
    staging_dir.mkdir()
    (staging_dir / "file.txt").write_text("hello", encoding="utf-8")

    final_dir = tmp_path / "final"

    publish_staged_directory(staging_dir, final_dir)

    assert final_dir.exists()
    assert (final_dir / "file.txt").read_text(encoding="utf-8") == "hello"
    assert not staging_dir.exists()


def test_publish_staged_directory_refuses_to_overwrite_existing_final(tmp_path):
    staging_dir = tmp_path / "staging_attempt"
    staging_dir.mkdir()
    (staging_dir / "file.txt").write_text("new", encoding="utf-8")

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "existing.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_staged_directory(staging_dir, final_dir)

    assert staging_dir.exists()
    assert (staging_dir / "file.txt").exists()
    assert (final_dir / "existing.txt").read_text(encoding="utf-8") == "old"
    assert not (final_dir / "file.txt").exists()


def test_publish_staged_directory_propagates_rename_oserror(tmp_path, monkeypatch):
    staging_dir = tmp_path / "staging_attempt"
    staging_dir.mkdir()

    final_dir = tmp_path / "final"

    def _raise_rename(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "rename", _raise_rename)

    with pytest.raises(OSError):
        publish_staged_directory(staging_dir, final_dir)


def test_publish_staged_directory_leaves_staging_intact_and_final_absent_after_failure(
    tmp_path, monkeypatch
):
    staging_dir = tmp_path / "staging_attempt"
    staging_dir.mkdir()
    (staging_dir / "file.txt").write_text("hello", encoding="utf-8")

    final_dir = tmp_path / "final"

    def _raise_rename(*args, **kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "rename", _raise_rename)

    with pytest.raises(OSError):
        publish_staged_directory(staging_dir, final_dir)

    assert staging_dir.exists()
    assert (staging_dir / "file.txt").read_text(encoding="utf-8") == "hello"
    assert not final_dir.exists()


# =============================================================================
# Route B endurecida (Section 8 of the frozen final specification,
# ``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``).
# Covers Section 19.3 (items 39-66) and Section 19.8 (items 136-150) of the
# frozen spec's normative test matrix.
# =============================================================================

_INPUT_ID = "curcumin-pubchem"
_CONTENT = b"the quick brown fox jumps over the lazy dog\n"


def _make_raw_input(tmp_path, *, input_id=_INPUT_ID, filename="raw.sdf", content=_CONTENT):
    """Build a minimal `inputs/ligands/<input_id>/<filename>` RAW artifact
    under a fresh project_dir, returning (project_dir, raw_artifact_path,
    checksum, snapshot_path)."""

    project_dir = tmp_path / "project"
    root = project_dir / "inputs" / "ligands" / input_id
    root.mkdir(parents=True)
    raw_file = root / filename
    raw_file.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()
    raw_artifact_path = f"inputs/ligands/{input_id}/{filename}"
    snapshot_path = tmp_path / "snapshot.sdf"
    return project_dir, raw_artifact_path, checksum, snapshot_path


def _snapshot(project_dir, raw_artifact_path, checksum, snapshot_path, *, input_id=_INPUT_ID):
    return create_verified_raw_snapshot(
        project_dir=project_dir,
        input_id=input_id,
        raw_artifact_path=raw_artifact_path,
        raw_artifact_checksum=checksum,
        snapshot_path=snapshot_path,
    )


def _make_junction(link_path: Path, target: Path) -> None:
    """A junction is constructible without privilege on Windows
    (confirmed by FS-018-01) - used here as the real, reproducible stand-in
    for "any reparse point", since a symlink is not constructible without
    elevation/Developer Mode on this platform."""

    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
        check=True,
        capture_output=True,
    )


# --- 39-44: lexical validation ---


def test_happy_path_creates_verified_snapshot(tmp_path):
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)

    result = _snapshot(project_dir, raw_path, checksum, snapshot_path)

    assert result.snapshot_path == snapshot_path
    assert result.snapshot_checksum == checksum
    assert result.snapshot_size_bytes == len(_CONTENT)
    assert snapshot_path.read_bytes() == _CONTENT


def test_rejects_posix_absolute_path(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, "/etc/passwd", checksum, snapshot_path)


def test_rejects_windows_drive_absolute_path(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, "C:/evil.sdf", checksum, snapshot_path)


def test_rejects_unc_path(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, "//server/share/evil.sdf", checksum, snapshot_path)


def test_rejects_drive_relative_path(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, "C:evil.sdf", checksum, snapshot_path)


def test_rejects_dot_and_dotdot_components(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(
            project_dir,
            f"inputs/ligands/{_INPUT_ID}/../../../evil.sdf",
            checksum,
            snapshot_path,
        )
    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, f"inputs/ligands/{_INPUT_ID}/./raw.sdf", checksum, snapshot_path)


def test_rejects_backslash_in_canonical_path(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(
            project_dir,
            f"inputs\\ligands\\{_INPUT_ID}\\raw.sdf",
            checksum,
            snapshot_path,
        )


def test_rejects_empty_path(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, "", checksum, snapshot_path)


def test_rejects_wrong_input_id_prefix(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, "inputs/ligands/other-input/raw.sdf", checksum, snapshot_path)


# --- 45-46: containment via is_relative_to(), never startswith() ---


def test_require_relative_to_root_uses_is_relative_to_not_startswith(tmp_path):
    """Item 45: directly exercise the internal containment primitive and
    confirm it is `is_relative_to()`-based - the naive `startswith()` trap
    this module must never reproduce is demonstrated failing first."""

    root = tmp_path / "allowed_root"
    evil = tmp_path / "allowed_root_EVIL"
    root.mkdir()
    evil.mkdir()
    evil_file = evil / "file.sdf"
    evil_file.write_bytes(b"x")

    resolved_root = root.resolve(strict=True)

    # the naive trap this module must NOT reproduce
    assert str(evil_file.resolve()).startswith(str(resolved_root))

    with pytest.raises(UnsafeRawArtifactPathError):
        atomic_module._require_relative_to_root(evil_file, resolved_root)


def test_sibling_prefix_collision_rejected_end_to_end(tmp_path):
    """Item 46: `allowed_root_EVIL`-style sibling collision rejected by the
    full `create_verified_raw_snapshot` flow, not only the internal
    primitive above."""

    project_dir = tmp_path / "project"
    evil_root = project_dir / "inputs" / "ligands" / f"{_INPUT_ID}_EVIL"
    evil_root.mkdir(parents=True)
    (evil_root / "raw.sdf").write_bytes(_CONTENT)
    checksum = hashlib.sha256(_CONTENT).hexdigest()

    # the lexical prefix check alone already rejects this (no trailing
    # slash match) - confirming the OUTER contract still refuses safely,
    # end to end, even though the deeper is_relative_to() check is proven
    # independently above.
    with pytest.raises(UnsafeRawArtifactPathError):
        create_verified_raw_snapshot(
            project_dir=project_dir,
            input_id=_INPUT_ID,
            raw_artifact_path=f"inputs/ligands/{_INPUT_ID}_EVIL/raw.sdf",
            raw_artifact_checksum=checksum,
            snapshot_path=tmp_path / "snap.sdf",
        )


# --- 47/48/142-144: symlink, junction, reparse point rejection ---


def test_rejects_junction_as_root_ancestor(tmp_path):
    """Item 51/142-144 (junction case): the authorized root itself,
    substituted by a junction, is rejected - junctions are constructible
    without privilege (FS-018-01)."""

    project_dir = tmp_path / "project"
    real_target = tmp_path / "real_target"
    real_target.mkdir(parents=True)
    (real_target / "raw.sdf").write_bytes(_CONTENT)
    checksum = hashlib.sha256(_CONTENT).hexdigest()

    junction_path = project_dir / "inputs" / "ligands" / _INPUT_ID
    junction_path.parent.mkdir(parents=True)
    _make_junction(junction_path, real_target)

    with pytest.raises(UnsafeRawArtifactPathError):
        create_verified_raw_snapshot(
            project_dir=project_dir,
            input_id=_INPUT_ID,
            raw_artifact_path=f"inputs/ligands/{_INPUT_ID}/raw.sdf",
            raw_artifact_checksum=checksum,
            snapshot_path=tmp_path / "snap.sdf",
        )


def test_rejects_junction_as_intermediate_ancestor(tmp_path):
    """Item 140: an ancestor that is neither the root nor the immediate
    parent is also rejected when it is a junction."""

    project_dir = tmp_path / "project"
    root = project_dir / "inputs" / "ligands" / _INPUT_ID
    root.mkdir(parents=True)
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    (real_target / "raw.sdf").write_bytes(_CONTENT)
    checksum = hashlib.sha256(_CONTENT).hexdigest()

    _make_junction(root / "subdir", real_target)

    with pytest.raises(UnsafeRawArtifactPathError):
        create_verified_raw_snapshot(
            project_dir=project_dir,
            input_id=_INPUT_ID,
            raw_artifact_path=f"inputs/ligands/{_INPUT_ID}/subdir/raw.sdf",
            raw_artifact_checksum=checksum,
            snapshot_path=tmp_path / "snap.sdf",
        )


def test_rejects_reparse_point_final_file_via_injection(tmp_path, monkeypatch):
    """Item 47: symlink-on-final-file rejection, exercised via localized
    injection - a real file symlink is not constructible without
    elevation/Developer Mode on this platform (FS-018-01 confirmed this
    empirically; this is not a new spike, only the same, already-accepted
    environment limitation). `_capture_identity` is patched to report the
    final file as a reparse point on its first call only, leaving every
    other path (ancestors) using the real implementation."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    real_capture = atomic_module._capture_identity

    def _fake_capture(path):
        token = real_capture(path)
        if Path(path) == final_path:
            return atomic_module._IdentityToken(
                st_dev=token.st_dev,
                st_ino=token.st_ino,
                is_reparse_point=True,
                file_attributes=(token.file_attributes or 0) | 0x400,
                reparse_tag=0xA000000C,  # IO_REPARSE_TAG_SYMLINK
                is_regular_file=token.is_regular_file,
                is_directory=token.is_directory,
                size=token.size,
            )
        return token

    monkeypatch.setattr(atomic_module, "_capture_identity", _fake_capture)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_rejects_reparse_point_ancestor_via_injection(tmp_path, monkeypatch):
    """Item 142: symlink-as-ancestor, via the same localized injection
    technique as the final-file case above, for the case a real symlink
    cannot stand in for."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    root = project_dir / "inputs" / "ligands" / _INPUT_ID
    real_capture = atomic_module._capture_identity

    def _fake_capture(path):
        token = real_capture(path)
        if Path(path) == root:
            return atomic_module._IdentityToken(
                st_dev=token.st_dev,
                st_ino=token.st_ino,
                is_reparse_point=True,
                file_attributes=(token.file_attributes or 0) | 0x400,
                reparse_tag=0xA000000C,
                is_regular_file=token.is_regular_file,
                is_directory=token.is_directory,
                size=token.size,
            )
        return token

    monkeypatch.setattr(atomic_module, "_capture_identity", _fake_capture)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


# --- 49: non-regular file ---


def test_rejects_directory_as_final_path(tmp_path):
    project_dir = tmp_path / "project"
    root = project_dir / "inputs" / "ligands" / _INPUT_ID
    root.mkdir(parents=True)
    (root / "raw.sdf").mkdir()  # a directory, not a regular file

    with pytest.raises(RawArtifactTypeError):
        create_verified_raw_snapshot(
            project_dir=project_dir,
            input_id=_INPUT_ID,
            raw_artifact_path=f"inputs/ligands/{_INPUT_ID}/raw.sdf",
            raw_artifact_checksum="0" * 64,
            snapshot_path=tmp_path / "snap.sdf",
        )


def test_rejects_missing_raw_file(tmp_path):
    project_dir = tmp_path / "project"
    root = project_dir / "inputs" / "ligands" / _INPUT_ID
    root.mkdir(parents=True)

    with pytest.raises(RawArtifactTypeError):
        create_verified_raw_snapshot(
            project_dir=project_dir,
            input_id=_INPUT_ID,
            raw_artifact_path=f"inputs/ligands/{_INPUT_ID}/missing.sdf",
            raw_artifact_checksum="0" * 64,
            snapshot_path=tmp_path / "snap.sdf",
        )


# --- 50/136-141/145-147: ancestor identity capture and reverification ---


def test_ancestor_swapped_before_open_is_detected(tmp_path, monkeypatch):
    """Item 138: the immediate parent (here, the root itself) is replaced
    - removed and recreated with the same name - between initial
    validation and opening the final file. Injected by wrapping the
    module's own pre-open reverification call."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    root = project_dir / "inputs" / "ligands" / _INPUT_ID
    real_reverify = atomic_module._reverify_ancestor_chain
    call_count = {"n": 0}

    def _mutating_reverify(ancestor_paths, previous_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # first reverify call happens right before opening the final
            # file (Section 8.3.2, step 4) - swap the root out from under
            # it, preserving the file inside under the same relative name.
            import shutil as _shutil

            _shutil.rmtree(root)
            root.mkdir()
            (root / "raw.sdf").write_bytes(_CONTENT)
        return real_reverify(ancestor_paths, previous_tokens)

    monkeypatch.setattr(atomic_module, "_reverify_ancestor_chain", _mutating_reverify)

    with pytest.raises(RawArtifactIdentityUnstableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_ancestor_swapped_after_open_before_read_is_detected(tmp_path, monkeypatch):
    """Item 139: the root's identity diverges between opening the
    descriptor and reading it - caught by the second reverification call
    (Section 8.3.2, step 6), immediately after opening. Since the final
    file is already open by this point, a real filesystem swap of its
    parent directory is not obtainable on Windows (renaming or removing a
    directory containing an open file handle is itself denied by the OS,
    confirmed empirically) - so this uses the same localized
    `_capture_identity` fabrication technique as the reparse-point
    injection tests above, changing only `st_ino` for the root, starting
    from its 3rd call (the one made by the reverify inside `with file:`)."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    root = project_dir / "inputs" / "ligands" / _INPUT_ID
    real_capture = atomic_module._capture_identity
    root_calls = {"n": 0}

    def _fake_capture(path):
        token = real_capture(path)
        if Path(path) == root:
            root_calls["n"] += 1
            if root_calls["n"] >= 3:
                return atomic_module._IdentityToken(
                    st_dev=token.st_dev,
                    st_ino=token.st_ino + 1,
                    is_reparse_point=token.is_reparse_point,
                    file_attributes=token.file_attributes,
                    reparse_tag=token.reparse_tag,
                    is_regular_file=token.is_regular_file,
                    is_directory=token.is_directory,
                    size=token.size,
                )
        return token

    monkeypatch.setattr(atomic_module, "_capture_identity", _fake_capture)

    with pytest.raises(RawArtifactIdentityUnstableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_ancestor_swapped_during_read_is_detected_post_read(tmp_path, monkeypatch):
    """Item 140/145 timing variant: the root is swapped WHILE the file is
    being read (between the pre-read reverify and the mandatory post-read
    reverify) - caught only by the post-read verification (Section
    8.3.2.1), the third and final reverify call."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    root = project_dir / "inputs" / "ligands" / _INPUT_ID
    real_reverify = atomic_module._reverify_ancestor_chain
    call_count = {"n": 0}

    def _mutating_reverify(ancestor_paths, previous_tokens):
        call_count["n"] += 1
        if call_count["n"] == 3:
            import shutil as _shutil

            _shutil.rmtree(root)
            root.mkdir()
            (root / "raw.sdf").write_bytes(_CONTENT)
        return real_reverify(ancestor_paths, previous_tokens)

    monkeypatch.setattr(atomic_module, "_reverify_ancestor_chain", _mutating_reverify)

    with pytest.raises(RawArtifactIdentityUnstableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_ancestor_removed_and_recreated_same_name_changes_identity(tmp_path):
    """Item 141: removing and recreating a directory with the same name
    always produces a new st_ino on this filesystem - confirmed directly,
    not assumed."""

    directory = tmp_path / "dir"
    directory.mkdir()
    token_before = atomic_module._capture_identity(directory)

    directory.rmdir()
    directory.mkdir()
    token_after = atomic_module._capture_identity(directory)

    assert not atomic_module._identity_matches(token_before, token_after)


def test_ancestor_chain_redirected_outside_root_rejected(tmp_path):
    """Item 145: after passing the initial lexical prefix check, an
    ancestor is a junction redirecting outside the authorized root -
    rejected both as a reparse point and, independently, on containment."""

    project_dir = tmp_path / "project"
    outside = tmp_path / "outside_root"
    outside.mkdir()
    (outside / "raw.sdf").write_bytes(_CONTENT)
    checksum = hashlib.sha256(_CONTENT).hexdigest()

    root_parent = project_dir / "inputs" / "ligands"
    root_parent.mkdir(parents=True)
    _make_junction(root_parent / _INPUT_ID, outside)

    with pytest.raises(UnsafeRawArtifactPathError):
        create_verified_raw_snapshot(
            project_dir=project_dir,
            input_id=_INPUT_ID,
            raw_artifact_path=f"inputs/ligands/{_INPUT_ID}/raw.sdf",
            raw_artifact_checksum=checksum,
            snapshot_path=tmp_path / "snap.sdf",
        )


def test_final_file_identity_preserved_ancestor_diverges(tmp_path, monkeypatch):
    """Item 146: only the ancestor's identity changes - the final file
    itself is untouched - still raises raw-identity-unstable."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    root = project_dir / "inputs" / "ligands" / _INPUT_ID
    real_reverify = atomic_module._reverify_ancestor_chain
    call_count = {"n": 0}

    def _mutating_reverify(ancestor_paths, previous_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # recreate the root in place (new inode) but copy the exact
            # same file back into it - the final file's own bytes/name are
            # untouched by this substitution.
            import shutil as _shutil

            _shutil.rmtree(root)
            root.mkdir()
            (root / "raw.sdf").write_bytes(_CONTENT)
        return real_reverify(ancestor_paths, previous_tokens)

    monkeypatch.setattr(atomic_module, "_reverify_ancestor_chain", _mutating_reverify)

    with pytest.raises(RawArtifactIdentityUnstableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_all_ancestors_preserved_final_file_diverges(tmp_path, monkeypatch):
    """Item 147/53: the ancestor chain is completely untouched, but the
    final file itself is swapped (same name, different content/inode)
    between the pre-open reverify and the post-open fstat reverify
    (Section 8.3.2 step 5). Injected by wrapping `_capture_identity` so
    that the SECOND time it is asked about the final path (the post-open
    reverification), it reflects a freshly-swapped file - every other
    call (ancestors, the first final-path check) uses the real
    implementation untouched."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    real_capture = atomic_module._capture_identity
    final_path_calls = {"n": 0}

    def _fake_capture(path):
        if Path(path) == final_path:
            final_path_calls["n"] += 1
            if final_path_calls["n"] == 2:
                final_path.unlink()
                final_path.write_bytes(b"substituted content, same name")
        return real_capture(path)

    monkeypatch.setattr(atomic_module, "_capture_identity", _fake_capture)

    with pytest.raises(RawArtifactIdentityUnstableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_final_file_swapped_between_pre_open_capture_and_open_is_detected(tmp_path, monkeypatch):
    """Item 136: the final file is swapped for a different one with the
    same name immediately after its pre-open lstat but before the actual
    `open()` call - caught by the post-open fstat comparison (Section
    8.3.2, step 5)."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    real_reverify = atomic_module._reverify_ancestor_chain

    def _swapping_reverify(ancestor_paths, previous_tokens):
        final_path.unlink()
        final_path.write_bytes(b"swapped before open")
        return real_reverify(ancestor_paths, previous_tokens)

    monkeypatch.setattr(atomic_module, "_reverify_ancestor_chain", _swapping_reverify)

    with pytest.raises(RawArtifactIdentityUnstableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_final_file_content_swapped_between_open_and_read_caught_by_checksum(tmp_path, monkeypatch):
    """Item 137: content is swapped (same size, so identity/size checks
    alone do not catch it) through the validated path itself, between the
    post-open identity check and the actual read - caught by the checksum
    comparison, the same class of defense as the in-place same-size
    mutation case below (Section 8.6, item 1)."""

    content = b"A" * 32
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path, content=content)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    real_reverify = atomic_module._reverify_ancestor_chain
    call_count = {"n": 0}

    def _mutating_reverify(ancestor_paths, previous_tokens):
        call_count["n"] += 1
        if call_count["n"] == 2:
            with open(final_path, "r+b") as writer:
                writer.seek(0)
                writer.write(b"B" * len(content))
        return real_reverify(ancestor_paths, previous_tokens)

    monkeypatch.setattr(atomic_module, "_reverify_ancestor_chain", _mutating_reverify)

    with pytest.raises(RawArtifactIntegrityError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


# --- 56: checksum/size come from bytes actually read, never a re-read ---


def test_checksum_computed_from_bytes_read_not_from_snapshot_file(tmp_path, monkeypatch):
    """Item 56: even though this module's own `compute_sha256` helper
    exists and is used elsewhere, Route B must never call it against
    `snapshot_path` (or anything else) - proven by poisoning it and
    confirming the result is unaffected."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)

    def _poisoned_compute_sha256(path):
        raise AssertionError("compute_sha256 must not be called by create_verified_raw_snapshot")

    monkeypatch.setattr(atomic_module, "compute_sha256", _poisoned_compute_sha256)

    result = _snapshot(project_dir, raw_path, checksum, snapshot_path)

    assert result.snapshot_checksum == checksum
    assert result.snapshot_size_bytes == len(_CONTENT)


# --- 57-59: same-size mutation / truncation / append, all in place ---


def test_same_size_in_place_content_mutation_caught_by_checksum(tmp_path, monkeypatch):
    """Item 57: content changed, size and inode both preserved - identity
    comparison alone cannot see this; only the checksum comparison does
    (Section 8.6, item 1, the defining case for RawArtifactIntegrityError)."""

    content = b"A" * 32
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path, content=content)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    mutated = {"done": False}

    def _wrapped_open(path, mode="r", *args, **kwargs):
        if Path(path) == final_path and mode == "rb" and not mutated["done"]:
            mutated["done"] = True
            final_path.write_bytes(b"B" * len(content))
        return open(path, mode, *args, **kwargs)

    monkeypatch.setattr(atomic_module, "open", _wrapped_open, raising=False)

    with pytest.raises(RawArtifactIntegrityError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_truncation_in_place_detected_via_size_mismatch(tmp_path, monkeypatch):
    """Item 58: the file is truncated in place (same inode, smaller size)
    after the pre-open lstat - the number of bytes actually read no longer
    matches the size observed before opening (Section 8.3.2.1, step 5)."""

    content = b"A" * 32
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path, content=content)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    mutated = {"done": False}

    def _wrapped_open(path, mode="r", *args, **kwargs):
        if Path(path) == final_path and mode == "rb" and not mutated["done"]:
            mutated["done"] = True
            with open(final_path, "r+b") as writer:
                writer.truncate(10)
        return open(path, mode, *args, **kwargs)

    monkeypatch.setattr(atomic_module, "open", _wrapped_open, raising=False)

    with pytest.raises(RawArtifactIdentityUnstableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_append_in_place_detected_via_size_mismatch(tmp_path, monkeypatch):
    """Item 59: the file is appended to in place (same inode, larger size)
    after the pre-open lstat - same size-mismatch defense as truncation."""

    content = b"A" * 32
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path, content=content)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    mutated = {"done": False}

    def _wrapped_open(path, mode="r", *args, **kwargs):
        if Path(path) == final_path and mode == "rb" and not mutated["done"]:
            mutated["done"] = True
            with open(final_path, "r+b") as writer:
                writer.seek(0, 2)
                writer.write(b"extra bytes appended")
        return open(path, mode, *args, **kwargs)

    monkeypatch.setattr(atomic_module, "open", _wrapped_open, raising=False)

    with pytest.raises(RawArtifactIdentityUnstableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


# --- 61-62: hard links ---


def test_raw_artifact_with_existing_hardlink_succeeds_normally(tmp_path):
    """Item 61: an additional hard link to the RAW artifact existing
    elsewhere (st_nlink > 1) is not, on its own, a rejection condition -
    Route B validates the identity and content of the validated path
    itself, never the link count."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    other_link = project_dir / "other_hardlink.sdf"
    os.link(final_path, other_link)

    result = _snapshot(project_dir, raw_path, checksum, snapshot_path)

    assert result.snapshot_checksum == checksum


def test_mutation_via_other_hardlink_before_open_is_caught_by_checksum(tmp_path, monkeypatch):
    """Item 62: content changed through a DIFFERENT hard-linked name
    pointing at the same inode is indistinguishable, by identity alone,
    from an in-place mutation through the validated name itself - still
    caught by the checksum comparison."""

    content = b"A" * 32
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path, content=content)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    other_link = project_dir / "other_hardlink.sdf"
    os.link(final_path, other_link)
    mutated = {"done": False}

    def _wrapped_open(path, mode="r", *args, **kwargs):
        if Path(path) == final_path and mode == "rb" and not mutated["done"]:
            mutated["done"] = True
            other_link.write_bytes(b"B" * len(content))
        return open(path, mode, *args, **kwargs)

    monkeypatch.setattr(atomic_module, "open", _wrapped_open, raising=False)

    with pytest.raises(RawArtifactIntegrityError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


# --- 63-65: platform/filesystem fail-closed ---


def test_non_windows_platform_fails_closed(tmp_path, monkeypatch):
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    monkeypatch.setattr(atomic_module.os, "name", "posix")

    with pytest.raises(RawArtifactNoFollowUnavailableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_non_ntfs_filesystem_fails_closed(tmp_path, monkeypatch):
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    monkeypatch.setattr(atomic_module, "_get_volume_filesystem", lambda path: "FAT32")

    with pytest.raises(RawArtifactNoFollowUnavailableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_non_fixed_drive_fails_closed(tmp_path, monkeypatch):
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    monkeypatch.setattr(atomic_module, "_get_drive_type", lambda path: 4)  # DRIVE_REMOTE

    with pytest.raises(RawArtifactNoFollowUnavailableError):
        _snapshot(project_dir, raw_path, checksum, snapshot_path)


def test_platform_check_runs_before_touching_raw_file(tmp_path, monkeypatch):
    """Items 63-65, fail-closed ordering: the platform/filesystem check
    happens before the RAW artifact's own root is even resolved - proven
    by never creating the `inputs/ligands/<id>/` directory at all and
    still getting the platform error, not a not-found error."""

    project_dir = tmp_path / "project"
    monkeypatch.setattr(atomic_module.os, "name", "posix")

    with pytest.raises(RawArtifactNoFollowUnavailableError):
        create_verified_raw_snapshot(
            project_dir=project_dir,
            input_id=_INPUT_ID,
            raw_artifact_path=f"inputs/ligands/{_INPUT_ID}/raw.sdf",
            raw_artifact_checksum="0" * 64,
            snapshot_path=tmp_path / "snap.sdf",
        )


# --- 66: RAW immutability and no-absolute-path-in-error ---


def test_raw_artifact_untouched_after_successful_snapshot(tmp_path):
    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    final_path = project_dir / "inputs" / "ligands" / _INPUT_ID / "raw.sdf"
    before_bytes = final_path.read_bytes()
    before_mtime = final_path.stat().st_mtime_ns

    _snapshot(project_dir, raw_path, checksum, snapshot_path)

    assert final_path.read_bytes() == before_bytes
    assert final_path.stat().st_mtime_ns == before_mtime


def test_unsafe_path_error_never_includes_absolute_path(tmp_path):
    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError) as exc_info:
        _snapshot(project_dir, "/etc/passwd", checksum, snapshot_path)

    assert str(project_dir) not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


# --- 148-150: snapshot_path never leaves a partial/stale artifact behind ---


def test_snapshot_path_removed_on_failure(tmp_path):
    """Item 148: on any failure, `snapshot_path` does not exist afterward -
    no partial snapshot is ever left for a caller to mistake for a
    complete one."""

    project_dir, _, checksum, snapshot_path = _make_raw_input(tmp_path)

    with pytest.raises(UnsafeRawArtifactPathError):
        _snapshot(project_dir, "/etc/passwd", checksum, snapshot_path)

    assert not snapshot_path.exists()


def test_snapshot_path_with_stale_preexisting_content_removed_on_failure(tmp_path):
    """Item 149: a stale file already sitting at `snapshot_path` before the
    call (e.g. left over from an unrelated prior run) is removed on
    failure, not left in place looking like a valid result."""

    project_dir, raw_path, checksum, snapshot_path = _make_raw_input(tmp_path)
    snapshot_path.write_bytes(b"stale leftover content")

    with pytest.raises(RawArtifactIntegrityError):
        _snapshot(project_dir, raw_path, "0" * 64, snapshot_path)

    assert not snapshot_path.exists()


def test_snapshot_absent_entirely_after_mid_read_failure(tmp_path):
    """Item 150: a failure detected only after the read loop has already
    written SOME bytes to `snapshot_path` (checksum mismatch discovered
    only at the very end, over a reasonably large file) still leaves
    `snapshot_path` completely absent - never a truncated, partially
    written file mistaken for a complete one."""

    project_dir, raw_path, _checksum, snapshot_path = _make_raw_input(
        tmp_path, content=b"X" * 200_000
    )

    with pytest.raises(RawArtifactIntegrityError):
        _snapshot(project_dir, raw_path, "0" * 64, snapshot_path)

    assert not snapshot_path.exists()
