import os
import tempfile

import pytest

from molecular.atomic import atomic_write_text, publish_staged_directory


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
