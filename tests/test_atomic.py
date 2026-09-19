import os
import tempfile

import pytest

from molecular.atomic import atomic_write_text


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
