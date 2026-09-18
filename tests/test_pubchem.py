import tempfile
import urllib.error
from pathlib import Path

import pytest

from molecular.pubchem import (
    PubChemAcquisitionError,
    PubChemNotFoundError,
    build_pubchem_sdf_url,
    cleanup_temp_artifact,
    fetch_pubchem_sdf_2d,
)


def test_build_pubchem_sdf_url_includes_cid_and_record_type_2d():
    url = build_pubchem_sdf_url(969516)

    assert url == (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/969516/SDF?record_type=2d"
    )


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_pubchem_sdf_2d_writes_temp_file_with_response_bytes(monkeypatch):
    fake_sdf = b"fake-sdf-content\n$$$$\n"

    def _fake_urlopen(url, timeout=None):
        assert "969516" in url
        assert "record_type=2d" in url
        return _FakeResponse(fake_sdf)

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    temp_path = fetch_pubchem_sdf_2d(969516)

    try:
        assert temp_path.exists()
        assert temp_path.suffix == ".sdf"
        assert temp_path.read_bytes() == fake_sdf
    finally:
        cleanup_temp_artifact(temp_path)


def test_fetch_pubchem_sdf_2d_uses_deterministic_basename(monkeypatch):
    def _fake_urlopen(url, timeout=None):
        return _FakeResponse(b"fake-sdf-content")

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    temp_path = fetch_pubchem_sdf_2d(969516)

    try:
        assert temp_path.name == "pubchem_cid_969516_2d.sdf"
    finally:
        cleanup_temp_artifact(temp_path)


def test_cleanup_temp_artifact_removes_whole_temp_directory(monkeypatch):
    def _fake_urlopen(url, timeout=None):
        return _FakeResponse(b"fake-sdf-content")

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    temp_path = fetch_pubchem_sdf_2d(969516)
    temp_dir = temp_path.parent
    assert temp_dir.exists()

    cleanup_temp_artifact(temp_path)

    assert not temp_dir.exists()


def test_fetch_pubchem_sdf_2d_cleans_up_temp_dir_when_write_fails(monkeypatch):
    """If writing the downloaded bytes to the deterministic temp file fails,
    the temp directory created for it must not be left behind."""

    def _fake_urlopen(url, timeout=None):
        return _FakeResponse(b"fake-sdf-content")

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    created_dirs = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        created = real_mkdtemp(*args, **kwargs)
        created_dirs.append(created)
        return created

    monkeypatch.setattr("molecular.pubchem.tempfile.mkdtemp", _tracking_mkdtemp)

    real_write_bytes = Path.write_bytes

    def _failing_write_bytes(self, data):
        if self.name.startswith("pubchem_cid_"):
            raise OSError("simulated disk failure while writing temp artifact")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", _failing_write_bytes)

    with pytest.raises(OSError):
        fetch_pubchem_sdf_2d(969516)

    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists()


def test_fetch_pubchem_sdf_2d_raises_not_found_on_404(monkeypatch):
    def _fake_urlopen(url, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(PubChemNotFoundError):
        fetch_pubchem_sdf_2d(999999999)


def test_fetch_pubchem_sdf_2d_raises_acquisition_error_on_other_http_status(monkeypatch):
    def _fake_urlopen(url, timeout=None):
        raise urllib.error.HTTPError(url, 500, "Server Error", hdrs=None, fp=None)

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(PubChemAcquisitionError):
        fetch_pubchem_sdf_2d(969516)


def test_fetch_pubchem_sdf_2d_raises_acquisition_error_on_timeout(monkeypatch):
    def _fake_urlopen(url, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(PubChemAcquisitionError):
        fetch_pubchem_sdf_2d(969516)


def test_fetch_pubchem_sdf_2d_raises_acquisition_error_on_wrapped_timeout(monkeypatch):
    def _fake_urlopen(url, timeout=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(PubChemAcquisitionError):
        fetch_pubchem_sdf_2d(969516)


def test_fetch_pubchem_sdf_2d_raises_acquisition_error_on_network_failure(monkeypatch):
    def _fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(PubChemAcquisitionError):
        fetch_pubchem_sdf_2d(969516)


def test_fetch_pubchem_sdf_2d_raises_acquisition_error_on_empty_response(monkeypatch):
    def _fake_urlopen(url, timeout=None):
        return _FakeResponse(b"")

    monkeypatch.setattr("molecular.pubchem.urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(PubChemAcquisitionError):
        fetch_pubchem_sdf_2d(969516)
