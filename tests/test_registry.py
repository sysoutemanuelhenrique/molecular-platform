import hashlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import molecular.registry as registry_module
from molecular.registry import load_ligand, load_protein, register_ligand, register_protein


def _make_raw_file(tmp_path, name="curcumin.sdf", content=b"fake-sdf-bytes"):
    raw_file = tmp_path / name
    raw_file.write_bytes(content)
    return raw_file


def test_register_ligand_uploaded_creates_metadata_and_copies_raw_file(tmp_path):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    ligand = register_ligand(
        project_dir,
        input_id="curcumin",
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
    )

    assert ligand.id == "curcumin"
    assert (project_dir / "inputs" / "ligands" / "curcumin.yaml").exists()

    stored_file = project_dir / "inputs" / "ligands" / "curcumin" / "curcumin.sdf"
    assert stored_file.exists()
    assert stored_file.read_bytes() == raw_file.read_bytes()


def test_register_ligand_downloaded_requires_and_persists_source_info(tmp_path):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    ligand = register_ligand(
        project_dir,
        input_id="curcumin",
        name="Curcumin",
        external_id="CID 969516",
        method="downloaded",
        source_name="PubChem",
        source_identifier="CID 969516",
        raw_file=raw_file,
    )

    assert ligand.acquisition.method == "downloaded"
    assert ligand.acquisition.source_name == "PubChem"

    reloaded = load_ligand(project_dir, "curcumin")
    assert reloaded.acquisition.source_identifier == "CID 969516"


def test_register_protein_downloaded_creates_metadata(tmp_path):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path, name="3la4.cif")

    protein = register_protein(
        project_dir,
        input_id="urease",
        name="Jack bean urease",
        external_id="3LA4",
        method="downloaded",
        source_name="RCSB PDB",
        source_identifier="3LA4",
        raw_file=raw_file,
    )

    assert protein.id == "urease"
    assert (project_dir / "inputs" / "proteins" / "urease.yaml").exists()

    reloaded = load_protein(project_dir, "urease")
    assert reloaded.name == "Jack bean urease"


def test_register_computes_and_persists_checksum(tmp_path):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path, content=b"specific-content-for-hash")
    expected_checksum = hashlib.sha256(b"specific-content-for-hash").hexdigest()

    register_ligand(
        project_dir,
        input_id="curcumin",
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
    )

    reloaded = load_ligand(project_dir, "curcumin")
    assert reloaded.acquisition.raw_artifact.checksum == expected_checksum


def test_persisted_checksum_matches_stored_artifact_bytes(tmp_path):
    """The persisted checksum must reflect the RAW copy kept inside the experiment,
    not merely the original source file (they happen to match because copyfile is
    a faithful byte copy, but this test hashes the stored artifact directly)."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path, content=b"content-for-stored-checksum-check")

    register_ligand(
        project_dir,
        input_id="curcumin",
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
    )

    reloaded = load_ligand(project_dir, "curcumin")

    stored_path = project_dir / reloaded.acquisition.raw_artifact.path
    stored_checksum = hashlib.sha256(stored_path.read_bytes()).hexdigest()

    assert reloaded.acquisition.raw_artifact.checksum == stored_checksum


def test_register_stores_forward_slash_path(tmp_path):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    ligand = register_ligand(
        project_dir,
        input_id="curcumin",
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
    )

    assert ligand.acquisition.raw_artifact.path == "inputs/ligands/curcumin/curcumin.sdf"
    assert "\\" not in ligand.acquisition.raw_artifact.path


def test_register_rejects_missing_raw_file(tmp_path):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        register_ligand(
            project_dir,
            input_id="curcumin",
            name="Curcumin",
            method="uploaded",
            raw_file=tmp_path / "missing.sdf",
        )


def test_register_rejects_duplicate_id(tmp_path):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    register_ligand(
        project_dir,
        input_id="curcumin",
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
    )

    with pytest.raises(FileExistsError):
        register_ligand(
            project_dir,
            input_id="curcumin",
            name="Curcumin (again)",
            method="uploaded",
            raw_file=raw_file,
        )


def test_register_rejects_downloaded_without_source_info_and_writes_nothing(tmp_path):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    with pytest.raises(ValidationError):
        register_ligand(
            project_dir,
            input_id="curcumin",
            name="Curcumin",
            method="downloaded",
            raw_file=raw_file,
        )

    assert not (project_dir / "inputs" / "ligands" / "curcumin.yaml").exists()
    assert not (project_dir / "inputs" / "ligands" / "curcumin").exists()


def test_acquired_at_is_computed_once_and_persisted_unchanged(tmp_path, monkeypatch):
    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    fixed_now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    class _FixedDateTime(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return fixed_now

    monkeypatch.setattr(registry_module, "datetime", _FixedDateTime)

    ligand = register_ligand(
        project_dir,
        input_id="curcumin",
        name="Curcumin",
        method="uploaded",
        raw_file=raw_file,
    )

    # datetime.now() must be called exactly once per registration, even though
    # _build_scientific_input is invoked twice (pre-write validation + final write).
    assert _FixedDateTime.calls == 1
    assert ligand.acquisition.acquired_at == fixed_now

    reloaded = load_ligand(project_dir, "curcumin")
    assert reloaded.acquisition.acquired_at == fixed_now
