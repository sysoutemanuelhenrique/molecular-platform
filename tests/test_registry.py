import hashlib
import logging
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import molecular.registry as registry_module
from molecular.registry import (
    InputIdentityConflictError,
    load_ligand,
    load_protein,
    register_ligand,
    register_protein,
)


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


# --- Incremento 009: Persistence Integrity Protocol v0.1 ---


def test_input_identity_conflict_error_is_file_exists_error_subclass():
    """(13) Existing callers that catch FileExistsError for 'already
    registered' - including test_register_rejects_duplicate_id above -
    keep working unchanged, since InputIdentityConflictError is-a
    FileExistsError."""

    assert issubclass(InputIdentityConflictError, FileExistsError)


def test_register_rejects_casefold_collision_with_existing_registration(tmp_path):
    """(14) 'curcumin' followed by 'Curcumin' is a storage identity
    conflict, not a second valid registration."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    register_ligand(
        project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
    )

    with pytest.raises(InputIdentityConflictError):
        register_ligand(
            project_dir,
            input_id="Curcumin",
            name="Curcumin (again)",
            method="uploaded",
            raw_file=raw_file,
        )


def test_casefold_collision_detected_before_any_filesystem_mutation_for_new_id(tmp_path):
    """(15) The conflict must be found by our own casefold comparison over
    a directory listing - a Python-level check - not by however the
    underlying OS/filesystem happens to treat case. Proven here by showing
    that rejecting 'Curcumin' leaves absolutely nothing new on disk: no
    directory or file for 'Curcumin' was ever attempted, because the
    conflict check runs before any staging or destination path is touched
    for the new id."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    register_ligand(
        project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
    )

    kind_dir = project_dir / "inputs" / "ligands"
    before = {entry.name for entry in kind_dir.iterdir()}

    with pytest.raises(InputIdentityConflictError):
        register_ligand(
            project_dir,
            input_id="Curcumin",
            name="Curcumin variant",
            method="uploaded",
            raw_file=raw_file,
        )

    after = {entry.name for entry in kind_dir.iterdir()}
    # the exact on-disk names are unchanged - nothing new was created for
    # 'Curcumin'. (Checking `.exists()` for a 'Curcumin.yaml'/'Curcumin'
    # path here would be misleading on a case-insensitive filesystem, since
    # it would resolve to the pre-existing 'curcumin' entries regardless of
    # whether anything new was ever created - the directory-listing
    # comparison above is the platform-independent proof.)
    assert after == before


def test_orphan_raw_directory_without_metadata_blocks_registration(tmp_path):
    """(16) A final RAW directory with no matching metadata is an orphan -
    not a registration - but it still occupies the identity and must block
    a new attempt rather than being silently overwritten."""

    project_dir = tmp_path / "exp001"
    kind_dir = project_dir / "inputs" / "ligands"
    orphan_dir = kind_dir / "curcumin"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "leftover.sdf").write_bytes(b"leftover-from-a-crashed-attempt")

    raw_file = _make_raw_file(tmp_path)

    with pytest.raises(InputIdentityConflictError):
        register_ligand(
            project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
        )

    assert not (kind_dir / "curcumin.yaml").exists()
    # the orphan itself is left exactly as it was - no automatic recovery
    assert (orphan_dir / "leftover.sdf").read_bytes() == b"leftover-from-a-crashed-attempt"


def test_orphan_raw_directory_casefold_collision_also_blocks(tmp_path):
    """(17) An orphan under one case still conflicts with a request under a
    different case."""

    project_dir = tmp_path / "exp001"
    kind_dir = project_dir / "inputs" / "ligands"
    (kind_dir / "curcumin").mkdir(parents=True)

    raw_file = _make_raw_file(tmp_path)

    with pytest.raises(InputIdentityConflictError):
        register_ligand(
            project_dir, input_id="Curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
        )


def test_staging_namespace_does_not_participate_in_collision_scan(tmp_path):
    """(18) A leftover directory that happens to be named after the
    requested id, but sits inside kind_dir/.staging, must never be treated
    as a conflict - .staging never represents a scientific input."""

    project_dir = tmp_path / "exp001"
    kind_dir = project_dir / "inputs" / "ligands"
    (kind_dir / ".staging" / "curcumin").mkdir(parents=True)

    raw_file = _make_raw_file(tmp_path)

    ligand = register_ligand(
        project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
    )

    assert ligand.id == "curcumin"
    assert (kind_dir / "curcumin.yaml").exists()


def test_copy_failure_leaves_no_final_raw_no_metadata_and_cleans_staging(tmp_path, monkeypatch):
    """(19) A failure while copying into staging must not publish anything,
    and the specific attempt's staging directory must be cleaned up."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    def _raise_copy(*args, **kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(registry_module.shutil, "copyfile", _raise_copy)

    with pytest.raises(OSError):
        register_ligand(
            project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
        )

    kind_dir = project_dir / "inputs" / "ligands"
    assert not (kind_dir / "curcumin.yaml").exists()
    assert not (kind_dir / "curcumin").exists()

    staging_root = kind_dir / ".staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


def test_checksum_mismatch_after_staging_leaves_nothing_published(tmp_path, monkeypatch):
    """(20) If the staged copy's checksum doesn't match the source's,
    nothing is published and the attempt's staging directory is removed."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    real_compute_checksum = registry_module._compute_checksum
    call_count = {"n": 0}

    def _flaky_checksum(path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # the second call is the staged-file check; force a mismatch
            return "0" * 64
        return real_compute_checksum(path)

    monkeypatch.setattr(registry_module, "_compute_checksum", _flaky_checksum)

    with pytest.raises(OSError, match="Checksum mismatch"):
        register_ligand(
            project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
        )

    kind_dir = project_dir / "inputs" / "ligands"
    assert not (kind_dir / "curcumin.yaml").exists()
    assert not (kind_dir / "curcumin").exists()

    staging_root = kind_dir / ".staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


def test_early_validation_failure_creates_no_staging_directory(tmp_path):
    """(21) Acquisition/config validation happens before any staging or
    destination path is touched at all - a rejected 'downloaded' input with
    no source info leaves not even kind_dir behind."""

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

    kind_dir = project_dir / "inputs" / "ligands"
    assert not kind_dir.exists()


def test_yaml_serialization_failure_before_publish_leaves_nothing_published(tmp_path, monkeypatch):
    """(22) A failure serializing the metadata to YAML happens after
    staging/checksum verification but strictly before publish - nothing
    must be published or committed."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    def _raise_dump(*args, **kwargs):
        raise TypeError("simulated serialization failure")

    monkeypatch.setattr(registry_module.yaml, "safe_dump", _raise_dump)

    with pytest.raises(TypeError):
        register_ligand(
            project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
        )

    kind_dir = project_dir / "inputs" / "ligands"
    assert not (kind_dir / "curcumin.yaml").exists()
    assert not (kind_dir / "curcumin").exists()

    staging_root = kind_dir / ".staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


def test_publish_failure_creates_no_metadata_and_no_final_and_cleans_staging(tmp_path, monkeypatch):
    """(23) If publish_staged_directory itself fails, metadata must never
    be created, no final RAW directory exists, and staging cleanup is
    attempted for this attempt's directory."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    def _raise_publish(*args, **kwargs):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(registry_module, "publish_staged_directory", _raise_publish)

    with pytest.raises(OSError):
        register_ligand(
            project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
        )

    kind_dir = project_dir / "inputs" / "ligands"
    assert not (kind_dir / "curcumin.yaml").exists()
    assert not (kind_dir / "curcumin").exists()

    staging_root = kind_dir / ".staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


def test_metadata_commit_failure_after_publish_removes_just_published_raw(tmp_path, monkeypatch):
    """(24) If the metadata atomic write fails after RAW was already
    published, the just-published RAW directory must be rolled back
    (removed), leaving no metadata and no orphan from this attempt."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    def _raise_atomic_write(*args, **kwargs):
        raise OSError("simulated metadata commit failure")

    monkeypatch.setattr(registry_module, "atomic_write_text", _raise_atomic_write)

    with pytest.raises(OSError):
        register_ligand(
            project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
        )

    kind_dir = project_dir / "inputs" / "ligands"
    assert not (kind_dir / "curcumin.yaml").exists()
    assert not (kind_dir / "curcumin").exists()


def test_cleanup_failure_still_propagates_original_error_and_logs_warning(
    tmp_path, monkeypatch, caplog
):
    """(25) A cleanup failure must never mask the original error, and must
    be logged as a warning."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    def _raise_copy(*args, **kwargs):
        raise OSError("simulated copy failure")

    def _raise_rmtree(*args, **kwargs):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(registry_module.shutil, "copyfile", _raise_copy)
    monkeypatch.setattr(registry_module.shutil, "rmtree", _raise_rmtree)

    with caplog.at_level(logging.WARNING, logger="molecular.registry"):
        with pytest.raises(OSError, match="simulated copy failure"):
            register_ligand(
                project_dir,
                input_id="curcumin",
                name="Curcumin",
                method="uploaded",
                raw_file=raw_file,
            )

    assert any("Failed to clean up" in record.getMessage() for record in caplog.records)


def test_successful_registration_metadata_and_raw_checksum_match(tmp_path):
    """(26) On success, metadata exists, RAW exists, and the checksum in
    metadata matches the checksum of the actual final RAW bytes."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    register_ligand(
        project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
    )

    kind_dir = project_dir / "inputs" / "ligands"
    assert (kind_dir / "curcumin.yaml").exists()

    reloaded = load_ligand(project_dir, "curcumin")
    stored_path = project_dir / reloaded.acquisition.raw_artifact.path
    assert stored_path.exists()

    actual_checksum = hashlib.sha256(stored_path.read_bytes()).hexdigest()
    assert reloaded.acquisition.raw_artifact.checksum == actual_checksum


def test_successful_registration_removes_its_own_staging_attempt_directory(tmp_path):
    """(27) The specific per-attempt staging directory must not survive a
    successful registration - publish_staged_directory renames it away
    rather than copying-and-leaving-behind. The .staging container itself
    may remain (empty), per the accepted staging layout."""

    project_dir = tmp_path / "exp001"
    project_dir.mkdir()
    raw_file = _make_raw_file(tmp_path)

    register_ligand(
        project_dir, input_id="curcumin", name="Curcumin", method="uploaded", raw_file=raw_file
    )

    staging_root = project_dir / "inputs" / "ligands" / ".staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


def test_ligand_registration_success_regression(tmp_path):
    """(28) Ligand registration end-to-end still works under the new
    stage/publish/commit protocol."""

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

    assert ligand.id == "curcumin"
    assert load_ligand(project_dir, "curcumin") == ligand


def test_protein_registration_success_regression(tmp_path):
    """(29) Protein registration end-to-end still works under the new
    stage/publish/commit protocol."""

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
    assert load_protein(project_dir, "urease") == protein


# (30) acquired_at computed exactly once is already covered by
# test_acquired_at_is_computed_once_and_persisted_unchanged above, which
# still passes unchanged under the new stage/publish/commit protocol.


# --- (33) Concurrency: documented residual risk, not tested here ---
#
# A genuinely deterministic, cross-platform test for two concurrent
# registrations racing on the same input_id would require either real
# multiprocessing/multithreading synchronized on filesystem timing (flaky
# by nature - exact interleaving of the check-then-act window is not
# controllable from a test) or replacing the filesystem with a fake that
# can inject a mid-operation pause (a much larger change than this
# increment's scope). Rather than ship a flaky or falsely-reassuring test,
# the race is left as a documented residual risk: the identity-conflict
# check in _check_storage_identity_conflict, and the final_dir.exists()
# check in publish_staged_directory, both have a TOCTOU window between the
# check and the corresponding filesystem mutation. No locking or retry is
# implemented (out of scope for this increment - see module docstring).
