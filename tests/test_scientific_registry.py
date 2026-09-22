"""Persistence tests for the scientific model registry: entity and
molecular-state registration under project_dir/scientific. Business rules
already covered by test_scientific.py (field-level model validation) are
not repeated here - these tests exercise storage behavior: conflicts,
atomicity, checksum verification, and the entity/state relationship."""

import logging

import pytest
import yaml
from pydantic import ValidationError

import molecular.scientific_registry as scientific_registry_module
from molecular.scientific import MolecularState, ScientificEntity
from molecular.scientific_registry import (
    EntityIdentityConflictError,
    StateIdentityConflictError,
    list_entity_states,
    load_entity,
    load_state,
    register_entity,
    register_state,
)


def _entity(entity_id="curcumin", name="Curcumin"):
    return ScientificEntity(schema_version="1.0", id=entity_id, name=name)


def _write_source_file(tmp_path, name="structure.sdf", content=b"fake-sdf-bytes"):
    source_file = tmp_path / name
    source_file.write_bytes(content)
    return source_file


# --- register_entity / load_entity ---


def test_register_entity_persists_metadata(tmp_path):
    registered = register_entity(tmp_path, _entity())

    metadata_file = tmp_path / "scientific" / "entities" / "curcumin.yaml"
    assert metadata_file.exists()
    assert registered.id == "curcumin"


def test_load_entity_round_trips(tmp_path):
    register_entity(tmp_path, _entity())

    loaded = load_entity(tmp_path, "curcumin")

    assert loaded == _entity()


def test_entity_serialization_matches_expected_shape(tmp_path):
    register_entity(tmp_path, _entity())

    metadata_file = tmp_path / "scientific" / "entities" / "curcumin.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))

    assert data == {"schema_version": "1.0", "id": "curcumin", "name": "Curcumin"}


def test_load_entity_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_entity(tmp_path, "nao_existe")


def test_register_entity_rejects_exact_duplicate(tmp_path):
    register_entity(tmp_path, _entity())

    with pytest.raises(FileExistsError):
        register_entity(tmp_path, _entity())


def test_register_entity_rejects_casefold_collision(tmp_path):
    register_entity(tmp_path, _entity(entity_id="curcumin"))

    with pytest.raises(EntityIdentityConflictError):
        register_entity(tmp_path, _entity(entity_id="Curcumin"))


def test_register_entity_duplicate_leaves_original_untouched(tmp_path):
    register_entity(tmp_path, _entity(name="Curcumin"))

    with pytest.raises(FileExistsError):
        register_entity(tmp_path, _entity(name="Overwritten"))

    assert load_entity(tmp_path, "curcumin").name == "Curcumin"


# --- register_state / load_state ---


def test_register_state_requires_existing_entity(tmp_path):
    source_file = _write_source_file(tmp_path)

    with pytest.raises(FileNotFoundError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )


def test_register_state_persists_metadata_and_artifact(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    state = register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    assert state.id == "curcumin-initial"
    assert state.entity_id == "curcumin"

    metadata_file = tmp_path / "scientific" / "states" / "curcumin-initial.yaml"
    assert metadata_file.exists()

    artifact_file = tmp_path / "scientific" / "artifacts" / "states" / "curcumin-initial" / "structure.sdf"
    assert artifact_file.exists()
    assert artifact_file.read_bytes() == b"fake-sdf-bytes"


def test_register_state_computes_correct_checksum(tmp_path):
    import hashlib

    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path, content=b"deterministic-content")

    state = register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    expected = hashlib.sha256(b"deterministic-content").hexdigest()
    assert state.representation.checksum == expected


def test_register_state_relative_representation_path_uses_posix_style(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    state = register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    assert state.representation.path == (
        "scientific/artifacts/states/curcumin-initial/structure.sdf"
    )
    assert "\\" not in state.representation.path


def test_load_state_round_trips(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    loaded = load_state(tmp_path, "curcumin-initial")

    assert loaded.id == "curcumin-initial"
    assert loaded.entity_id == "curcumin"
    assert loaded.representation.format == "sdf"


def test_state_serialization_matches_expected_shape(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    metadata_file = tmp_path / "scientific" / "states" / "curcumin-initial.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.0"
    assert data["id"] == "curcumin-initial"
    assert data["entity_id"] == "curcumin"
    assert data["representation"]["format"] == "sdf"
    assert data["representation"]["path"] == (
        "scientific/artifacts/states/curcumin-initial/structure.sdf"
    )


def test_load_state_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_state(tmp_path, "nao_existe")


def test_register_state_rejects_duplicate_state_id(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    with pytest.raises(FileExistsError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )


def test_register_state_rejects_casefold_collision(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    with pytest.raises(StateIdentityConflictError):
        register_state(
            tmp_path,
            state_id="Curcumin-Initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )


def test_multiple_states_for_same_entity(tmp_path):
    register_entity(tmp_path, _entity())
    source_a = _write_source_file(tmp_path, name="a.sdf", content=b"state-a")
    source_b = _write_source_file(tmp_path, name="b.sdf", content=b"state-b")

    register_state(
        tmp_path,
        state_id="curcumin-a",
        entity_id="curcumin",
        source_file=source_a,
        format="sdf",
    )
    register_state(
        tmp_path,
        state_id="curcumin-b",
        entity_id="curcumin",
        source_file=source_b,
        format="sdf",
    )

    states = list_entity_states(tmp_path, "curcumin")

    assert {state.id for state in states} == {"curcumin-a", "curcumin-b"}


def test_entity_unchanged_after_state_registration(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    assert load_entity(tmp_path, "curcumin") == _entity()


def test_register_state_rejects_missing_source_file(tmp_path):
    register_entity(tmp_path, _entity())

    with pytest.raises(FileNotFoundError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=tmp_path / "does-not-exist.sdf",
            format="sdf",
        )


def test_register_state_rejects_empty_format(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    with pytest.raises(ValueError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="",
        )


def test_register_state_rejects_invalid_state_id(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    with pytest.raises(ValueError):
        register_state(
            tmp_path,
            state_id="a/b",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )


def test_failed_state_registration_leaves_no_metadata(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    with pytest.raises(FileExistsError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )

    # the original registration must remain the only metadata on disk
    states_dir = tmp_path / "scientific" / "states"
    assert [entry.name for entry in states_dir.iterdir()] == ["curcumin-initial.yaml"]


def test_failed_state_registration_does_not_publish_artifact_twice(tmp_path):
    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    register_state(
        tmp_path,
        state_id="curcumin-initial",
        entity_id="curcumin",
        source_file=source_file,
        format="sdf",
    )

    with pytest.raises(FileExistsError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )

    # the failed attempt's staging directory must have been cleaned up
    staging_root = tmp_path / "scientific" / "artifacts" / "states" / ".staging"
    assert not staging_root.exists() or list(staging_root.iterdir()) == []


# --- fault injection: stage -> verify -> publish -> commit, adapted from
# the equivalent protocol tests in tests/test_registry.py ---


def test_register_state_checksum_mismatch_after_staging_leaves_nothing_published(
    tmp_path, monkeypatch
):
    """If the staged copy's checksum doesn't match the source's, nothing is
    published, no metadata is committed, and the attempt's staging
    directory is removed."""

    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    real_compute_sha256 = scientific_registry_module.compute_sha256
    call_count = {"n": 0}

    def _flaky_checksum(path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # the second call is the staged-file check; force a mismatch
            return "0" * 64
        return real_compute_sha256(path)

    monkeypatch.setattr(scientific_registry_module, "compute_sha256", _flaky_checksum)

    with pytest.raises(OSError, match="Checksum mismatch"):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )

    states_dir = tmp_path / "scientific" / "states"
    assert not (states_dir / "curcumin-initial.yaml").exists()

    final_dir = tmp_path / "scientific" / "artifacts" / "states" / "curcumin-initial"
    assert not final_dir.exists()

    staging_root = tmp_path / "scientific" / "artifacts" / "states" / ".staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


def test_register_state_copy_failure_leaves_no_final_artifact_no_metadata_and_cleans_staging(
    tmp_path, monkeypatch
):
    """A failure while copying into staging must not publish anything, and
    the specific attempt's staging directory must be cleaned up."""

    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    def _raise_copy(*args, **kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(scientific_registry_module.shutil, "copyfile", _raise_copy)

    with pytest.raises(OSError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )

    states_dir = tmp_path / "scientific" / "states"
    assert not (states_dir / "curcumin-initial.yaml").exists()

    final_dir = tmp_path / "scientific" / "artifacts" / "states" / "curcumin-initial"
    assert not final_dir.exists()

    staging_root = tmp_path / "scientific" / "artifacts" / "states" / ".staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


def test_register_state_publish_failure_creates_no_metadata_and_no_final_and_cleans_staging(
    tmp_path, monkeypatch
):
    """If publish_staged_directory itself fails, metadata must never be
    created, no final artifact directory exists, and staging cleanup is
    attempted for this attempt's directory."""

    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    def _raise_publish(*args, **kwargs):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(scientific_registry_module, "publish_staged_directory", _raise_publish)

    with pytest.raises(OSError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )

    states_dir = tmp_path / "scientific" / "states"
    assert not (states_dir / "curcumin-initial.yaml").exists()

    final_dir = tmp_path / "scientific" / "artifacts" / "states" / "curcumin-initial"
    assert not final_dir.exists()

    staging_root = tmp_path / "scientific" / "artifacts" / "states" / ".staging"
    if staging_root.exists():
        assert list(staging_root.iterdir()) == []


def test_register_state_metadata_commit_failure_after_publish_removes_just_published_artifact(
    tmp_path, monkeypatch
):
    """If the metadata atomic write fails after the representation was
    already published, the just-published artifact directory must be
    rolled back (removed), leaving no metadata and no valid registration
    from this attempt."""

    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    def _raise_atomic_write(*args, **kwargs):
        raise OSError("simulated metadata commit failure")

    monkeypatch.setattr(scientific_registry_module, "atomic_write_text", _raise_atomic_write)

    with pytest.raises(OSError):
        register_state(
            tmp_path,
            state_id="curcumin-initial",
            entity_id="curcumin",
            source_file=source_file,
            format="sdf",
        )

    states_dir = tmp_path / "scientific" / "states"
    assert not (states_dir / "curcumin-initial.yaml").exists()

    final_dir = tmp_path / "scientific" / "artifacts" / "states" / "curcumin-initial"
    assert not final_dir.exists()


def test_register_state_cleanup_failure_still_propagates_original_error_and_logs_warning(
    tmp_path, monkeypatch, caplog
):
    """A cleanup failure must never mask the original error, and must be
    logged as a warning."""

    register_entity(tmp_path, _entity())
    source_file = _write_source_file(tmp_path)

    def _raise_copy(*args, **kwargs):
        raise OSError("simulated copy failure")

    def _raise_rmtree(*args, **kwargs):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(scientific_registry_module.shutil, "copyfile", _raise_copy)
    monkeypatch.setattr(scientific_registry_module.shutil, "rmtree", _raise_rmtree)

    with caplog.at_level(logging.WARNING, logger="molecular.scientific_registry"):
        with pytest.raises(OSError, match="simulated copy failure"):
            register_state(
                tmp_path,
                state_id="curcumin-initial",
                entity_id="curcumin",
                source_file=source_file,
                format="sdf",
            )

    assert any("Failed to clean up" in record.getMessage() for record in caplog.records)
