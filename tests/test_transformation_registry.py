"""Persistence tests for state transformation provenance registration under
project_dir/scientific/transformations. Business rules already covered by
test_transformation.py (field-level model validation) are not repeated
here - these tests exercise storage behavior: parent/child resolution,
same-entity enforcement, conflicts, atomicity, cycles, and multiple
parents/children."""

from datetime import datetime, timezone

import pytest
import yaml

import molecular.transformation_registry as transformation_registry_module
from molecular.scientific import ScientificEntity
from molecular.scientific_registry import register_entity, register_state
from molecular.transformation import (
    StateTransformationRecord,
    TransformationParameter,
    TransformationProducer,
)
from molecular.transformation_registry import (
    TransformationIdentityConflictError,
    list_state_transformations,
    load_state_transformation,
    register_state_transformation,
)


def _entity(entity_id="curcumin", name="Curcumin"):
    return ScientificEntity(schema_version="1.0", id=entity_id, name=name)


def _write_source_file(tmp_path, name="structure.sdf", content=b"fake-sdf-bytes"):
    source_file = tmp_path / name
    source_file.write_bytes(content)
    return source_file


def _setup_entity_with_states(tmp_path, entity_id="curcumin", state_ids=("state-a", "state-b")):
    """Register one entity and one MolecularState per id in state_ids,
    all belonging to that entity."""

    register_entity(tmp_path, _entity(entity_id=entity_id))

    for state_id in state_ids:
        register_state(
            tmp_path,
            state_id=state_id,
            entity_id=entity_id,
            source_file=_write_source_file(tmp_path, name=f"{state_id}.sdf"),
            format="sdf",
        )


def _record(**overrides):
    defaults = dict(
        schema_version="1.0",
        id="state-a-to-state-b",
        parent_state_id="state-a",
        child_state_id="state-b",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool", version="1.0"),
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return StateTransformationRecord(**defaults)


# --- register / load ---


def test_register_state_transformation_persists_metadata(tmp_path):
    _setup_entity_with_states(tmp_path)

    registered = register_state_transformation(tmp_path, _record())

    metadata_file = tmp_path / "scientific" / "transformations" / "state-a-to-state-b.yaml"
    assert metadata_file.exists()
    assert registered.id == "state-a-to-state-b"


def test_transformation_serialization_matches_expected_shape(tmp_path):
    _setup_entity_with_states(tmp_path)
    register_state_transformation(tmp_path, _record())

    metadata_file = tmp_path / "scientific" / "transformations" / "state-a-to-state-b.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.0"
    assert data["id"] == "state-a-to-state-b"
    assert data["parent_state_id"] == "state-a"
    assert data["child_state_id"] == "state-b"
    assert data["operation"] == "tautomerization"
    assert data["producer"] == {"name": "ExternalTool", "version": "1.0"}
    assert data["method_description"] is None
    assert data["parameters"] == []


def test_load_state_transformation_round_trips(tmp_path):
    _setup_entity_with_states(tmp_path)
    register_state_transformation(tmp_path, _record())

    loaded = load_state_transformation(tmp_path, "state-a-to-state-b")

    assert loaded.parent_state_id == "state-a"
    assert loaded.child_state_id == "state-b"
    assert loaded.operation == "tautomerization"


def test_load_state_transformation_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_state_transformation(tmp_path, "nao_existe")


# --- conflicts ---


def test_register_state_transformation_rejects_duplicate_id(tmp_path):
    _setup_entity_with_states(tmp_path)
    register_state_transformation(tmp_path, _record())

    with pytest.raises(FileExistsError):
        register_state_transformation(tmp_path, _record())


def test_register_state_transformation_rejects_casefold_collision(tmp_path):
    _setup_entity_with_states(tmp_path)
    register_state_transformation(tmp_path, _record(id="state-a-to-state-b"))

    with pytest.raises(TransformationIdentityConflictError):
        register_state_transformation(tmp_path, _record(id="State-A-To-State-B"))


def test_register_state_transformation_never_overwrites_existing(tmp_path):
    _setup_entity_with_states(tmp_path)
    register_state_transformation(
        tmp_path, _record(method_description="original", producer=None)
    )

    with pytest.raises(FileExistsError):
        register_state_transformation(
            tmp_path, _record(method_description="overwritten attempt", producer=None)
        )

    loaded = load_state_transformation(tmp_path, "state-a-to-state-b")
    assert loaded.method_description == "original"


# --- parent/child resolution ---


def test_register_state_transformation_rejects_missing_parent(tmp_path):
    register_entity(tmp_path, _entity())
    register_state(
        tmp_path,
        state_id="state-b",
        entity_id="curcumin",
        source_file=_write_source_file(tmp_path),
        format="sdf",
    )

    with pytest.raises(FileNotFoundError):
        register_state_transformation(tmp_path, _record())


def test_register_state_transformation_rejects_missing_child(tmp_path):
    register_entity(tmp_path, _entity())
    register_state(
        tmp_path,
        state_id="state-a",
        entity_id="curcumin",
        source_file=_write_source_file(tmp_path),
        format="sdf",
    )

    with pytest.raises(FileNotFoundError):
        register_state_transformation(tmp_path, _record())


def test_register_state_transformation_rejects_cross_entity_parent_child(tmp_path):
    register_entity(tmp_path, _entity(entity_id="curcumin", name="Curcumin"))
    register_entity(tmp_path, _entity(entity_id="ibuprofen", name="Ibuprofen"))

    register_state(
        tmp_path,
        state_id="state-a",
        entity_id="curcumin",
        source_file=_write_source_file(tmp_path, name="a.sdf"),
        format="sdf",
    )
    register_state(
        tmp_path,
        state_id="state-b",
        entity_id="ibuprofen",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
    )

    with pytest.raises(ValueError):
        register_state_transformation(tmp_path, _record())


# --- fan-in / fan-out / cycles ---


def test_multiple_children_from_same_parent(tmp_path):
    _setup_entity_with_states(tmp_path, state_ids=("state-a", "state-b", "state-c"))

    register_state_transformation(
        tmp_path,
        _record(id="a-to-b", parent_state_id="state-a", child_state_id="state-b"),
    )
    register_state_transformation(
        tmp_path,
        _record(id="a-to-c", parent_state_id="state-a", child_state_id="state-c"),
    )

    children = list_state_transformations(tmp_path, parent_state_id="state-a")
    assert {t.child_state_id for t in children} == {"state-b", "state-c"}


def test_multiple_parents_to_same_child(tmp_path):
    _setup_entity_with_states(tmp_path, state_ids=("state-a", "state-b", "state-c"))

    register_state_transformation(
        tmp_path,
        _record(id="a-to-c", parent_state_id="state-a", child_state_id="state-c"),
    )
    register_state_transformation(
        tmp_path,
        _record(id="b-to-c", parent_state_id="state-b", child_state_id="state-c"),
    )

    parents = list_state_transformations(tmp_path, child_state_id="state-c")
    assert {t.parent_state_id for t in parents} == {"state-a", "state-b"}


def test_cycles_are_accepted(tmp_path):
    _setup_entity_with_states(tmp_path, state_ids=("state-a", "state-b"))

    register_state_transformation(
        tmp_path,
        _record(id="a-to-b", parent_state_id="state-a", child_state_id="state-b"),
    )
    registered_back = register_state_transformation(
        tmp_path,
        _record(id="b-to-a", parent_state_id="state-b", child_state_id="state-a"),
    )

    assert registered_back.parent_state_id == "state-b"
    assert registered_back.child_state_id == "state-a"


# --- atomicity / non-mutation ---


def test_atomic_metadata_write_failure_creates_no_transformation(tmp_path, monkeypatch):
    _setup_entity_with_states(tmp_path)

    def _raise_atomic_write(*args, **kwargs):
        raise OSError("simulated metadata commit failure")

    monkeypatch.setattr(transformation_registry_module, "atomic_write_text", _raise_atomic_write)

    with pytest.raises(OSError):
        register_state_transformation(tmp_path, _record())

    metadata_file = tmp_path / "scientific" / "transformations" / "state-a-to-state-b.yaml"
    assert not metadata_file.exists()


def test_parent_state_unchanged_after_transformation_registration(tmp_path):
    _setup_entity_with_states(tmp_path)

    from molecular.scientific_registry import load_state

    before = load_state(tmp_path, "state-a")
    register_state_transformation(tmp_path, _record())
    after = load_state(tmp_path, "state-a")

    assert before == after


def test_child_state_unchanged_after_transformation_registration(tmp_path):
    _setup_entity_with_states(tmp_path)

    from molecular.scientific_registry import load_state

    before = load_state(tmp_path, "state-b")
    register_state_transformation(tmp_path, _record())
    after = load_state(tmp_path, "state-b")

    assert before == after


# --- F-012-C-06: bool/int survive the real registry round-trip ---


def test_bool_and_int_parameter_values_survive_registry_round_trip(tmp_path):
    """Exercises the real path - Pydantic -> model_dump(mode="json") ->
    yaml.safe_dump -> yaml.safe_load -> model_validate - not just direct
    model construction, to prove bool is never silently coerced to int (or
    vice versa) once the value has actually been through YAML."""

    _setup_entity_with_states(tmp_path)

    register_state_transformation(
        tmp_path,
        _record(
            parameters=(
                TransformationParameter(name="enabled", value=True),
                TransformationParameter(name="disabled", value=False),
                TransformationParameter(name="count", value=1),
                TransformationParameter(name="zero_count", value=0),
            )
        ),
    )

    loaded = load_state_transformation(tmp_path, "state-a-to-state-b")
    by_name = {parameter.name: parameter for parameter in loaded.parameters}

    assert by_name["enabled"].value is True
    assert type(by_name["enabled"].value) is bool

    assert by_name["disabled"].value is False
    assert type(by_name["disabled"].value) is bool

    assert by_name["count"].value == 1
    assert type(by_name["count"].value) is int

    assert by_name["zero_count"].value == 0
    assert type(by_name["zero_count"].value) is int
