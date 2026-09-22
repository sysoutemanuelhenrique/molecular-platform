"""Model-level tests for the scientific domain foundation: ScientificEntity,
MolecularRepresentationRef, MolecularState, and Conformer. No filesystem
I/O here - see test_scientific_registry.py for persistence behavior."""

import pytest
from pydantic import ValidationError

from molecular.scientific import (
    Conformer,
    MolecularRepresentationRef,
    MolecularState,
    ScientificEntity,
)


def _representation(**overrides):
    defaults = dict(
        path="scientific/artifacts/states/curcumin-initial/structure.sdf",
        format="sdf",
        checksum="a" * 64,
    )
    defaults.update(overrides)
    return MolecularRepresentationRef(**defaults)


# --- ScientificEntity ---


def test_scientific_entity_accepts_valid_data():
    entity = ScientificEntity(schema_version="1.0", id="curcumin", name="Curcumin")

    assert entity.id == "curcumin"
    assert entity.name == "Curcumin"


def test_scientific_entity_requires_schema_version():
    with pytest.raises(ValidationError):
        ScientificEntity(id="curcumin", name="Curcumin")


def test_scientific_entity_rejects_unknown_schema_version():
    with pytest.raises(ValidationError):
        ScientificEntity(schema_version="9.9", id="curcumin", name="Curcumin")


def test_scientific_entity_rejects_empty_id():
    with pytest.raises(ValidationError):
        ScientificEntity(schema_version="1.0", id="", name="Curcumin")


def test_scientific_entity_rejects_id_with_path_separator():
    with pytest.raises(ValidationError):
        ScientificEntity(schema_version="1.0", id="a/b", name="Curcumin")


def test_scientific_entity_rejects_id_with_leading_dot():
    with pytest.raises(ValidationError):
        ScientificEntity(schema_version="1.0", id=".hidden", name="Curcumin")


def test_scientific_entity_rejects_empty_name():
    with pytest.raises(ValidationError):
        ScientificEntity(schema_version="1.0", id="curcumin", name="")


def test_scientific_entity_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ScientificEntity(
            schema_version="1.0", id="curcumin", name="Curcumin", smiles="CC(=O)Oc1ccccc1C(=O)O"
        )


def test_scientific_entity_is_immutable():
    entity = ScientificEntity(schema_version="1.0", id="curcumin", name="Curcumin")

    with pytest.raises(ValidationError):
        entity.name = "Something else"


# --- MolecularRepresentationRef ---


def test_molecular_representation_ref_accepts_valid_relative_path():
    representation = _representation()

    assert representation.path == "scientific/artifacts/states/curcumin-initial/structure.sdf"
    assert representation.format == "sdf"


def test_molecular_representation_ref_rejects_absolute_posix_path():
    with pytest.raises(ValidationError):
        _representation(path="/etc/passwd")


def test_molecular_representation_ref_rejects_absolute_windows_path():
    with pytest.raises(ValidationError):
        _representation(path="C:/Users/someone/structure.sdf")


def test_molecular_representation_ref_rejects_unc_path():
    with pytest.raises(ValidationError):
        _representation(path=r"\\server\share\structure.sdf")


@pytest.mark.parametrize(
    "path",
    [
        "../foo",
        "foo/../../bar",
        "foo\\..\\bar",
        "..",
        "foo/..",
        ".",
    ],
)
def test_molecular_representation_ref_rejects_path_traversal(path):
    with pytest.raises(ValidationError):
        _representation(path=path)


def test_molecular_representation_ref_rejects_empty_format():
    with pytest.raises(ValidationError):
        _representation(format="")


def test_molecular_representation_ref_rejects_empty_checksum():
    with pytest.raises(ValidationError):
        _representation(checksum="")


def test_molecular_representation_ref_is_immutable():
    representation = _representation()

    with pytest.raises(ValidationError):
        representation.format = "mol2"


# --- MolecularState ---


def test_molecular_state_accepts_valid_data():
    state = MolecularState(
        schema_version="1.0",
        id="curcumin-initial",
        entity_id="curcumin",
        representation=_representation(),
    )

    assert state.id == "curcumin-initial"
    assert state.entity_id == "curcumin"
    assert state.representation.format == "sdf"


def test_molecular_state_requires_entity_id():
    with pytest.raises(ValidationError):
        MolecularState(
            schema_version="1.0",
            id="curcumin-initial",
            representation=_representation(),
        )


def test_molecular_state_requires_representation():
    with pytest.raises(ValidationError):
        MolecularState(
            schema_version="1.0",
            id="curcumin-initial",
            entity_id="curcumin",
        )


def test_molecular_state_rejects_extra_fields():
    with pytest.raises(ValidationError):
        MolecularState(
            schema_version="1.0",
            id="curcumin-initial",
            entity_id="curcumin",
            representation=_representation(),
            protonation="neutral",
        )


def test_molecular_state_is_immutable():
    state = MolecularState(
        schema_version="1.0",
        id="curcumin-initial",
        entity_id="curcumin",
        representation=_representation(),
    )

    with pytest.raises(ValidationError):
        state.entity_id = "ibuprofen"


# --- Conformer ---


def test_conformer_accepts_valid_data():
    conformer = Conformer(
        schema_version="1.0",
        id="curcumin-initial-conf-001",
        state_id="curcumin-initial",
        representation=_representation(),
    )

    assert conformer.id == "curcumin-initial-conf-001"
    assert conformer.state_id == "curcumin-initial"
    assert conformer.representation.format == "sdf"


def test_conformer_requires_schema_version():
    with pytest.raises(ValidationError):
        Conformer(
            id="curcumin-initial-conf-001",
            state_id="curcumin-initial",
            representation=_representation(),
        )


def test_conformer_rejects_unknown_schema_version():
    with pytest.raises(ValidationError):
        Conformer(
            schema_version="9.9",
            id="curcumin-initial-conf-001",
            state_id="curcumin-initial",
            representation=_representation(),
        )


def test_conformer_rejects_empty_id():
    with pytest.raises(ValidationError):
        Conformer(
            schema_version="1.0",
            id="",
            state_id="curcumin-initial",
            representation=_representation(),
        )


def test_conformer_rejects_id_with_path_separator():
    with pytest.raises(ValidationError):
        Conformer(
            schema_version="1.0",
            id="a/b",
            state_id="curcumin-initial",
            representation=_representation(),
        )


def test_conformer_rejects_empty_state_id():
    with pytest.raises(ValidationError):
        Conformer(
            schema_version="1.0",
            id="curcumin-initial-conf-001",
            state_id="",
            representation=_representation(),
        )


def test_conformer_rejects_state_id_with_leading_dot():
    with pytest.raises(ValidationError):
        Conformer(
            schema_version="1.0",
            id="curcumin-initial-conf-001",
            state_id=".hidden",
            representation=_representation(),
        )


def test_conformer_requires_representation():
    with pytest.raises(ValidationError):
        Conformer(
            schema_version="1.0",
            id="curcumin-initial-conf-001",
            state_id="curcumin-initial",
        )


def test_conformer_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Conformer(
            schema_version="1.0",
            id="curcumin-initial-conf-001",
            state_id="curcumin-initial",
            representation=_representation(),
            energy=-42.0,
        )


def test_conformer_is_immutable():
    conformer = Conformer(
        schema_version="1.0",
        id="curcumin-initial-conf-001",
        state_id="curcumin-initial",
        representation=_representation(),
    )

    with pytest.raises(ValidationError):
        conformer.state_id = "ibuprofen-initial"


def test_conformer_reuses_molecular_representation_ref_validation():
    """Smoke test only - MolecularRepresentationRef's own validation rules
    (absolute path, traversal components, empty format/checksum) are
    already exhaustively covered by its own tests above; this just proves
    Conformer actually wires the same class in, unmodified."""

    with pytest.raises(ValidationError):
        Conformer(
            schema_version="1.0",
            id="curcumin-initial-conf-001",
            state_id="curcumin-initial",
            representation=_representation(path="/etc/passwd"),
        )

    with pytest.raises(ValidationError):
        Conformer(
            schema_version="1.0",
            id="curcumin-initial-conf-001",
            state_id="curcumin-initial",
            representation=_representation(path="../escape.sdf"),
        )


def test_conformer_accepts_any_non_empty_format():
    """Normative decision: no closed/allowed list of '3D-capable' formats
    in this increment - format is an unvalidated hint, exactly like
    MolecularState's representation. A format such as 'smiles' - which
    could never actually carry 3D coordinates - is not rejected here."""

    conformer = Conformer(
        schema_version="1.0",
        id="curcumin-initial-conf-001",
        state_id="curcumin-initial",
        representation=_representation(format="smiles"),
    )

    assert conformer.representation.format == "smiles"
