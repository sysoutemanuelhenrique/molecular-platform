from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from molecular.inputs import (
    AcquisitionRecord,
    LigandInput,
    ProteinInput,
    RawArtifact,
    validate_input_id,
)


def _raw_artifact(**overrides):
    defaults = dict(path="inputs/ligands/curcumin/curcumin.sdf", format="sdf", checksum="a" * 64)
    defaults.update(overrides)
    return RawArtifact(**defaults)


def test_raw_artifact_accepts_valid_data():
    artifact = _raw_artifact()

    assert artifact.format == "sdf"
    assert artifact.checksum == "a" * 64


def test_acquisition_record_accepts_uploaded_without_source_info():
    record = AcquisitionRecord(
        method="uploaded",
        acquired_at=datetime.now(timezone.utc),
        raw_artifact=_raw_artifact(),
    )

    assert record.method == "uploaded"
    assert record.source_name is None


def test_acquisition_record_accepts_downloaded_with_source_info():
    record = AcquisitionRecord(
        method="downloaded",
        source_name="PubChem",
        source_identifier="CID 969516",
        acquired_at=datetime.now(timezone.utc),
        raw_artifact=_raw_artifact(),
    )

    assert record.source_name == "PubChem"


def test_acquisition_record_rejects_downloaded_without_source_name():
    with pytest.raises(ValidationError):
        AcquisitionRecord(
            method="downloaded",
            source_identifier="CID 969516",
            acquired_at=datetime.now(timezone.utc),
            raw_artifact=_raw_artifact(),
        )


def test_acquisition_record_rejects_downloaded_without_source_identifier_or_uri():
    with pytest.raises(ValidationError):
        AcquisitionRecord(
            method="downloaded",
            source_name="PubChem",
            acquired_at=datetime.now(timezone.utc),
            raw_artifact=_raw_artifact(),
        )


def test_acquisition_record_rejects_invalid_method():
    with pytest.raises(ValidationError):
        AcquisitionRecord(
            method="synthesized",
            acquired_at=datetime.now(timezone.utc),
            raw_artifact=_raw_artifact(),
        )


def test_ligand_input_accepts_valid_data():
    ligand = LigandInput(
        id="curcumin",
        name="Curcumin",
        external_id="CID 969516",
        acquisition=AcquisitionRecord(
            method="downloaded",
            source_name="PubChem",
            source_identifier="CID 969516",
            acquired_at=datetime.now(timezone.utc),
            raw_artifact=_raw_artifact(),
        ),
    )

    assert ligand.id == "curcumin"
    assert ligand.acquisition.method == "downloaded"


def test_protein_input_accepts_valid_data():
    protein = ProteinInput(
        id="urease",
        name="Jack bean urease",
        external_id="3LA4",
        acquisition=AcquisitionRecord(
            method="downloaded",
            source_name="RCSB PDB",
            source_identifier="3LA4",
            acquired_at=datetime.now(timezone.utc),
            raw_artifact=_raw_artifact(
                path="inputs/proteins/urease/3la4.cif", format="cif"
            ),
        ),
    )

    assert protein.external_id == "3LA4"


def test_scientific_input_rejects_missing_name():
    with pytest.raises(ValidationError):
        LigandInput(
            id="curcumin",
            acquisition=AcquisitionRecord(
                method="uploaded",
                acquired_at=datetime.now(timezone.utc),
                raw_artifact=_raw_artifact(),
            ),
        )


# --- validate_input_id: the single canonical authority for input_id format ---


def test_validate_input_id_accepts_valid_value():
    assert validate_input_id("my_ligand-2") == "my_ligand-2"


def test_validate_input_id_rejects_empty_string():
    with pytest.raises(ValueError):
        validate_input_id("")


def test_validate_input_id_rejects_parent_directory_traversal():
    with pytest.raises(ValueError):
        validate_input_id("../etc")


def test_validate_input_id_rejects_forward_slash():
    with pytest.raises(ValueError):
        validate_input_id("a/b")


def test_validate_input_id_rejects_backslash():
    with pytest.raises(ValueError):
        validate_input_id("a\\b")


def test_validate_input_id_rejects_leading_dot():
    with pytest.raises(ValueError):
        validate_input_id(".hidden")


def test_validate_input_id_rejects_whitespace():
    with pytest.raises(ValueError):
        validate_input_id("curcumin 2")


def test_validate_input_id_rejects_trailing_newline():
    """`$` can match immediately before a trailing newline, so the pattern
    must be applied with fullmatch, not match, to reject the whole string."""

    with pytest.raises(ValueError):
        validate_input_id("curcumin\n")


def test_validate_input_id_rejects_bare_dot_and_dotdot():
    with pytest.raises(ValueError):
        validate_input_id(".")

    with pytest.raises(ValueError):
        validate_input_id("..")


def test_validate_input_id_preserves_exact_case_of_valid_values():
    """validate_input_id never sanitizes or normalizes - a value that
    passes comes back byte-for-byte identical."""

    for value in ("curcumin", "CURCUMIN_48", "ligand-2", "a1"):
        assert validate_input_id(value) == value


def test_scientific_input_id_field_uses_validate_input_id():
    """ScientificInput.id must reject exactly what validate_input_id
    rejects - proving both call sites share the one canonical rule."""

    with pytest.raises(ValidationError):
        LigandInput(
            id="a/b",
            name="Curcumin",
            acquisition=AcquisitionRecord(
                method="uploaded",
                acquired_at=datetime.now(timezone.utc),
                raw_artifact=_raw_artifact(),
            ),
        )
