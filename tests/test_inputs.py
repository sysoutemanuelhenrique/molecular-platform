from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from molecular.inputs import AcquisitionRecord, LigandInput, ProteinInput, RawArtifact


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
