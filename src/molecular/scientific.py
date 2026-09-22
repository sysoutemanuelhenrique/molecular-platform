"""Scientific model foundation: entity and molecular-state identity.

This module belongs to the "Represent" stage of the project's conceptual
pipeline (Represent -> Prepare -> Translate -> Run -> Trace): it establishes
the object boundary of the scientific domain, not the chemistry itself.

ScientificEntity is the stable scientific identity of the molecular object
being followed within an experiment, independently from particular
molecular states, conformations, file formats, or external database
identifiers. It is deliberately NOT a file, a SMILES/InChI string, a
PubChem CID, an RDKit Mol, or a docking ligand - those are external
representations or acquisition provenance (see molecular.inputs), never
this module's concern.

MolecularState is an explicitly identified chemical realization of a
ScientificEntity, prior to selecting a particular conformational geometry
(e.g. a named protonation or tautomer state - which states exist is not
enumerated here, only the shape of "a state"). It is deliberately NOT a
conformer, a PDBQT file, or a docking pose - those belong to later stages
(Conformer, Pose) that this increment does not implement.

MolecularRepresentationRef references a persisted scientific representation
used by a MolecularState. It is NOT a RawArtifact (molecular.inputs):
RawArtifact is the original acquired material; this is the scientific
representation derived from - and stored separately from - that material.

No filesystem behavior lives here - see scientific_registry.py for
persistence. No CLI, HTTP, or chemistry-engine imports.
"""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from molecular.inputs import validate_input_id


class ScientificEntity(BaseModel):
    """The stable scientific identity of a molecular object followed within
    an experiment, independent of any particular molecular state,
    conformation, file format, or external database identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)


class MolecularRepresentationRef(BaseModel):
    """A reference to a persisted molecular representation associated with a
    MolecularState - the scientific representation the state actually uses,
    not the RAW acquisition artifact it may have been derived from. `path`
    is a POSIX-style path relative to the project root; it identifies where
    the representation is stored, never the chemical identity itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    format: str = Field(min_length=1)
    checksum: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def _validate_path_is_relative(cls, value: str) -> str:
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError(
                f"path must be relative to the project root, not absolute (got {value!r})"
            )
        return value


class MolecularState(BaseModel):
    """An explicitly identified chemical realization of a ScientificEntity,
    prior to selecting a particular conformational geometry. `entity_id`
    references its ScientificEntity explicitly - the relationship is never
    implied by directory hierarchy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    representation: MolecularRepresentationRef

    @field_validator("id", "entity_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)
