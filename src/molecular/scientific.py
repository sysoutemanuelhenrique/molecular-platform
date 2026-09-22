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

Conformer is a specific, immutable 3D geometry of exactly one
MolecularState - one of a set of stereoisomers each characterized by a
conformation corresponding to a distinct potential energy minimum,
reachable from any other conformer of the same state by rotation about
formally single bonds alone (IUPAC Gold Book C01258/C01262). It is
deliberately NOT a PDBQT file, a docking pose, or proof that the geometry
is chemically valid or genuinely three-dimensional - see its own docstring
below for the exact boundary. How a Conformer came to exist (which engine,
which parameters, whether it was optimized from another conformer) is
provenance, not identity, and is not modeled by this increment.

MolecularRepresentationRef references a persisted scientific representation
used by a MolecularState or a Conformer. It is NOT a RawArtifact
(molecular.inputs): RawArtifact is the original acquired material; this is
the scientific representation derived from - and stored separately from -
that material.

No filesystem behavior lives here - see scientific_registry.py for
persistence. No CLI, HTTP, or chemistry-engine imports.
"""

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from molecular.inputs import validate_input_id

_PATH_SEPARATOR_PATTERN = re.compile(r"[\\/]+")
_TRAVERSAL_COMPONENTS = {".", ".."}


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
    MolecularState or a Conformer - the scientific representation the state
    or conformer actually uses, not the RAW acquisition artifact it may
    have been derived from. `path` is a POSIX-style path relative to the
    project root; it identifies where the representation is stored, never
    the chemical identity itself. `path`
    is rejected outright (never resolved, normalized, or filesystem-checked)
    if it is absolute in either POSIX or Windows form, or if any of its
    `/`- or `\\`-separated components is exactly '.' or '..' - a purely
    lexical check, so no path can traverse outside the project root."""

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

        components = _PATH_SEPARATOR_PATTERN.split(value)
        if any(component in _TRAVERSAL_COMPONENTS for component in components):
            raise ValueError(
                "path must not contain '.' or '..' path traversal components "
                f"(got {value!r})"
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


class Conformer(BaseModel):
    """A specific, immutable 3D geometry of exactly one MolecularState.
    `state_id` references its parent MolecularState explicitly - the
    relationship is never implied by directory hierarchy, and a Conformer
    always belongs to exactly one state (a MolecularState may have zero,
    one, or many Conformers, discovered by listing, never by an embedded
    id list on the state itself).

    Registering a Conformer does NOT assert that its geometry has been
    chemically validated, and does NOT assert that its representation is
    genuinely three-dimensional - `representation.format` is a declared
    hint, not a verified guarantee (this increment has no chemistry engine
    capable of checking either). The caller is only declaring: "this
    artifact represents a conformer of this state."

    The caller is responsible for ensuring that every Conformer registered
    under the same state_id corresponds to the same configurational
    identity of that MolecularState. This increment performs no chemical
    or stereochemical validation of that relationship. In particular, a
    configurational change - R to S, E to Z, or any other stereochemical
    change that cannot be reached by rotation about a single bond alone -
    is NOT simply another Conformer of the same state; by the IUPAC
    definition of conformer, such a change falls outside what a Conformer
    relationship can represent at all. Where a configurational change
    belongs in this domain model is not resolved by this increment.

    Once registered, a Conformer is immutable and its artifact is never
    edited in place: an optimized geometry is persisted as a new Conformer,
    never as a mutation of an existing one - a deliberate persistence
    decision of this platform, not the native in-memory behavior of any
    particular chemistry toolkit. How one Conformer relates to another
    (e.g. "this one is an optimized version of that one") is provenance,
    and is not modeled by this increment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    representation: MolecularRepresentationRef

    @field_validator("id", "state_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)
