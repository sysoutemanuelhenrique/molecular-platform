"""State transformation provenance: the edge between two MolecularStates.

MolecularState (molecular.scientific) is the node: a chemical realization
of a ScientificEntity. StateTransformationRecord is the edge: a fact of
provenance recording that one already-registered MolecularState (the
parent) gave rise to another already-registered MolecularState (the
child), through a named, closed-vocabulary operation.

StateTransformationRecord is deliberately NOT an engine, a workflow, a job,
a run, a state generator, a file artifact, a MolecularState, or a
Conformer. It carries no input/output artifact path or checksum of its own
- those already live on the parent's and child's own
MolecularRepresentationRef, recoverable via parent_state_id/child_state_id;
duplicating them here would create a second source of truth.

Multiple StateTransformationRecords may point at the same child_state_id
(independent provenance claims about how the same state came to be) or
form a cycle (A -> B -> A) - this module does not enforce a DAG, and any
future graph traversal over these edges must be cycle-safe.

No filesystem behavior lives here - see transformation_registry.py for
persistence. No CLI, HTTP, or chemistry-engine imports.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from molecular.inputs import validate_input_id


class TransformationProducer(BaseModel):
    """The identifiable tool that produced a state transformation, when one
    exists. Deliberately has no taxonomy of producer kinds (manual, tool,
    external, ...) - if a tool is identifiable, name it here; otherwise
    leave producer unset and use method_description instead."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("version must not be blank")
        return value


class TransformationParameter(BaseModel):
    """One named parameter of a state transformation. `value` is restricted
    to the JSON-scalar shapes str/int/float/bool - never None, never a
    nested dict or list - so a parameter can never become an untyped
    escape hatch for arbitrary metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    value: str | int | float | bool

    @field_validator("name")
    @classmethod
    def _validate_name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value


class StateTransformationRecord(BaseModel):
    """A fact of provenance: MolecularState `parent_state_id` gave rise to
    MolecularState `child_state_id` through `operation`. Both ids reference
    already-registered states explicitly - never implied by directory
    hierarchy - and are resolved/cross-checked by transformation_registry.py,
    not here. This model only enforces what is true by construction, with
    no filesystem access: parent and child cannot be the same id, the
    operation is a closed vocabulary, and either a producer tool or a
    manual method_description is required so every transformation carries
    usable provenance even without a structured chemistry-engine
    integration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    id: str = Field(min_length=1)
    parent_state_id: str = Field(min_length=1)
    child_state_id: str = Field(min_length=1)
    operation: Literal["protonation", "tautomerization"]
    producer: TransformationProducer | None = None
    method_description: str | None = None
    parameters: tuple[TransformationParameter, ...] = ()
    created_at: datetime

    @field_validator("id", "parent_state_id", "child_state_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("method_description")
    @classmethod
    def _validate_method_description_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("method_description must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_cross_field_invariants(self) -> "StateTransformationRecord":
        if self.parent_state_id == self.child_state_id:
            raise ValueError(
                "parent_state_id and child_state_id must differ "
                f"(got '{self.parent_state_id}' for both) - a state cannot "
                "be recorded as its own transformation source"
            )

        if self.producer is None and self.method_description is None:
            raise ValueError(
                "either producer or method_description is required - a "
                "transformation must carry some usable provenance"
            )

        # Intentionally case-sensitive: "pH" and "ph" are distinct
        # parameter names here. Parameter names are not storage
        # identities - unlike the id/parent_state_id/child_state_id
        # fields above (whose storage-layer conflict checks are
        # casefold-normalized for cross-filesystem portability), a
        # parameter name is scoped to this single record and never
        # addresses anything on disk, so there is no portability reason
        # to fold its case.
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameters must not contain duplicate names")

        return self
