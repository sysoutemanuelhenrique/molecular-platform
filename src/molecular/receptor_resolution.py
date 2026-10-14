"""Domain model for a resolved receptor selection (Increment 020,
Checkpoint 2).

`ResolvedReceptorSelection` is the deterministic, create-once outcome of
resolving a `ReceptorSelectionPolicy` against the RAW and against the
`ReceptorInspectionReport` it references by `inspection_id` (Section 22 of
the frozen spec, correcting F-04) - it never embeds the full inspection
report, only the reference and the already-bounded subset of chains/
dispositions the scientist's own policy actually named (Section 14.3, rule
5). Resolving is itself Checkpoint 3's Application Service responsibility
(`resolve_receptor_selection`); this module only defines the shape of the
result. It never loads a `ReceptorInspectionReport`, never reopens the
RAW, and never raises `ModelNotFoundError`/`AmbiguousModelSelectionError`/
`AssemblyNotFoundError`/`ChainNotFoundError` - those are Service-level
failures over data this model never touches (Section 25.4-25.5); T-061
(Section 35.4) migrates exactly this kind of check from "model validator"
to "Service behavior" for the same reason.

`ModelResolutionBasis` is referenced by Section 22.2 of the frozen spec
without its full closed vocabulary restated there (inherited from the
Draft original). The only value the frozen spec names explicitly is
`"only-model-present"` (Section 25.5/35.4, T-065) - the case where
`selected_model=None` resolves transparently because the report carries
exactly one model. The only other successful resolution path the same
table describes is a scientist-declared `selected_model` that the report
confirms - named here `"explicitly-selected"`, the minimal, mechanical
completion of the two-row success set in Section 25.5's table; no other
value is introduced.

This module is domain-only: no filesystem, CLI, or Gemmi import.
Persistence lives in `receptor_registry.py`.
"""

from datetime import datetime
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from molecular.inputs import validate_input_id
from molecular.promotion import validate_semantic_key
from molecular.receptor_inspection import NonPolymerClassification, ParserProvenance, StructureFormat
from molecular.receptor_selection import Disposition

ResolvedReceptorSelectionSchemaVersion = Literal["1.0"]
DispositionSource = Literal["water_disposition", "metal_disposition", "component_dispositions"]
ModelResolutionBasis = Literal["only-model-present", "explicitly-selected"]


class ResolvedChainSelection(BaseModel):
    """`label_asym_id`/`label_entity_id` are always present - never
    `Optional` - because every `ResolvedReceptorSelection` is, by
    construction, the product of resolving against an inspection whose
    `selection_capability` was `"full"` (Section 20.4); `auth_asym_id` is
    preserved when known, but never required. There is no
    `label_asym_id_is_auth_equivalent` field here - it was retired as dead
    weight under this same guarantee (Section 15.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: str = Field(min_length=1)
    auth_asym_id: Optional[str] = None
    label_entity_id: str = Field(min_length=1)
    model_number: int = Field(ge=1)


class ResolvedComponentDisposition(BaseModel):
    """`classification` is copied, read-only, from the referenced
    inspection's own mechanical classification (Section 17) - this model
    never recomputes or overrides it. `disposition`/`disposition_source`
    are `None` together whenever no disposition from the policy actually
    applied to this specific component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: str = Field(min_length=1)
    ccd_code: str = Field(min_length=1)
    classification: NonPolymerClassification
    disposition: Optional[Disposition] = None
    disposition_source: Optional[DispositionSource] = None


class ResolvedReceptorSelection(BaseModel):
    """The deterministic, create-once resolution of a
    `ReceptorSelectionPolicy` (schema_version "1.0") - 1:1 with the policy
    it resolves (`resolved_selection_id == policy_selection_id`),
    referencing its inspection by `inspection_id` alone (never embedding
    the full report, Section 22.2, correcting F-04). Not a molecular
    representation; no field names an output coordinate file (Section
    22.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: ResolvedReceptorSelectionSchemaVersion
    resolved_selection_id: str = Field(min_length=1)
    policy_selection_id: str = Field(min_length=1)
    protein_input_id: str = Field(min_length=1)
    inspection_id: str = Field(min_length=64, max_length=64)
    raw_checksum: str = Field(min_length=64, max_length=64)
    structure_format: StructureFormat
    parser: ParserProvenance
    resolved_model: int = Field(ge=1)
    model_resolution_basis: ModelResolutionBasis
    resolved_assembly: Optional[str] = Field(default=None, min_length=1, max_length=50)
    resolved_chains: Tuple[ResolvedChainSelection, ...]
    resolved_component_dispositions: Tuple[ResolvedComponentDisposition, ...]
    resolved_at: datetime

    @field_validator("resolved_selection_id", "policy_selection_id", "protein_input_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("inspection_id", "raw_checksum")
    @classmethod
    def _validate_hex64_format(cls, value: str) -> str:
        return validate_semantic_key(value)

    @field_validator("resolved_at")
    @classmethod
    def _validate_resolved_at_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("resolved_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_internal_consistency(self) -> "ResolvedReceptorSelection":
        if self.resolved_selection_id != self.policy_selection_id:
            raise ValueError(
                "resolved_selection_id must always equal policy_selection_id "
                f"(got {self.resolved_selection_id!r} != {self.policy_selection_id!r})"
            )
        seen = [chain.label_asym_id for chain in self.resolved_chains]
        if len(seen) != len(set(seen)):
            raise ValueError("resolved_chains must not contain a duplicate label_asym_id")
        return self
