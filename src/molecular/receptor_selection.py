"""Domain model for scientist-declared receptor selection (Increment 020,
Checkpoint 2).

`ReceptorSelectionPolicy` is the scientist's explicit, immutable
declaration of which model/assembly/chains to keep, and what to do with
water/metal/named components, against a specific `ReceptorInspectionReport`
- identified by `inspection_id` - and the RAW checksum observed at
declaration time (Section 21.2 of the frozen spec, correcting F-04). It is
never resolved by this module: resolving a policy against the RAW and the
inspection report it references is `receptor_resolution.py`'s
`ResolvedReceptorSelection`, produced by a later checkpoint's Application
Service (`resolve_receptor_selection`, Checkpoint 3) - never by this
domain model.

This module is domain-only: no filesystem, CLI, or Gemmi import.
Persistence lives in `receptor_registry.py`; the orchestration that
invokes `inspect_receptor` internally, enforces `selection_capability ==
"full"` before any field is even examined, and decides retry-vs-conflict
under a colliding `selection_id` is `receptor_service.py`'s responsibility
(Checkpoint 3), never this module's.

`inspection_id`/`raw_checksum_observed` are always filled in by the
Application Service from the result of its own internal `inspect_receptor`
call (Section 21.2) - no caller of this model ever supplies them freely;
this module only validates their shape (64 lowercase hex characters, the
same format `validate_semantic_key` already enforces for `inspection_id`
elsewhere in this codebase), never their truth against a real inspection
report (that cross-check belongs to the Service, Section 25.3).
"""

from datetime import datetime
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from molecular.inputs import validate_input_id
from molecular.promotion import validate_semantic_key

ReceptorSelectionPolicySchemaVersion = Literal["1.0"]
ComponentRole = Literal["cofactor", "crystallographic_ligand"]
Disposition = Literal["keep", "remove"]


class ChainSelector(BaseModel):
    """A scientist-declared chain reference. `label_asym_id` is always the
    canonical, required identity (Section 21.1); `auth_asym_id` is
    preserved when the scientist supplies it, but never becomes, by
    itself, a valid selector (Section 20.3) - there is no construction
    path that accepts `auth_asym_id` while omitting `label_asym_id`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: str = Field(min_length=1, max_length=10)
    auth_asym_id: Optional[str] = Field(default=None, max_length=10)


class ComponentDisposition(BaseModel):
    """A scientist-declared disposition for one named non-polymer
    component. `role` is a scientific label the scientist assigns here -
    never inferred from `NonPolymerComponent.classification` by this
    increment (Section 17.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: str = Field(min_length=1, max_length=10)
    ccd_code: str = Field(min_length=1, max_length=10)
    role: ComponentRole
    disposition: Disposition


class ReceptorSelectionPolicy(BaseModel):
    """The scientist's explicit, immutable declaration (schema_version
    "1.0") of model/assembly/chain/water/metal/component selection, always
    pinned to a specific `inspection_id` and the `raw_checksum_observed`
    at the moment of declaration (Section 21.2, correcting F-04) - never a
    structural transformation, never a materialized artifact, never a
    coordinate file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: ReceptorSelectionPolicySchemaVersion
    selection_id: str = Field(min_length=1)
    protein_input_id: str = Field(min_length=1)
    inspection_id: str = Field(min_length=64, max_length=64)
    raw_checksum_observed: str = Field(min_length=64, max_length=64)
    selected_model: Optional[int] = Field(default=None, ge=1)
    selected_assembly: Optional[str] = Field(default=None, min_length=1, max_length=50)
    selected_chains: Tuple[ChainSelector, ...] = ()
    water_disposition: Optional[Disposition] = None
    metal_disposition: Optional[Disposition] = None
    component_dispositions: Tuple[ComponentDisposition, ...] = ()
    declared_by: str = Field(min_length=1, max_length=200)
    label: Optional[str] = Field(default=None, max_length=200)
    rationale: Optional[str] = Field(default=None, max_length=4000)
    declared_at: datetime

    @field_validator("selection_id", "protein_input_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("inspection_id", "raw_checksum_observed")
    @classmethod
    def _validate_hex64_format(cls, value: str) -> str:
        return validate_semantic_key(value)

    @field_validator("declared_by")
    @classmethod
    def _validate_declared_by_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("declared_by must not be blank")
        return value

    @field_validator("label")
    @classmethod
    def _validate_label_not_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("label must not be blank")
        return value

    @field_validator("rationale")
    @classmethod
    def _validate_rationale_not_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("rationale must not be blank")
        return value

    @field_validator("declared_at")
    @classmethod
    def _validate_declared_at_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("declared_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_no_duplicate_chains(self) -> "ReceptorSelectionPolicy":
        seen = [chain.label_asym_id for chain in self.selected_chains]
        if len(seen) != len(set(seen)):
            raise ValueError("selected_chains must not contain a duplicate label_asym_id")
        return self

    @model_validator(mode="after")
    def _validate_no_duplicate_component_dispositions(self) -> "ReceptorSelectionPolicy":
        seen = [(d.label_asym_id, d.ccd_code) for d in self.component_dispositions]
        if len(seen) != len(set(seen)):
            raise ValueError(
                "component_dispositions must not contain a duplicate "
                "(label_asym_id, ccd_code) pair"
            )
        return self
