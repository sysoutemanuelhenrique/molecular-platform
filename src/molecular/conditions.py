"""Domain model for scientist-declared experimental conditions.

ScientificConditionsDeclaration (Increment 019, "Experimental Conditions
Foundation") is an independent, immutable, scientist-authored declaration of
an intended pH (point or range) for a future simulation input. It is
created once and never edited in place; a revision is a brand-new record
whose supersedes_id names the record it revises - never an in-place change
to that record (see INCREMENT_019_FINAL_SPECIFICATION.md, Sections 15-17).

This module is domain-only: no filesystem, CLI, or chemistry-engine import.
Persistence lives in conditions_registry.py; identity/idempotency
recognition and revision resolution are orchestrated by conditions_service.py
(a later checkpoint), never by this module.

ScientificConditionsDeclaration is not Evidence, not EngineParameters, not
ExecutionResult, and not a ScientificPolicy (Section 7.2 of the frozen
spec) - it is a standing fact a scientist asserts, independent of whether
any execution ever references it. It never infers an absent pH, never
defaults one, and this platform never treats any declaration as "active."

The max_digits/decimal_places bound on every pH-bearing field below is a
purely computational/representational limit (Section 14 of the frozen
spec) - it carries no scientific meaning and is never equivalent to, or a
substitute for, the deliberate absence of any universal 0-14 range on pH
itself.
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from molecular.inputs import validate_input_id

ScientificConditionsDeclarationSchemaVersion = Literal["1.0"]


class PhPoint(BaseModel):
    """A single, finite pH value.

    `Field(max_digits=12, decimal_places=6)` enforces three distinct native
    Pydantic constraints - decimal_max_digits, decimal_max_places, and
    decimal_whole_digits (integer-part digits <= 12 - 6 = 6) - and already
    rejects boolean, NaN, and infinite input natively, with no custom
    validator needed (Section 14.3 of the frozen spec).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: Decimal = Field(max_digits=12, decimal_places=6)


class PhRange(BaseModel):
    """A {minimum, maximum} pH range. Both bounds inclusive; minimum <= maximum."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum: Decimal = Field(max_digits=12, decimal_places=6)
    maximum: Decimal = Field(max_digits=12, decimal_places=6)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "PhRange":
        if self.minimum > self.maximum:
            raise ValueError(
                f"minimum ({self.minimum}) must be <= maximum ({self.maximum})"
            )
        return self


class PhDeclaration(BaseModel):
    """Exactly one of a point or a range pH value - never both, never neither.

    This is an exclusive-OR cross-field rule, distinct from (though stylistically
    inspired by) StateTransformationRecord's own inclusive-OR "producer or
    method_description" validator in transformation.py - a pH declaration
    cannot simultaneously be a point and a range, which that earlier
    precedent does not itself require (Section 13.4 of the frozen spec).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: Optional[PhPoint] = None
    range: Optional[PhRange] = None

    @model_validator(mode="after")
    def _validate_exactly_one_form(self) -> "PhDeclaration":
        if (self.point is None) == (self.range is None):
            raise ValueError(
                "exactly one of 'point' or 'range' must be provided, never both, never neither"
            )
        return self


class ScientificConditionsDeclaration(BaseModel):
    """An independent, immutable, scientist-authored declaration of intended
    pH for a future simulation input (schema_version "1.0").

    Create-once: once persisted under `id` (conditions_registry.py), a
    record is never edited in place. A revision is a brand-new record
    naming the one it supersedes via `supersedes_id` - this model only
    enforces that a record never supersedes itself; predecessor existence,
    retry/conflict recognition, and the full acyclicity guarantee are the
    registry's and the Application Service's responsibility, not this
    domain model's (Sections 15-17 of the frozen spec).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: ScientificConditionsDeclarationSchemaVersion
    id: str = Field(min_length=1)
    ph: PhDeclaration
    purpose: Literal["simulation_input"]
    declared_by: str = Field(min_length=1, max_length=200)
    label: Optional[str] = Field(default=None, max_length=200)
    rationale: Optional[str] = Field(default=None, max_length=4000)
    supersedes_id: Optional[str] = None
    declared_at: datetime

    @field_validator("id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("supersedes_id")
    @classmethod
    def _validate_supersedes_id_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return validate_input_id(value)

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
    def _validate_no_self_supersession(self) -> "ScientificConditionsDeclaration":
        if self.supersedes_id is not None and self.supersedes_id == self.id:
            raise ValueError("a declaration must not supersede itself")
        return self
