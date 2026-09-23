"""Execution attempt foundation: the platform's own future intent to
execute a scientific operation, and the state machine governing how that
intent resolves - before any chemistry engine exists in this codebase.

This is deliberately NOT chemistry. Nothing in this module runs, computes,
or generates anything. An ExecutionAttempt records that the platform
*intends* to carry out a named operation (today: tautomer enumeration via
TautomerEnumerationPolicy), and tracks, with enforced invariants, whether
that intent was resolved into a concrete configuration and whether that
configuration then succeeded or failed - all as persisted provenance, not
as an executor.

ExecutionAttempt is a different mechanism from StateTransformationRecord
(molecular.transformation) and register_derived_state
(molecular.application), not a replacement for either:

- `state derive` / StateTransformationRecord record, after the fact, that
  a transformation already performed by the scientist or some external
  tool produced a child MolecularState from a parent one. The artifact
  already exists when that registration happens.
- ExecutionAttempt records a transformation the *platform itself* intends
  to carry out, before it has happened, and tracks the attempt through
  resolution and execution. Manual external registration (`state derive`)
  is never itself an execution attempt, and beginning an ExecutionAttempt
  never creates, requires, or implies a StateTransformationRecord - that
  connection, if any, is Increment 016's concern (an Application Service
  orchestrating an attempt through to a real StateTransformationRecord),
  not this one's.

No filesystem behavior lives here - see execution_registry.py for
persistence. No CLI, HTTP, or chemistry-engine imports.
"""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from molecular.inputs import validate_input_id

ExecutionStatus = Literal[
    "started",
    "resolution_failed",
    "resolved",
    "execution_failed",
    "succeeded",
]


class TautomerEnumerationPolicy(BaseModel):
    """A scientist's declared policy for tautomer enumeration.

    The policy's existence already means enumeration was requested - there
    is deliberately no `enumerate` flag to toggle. `max_states` is an
    optional ceiling the scientist may declare; `None` means the scientist
    declared no ceiling at all, not that some numeric default applies -
    this module carries no scientific default for how many tautomers to
    enumerate, and none is invented here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    max_states: Optional[int] = None
    source: Literal["scientist"]

    @field_validator("max_states")
    @classmethod
    def _validate_max_states_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError(f"max_states must be >= 1 when provided (got {value})")
        return value


class ResolvedExecutionSnapshot(BaseModel):
    """A PROVISIONAL snapshot of the concrete engine configuration an
    execution attempt resolved to.

    This shape is deliberately minimal and exists only to let the state
    machine in this increment carry *some* resolved configuration forward
    without inventing engine-specific structure ahead of time. It is not a
    final engine contract: no chemistry engine is integrated in this
    increment, and this schema is expected to be revisited in Increment
    016 against the real API of whichever engine is integrated first. No
    field here should be treated as load-bearing beyond driving this
    increment's own state machine.

    `parameters` is restricted to the JSON-scalar shapes str/int/float/bool
    - never None, never a nested dict or list - for the same reason
    TransformationParameter.value is so restricted in molecular.transformation:
    it must never become an untyped escape hatch for arbitrary metadata.

    `parameters` is exposed as a read-only `MappingProxyType`, not a plain
    `dict`: `model_config = ConfigDict(frozen=True)` only stops a field
    from being *reassigned* (`snapshot.parameters = {...}`) - it does
    nothing to stop a caller from mutating a mutable object already
    sitting in one of the model's fields (`snapshot.parameters["x"] = 1`
    would otherwise succeed silently). Wrapping the validated dict in a
    `MappingProxyType` closes that gap, so this provenance record is
    genuinely immutable end to end, not just at the top level. Validation,
    YAML serialization, and round-trip loading are unaffected: the
    `_freeze_parameters` validator only changes the in-memory container
    after validation has already run, and `_serialize_parameters` maps it
    straight back to a plain `dict` for `model_dump`/YAML, so the
    persisted shape on disk is identical to before.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    engine_name: str = Field(min_length=1)
    engine_version: str = Field(min_length=1)
    parameters: Mapping[str, str | int | float | bool] = Field(
        default_factory=dict, validate_default=True
    )

    @field_validator("engine_name", "engine_version")
    @classmethod
    def _validate_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("parameters", mode="after")
    @classmethod
    def _freeze_parameters(
        cls, value: Mapping[str, str | int | float | bool]
    ) -> Mapping[str, str | int | float | bool]:
        """Wrap the already-validated mapping in a read-only view. Runs
        with validate_default=True so the empty default is frozen too,
        not just an explicitly-passed value."""

        return MappingProxyType(dict(value))

    @field_serializer("parameters")
    def _serialize_parameters(
        self, value: Mapping[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        """model_dump/YAML need a plain dict, not a MappingProxyType -
        this is presentation only and never changes the frozen in-memory
        value itself."""

        return dict(value)


class ExecutionAttemptError(BaseModel):
    """The provenance of a failure at one specific phase of an execution
    attempt. Deliberately carries only a phase and a human-readable
    message - no traceback or exception object, matching the level of
    detail already carried by other provenance records in this domain
    (e.g. StateTransformationRecord's method_description)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: Literal["resolution", "execution"]
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def _validate_message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ExecutionAttempt(BaseModel):
    """One attempt by the platform to execute a named scientific operation.

    AMENDMENT TO D-015-11 (post-freeze; see execution_registry.py's module
    docstring for the full amendment text). The frozen spec's Section 12
    and decision D-015-11 originally modeled the target as a single
    `operation_id: str` and stated that it becomes "guaranteed
    dereferenceable" once status=="succeeded". Two problems with that as
    originally written, both resolved by this amendment rather than by a
    docstring alone:

    1. Cardinality: a single tautomer-enumeration attempt (one engine
       invocation, one resolved configuration, one success/failure
       outcome) may legitimately produce more than one resulting
       MolecularState, and - following the existing one-child-per-record
       shape of StateTransformationRecord in molecular.transformation -
       that means more than one StateTransformationRecord per attempt.
       `operation_id` (singular) could not name that. `operation_ids` (a
       non-empty tuple, each entry independently validated for
       filesystem-safe format, no duplicates permitted) now names the
       *complete set* of StateTransformationRecord ids this one attempt
       intends to produce. This still never requires any of them to
       exist at any point this model or execution_registry.py can see -
       an attempt begins before the transformations it targets exist, by
       design, exactly as before.

    2. Guarantee vs. contract: "succeeded" does not, by itself, prove
       every id in operation_ids resolves to a real StateTransformationRecord.
       Neither this model nor execution_registry.mark_succeeded can check
       that: no producer of StateTransformationRecords participates in
       this increment, and coupling this minimal state machine to
       molecular.transformation_registry just to verify a guarantee it
       cannot yet fulfil would be premature, contradicting this
       increment's own "no Application Service, no engine" scope
       decision. D-015-11's guarantee is therefore reclassified from a
       system-enforced invariant of the persisted record to a CALLER
       CONTRACT: whoever transitions an attempt to "succeeded" (in
       Increment 016, the orchestrating Application Service) MUST have
       already registered every StateTransformationRecord named in
       operation_ids *before* calling mark_succeeded, never after. Until
       that Application Service exists and is exercised, a persisted
       ExecutionAttempt with status=="succeeded" asserts only the
       platform's own belief that its intended operation completed - not
       a verified fact. See mark_succeeded's own docstring in
       execution_registry.py for the same contract restated at the call
       site.

    `id` is always generated by the registry (uuid4().hex) - never
    supplied by a caller, and never derived from scientific content.
    Multiple attempts (retries) may legitimately share one or more
    operation_ids, each carrying its own unique `id`.

    Instances are frozen: a status transition is never an in-place field
    assignment. See execution_registry.py for how a persisted attempt
    moves from one status to the next - always by constructing a whole new
    validated ExecutionAttempt snapshot and atomically overwriting the
    previous one, never by mutating this object.

    The `_validate_status_invariants` model validator below is this
    model's only enforcement of the state machine's per-status shape; the
    state *machine* itself (which transitions are reachable from which)
    is enforced by execution_registry.py, not here - this model only ever
    validates that one given snapshot is internally consistent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    id: str = Field(min_length=1)
    operation_type: Literal["state-transformation"]
    operation_ids: tuple[str, ...] = Field(min_length=1)
    status: ExecutionStatus
    requested: TautomerEnumerationPolicy
    resolved: Optional[ResolvedExecutionSnapshot] = None
    error: Optional[ExecutionAttemptError] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @field_validator("id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("operation_ids")
    @classmethod
    def _validate_operation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for candidate in value:
            validate_input_id(candidate)
        if len(set(value)) != len(value):
            raise ValueError("operation_ids must not contain duplicate entries")
        return value

    @model_validator(mode="after")
    def _validate_status_invariants(self) -> "ExecutionAttempt":
        if self.status == "started":
            if self.resolved is not None:
                raise ValueError("a 'started' attempt must not have a resolved snapshot")
            if self.error is not None:
                raise ValueError("a 'started' attempt must not have an error")
            if self.resolved_at is not None:
                raise ValueError("a 'started' attempt must not have resolved_at set")
            if self.finished_at is not None:
                raise ValueError("a 'started' attempt must not have finished_at set")

        elif self.status == "resolution_failed":
            if self.resolved is not None:
                raise ValueError(
                    "a 'resolution_failed' attempt must not have a resolved snapshot"
                )
            if self.error is None:
                raise ValueError("a 'resolution_failed' attempt must carry an error")
            if self.error.phase != "resolution":
                raise ValueError(
                    "a 'resolution_failed' attempt's error phase must be 'resolution' "
                    f"(got {self.error.phase!r})"
                )
            if self.resolved_at is not None:
                raise ValueError("a 'resolution_failed' attempt must not have resolved_at set")
            if self.finished_at is None:
                raise ValueError("a 'resolution_failed' attempt must have finished_at set")

        elif self.status == "resolved":
            if self.resolved is None:
                raise ValueError("a 'resolved' attempt must carry a resolved snapshot")
            if self.error is not None:
                raise ValueError("a 'resolved' attempt must not have an error")
            if self.resolved_at is None:
                raise ValueError("a 'resolved' attempt must have resolved_at set")
            if self.finished_at is not None:
                raise ValueError("a 'resolved' attempt must not have finished_at set")

        elif self.status == "execution_failed":
            if self.resolved is None:
                raise ValueError("an 'execution_failed' attempt must carry a resolved snapshot")
            if self.error is None:
                raise ValueError("an 'execution_failed' attempt must carry an error")
            if self.error.phase != "execution":
                raise ValueError(
                    "an 'execution_failed' attempt's error phase must be 'execution' "
                    f"(got {self.error.phase!r})"
                )
            if self.resolved_at is None:
                raise ValueError("an 'execution_failed' attempt must have resolved_at set")
            if self.finished_at is None:
                raise ValueError("an 'execution_failed' attempt must have finished_at set")

        elif self.status == "succeeded":
            if self.resolved is None:
                raise ValueError("a 'succeeded' attempt must carry a resolved snapshot")
            if self.error is not None:
                raise ValueError("a 'succeeded' attempt must not have an error")
            if self.resolved_at is None:
                raise ValueError("a 'succeeded' attempt must have resolved_at set")
            if self.finished_at is None:
                raise ValueError("a 'succeeded' attempt must have finished_at set")

        return self
