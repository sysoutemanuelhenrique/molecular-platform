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

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"]
    engine_name: str = Field(min_length=1)
    engine_version: str = Field(min_length=1)
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("engine_name", "engine_version")
    @classmethod
    def _validate_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


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

    `operation_id` names the id of the StateTransformationRecord this
    attempt intends to eventually produce. It is validated only for
    filesystem-safe format (via the same validate_input_id used
    everywhere else in this domain) - never checked against an existing
    record, and never required to resolve to one: an attempt begins
    before the transformation it aims to produce exists, by design.
    Multiple attempts (retries) may legitimately share the same
    operation_id, each carrying its own unique `id`.

    `id` is always generated by the registry (uuid4().hex) - never
    supplied by a caller, and never derived from scientific content.

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
    operation_id: str = Field(min_length=1)
    status: ExecutionStatus
    requested: TautomerEnumerationPolicy
    resolved: Optional[ResolvedExecutionSnapshot] = None
    error: Optional[ExecutionAttemptError] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @field_validator("id", "operation_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)

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
