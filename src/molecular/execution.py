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

ExecutionAttemptSchemaVersion = Literal["1.0", "1.1"]


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
    """max_states é o teto opcional de novos MolecularStates publicados
    após a enumeração ter concluído (status=Completed), a sanitização e a
    verificação de invariantes químicos de cada candidato, a deduplicação
    por SMILES canônico isomérico, a exclusão do estado equivalente à
    entrada, e a ordenação determinística. Ele não limita a busca interna
    do motor e não é mapeado para maxTautomers ou maxTransforms do RDKit.
    None significa que o cientista não declarou teto algum - não que algum
    default numérico da plataforma se aplica; este módulo não carrega
    nenhum default científico para quantos tautômeros enumerar, e nenhum é
    inventado aqui."""
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
    attempt. Carries a phase and a human-readable message - no traceback
    or exception object, matching the level of detail already carried by
    other provenance records in this domain (e.g.
    StateTransformationRecord's method_description) - plus, since
    Increment 016, two residual-id tuples (Section 10.4 of the frozen
    spec) recording ids that best-effort compensation (Section 17.3)
    attempted to remove but could not confirm as removed.

    residual_state_ids/residual_operation_ids are permitted only when
    phase == "execution" (nothing can have been published before
    resolution, so a 'resolution'-phase error can never carry residuals);
    every id is independently validated and duplicate-free within its own
    tuple. This model has no access to the enclosing ExecutionAttempt's
    schema_version - it validates only its own internal content. The
    cross-check that a schema_version="1.0" attempt's error must never
    carry residuals at all belongs to ExecutionAttempt._validate_status_
    invariants (Section 13 of the frozen spec), not here - see that
    validator's own docstring for the full separation of responsibilities.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: Literal["resolution", "execution"]
    message: str = Field(min_length=1)
    residual_state_ids: tuple[str, ...] = ()
    residual_operation_ids: tuple[str, ...] = ()

    @field_validator("message")
    @classmethod
    def _validate_message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_residuals(self) -> "ExecutionAttemptError":
        for id_value in (*self.residual_state_ids, *self.residual_operation_ids):
            validate_input_id(id_value)

        if len(set(self.residual_state_ids)) != len(self.residual_state_ids):
            raise ValueError("residual_state_ids must not contain duplicate entries")
        if len(set(self.residual_operation_ids)) != len(self.residual_operation_ids):
            raise ValueError("residual_operation_ids must not contain duplicate entries")

        if self.phase == "resolution":
            if self.residual_state_ids or self.residual_operation_ids:
                raise ValueError(
                    "a 'resolution'-phase error must not carry any residual ids "
                    "(nothing can have been published before resolution)"
                )

        return self


class ExecutionAttempt(BaseModel):
    """One attempt by the platform to execute a named scientific operation.

    schema_version accepts "1.0" or "1.1" (Increment 016, Section 10.1 of
    the frozen spec). Only this model is versioned by that change -
    TautomerEnumerationPolicy and ResolvedExecutionSnapshot remain
    exclusively schema_version="1.0", unaffected. Every new attempt this
    codebase creates is schema_version="1.1"; "1.0" documents persisted by
    Increment 015 remain readable, unmigrated, forever.

    input_state_id (new in "1.1") declares, from the moment the attempt is
    created (status "started"), which MolecularState the operation acts
    upon - a sujeito auditável even for an attempt that never reaches the
    engine. It is None for every "1.0" attempt (the field predates it) and
    non-None, identical across every status, for every "1.1" attempt - see
    _validate_status_invariants below for the enforced cross-check
    (Section 10.3 of the frozen spec).

    operation_ids names the StateTransformationRecord ids this attempt
    actually produced. It is `None` for every status except "succeeded":
    which ids (if any) an engine invocation actually produces is only
    knowable once execution has finished, so the field carries no value
    at all before then - not a placeholder, not a pre-declared intent.
    Once status=="succeeded", operation_ids is a tuple - possibly empty,
    meaning the operation completed but produced no new transformations -
    with every entry independently validated for filesystem-safe format
    and no duplicates permitted. See the D-015-11/D-015-12 amendment
    recorded in execution_registry.py's module docstring for the full
    rationale and history.

    Reaching status=="succeeded" does not, by itself, prove any id in
    operation_ids resolves to a real StateTransformationRecord. Neither
    this model nor execution_registry.mark_succeeded can check that: no
    producer of StateTransformationRecords participates in this
    increment. Ensuring it is true - registering every named
    StateTransformationRecord *before* calling mark_succeeded - is a
    caller contract placed on Increment 016's future orchestrating
    Application Service, not something enforced here. See
    mark_succeeded's own docstring in execution_registry.py for the same
    contract restated at the call site.

    `id` is always generated by the registry (uuid4().hex) - never
    supplied by a caller, and never derived from scientific content.

    Instances are frozen: a status transition is never an in-place field
    assignment. See execution_registry.py for how a persisted attempt
    moves from one status to the next - always by constructing a whole new
    validated ExecutionAttempt snapshot and atomically overwriting the
    previous one, never by mutating this object.

    The `_validate_status_invariants` model validator below is this
    model's only enforcement of the state machine's per-status shape, PLUS
    (since Increment 016) the schema_version x status x input_state_id
    cross-check (Section 10.3) and the schema_version x residuals
    cross-check (Section 12) - both necessarily live here, not on
    ExecutionAttemptError, because both depend on schema_version, a field
    ExecutionAttemptError does not have and must never be given just to
    duplicate a check that belongs at this level (Section 13 of the frozen
    spec). The state *machine* itself (which transitions are reachable
    from which) is enforced by execution_registry.py, not here - this
    model only ever validates that one given snapshot is internally
    consistent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: ExecutionAttemptSchemaVersion
    id: str = Field(min_length=1)
    operation_type: Literal["state-transformation"]
    input_state_id: Optional[str] = None
    operation_ids: Optional[tuple[str, ...]] = None
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

    @field_validator("input_state_id")
    @classmethod
    def _validate_input_state_id_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return validate_input_id(value)

    @field_validator("operation_ids")
    @classmethod
    def _validate_operation_ids(
        cls, value: Optional[tuple[str, ...]]
    ) -> Optional[tuple[str, ...]]:
        if value is None:
            return None
        for candidate in value:
            validate_input_id(candidate)
        if len(set(value)) != len(value):
            raise ValueError("operation_ids must not contain duplicate entries")
        return value

    @model_validator(mode="after")
    def _validate_status_invariants(self) -> "ExecutionAttempt":
        # --- schema_version x input_state_id (Section 10.3) ---
        if self.schema_version == "1.0":
            if self.input_state_id is not None:
                raise ValueError(
                    "a schema_version '1.0' attempt must not carry input_state_id "
                    "(it predates this field)"
                )
        else:  # "1.1"
            if self.input_state_id is None:
                raise ValueError(
                    "a schema_version '1.1' attempt must carry a non-null "
                    "input_state_id in every status"
                )

        # --- schema_version x residuals (Section 12) ---
        if self.schema_version == "1.0" and self.error is not None:
            if self.error.residual_state_ids or self.error.residual_operation_ids:
                raise ValueError(
                    "a schema_version '1.0' attempt's error must not carry "
                    "residual ids (residuals did not exist in this schema)"
                )

        if self.status == "started":
            if self.resolved is not None:
                raise ValueError("a 'started' attempt must not have a resolved snapshot")
            if self.error is not None:
                raise ValueError("a 'started' attempt must not have an error")
            if self.resolved_at is not None:
                raise ValueError("a 'started' attempt must not have resolved_at set")
            if self.finished_at is not None:
                raise ValueError("a 'started' attempt must not have finished_at set")
            if self.operation_ids is not None:
                raise ValueError("a 'started' attempt must not have operation_ids set")

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
            if self.operation_ids is not None:
                raise ValueError("a 'resolution_failed' attempt must not have operation_ids set")

        elif self.status == "resolved":
            if self.resolved is None:
                raise ValueError("a 'resolved' attempt must carry a resolved snapshot")
            if self.error is not None:
                raise ValueError("a 'resolved' attempt must not have an error")
            if self.resolved_at is None:
                raise ValueError("a 'resolved' attempt must have resolved_at set")
            if self.finished_at is not None:
                raise ValueError("a 'resolved' attempt must not have finished_at set")
            if self.operation_ids is not None:
                raise ValueError("a 'resolved' attempt must not have operation_ids set")

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
            if self.operation_ids is not None:
                raise ValueError("an 'execution_failed' attempt must not have operation_ids set")

        elif self.status == "succeeded":
            if self.resolved is None:
                raise ValueError("a 'succeeded' attempt must carry a resolved snapshot")
            if self.error is not None:
                raise ValueError("a 'succeeded' attempt must not have an error")
            if self.resolved_at is None:
                raise ValueError("a 'succeeded' attempt must have resolved_at set")
            if self.finished_at is None:
                raise ValueError("a 'succeeded' attempt must have finished_at set")
            if self.operation_ids is None:
                raise ValueError("a 'succeeded' attempt must have operation_ids set")

        return self
