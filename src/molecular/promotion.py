"""Ligand promotion domain foundation: turning an already-registered
LigandInput RAW artifact into an auditable ScientificEntity/MolecularState
pair, and the lineage record that proves it happened.

This module is Checkpoint 1 of Increment 018's frozen final specification
(``increments/018/11-final-specification/INCREMENT_018_FINAL_SPECIFICATION.md``,
SHA-256 ``347632175991b1d72b9f193312b9629ce71697fb0c3d0de2285e52b7299d8023``).
It implements exactly Section 6 (domain models) and Section 10 (semantic key
canonicalization) of that specification - the models and validators, and the
pure function that derives a promotion's identity from its content. It is
deliberately NOT the RAW-opening route (Section 8), NOT the RDKit inspection
boundary (Section 9), NOT the cross-process reservation mechanism
(Section 11.2-11.14), and NOT the Application Service (Section 14) - those
are later checkpoints. No filesystem behavior lives here - see
promotion_registry.py for persistence of the two models that are actually
written to disk (LigandPromotionAttempt, ScientificInputPromotionRecord).

LigandPromotionPolicy's three RDKit flags (sanitize, remove_hs,
strict_parsing) are plain `bool`, not `Literal` - Increment 018's RD-018-01
spike could only construct and exercise its full 8-combination matrix
because the model already accepted arbitrary boolean combinations (Section
9.3 of the frozen spec). The single combination the production CLI will
ever construct is the default shown below (`sanitize=True, remove_hs=False,
strict_parsing=True`) - RD-018-01 confirmed this is the only one of the 8
that rejects an invalid-valence structure without RDKit's own
`removeHs=True` masking it into a different, silently-substituted molecule
(Section 9.3.1 of the frozen spec). Nothing in this module enforces that
restriction; it is a CLI/Application-Service-level contract (Section 15.1),
not a domain-model invariant, because the domain model must remain able to
represent all 8 combinations for spikes and tests (Section 9.6 of the frozen
spec).

The semantic_key algorithm (Section 10.2) is a pure function over a plain
Python dict, never over `model_dump()` of a Pydantic object directly
(Section 10.1) - the payload shape is a deliberate, narrow, versioned
contract (`semantic_key_algorithm_version`), not an accidental byproduct of
whatever fields a model happens to carry.
"""

import hashlib
import json
import re
import unicodedata
import uuid
from datetime import datetime
from typing import Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from molecular.inputs import validate_input_id
from molecular.scientific import MolecularRepresentationRef

PromotionStatus = Literal[
    "started",
    "resolved",
    "resolution_failed",
    "execution_failed",
    "succeeded",
]

PromotionPhase = Literal["resolution", "execution"]

_SEMANTIC_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# The complete error code taxonomy of Section 17 of the frozen spec. Every
# LigandPromotionError.code must be a member of this set - codes are never
# invented ad hoc by a caller, and no code here is wired to an exception
# class yet (that belongs to the modules that actually raise them: atomic.py,
# rdkit_tautomer.py, promotion_service.py - all later checkpoints).
PROMOTION_ERROR_CODES = frozenset(
    {
        "input-not-found",
        "unsupported-format",
        "unsafe-raw-path",
        "raw-not-regular-file",
        "raw-checksum-mismatch",
        "raw-nofollow-unavailable",
        "raw-identity-unstable",
        "raw-permission-denied",
        "rdkit-unavailable",
        "invalid-sdf",
        "record-count",
        "disconnected-structure",
        "revision-required",
        "identity-conflict",
        "reservation-conflict",
        "promotion-primitive-unavailable",
        "consistency-error",
        "publication-failed",
        "promotion-persistence-failed",
    }
)


def validate_semantic_key(value: str) -> str:
    """Validate that `value` is a lowercase, 64-character hexadecimal
    SHA-256 digest string - the only shape a semantic_key (Section 10.2,
    item 11 of the frozen spec) or a promotion record's `id` may ever take.
    Never uppercases, strips, or otherwise repairs the value: a malformed
    value is rejected outright, matching validate_input_id's own contract
    in molecular.inputs."""

    if not isinstance(value, str) or not _SEMANTIC_KEY_PATTERN.fullmatch(value):
        raise ValueError(
            "semantic_key must be a lowercase 64-character hexadecimal SHA-256 "
            f"digest (got {value!r})"
        )

    return value


def _validate_uuid4_hex(value: str) -> str:
    """Validate that `value` is the lowercase hex form of a UUID4 - the only
    shape a LigandPromotionAttempt.id may take (Section 6.7 of the frozen
    spec: "id é UUID4 hex validado")."""

    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"id must be a valid UUID4 hex string (got {value!r})") from exc

    if parsed.version != 4 or parsed.hex != value:
        raise ValueError(f"id must be a lowercase UUID4 hex string (got {value!r})")

    return value


def _validate_no_duplicates(values: tuple[str, ...], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicate entries")


def _normalize_payload_strings(value: object) -> object:
    """Recursively normalize every string leaf of `value` to Unicode NFC
    (Section 10.2, item 2 of the frozen spec), preserving dict/list
    structure and list order (item 4) untouched. Raises ValueError on any
    `None` leaf (item 5: "nenhum campo pode resolver para None") - a
    programming error in the caller, never silently serialized as `null`."""

    if value is None:
        raise ValueError("semantic_key payload must not contain a null value")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {key: _normalize_payload_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_payload_strings(item) for item in value]
    return value


def canonicalize_payload(payload: Mapping[str, object]) -> str:
    """Render `payload` as the canonical JSON string defined by Section
    10.2 of the frozen spec: NFC-normalized strings, recursively
    sort_keys=True, compact separators, non-ASCII preserved, NaN/Infinity
    rejected. This is a pure function over a plain dict - never over
    `model_dump()` of a Pydantic object (Section 10.1)."""

    normalized = _normalize_payload_strings(dict(payload))

    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def compute_semantic_key(payload: Mapping[str, object]) -> str:
    """Compute the semantic_key digest for `payload`: SHA-256 over the
    canonical JSON's UTF-8 bytes, as a lowercase 64-character hex string
    (Section 10.2, items 7 and 10-11 of the frozen spec)."""

    canonical_json = canonicalize_payload(payload)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_semantic_key_payload(
    *,
    input_id: str,
    snapshot_checksum: str,
    engine_name: str,
    engine_version: str,
    policy: "LigandPromotionPolicy",
) -> dict:
    """Build the semantic_key payload dict exactly as Section 10.1 of the
    frozen spec defines it: constructed field by field, never via
    `policy.model_dump()` or any other whole-object dump. Only the eight
    named policy fields participate - `schema_version` and `source` are
    deliberately excluded (Section 10.4)."""

    return {
        "semantic_key_algorithm_version": "1",
        "input_id": input_id,
        "snapshot_checksum": snapshot_checksum,
        "engine_name": engine_name,
        "engine_version": engine_version,
        "policy": {
            "sanitize": policy.sanitize,
            "remove_hs": policy.remove_hs,
            "strict_parsing": policy.strict_parsing,
            "record_policy": policy.record_policy,
            "component_policy": policy.component_policy,
            "charge_policy": policy.charge_policy,
            "stereo_policy": policy.stereo_policy,
            "representation_policy": policy.representation_policy,
        },
    }


class LigandPromotionPolicy(BaseModel):
    """A scientist's declared policy for promoting a ligand RAW artifact.

    `sanitize`, `remove_hs`, and `strict_parsing` are plain `bool` - the
    representable space is all 8 boolean combinations (Section 9.3 of the
    frozen spec); the default shown here is the only combination the
    production CLI will ever construct (Section 6.1/7.1). The 7 non-default
    combinations exist solely for spike/test exercise via this same model,
    never for a production caller of `promote_ligand_input`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    source: Literal["scientist"] = "scientist"
    sanitize: bool = True
    remove_hs: bool = False
    strict_parsing: bool = True
    record_policy: Literal["exactly-one"] = "exactly-one"
    component_policy: Literal["reject-disconnected"] = "reject-disconnected"
    charge_policy: Literal["preserve"] = "preserve"
    stereo_policy: Literal["preserve"] = "preserve"
    representation_policy: Literal["verified-source-bytes"] = "verified-source-bytes"


class ResolvedLigandPromotionSnapshot(BaseModel):
    """The concrete, verified outcome of resolving a RAW artifact into a
    private snapshot and inspecting it with RDKit (Section 6.3 of the
    frozen spec). `raw_checksum_persisted` and `snapshot_checksum` must be
    equal - both are SHA-256 hex digests of the RAW's registered checksum
    and of the bytes actually read from the snapshot, and the promotion
    contract requires them to match before this snapshot is ever
    considered valid (Section 8.3.2.1). None of these fields participate
    in the semantic_key payload except via `engine_name`, `engine_version`,
    and `snapshot_checksum`, passed explicitly (Section 10.1) - never via
    this model's own `model_dump()`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    engine_name: Literal["rdkit"]
    engine_version: str = Field(min_length=1)
    input_format: Literal["sdf"]
    raw_checksum_persisted: str
    snapshot_checksum: str
    snapshot_size_bytes: int = Field(ge=1)
    sanitize: bool
    remove_hs: bool
    strict_parsing: bool
    record_count: Literal[1]
    fragment_count: Literal[1]
    formal_charge: int
    atom_count: int = Field(ge=1)
    explicit_hydrogen_count: int = Field(ge=0)
    isotopic_hydrogen_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()

    @field_validator("raw_checksum_persisted", "snapshot_checksum")
    @classmethod
    def _validate_checksum_format(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                "checksum must be a lowercase 64-character hexadecimal SHA-256 "
                f"digest (got {value!r})"
            )
        return value

    @field_validator("warnings")
    @classmethod
    def _validate_warnings_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for warning in value:
            if not warning.strip():
                raise ValueError("warnings must not contain blank entries")
        return value

    @model_validator(mode="after")
    def _validate_checksums_match(self) -> "ResolvedLigandPromotionSnapshot":
        if self.raw_checksum_persisted != self.snapshot_checksum:
            raise ValueError(
                "raw_checksum_persisted and snapshot_checksum must be equal "
                f"(got {self.raw_checksum_persisted!r} and {self.snapshot_checksum!r})"
            )
        return self


class AcquisitionProvenanceSnapshot(BaseModel):
    """A copy of already-persisted acquisition provenance (Section 6.4 of
    the frozen spec) - this snapshot never performs a new external query;
    `pubchem_cid`/`pubchem_record_type` are only populated when already
    known from the input's own persisted provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    method: Literal["uploaded", "downloaded"]
    source_name: Optional[str] = None
    source_identifier: Optional[str] = None
    source_uri: Optional[str] = None
    acquired_at: datetime
    raw_artifact_path: str = Field(min_length=1)
    raw_artifact_format: str = Field(min_length=1)
    raw_artifact_checksum: str = Field(min_length=1)
    pubchem_cid: Optional[int] = Field(default=None, gt=0)
    pubchem_record_type: Optional[Literal["2d"]] = None


class PromotionResultSnapshot(BaseModel):
    """The concrete outcome of a completed promotion (Section 6.5 of the
    frozen spec). If `reused_existing_promotion` is True, this attempt did
    not create anything new - both `entity_created` and `state_created`
    must be False."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    promotion_id: str
    entity_id: str
    state_id: str
    representation: MolecularRepresentationRef
    entity_created: bool
    state_created: bool
    reused_existing_promotion: bool

    @field_validator("promotion_id")
    @classmethod
    def _validate_promotion_id_format(cls, value: str) -> str:
        return validate_semantic_key(value)

    @field_validator("entity_id", "state_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @model_validator(mode="after")
    def _validate_reuse_implies_nothing_created(self) -> "PromotionResultSnapshot":
        if self.reused_existing_promotion and (self.entity_created or self.state_created):
            raise ValueError(
                "reused_existing_promotion=True requires entity_created=False "
                "and state_created=False"
            )
        return self


class LigandPromotionError(BaseModel):
    """The provenance of a failure at one specific phase of a promotion
    attempt (Section 6.6 of the frozen spec). `code` must belong to the
    Section 17 taxonomy (`PROMOTION_ERROR_CODES`). Residual ids/keys are
    forbidden in phase="resolution" (nothing can have been published
    before resolution), mirroring ExecutionAttemptError's own rule in
    molecular.execution - extended here to three residual tuples instead
    of two."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    phase: PromotionPhase
    code: str
    message: str = Field(min_length=1)
    residual_entity_ids: tuple[str, ...] = ()
    residual_state_ids: tuple[str, ...] = ()
    residual_reservation_keys: tuple[str, ...] = ()

    @field_validator("code")
    @classmethod
    def _validate_code_in_taxonomy(cls, value: str) -> str:
        if value not in PROMOTION_ERROR_CODES:
            raise ValueError(
                f"code {value!r} is not a member of the Section 17 error taxonomy"
            )
        return value

    @field_validator("message")
    @classmethod
    def _validate_message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @field_validator("residual_entity_ids", "residual_state_ids")
    @classmethod
    def _validate_residual_domain_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for candidate in value:
            validate_input_id(candidate)
        return value

    @field_validator("residual_reservation_keys")
    @classmethod
    def _validate_residual_reservation_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for candidate in value:
            validate_input_id(candidate)
        return value

    @model_validator(mode="after")
    def _validate_residuals(self) -> "LigandPromotionError":
        _validate_no_duplicates(self.residual_entity_ids, field_name="residual_entity_ids")
        _validate_no_duplicates(self.residual_state_ids, field_name="residual_state_ids")
        _validate_no_duplicates(
            self.residual_reservation_keys, field_name="residual_reservation_keys"
        )

        if self.phase == "resolution":
            if (
                self.residual_entity_ids
                or self.residual_state_ids
                or self.residual_reservation_keys
            ):
                raise ValueError(
                    "a 'resolution'-phase error must not carry any residual ids/keys "
                    "(nothing can have been published before resolution)"
                )

        return self


class LigandPromotionAttempt(BaseModel):
    """One attempt by the platform to promote a LigandInput RAW artifact
    into a ScientificEntity/MolecularState pair (Section 6.7 of the frozen
    spec). `id` is always generated by the registry (uuid4().hex) - never
    supplied by a caller, never derived from scientific content, mirroring
    ExecutionAttempt.id in molecular.execution.

    None of these fields - including `requested_entity_id`,
    `requested_state_id`, `requested_entity_name`, this attempt's own `id`,
    or any timestamp - participate in the semantic_key payload (Section
    10.1/10.4); `semantic_key` itself is computed separately (Section
    10.2) once `resolved` is known, and only then persisted here.

    Instances are frozen: a status transition is never an in-place field
    assignment - see promotion_registry.py for how a persisted attempt
    moves from one status to the next by constructing a whole new
    validated snapshot and atomically overwriting the previous one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    id: str
    input_id: str
    requested_entity_id: str
    requested_state_id: str
    requested_entity_name: str = Field(min_length=1)
    intentional_revision: Optional[str] = None
    requested: LigandPromotionPolicy
    status: PromotionStatus
    semantic_key: Optional[str] = None
    resolved: Optional[ResolvedLigandPromotionSnapshot] = None
    result: Optional[PromotionResultSnapshot] = None
    error: Optional[LigandPromotionError] = None
    created_at: datetime
    resolved_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @field_validator("id")
    @classmethod
    def _validate_id_is_uuid4_hex(cls, value: str) -> str:
        return _validate_uuid4_hex(value)

    @field_validator("input_id", "requested_entity_id", "requested_state_id")
    @classmethod
    def _validate_domain_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("intentional_revision")
    @classmethod
    def _validate_intentional_revision_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return validate_input_id(value)

    @field_validator("semantic_key")
    @classmethod
    def _validate_semantic_key_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return validate_semantic_key(value)

    @model_validator(mode="after")
    def _validate_status_invariants(self) -> "LigandPromotionAttempt":
        """Enforce the Section 6.9 matrix: which of semantic_key/resolved/
        result/error/resolved_at/finished_at are required, forbidden, or
        optional at each status. Mirrors ExecutionAttempt._validate_status_
        invariants in molecular.execution."""

        if self.status == "started":
            if self.semantic_key is not None:
                raise ValueError("a 'started' attempt must not have semantic_key set")
            if self.resolved is not None:
                raise ValueError("a 'started' attempt must not have a resolved snapshot")
            if self.result is not None:
                raise ValueError("a 'started' attempt must not have a result")
            if self.error is not None:
                raise ValueError("a 'started' attempt must not have an error")
            if self.resolved_at is not None:
                raise ValueError("a 'started' attempt must not have resolved_at set")
            if self.finished_at is not None:
                raise ValueError("a 'started' attempt must not have finished_at set")

        elif self.status == "resolved":
            if self.semantic_key is None:
                raise ValueError("a 'resolved' attempt must have semantic_key set")
            if self.resolved is None:
                raise ValueError("a 'resolved' attempt must carry a resolved snapshot")
            if self.result is not None:
                raise ValueError("a 'resolved' attempt must not have a result")
            if self.error is not None:
                raise ValueError("a 'resolved' attempt must not have an error")
            if self.resolved_at is None:
                raise ValueError("a 'resolved' attempt must have resolved_at set")
            if self.finished_at is not None:
                raise ValueError("a 'resolved' attempt must not have finished_at set")

        elif self.status == "resolution_failed":
            if self.resolved is not None:
                raise ValueError(
                    "a 'resolution_failed' attempt must not have a resolved snapshot"
                )
            if self.result is not None:
                raise ValueError("a 'resolution_failed' attempt must not have a result")
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

        elif self.status == "execution_failed":
            if self.semantic_key is None:
                raise ValueError("an 'execution_failed' attempt must have semantic_key set")
            if self.resolved is None:
                raise ValueError(
                    "an 'execution_failed' attempt must carry a resolved snapshot"
                )
            if self.result is not None:
                raise ValueError("an 'execution_failed' attempt must not have a result")
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
            if self.semantic_key is None:
                raise ValueError("a 'succeeded' attempt must have semantic_key set")
            if self.resolved is None:
                raise ValueError("a 'succeeded' attempt must carry a resolved snapshot")
            if self.result is None:
                raise ValueError("a 'succeeded' attempt must carry a result")
            if self.error is not None:
                raise ValueError("a 'succeeded' attempt must not have an error")
            if self.resolved_at is None:
                raise ValueError("a 'succeeded' attempt must have resolved_at set")
            if self.finished_at is None:
                raise ValueError("a 'succeeded' attempt must have finished_at set")

        return self


class ScientificInputPromotionRecord(BaseModel):
    """The commit marker and formal lineage edge from a LigandInput RAW
    artifact to a ScientificEntity/MolecularState pair (Section 6.8 of the
    frozen spec). Exists only in success - there is no status, no error, no
    residual field on this model at all: its mere presence, once published,
    already means the promotion succeeded. `id` always equals
    `semantic_key`. Publication (never overwritten once it exists) is a
    persistence-layer guarantee - see promotion_registry.register_promotion
    - not something this model itself can enforce.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    id: str
    semantic_key: str
    input_id: str
    acquisition: AcquisitionProvenanceSnapshot
    policy: LigandPromotionPolicy
    resolved: ResolvedLigandPromotionSnapshot
    entity_id: str
    state_id: str
    representation: MolecularRepresentationRef
    created_by_attempt_id: str
    intentional_revision: Optional[str] = None
    created_at: datetime

    @field_validator("id", "semantic_key")
    @classmethod
    def _validate_semantic_key_fields(cls, value: str) -> str:
        return validate_semantic_key(value)

    @field_validator("input_id", "entity_id", "state_id")
    @classmethod
    def _validate_domain_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("created_by_attempt_id")
    @classmethod
    def _validate_created_by_attempt_id(cls, value: str) -> str:
        return _validate_uuid4_hex(value)

    @field_validator("intentional_revision")
    @classmethod
    def _validate_intentional_revision_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return validate_input_id(value)

    @model_validator(mode="after")
    def _validate_identity_and_representation(self) -> "ScientificInputPromotionRecord":
        if self.id != self.semantic_key:
            raise ValueError(
                f"id must equal semantic_key (got id={self.id!r}, "
                f"semantic_key={self.semantic_key!r})"
            )

        if self.policy.representation_policy == "verified-source-bytes":
            if self.representation.checksum != self.resolved.snapshot_checksum:
                raise ValueError(
                    "representation.checksum must equal resolved.snapshot_checksum "
                    "when representation_policy is 'verified-source-bytes' "
                    f"(got {self.representation.checksum!r} and "
                    f"{self.resolved.snapshot_checksum!r})"
                )

        return self
