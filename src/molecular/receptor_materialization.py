"""Domain model, canonicalization and pure planner/resolver for receptor
materialization (Increment 021, Checkpoints 1-2 - frozen specification at
increments/021/15-final-specification/INCREMENT_021_FINAL_SPECIFICATION.md,
governance commit b0e60f6528f6fcb828aa6dd5577824e08c7c9fac).

This module is the *domain-only* slice of Increment 021: DTOs, hash
canonicalization, the pure planner that derives the S/D/U/X/O decisions
(Sections 7-8), the `struct_conn`/symmetry/IdentityBasis resolver
(`resolve_connection_decisions` - Sections 9-10/10.1), the pinned
94-category catalogue engine (`compute_category_decisions` - Section 11),
and the acknowledgement-checked `PolicyRecord` (Section 12-13). It never
imports `gemmi`/`rdkit`, never opens a RAW artifact or a 020 YAML record,
and never writes to storage - those are Checkpoint 2's adapter boundary
(`receptor_parsing.materialize_receptor_cif021`, the only place both
`gemmi` and this module are imported together), Checkpoint 3 (storage/
registry/service) and Checkpoint 4 (CLI) responsibilities (Section 19).

`AtomObservation`, `RawConnectionRow`, `SymmetryOperatorTables` and
`RawCategoryTable` are the explicit boundary: the Checkpoint 2 Gemmi
adapter in `receptor_parsing.py` is the only code that ever constructs
these from a real RAW document (via a late, local reference to `gemmi`
inside that other, dedicated module - never from this one). Domain tests
construct them directly, in memory, to exercise the planner/resolver/
catalogue engine - never a stub that fabricates a constant "scientific
success".

Deliberately not implemented here (Checkpoint 3-4, Section 19): `Summary`/
`PreviewOutcome`/`ManifestRecord`/`AuditRecord`/`ArtifactOutcome` (they
need real output byte counts and a persisted audit trail Checkpoint 2
never writes to storage), storage/reservation/registry, and the CLI.

All new records are frozen, `extra="forbid"`, and constructed with
`strict=True` (Section 12: "rejeitam coerção: str/bytes, bool/int, int/str
e enum inválido não passam por conversão conveniente") - Pydantic v2's lax
mode alone does not reject `bool` for `int` fields nor `bytes` for `str`
fields, so `strict=True` is required on every model here, not merely
`frozen`/`extra="forbid"`.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Literal, Mapping, NamedTuple, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from molecular.inputs import ProteinInput, validate_input_id
from molecular.promotion import compute_semantic_key
from molecular.receptor_selection import ReceptorSelectionPolicy

if TYPE_CHECKING:
    # ARCH-01 (corrections reconciliation 521500c): `receptor_inspection` and
    # `receptor_resolution` both transitively import Gemmi via
    # `receptor_parsing` (`import gemmi` at module level). This pure domain
    # module must stay importable with Gemmi/RDKit entirely absent (Section
    # 3/19; T021-040), so these two names are annotation-only - never
    # imported at runtime. `from __future__ import annotations` (above)
    # already makes every annotation a string; nothing here ever calls
    # `typing.get_type_hints()` on them. Duck-typed attribute access on the
    # real objects a caller passes in is all this module ever needs.
    from molecular.receptor_inspection import ReceptorInspectionReport
    from molecular.receptor_resolution import ResolvedReceptorSelection

MATERIALIZATION_SCHEMA_VERSION = "1.0"
MATERIALIZATION_PROFILE_VERSION = "1.0"
MATERIALIZATION_ALGORITHM_VERSION = "1.0"

# The frozen bytes of increments/021/15-final-specification/evidence/
# CATALOG_021.json (Section 1/11 of the frozen spec: "O catalogue_sha256 é
# dos bytes exatos do anexo publicado"). Checkpoint 1 pins this identity
# without reading or applying the catalogue itself - that is Checkpoint 2.
CATALOGUE_SHA256 = "8561039fd15137811b60316336dff04539df3732710dca6ec4009f23150d6d03"

ENGINE_NAME = "gemmi"
ENGINE_VERSION = "0.7.5"

# MAJOR-A residual (reconciliation 155661115851c2a5fb70f5d905425baeb4d23c9f):
# the same declared-format label aliasing the 020 baseline already applies
# (receptor_inspection.py's `_DECLARED_FORMAT_ALIASES`, `raw_format_declared.
# strip().lower()`) - projected here, not imported, so this module never
# loads receptor_inspection at runtime (ARCH-01). This alias rule is for
# format *labels* only, never for scientific tokens/ids/checksums.
_DECLARED_FORMAT_ALIASES = {"cif": "mmcif", "mmcif": "mmcif", "pdb": "pdb", "ent": "pdb"}

# The 94 canonical category names of the pinned catalogue (CATALOGUE_SHA256
# above) plus the three explicitly approved omission-family prefixes
# (Section 11: "_pdbx_validate_, _pdbx_nmr_ e _em_"). MINOR-3: a
# `CategoryDecision.category` must be one or the other - never an arbitrary
# string - even though CP1 never reads the catalogue file itself.
_CATALOGUE_CATEGORY_NAMES = frozenset(
    {
        "_atom_site", "_atom_site_anisotrop", "_atom_sites", "_atom_type", "_audit_author", "_audit_conform",
        "_cell", "_chem_comp", "_chem_comp_atom", "_chem_comp_bond", "_citation", "_citation_author",
        "_database_2", "_database_pdb_matrix", "_diffrn", "_diffrn_detector", "_diffrn_radiation",
        "_diffrn_radiation_wavelength", "_diffrn_source", "_entity", "_entity_name_com", "_entity_poly",
        "_entity_poly_seq", "_entity_src_gen", "_entity_src_nat", "_entry", "_exptl", "_exptl_crystal",
        "_exptl_crystal_grow", "_exptl_crystal_grow_comp", "_ndb_struct_conf_na", "_ndb_struct_na_base_pair",
        "_ndb_struct_na_base_pair_step", "_pdbx_audit_revision_details", "_pdbx_audit_revision_group",
        "_pdbx_audit_revision_history", "_pdbx_audit_support", "_pdbx_branch_scheme",
        "_pdbx_database_pdb_obs_spr", "_pdbx_database_related", "_pdbx_database_status",
        "_pdbx_entity_branch", "_pdbx_entity_branch_link", "_pdbx_entity_branch_list", "_pdbx_entity_nonpoly",
        "_pdbx_entity_src_syn", "_pdbx_entry_details", "_pdbx_molecule", "_pdbx_molecule_features",
        "_pdbx_nonpoly_scheme", "_pdbx_poly_seq_scheme", "_pdbx_refine", "_pdbx_refine_tls",
        "_pdbx_refine_tls_group", "_pdbx_struct_assembly", "_pdbx_struct_assembly_gen",
        "_pdbx_struct_assembly_prop", "_pdbx_struct_conn_angle", "_pdbx_struct_mod_residue",
        "_pdbx_struct_oper_list", "_pdbx_struct_special_symmetry", "_pdbx_unobs_or_zero_occ_atoms",
        "_pdbx_unobs_or_zero_occ_residues", "_refine", "_refine_analyze", "_refine_hist",
        "_refine_ls_restr", "_refine_ls_shell", "_reflns", "_reflns_shell", "_software",
        "_space_group_symop", "_struct", "_struct_asym", "_struct_biol", "_struct_conf",
        "_struct_conf_type", "_struct_conn", "_struct_conn_type", "_struct_keywords",
        "_struct_mon_prot_cis", "_struct_ncs_oper", "_struct_ref", "_struct_ref_seq",
        "_struct_ref_seq_dif", "_struct_sheet", "_struct_sheet_hbond", "_struct_sheet_order",
        "_struct_sheet_range", "_struct_site", "_struct_site_gen", "_struct_site_keywords",
        "_symmetry", "_symmetry_equiv",
    }
)
# MINOR-3 residual (reconciliation 155661115851c2a5fb70f5d905425baeb4d23c9f):
# a delimited category-name grammar for the three approved omission
# families - lowercase/digits/underscore only, after the prefix, with at
# least one such character. `re.fullmatch` (never `startswith` alone, never
# a `$`-anchored pattern that would admit a trailing LF) rejects a bare
# prefix, an item name (contains '.'), whitespace/control characters and
# uppercase. This is not a general CIF category parser.
_CATALOGUE_OMISSION_FAMILY_PATTERN = re.compile(r"(?:_pdbx_validate_|_pdbx_nmr_|_em_)[a-z0-9_]+")

_HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TIME_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


UnaddressedComponentsPolicy = Literal["preserve", "require-disposition"]
EntityTypeToken = Literal["polymer", "non-polymer", "macrolide", "water", "branched", "unknown", "unrecognized"]
AtomReason = Literal[
    "model-filter", "selected-asym", "disposition-keep", "disposition-remove",
    "unaddressed-preserve", "chain-complement",
]
DecisionSource = Literal["disposition-keep", "unaddressed-preserve"]
AuthorIdCoincidence = Literal["retained-polymer", "excluded-polymer", "multiple", "none", "unavailable"]
RecordKind = Literal["protein", "inspection", "selection", "resolution"]
CategoryAction = Literal["filter", "omit", "generate", "guard_then_omit"]
WarningCode = Literal[
    "all-altlocs-retained", "outside-selected-asym-retained", "declared-connections-unavailable",
    "source-annotations-omitted", "coordination-connections-omitted", "hydrogen-bonds-omitted",
    "source-symmetry-context-only", "other-models-excluded", "source-ncs-annotations-omitted",
    "special-position-annotations-omitted",
]

_RECOGNIZED_ENTITY_TYPES = frozenset({"polymer", "non-polymer", "macrolide", "water", "branched"})


class MaterializationInputError(ValueError):
    """A scientist-facing input/consent error (Section 16: `code` belongs to
    `MaterializationInputError`'s closed set - INVALID_ID, MISSING_ACK,
    ACK_MISMATCH, SELECTION_CONFLICT, NONUNIFORM_ASYM, EMPTY_SELECTION,
    UNSUPPORTED_REQUEST). Raised by the pure planner and by
    `build_policy_record`; never by a DTO's own `__init__` (that is always
    a plain pydantic `ValidationError`)."""

    _CODES = frozenset(
        {
            "INVALID_ID", "MISSING_ACK", "ACK_MISMATCH", "SELECTION_CONFLICT",
            "NONUNIFORM_ASYM", "EMPTY_SELECTION", "UNSUPPORTED_REQUEST",
        }
    )

    def __init__(self, code: str, message: str, *, phase: str = "plan") -> None:
        # MINOR-6 (corrections reconciliation 521500c): an explicit `raise`,
        # never `assert` - `assert` is stripped under `python -O`, which
        # would silently accept a code outside the closed set in production.
        if code not in self._CODES:
            raise ValueError(f"unknown MaterializationInputError code {code!r}")
        super().__init__(message)
        self.code = code
        self.phase = phase


class MaterializationConfigurationError(ValueError):
    """A configuration/identity error over the four captured source records
    (Section 16: INVALID_RECORD, IDENTITY_MISMATCH, SOURCE_CHANGED,
    CHECKSUM_MISMATCH, INCOMPLETE_INSPECTION, ENGINE_UNAVAILABLE,
    ENGINE_VERSION, UNSUPPORTED_PLATFORM). Checkpoint 1 only ever raises
    the subset that a pure, in-memory cross-check of already-loaded
    records can detect; the storage-level codes (ENGINE_UNAVAILABLE,
    UNSUPPORTED_PLATFORM) belong to Checkpoint 2-3's adapter/host
    detection and are never raised from this module."""

    _CODES = frozenset(
        {
            "INVALID_RECORD", "IDENTITY_MISMATCH", "SOURCE_CHANGED", "CHECKSUM_MISMATCH",
            "INCOMPLETE_INSPECTION", "ENGINE_UNAVAILABLE", "ENGINE_VERSION", "UNSUPPORTED_PLATFORM",
        }
    )

    def __init__(self, code: str, message: str, *, phase: str = "metadata") -> None:
        # MINOR-6: see MaterializationInputError.__init__ - `raise`, never
        # `assert`, so this survives `python -O`.
        if code not in self._CODES:
            raise ValueError(f"unknown MaterializationConfigurationError code {code!r}")
        super().__init__(message)
        self.code = code
        self.phase = phase


# --------------------------------------------------------------------------
# Section 1/13: shared value validation and canonical-JSON hashing
# --------------------------------------------------------------------------


def validate_hex64(value: str) -> str:
    """Hex64 (SCHEMAS Section 1): a real `str` of exactly 64 lowercase
    hexadecimal characters - never uppercase, never a numeric type."""

    if not isinstance(value, str) or not _HEX64_PATTERN.fullmatch(value):
        raise ValueError(f"expected a lowercase 64-character hexadecimal digest (got {value!r})")
    return value


def validate_native_id(value: str) -> str:
    """NativeId (SCHEMAS Section 1): a non-empty, case-preserving `str`
    decoded from the CIF DOM, up to 256 characters, with no C0/C1 control
    characters. Never a filesystem path component - `validate_input_id`
    is the wrong validator for this family of fields."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"NativeId must be a non-empty str (got {value!r})")
    if len(value) > 256:
        raise ValueError(f"NativeId must not exceed 256 characters (got {len(value)})")
    if any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in value):
        raise ValueError("NativeId must not contain C0/C1 control characters")
    return value


def validate_time_utc(value: str) -> str:
    """TimeUTC (SCHEMAS Section 1): a real `str` of the exact form
    `YYYY-MM-DDTHH:MM:SS.ffffffZ`, naming a real calendar date/time - never
    a naive `datetime` object, and never an impossible date such as a
    February 30th (T021-004)."""

    if not isinstance(value, str) or not _TIME_UTC_PATTERN.fullmatch(value):
        raise ValueError(
            f"expected TimeUTC in YYYY-MM-DDTHH:MM:SS.ffffffZ form (got {value!r})"
        )
    try:
        datetime.strptime(value[:-1], "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError as exc:
        raise ValueError(f"TimeUTC value is not a real calendar date/time (got {value!r})") from exc
    return value


def _reject_float_leaves(value: object) -> None:
    """Section 13: "JSON só usa null/bool/int/str/arrays/objetos; sem
    floats" - a plain `json.dumps(allow_nan=False)` only rejects NaN/Inf,
    never an ordinary finite float (T021-033), so every leaf is walked
    explicitly here. `bool` is intentionally not rejected - it is not a
    `float` instance."""

    if isinstance(value, float):
        raise ValueError(f"materialization hash payloads must not contain float values (got {value!r})")
    if isinstance(value, dict):
        for item in value.values():
            _reject_float_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_float_leaves(item)


def canonicalize_materialization_payload(payload: dict) -> str:
    """H(payload)'s canonical JSON rendering (Section 13): UTF-8,
    `sort_keys=True` (object key order never matters, T021-032),
    compact separators, non-ASCII preserved, no float anywhere, no trailing
    LF. List order is never sorted - it is preserved exactly as given,
    because list order carries scientific meaning (Section 13)."""

    _reject_float_leaves(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def compute_materialization_hash(payload: dict) -> str:
    """H(payload) = SHA-256 of the canonical JSON's UTF-8 bytes (Section
    13), as a lowercase 64-character hex digest."""

    canonical = canonicalize_materialization_payload(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_LEGACY_INSPECTION_SCHEMA_VERSION = "1.0"
_LEGACY_INSPECTION_PROFILE_VERSION = "1.0"


def _compute_legacy_inspection_id(
    *, parser_name: str, parser_version: str, protein_input_id: str, raw_checksum: str, raw_format: str,
) -> str:
    """ARCH-01 (corrections reconciliation 521500c): reproduces
    `receptor_inspection.compute_inspection_id`'s exact seven-field payload
    and exact backend (`promotion.compute_semantic_key` - NFC-normalizing,
    per the 020 frozen spec Section 15.1-15.2) *without* importing
    `receptor_inspection`, which transitively pulls Gemmi. This is a
    narrow, versioned duplication of one seven-key payload projection, not
    a reimplementation of inspection logic, and it deliberately keeps the
    legacy NFC-normalizing canonicalization - never the 021 token-preserving
    `compute_materialization_hash` - because identity compatibility with
    the already-published `inspection_id` values requires the exact same
    bytes, not a new algorithm."""

    payload = {
        "inspection_schema_version": _LEGACY_INSPECTION_SCHEMA_VERSION,
        "inspection_profile_version": _LEGACY_INSPECTION_PROFILE_VERSION,
        "parser_name": parser_name,
        "parser_version": parser_version,
        "protein_input_id": protein_input_id,
        "raw_checksum": raw_checksum,
        "raw_format": raw_format,
    }
    return compute_semantic_key(payload)


def _build_nfc_injective_index(tokens: Sequence[str]) -> Dict[str, str]:
    """Section 6: "Uma normalização NFC... só pode servir à correspondência
    se for injetiva entre tokens fonte envolvidos; colisão de tokens
    distintos é recusa, nunca fusão" (T021-013). Returns NFC form -> the
    one source token observed for it; raises if two *distinct* source
    tokens collide under NFC."""

    index: Dict[str, str] = {}
    for token in tokens:
        nfc = unicodedata.normalize("NFC", token)
        existing = index.get(nfc)
        if existing is not None and existing != token:
            raise MaterializationInputError(
                "UNSUPPORTED_REQUEST",
                f"distinct source tokens {existing!r} and {token!r} collide under NFC "
                f"normalization ({nfc!r}); this is an ambiguity, never a silent merge",
            )
        index[nfc] = token
    return index


# --------------------------------------------------------------------------
# Section 2/13: Engine and RecordSnapshot
# --------------------------------------------------------------------------


class Engine(BaseModel):
    """Engine (SCHEMAS Section 1): pinned exactly to gemmi/0.7.5 (Section
    3). A divergent name/version is rejected by the closed `Literal`
    itself, never coerced (T021-005)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: Literal["gemmi"] = "gemmi"
    version: Literal["0.7.5"] = "0.7.5"


class RecordSnapshot(BaseModel):
    """RecordSnapshot (SCHEMAS Section 2): the documentary identity of one
    of the four captured 020 records. `schema_version` is `null` only for
    `kind="protein"` (`ProteinInput` has no schema_version field); it is
    `"1.0"` for the other three kinds."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: RecordKind
    id: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    schema_version: Optional[Literal["1.0"]] = None

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        return validate_hex64(value)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str, info) -> str:
        kind = info.data.get("kind")
        if kind == "inspection":
            return validate_hex64(value)
        return validate_input_id(value)

    @model_validator(mode="after")
    def _validate_schema_version_rule(self) -> "RecordSnapshot":
        if self.kind == "protein":
            if self.schema_version is not None:
                raise ValueError("a ProteinInput RecordSnapshot must have schema_version=null")
        elif self.schema_version != "1.0":
            raise ValueError(f"a {self.kind!r} RecordSnapshot requires schema_version='1.0'")
        return self


_RECORD_KIND_ORDER: Tuple[RecordKind, ...] = ("protein", "inspection", "selection", "resolution")


class SourceContext(BaseModel):
    """SourceContext (SCHEMAS Section 2): the documentary snapshot of the
    four captured 020 records plus the RAW identity and the resolved
    model. Its own `model_validator` enforces the *shape* invariant
    (exactly four snapshots, in order, each cross-referencing this
    object's own ids); the full cross-record authenticity check (against a
    `ReceptorInspectionReport`/`ReceptorSelectionPolicy`/
    `ResolvedReceptorSelection` triplet) is `build_source_context`'s job,
    not this constructor's."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    protein_input_id: str = Field(min_length=1)
    selection_id: str = Field(min_length=1)
    inspection_id: str = Field(min_length=64, max_length=64)
    model: int = Field(ge=1)
    records: Tuple[RecordSnapshot, ...]
    raw_sha256: str = Field(min_length=64, max_length=64)
    raw_size_bytes: int = Field(ge=0)
    raw_format: Literal["mmcif"] = "mmcif"
    inspection_schema: Literal["1.0"] = "1.0"
    inspection_profile: Literal["1.0"] = "1.0"
    parser: Engine

    @field_validator("protein_input_id", "selection_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("inspection_id", "raw_sha256")
    @classmethod
    def _validate_hex_fields(cls, value: str) -> str:
        return validate_hex64(value)

    @model_validator(mode="after")
    def _validate_records_shape(self) -> "SourceContext":
        kinds = tuple(record.kind for record in self.records)
        if kinds != _RECORD_KIND_ORDER:
            raise ValueError(
                f"SourceContext.records must carry exactly the four kinds in order "
                f"{_RECORD_KIND_ORDER}, got {kinds}"
            )
        by_kind = {record.kind: record for record in self.records}
        if by_kind["protein"].id != self.protein_input_id:
            raise ValueError("the protein RecordSnapshot.id must equal protein_input_id")
        if by_kind["inspection"].id != self.inspection_id:
            raise ValueError("the inspection RecordSnapshot.id must equal inspection_id")
        if by_kind["selection"].id != self.selection_id:
            raise ValueError("the selection RecordSnapshot.id must equal selection_id")
        if by_kind["resolution"].id != self.selection_id:
            raise ValueError(
                "the resolution RecordSnapshot.id must equal selection_id "
                "(resolved_selection_id == policy_selection_id is already a 020 invariant)"
            )
        return self


def compute_source_context_key(source_context: SourceContext) -> str:
    """source_context_key = H({kind:'receptor-materialization-sources',
    version:'1.0', sources:SourceContext}) (Section 2/13)."""

    payload = {
        "kind": "receptor-materialization-sources",
        "version": "1.0",
        "sources": source_context.model_dump(mode="json"),
    }
    return compute_materialization_hash(payload)


def build_source_context(
    *,
    protein_input: ProteinInput,
    inspection_report: ReceptorInspectionReport,
    policy: ReceptorSelectionPolicy,
    resolved_selection: ResolvedReceptorSelection,
    raw_size_bytes: int,
    records: Tuple[RecordSnapshot, ...],
) -> SourceContext:
    """The pure, in-memory cross-check of the four already-captured 020
    records (Section 5-6, NOTE-F1 minus the actual byte capture - that is
    Checkpoint 2-3's storage reader). Never opens a file and never calls a
    020 loader; every argument is an already-validated, in-memory 020
    record the (future) storage reader will have loaded, or a plain scalar
    it will have measured.

    MAJOR-A (corrections reconciliation 521500c): receives the full typed
    `ProteinInput`/`ReceptorSelectionPolicy` - not bare scalars - so every
    field the frozen spec's Section 5 names ("conferir protein_input_id,
    inspection_id, checksums acquisition/raw_checksum_observed/raw_checksum,
    formato e parser") is actually cross-checked, including the inspection's
    own parser (previously only `resolved_selection.parser` was checked),
    the resolution's `structure_format` against the inspection's, the
    policy's own `protein_input_id`, and the `ProteinInput.acquisition.
    raw_artifact.checksum` against the observed raw_checksum - none of
    which can be inferred from the records already being cross-checked."""

    # Order: engine -> sources -> selection (Section 16), preserved exactly.
    if inspection_report.parser.name != ENGINE_NAME or inspection_report.parser.version != ENGINE_VERSION:
        raise MaterializationConfigurationError(
            "ENGINE_VERSION",
            f"inspection parser {inspection_report.parser.name}/{inspection_report.parser.version} "
            f"does not match the pinned {ENGINE_NAME}/{ENGINE_VERSION}",
        )

    if resolved_selection.parser.name != ENGINE_NAME or resolved_selection.parser.version != ENGINE_VERSION:
        raise MaterializationConfigurationError(
            "ENGINE_VERSION",
            f"resolved selection parser {resolved_selection.parser.name}/{resolved_selection.parser.version} "
            f"does not match the pinned {ENGINE_NAME}/{ENGINE_VERSION}",
        )

    if inspection_report.inspection_schema_version != "1.0" or inspection_report.inspection_profile_version != "1.0":
        raise MaterializationConfigurationError(
            "INVALID_RECORD",
            "inspection_schema_version/inspection_profile_version must both be '1.0'",
        )

    if inspection_report.structure_format != "mmcif":
        raise MaterializationConfigurationError(
            "INVALID_RECORD",
            f"receptor materialization requires full mmCIF input, got {inspection_report.structure_format!r}",
        )

    if resolved_selection.structure_format != inspection_report.structure_format:
        raise MaterializationConfigurationError(
            "INVALID_RECORD",
            f"resolved selection structure_format {resolved_selection.structure_format!r} disagrees "
            f"with the inspection's {inspection_report.structure_format!r}",
        )

    # MAJOR-A residual: cross-check the acquisition's own declared format
    # label and the inspection's `raw_format_declared` - neither was
    # compared before. Both are normalized via the 020 alias rule
    # (strip().lower() + the same four-entry map); an unrecognized label on
    # either side, or a recognized-but-contradictory pair between them or
    # against the detected canonical mmcif, is a configuration error. The
    # `declared-format-mismatch` warning the 020 inspection may already
    # carry is never read as authorization to materialize with
    # contradictory provenance - nothing here is rewritten or reclassified.
    acquisition_format = _DECLARED_FORMAT_ALIASES.get(
        protein_input.acquisition.raw_artifact.format.strip().lower()
    )
    declared_format = _DECLARED_FORMAT_ALIASES.get(
        inspection_report.raw_format_declared.strip().lower()
    )
    if (
        acquisition_format is None
        or declared_format is None
        or acquisition_format != declared_format
        or acquisition_format != "mmcif"
    ):
        raise MaterializationConfigurationError(
            "INVALID_RECORD",
            f"acquisition format {protein_input.acquisition.raw_artifact.format!r} and declared "
            f"format {inspection_report.raw_format_declared!r} must both be recognized aliases of "
            "the detected canonical 'mmcif', and must agree with each other",
        )

    if inspection_report.selection_capability != "full":
        raise MaterializationConfigurationError(
            "INCOMPLETE_INSPECTION",
            f"selection_capability must be 'full', got {inspection_report.selection_capability!r}",
        )

    if any(warning.code == "parser-limitation" for warning in inspection_report.warnings):
        raise MaterializationConfigurationError(
            "INCOMPLETE_INSPECTION",
            "the referenced inspection report carries a parser-limitation warning",
        )

    if not (
        protein_input.id
        == inspection_report.protein_input_id
        == resolved_selection.protein_input_id
        == policy.protein_input_id
    ):
        raise MaterializationConfigurationError(
            "IDENTITY_MISMATCH", "protein_input_id disagrees across the captured records"
        )

    if not (policy.selection_id == resolved_selection.policy_selection_id == resolved_selection.resolved_selection_id):
        raise MaterializationConfigurationError(
            "IDENTITY_MISMATCH", "selection_id disagrees across the captured records"
        )

    if not (policy.inspection_id == inspection_report.inspection_id == resolved_selection.inspection_id):
        raise MaterializationConfigurationError(
            "IDENTITY_MISMATCH", "inspection_id disagrees across the captured records"
        )

    if not (
        protein_input.acquisition.raw_artifact.checksum
        == policy.raw_checksum_observed
        == inspection_report.raw_checksum
        == resolved_selection.raw_checksum
    ):
        raise MaterializationConfigurationError(
            "CHECKSUM_MISMATCH", "raw_checksum disagrees across the captured records"
        )

    recomputed_inspection_id = _compute_legacy_inspection_id(
        parser_name=inspection_report.parser.name,
        parser_version=inspection_report.parser.version,
        protein_input_id=protein_input.id,
        raw_checksum=inspection_report.raw_checksum,
        raw_format=inspection_report.structure_format,
    )
    if recomputed_inspection_id != inspection_report.inspection_id:
        raise MaterializationConfigurationError(
            "IDENTITY_MISMATCH",
            "the inspection payload does not reproduce its own declared inspection_id",
        )

    if resolved_selection.resolved_assembly is not None:
        raise MaterializationInputError(
            "UNSUPPORTED_REQUEST",
            "receptor materialization does not apply assembly operators (resolved_assembly must be null)",
        )

    if not resolved_selection.resolved_chains:
        raise MaterializationInputError(
            "EMPTY_SELECTION",
            "resolved_chains is empty; receptor materialization never selects everything by default",
        )

    return SourceContext(
        protein_input_id=protein_input.id,
        selection_id=policy.selection_id,
        inspection_id=inspection_report.inspection_id,
        model=resolved_selection.resolved_model,
        records=records,
        raw_sha256=inspection_report.raw_checksum,
        raw_size_bytes=raw_size_bytes,
        raw_format="mmcif",
        inspection_schema=inspection_report.inspection_schema_version,
        inspection_profile=inspection_report.inspection_profile_version,
        parser=Engine(name=resolved_selection.parser.name, version=resolved_selection.parser.version),
    )


# --------------------------------------------------------------------------
# Section 3/4: OccurrenceIdentity, XEntry, OutsideEntry, AtomDecision
# --------------------------------------------------------------------------


class OccurrenceIdentity(BaseModel):
    """OccurrenceIdentity (SCHEMAS Section 3): the residue-occurrence
    identity of Section 6 - (model, label_asym_id, ccd, label_seq_id when
    not null; otherwise auth_seq_id + insertion_token), with auth_asym_id
    kept only as a consistency reference, never as biological identity."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: int = Field(ge=1)
    label_asym_id: str = Field(min_length=1, max_length=256)
    entity_id: str = Field(min_length=1, max_length=256)
    ccd: str = Field(min_length=1, max_length=256)
    label_seq_id: Optional[str] = None
    auth_asym_id: Optional[str] = None
    auth_seq_id: Optional[str] = None
    insertion_token: Optional[str] = None

    @field_validator("label_asym_id", "entity_id", "ccd", "label_seq_id", "auth_asym_id", "auth_seq_id")
    @classmethod
    def _validate_native_ids(cls, value: Optional[str]) -> Optional[str]:
        return value if value is None else validate_native_id(value)

    @model_validator(mode="after")
    def _validate_distinguishable(self) -> "OccurrenceIdentity":
        if self.label_seq_id is None and self.auth_seq_id is None:
            raise ValueError(
                "an occurrence must be distinguishable by label_seq_id, or else by "
                "auth_seq_id (optionally with insertion_token); both were absent"
            )
        return self


def compute_occurrence_key(identity: OccurrenceIdentity) -> str:
    """occurrence key = H({kind:'receptor-materialization-occurrence',
    version:'1.0', identity:OccurrenceIdentity}) (SCHEMAS Section 3)."""

    payload = {
        "kind": "receptor-materialization-occurrence",
        "version": "1.0",
        "identity": identity.model_dump(mode="json"),
    }
    return compute_materialization_hash(payload)


class AtomObservation(BaseModel):
    """One `atom_site` row, from any model, as the (future) Checkpoint 2
    Gemmi adapter will hand it to this pure planner. Checkpoint 1 domain
    tests construct this directly in memory - never from a real RAW
    artifact, never a stand-in that always reports scientific success.

    `entity_type` deliberately keeps `"unknown"`/`"unrecognized"` as valid
    *values* here (Section 7: "Tipos unknown/unrecognized... são
    recusados") - the DTO accepts them so the planner itself can reject
    them explicitly (T021-039), rather than the DTO silently narrowing the
    vocabulary before the planner ever sees the case it must refuse."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    ordinal: int = Field(ge=0)
    atom_id: str = Field(min_length=1, max_length=256)
    model: int = Field(ge=1)
    label_asym_id: str = Field(min_length=1, max_length=256)
    entity_id: str = Field(min_length=1, max_length=256)
    entity_type: EntityTypeToken
    ccd: str = Field(min_length=1, max_length=256)
    label_seq_id: Optional[str] = None
    auth_asym_id: Optional[str] = None
    auth_seq_id: Optional[str] = None
    insertion_token: Optional[str] = None
    # Checkpoint 2 additions (Section 9: struct_conn endpoint resolution
    # needs the atom's chemical name and altloc code, neither of which
    # Checkpoint 1's pure S/D/U/X/O planner ever reads). Optional, default
    # `None`, so every Checkpoint 1 fixture/test - none of which sets
    # these - keeps constructing unchanged.
    label_atom_id: Optional[str] = None
    altloc: Optional[str] = None

    @field_validator(
        "atom_id", "label_asym_id", "entity_id", "ccd", "label_seq_id", "auth_asym_id", "auth_seq_id",
        "label_atom_id", "altloc",
    )
    @classmethod
    def _validate_native_ids(cls, value: Optional[str]) -> Optional[str]:
        return value if value is None else validate_native_id(value)


def _build_occurrence_identity(atom: AtomObservation) -> OccurrenceIdentity:
    return OccurrenceIdentity(
        model=atom.model,
        label_asym_id=atom.label_asym_id,
        entity_id=atom.entity_id,
        ccd=atom.ccd,
        label_seq_id=atom.label_seq_id,
        auth_asym_id=atom.auth_asym_id,
        auth_seq_id=atom.auth_seq_id,
        insertion_token=atom.insertion_token,
    )


def _occurrence_group_key(atom: AtomObservation):
    if atom.label_seq_id is not None:
        seq_component = ("label", atom.label_seq_id)
    else:
        seq_component = ("auth", atom.auth_seq_id, atom.insertion_token)
    return (atom.model, atom.label_asym_id, atom.ccd, seq_component)


def compute_atom_ids_key(atoms: Sequence[AtomObservation]) -> str:
    """atom_ids_key = H({kind:'receptor-materialization-atom-ids',
    version:'1.0', atoms:[{ordinal,id},...]}) (SCHEMAS Section 4), in
    source ordinal order - never re-sorted by id."""

    payload = {
        "kind": "receptor-materialization-atom-ids",
        "version": "1.0",
        "atoms": [{"ordinal": atom.ordinal, "id": atom.atom_id} for atom in atoms],
    }
    return compute_materialization_hash(payload)


class XEntry(BaseModel):
    """XEntry (SCHEMAS Section 3): one whole polymer `label_asym_id`
    excluded from M by the complement of S."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: int = Field(ge=1)
    label_asym_id: str = Field(min_length=1, max_length=256)
    entity_id: str = Field(min_length=1, max_length=256)
    atom_count: int = Field(ge=1)
    atom_ids_key: str = Field(min_length=64, max_length=64)

    @field_validator("label_asym_id", "entity_id")
    @classmethod
    def _validate_native_ids(cls, value: str) -> str:
        return validate_native_id(value)

    @field_validator("atom_ids_key")
    @classmethod
    def _validate_hex(cls, value: str) -> str:
        return validate_hex64(value)


class OutsideEntry(BaseModel):
    """OutsideEntry (SCHEMAS Section 3): one retained occurrence of an
    asym outside S, kept by D=keep or by U=preserve."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    identity: OccurrenceIdentity
    occurrence_key: str = Field(min_length=64, max_length=64)
    decision_source: DecisionSource
    atom_count: int = Field(ge=1)
    atom_ids_key: str = Field(min_length=64, max_length=64)
    author_id_coincidence: AuthorIdCoincidence

    @field_validator("occurrence_key", "atom_ids_key")
    @classmethod
    def _validate_hex(cls, value: str) -> str:
        return validate_hex64(value)


class AtomDecision(BaseModel):
    """AtomDecision (SCHEMAS Section 4): one atom_site ordinal's retain/
    remove outcome and its reason. `retained` and `reason` must agree
    (Section 4: "retained equivale à máscara e aos motivos válidos")."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    ordinal: int = Field(ge=0)
    atom_id: str = Field(min_length=1, max_length=256)
    occurrence_key: str = Field(min_length=64, max_length=64)
    retained: bool
    reason: AtomReason

    @field_validator("atom_id")
    @classmethod
    def _validate_atom_id(cls, value: str) -> str:
        return validate_native_id(value)

    @field_validator("occurrence_key")
    @classmethod
    def _validate_occurrence_key(cls, value: str) -> str:
        return validate_hex64(value)

    _RETAINED_REASONS = frozenset({"selected-asym", "disposition-keep", "unaddressed-preserve"})
    _REMOVED_REASONS = frozenset({"model-filter", "disposition-remove", "chain-complement"})

    @model_validator(mode="after")
    def _validate_retained_matches_reason(self) -> "AtomDecision":
        if self.retained and self.reason not in self._RETAINED_REASONS:
            raise ValueError(f"retained=True is inconsistent with reason={self.reason!r}")
        if not self.retained and self.reason not in self._REMOVED_REASONS:
            raise ValueError(f"retained=False is inconsistent with reason={self.reason!r}")
        return self


def compute_exclusions_key(*, source_context_key: str, model: int, entries: Sequence[XEntry]) -> str:
    payload = {
        "kind": "receptor-materialization-exclusions",
        "version": "1.0",
        "source_context_key": source_context_key,
        "model": model,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    return compute_materialization_hash(payload)


def compute_outside_key(*, source_context_key: str, model: int, entries: Sequence[OutsideEntry]) -> str:
    payload = {
        "kind": "receptor-materialization-outside",
        "version": "1.0",
        "source_context_key": source_context_key,
        "model": model,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }
    return compute_materialization_hash(payload)


def compute_atom_decisions_key(*, source_context_key: str, decisions: Sequence[AtomDecision]) -> str:
    payload = {
        "kind": "receptor-materialization-atom-decisions",
        "version": "1.0",
        "source_context_key": source_context_key,
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
    }
    return compute_materialization_hash(payload)


def compute_connection_decisions_key(
    *, source_context_key: str, decisions: Sequence["ConnectionDecision"] = ()
) -> str:
    """Checkpoint 1 never resolved `struct_conn` (Section 9-10 was
    Checkpoint 2's own delivery): `decisions` defaults to `()` so every
    Checkpoint 1 call site - which never had a real connection vector to
    pass - keeps hashing the exact same empty-vector payload it always
    did. Checkpoint 2's real pipeline (`resolve_connection_decisions`)
    always passes its real, possibly non-empty vector here."""

    payload = {
        "kind": "receptor-materialization-connection-decisions",
        "version": "1.0",
        "source_context_key": source_context_key,
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
    }
    return compute_materialization_hash(payload)


# --------------------------------------------------------------------------
# Section 11 (T021-161 only): CategoryDecision, at the DTO level
# --------------------------------------------------------------------------


class CategoryDecision(BaseModel):
    """CategoryDecision (SCHEMAS Section 4): one category's row/tag
    disposition. Checkpoint 1 never computes these from a real catalogue
    lookup (Section 11 row-level rules are Checkpoint 2) - this class only
    carries the already-decided shape and its own closed-vocabulary
    invariants, exercised directly by domain tests (T021-161).

    Section 11: "em action=omit, os arrays vazios... significam retirar
    todas as tags/linhas... não uma sentinela textual all" - so
    `action="omit"` with every array empty is valid by construction, never
    rejected for "looking empty". `guard_then_omit` instead requires its
    guarded tag list to be enumerated (and validated) before the omission
    it protects."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    category: str = Field(min_length=1, max_length=256)
    action: CategoryAction
    kept_row_ordinals: Tuple[int, ...] = ()
    omitted_row_ordinals: Tuple[int, ...] = ()
    preserved_tags: Tuple[str, ...] = ()
    omitted_tags: Tuple[str, ...] = ()
    generated_tags: Tuple[str, ...] = ()
    source_rows_key: Optional[str] = None

    @field_validator("source_rows_key")
    @classmethod
    def _validate_source_rows_key(cls, value: Optional[str]) -> Optional[str]:
        return value if value is None else validate_hex64(value)

    @model_validator(mode="after")
    def _validate_action_invariants(self) -> "CategoryDecision":
        # MINOR-3 (corrections reconciliation 521500c): `category` must be
        # one of the pinned catalogue's 94 canonical names, or an approved
        # omission-family prefix (Section 11) - which is only ever valid
        # with action="omit". An arbitrary or non-catalogue name is refused
        # even though CP1 never reads the catalogue file itself.
        if self.category not in _CATALOGUE_CATEGORY_NAMES:
            # MINOR-3 residual: a delimited fullmatch grammar, never a bare
            # `startswith` - rejects a bare prefix, an item name ("_em_
            # entity.id"), embedded whitespace/LF, and uppercase, none of
            # which `startswith(prefix)` alone would have refused.
            if not _CATALOGUE_OMISSION_FAMILY_PATTERN.fullmatch(self.category):
                raise ValueError(
                    f"category {self.category!r} is neither a canonical catalogue category "
                    "nor an approved omission-family name (fullmatch "
                    f"{_CATALOGUE_OMISSION_FAMILY_PATTERN.pattern!r})"
                )
            if self.action != "omit":
                raise ValueError(
                    f"category {self.category!r} matches an approved omission-family name, "
                    "which is only ever valid with action='omit'"
                )

        # MINOR-3 residual: ordinals are real, non-negative source positions,
        # unique *within* each tuple (not only disjoint *between* the two
        # tuples) - (0, 0) previously passed because only the cross-tuple
        # overlap was checked.
        if len(set(self.kept_row_ordinals)) != len(self.kept_row_ordinals):
            raise ValueError("kept_row_ordinals must not contain a duplicate ordinal")
        if len(set(self.omitted_row_ordinals)) != len(self.omitted_row_ordinals):
            raise ValueError("omitted_row_ordinals must not contain a duplicate ordinal")
        if any(ordinal < 0 for ordinal in self.kept_row_ordinals) or any(
            ordinal < 0 for ordinal in self.omitted_row_ordinals
        ):
            raise ValueError("row ordinals must be non-negative")

        if set(self.kept_row_ordinals) & set(self.omitted_row_ordinals):
            raise ValueError("kept_row_ordinals and omitted_row_ordinals must not overlap")

        preserved, omitted, generated = set(self.preserved_tags), set(self.omitted_tags), set(self.generated_tags)
        if len(preserved) != len(self.preserved_tags) or len(omitted) != len(self.omitted_tags) or len(
            generated
        ) != len(self.generated_tags):
            raise ValueError("preserve/omit/generate tag arrays must not contain duplicates")
        if preserved & omitted or preserved & generated or omitted & generated:
            raise ValueError("preserve/omit/generate tags must always be disjoint arrays")

        # MINOR-3: "quando a categoria é omitida, nenhuma tag/linha
        # sobrevive" - omit/guard_then_omit must never retain a row.
        if self.action in ("omit", "guard_then_omit") and self.kept_row_ordinals:
            raise ValueError(f"action={self.action!r} must retain no row (kept_row_ordinals must be empty)")

        if self.action == "guard_then_omit":
            # The guard's own enumerated tag list must exist and be validated
            # before any omission it protects is accepted (Section 11).
            if not self.omitted_tags:
                raise ValueError(
                    "guard_then_omit requires its guarded tags to be enumerated in "
                    "omitted_tags before the omission it protects can be validated"
                )
            if self.preserved_tags or self.generated_tags:
                raise ValueError("guard_then_omit must not preserve or generate tags")
        elif self.action == "omit":
            if self.preserved_tags or self.generated_tags:
                raise ValueError("omit must not preserve or generate tags")
        elif self.action == "generate":
            if self.preserved_tags or self.omitted_tags:
                raise ValueError("generate must not preserve or omit tags")
        # ARCH-03 (corrections reconciliation 521500c): action="filter" has
        # no extra restriction beyond the generic disjointness check above -
        # the pinned catalogue's own _cell/_symmetry entries are filter with
        # non-empty preserve_tags, omit_tags AND generate_tags=['entry_id']
        # (DERIVED_ENTRY/SOURCE_FRAME_SINGLETON, Section 11); a blanket
        # "filter must not generate tags" rule contradicted that annex.

        # MINOR-3: filter/omit/guard_then_omit always reference real source
        # rows - even an empty source category still has a non-null hash
        # ("fonte presente vazia também tem hash não nulo"); only
        # action="generate" may legitimately have no source at all.
        if self.action in ("filter", "omit", "guard_then_omit") and self.source_rows_key is None:
            raise ValueError(
                f"action={self.action!r} always references real source rows; source_rows_key "
                "must not be null, even for an empty source category"
            )
        if self.source_rows_key is None and (self.kept_row_ordinals or self.omitted_row_ordinals):
            raise ValueError("row ordinals require a non-null source_rows_key")

        return self


def compute_category_decisions_key(*, source_context_key: str, decisions: Sequence[CategoryDecision]) -> str:
    payload = {
        "kind": "receptor-materialization-category-decisions",
        "version": "1.0",
        "source_context_key": source_context_key,
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
    }
    return compute_materialization_hash(payload)


# --------------------------------------------------------------------------
# Section 8/13: PlanBinding and the pure planner
# --------------------------------------------------------------------------


class PlanBinding(BaseModel):
    """PlanBinding (SCHEMAS Section 4): the hashed, normative plan. Only
    the three decision *keys* live here - the full decision vectors live
    in `MaterializationPlan` (an internal, non-normative container that is
    never itself hashed), "preservando bytes/DTOs suficientes para
    serializar sem reabrir RAW" (Section 8)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["receptor-materialization-plan"] = "receptor-materialization-plan"
    version: Literal["1.0"] = "1.0"
    profile: Literal["1.0"] = "1.0"
    algorithm: Literal["1.0"] = "1.0"
    # MINOR-4 (corrections reconciliation 521500c): pinned to the one
    # approved catalogue identity - a divergent value is rejected by the
    # closed `Literal` itself, never accepted as a configurable input
    # (T021-005 "catalogue divergente").
    catalogue_sha256: Literal[CATALOGUE_SHA256] = CATALOGUE_SHA256
    engine: Engine
    sources: SourceContext
    source_context_key: str = Field(min_length=64, max_length=64)
    selected_asym_ids: Tuple[str, ...]
    unaddressed_components: UnaddressedComponentsPolicy
    exclusions: Tuple[XEntry, ...]
    outside: Tuple[OutsideEntry, ...]
    exclusions_key: str = Field(min_length=64, max_length=64)
    outside_key: str = Field(min_length=64, max_length=64)
    atom_decisions_key: str = Field(min_length=64, max_length=64)
    connection_decisions_key: str = Field(min_length=64, max_length=64)
    category_decisions_key: str = Field(min_length=64, max_length=64)
    unavailable_categories: Tuple[str, ...] = ()
    warnings: Tuple[WarningCode, ...] = ()

    @field_validator("source_context_key", "exclusions_key", "outside_key",
                      "atom_decisions_key", "connection_decisions_key", "category_decisions_key")
    @classmethod
    def _validate_hex_fields(cls, value: str) -> str:
        return validate_hex64(value)

    @field_validator("selected_asym_ids")
    @classmethod
    def _validate_selected_asym_ids(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        for asym in value:
            validate_native_id(asym)
        if len(set(value)) != len(value):
            raise ValueError("selected_asym_ids must not contain a duplicate label_asym_id")
        return value

    @field_validator("warnings")
    @classmethod
    def _validate_warnings_sorted_unique(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("warnings must not repeat a code")
        if tuple(sorted(value)) != tuple(value):
            raise ValueError("warnings must be sorted lexicographically")
        return value


def compute_plan_key(plan: PlanBinding) -> str:
    """plan_key = H(PlanBinding) (Section 13) - the whole, already-shaped
    object is the envelope; it already carries its own `kind`/`version`."""

    return compute_materialization_hash(plan.model_dump(mode="json"))


class SpecialPositionSummary(BaseModel):
    """SpecialPositionSummary (SCHEMAS Section 8, AuditRecord field): the
    three frozen `_pdbx_struct_special_symmetry` counts (Section 11
    SPECIAL_POSITION_GUARD_THEN_OMIT, R-CP2-01) - source-annotation
    matches, never a computed site multiplicity. Zero counts when the
    category is absent. Deliberately not one of `PlanBinding`'s own
    hashed fields (SCHEMAS lists it only on `AuditRecord`, never on
    `PlanBinding` itself) - carried on `MaterializationPlan` as extra,
    non-normative transported data, exactly like every other vector it
    already carries without being part of `plan_key`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_annotation_count: int = Field(ge=0)
    retained_annotation_count: int = Field(ge=0)
    retained_unique_atom_count: int = Field(ge=0)


_ZERO_SPECIAL_POSITION_SUMMARY = SpecialPositionSummary(
    source_annotation_count=0, retained_annotation_count=0, retained_unique_atom_count=0,
)


class MaterializationPlan(BaseModel):
    """The internal, non-normative plan: `PlanBinding` plus the full
    decision vectors it was hashed from (Section 8). Never itself hashed
    or persisted - only `PlanBinding.plan_key` is."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    plan: PlanBinding
    plan_key: str = Field(min_length=64, max_length=64)
    atom_decisions: Tuple[AtomDecision, ...]
    category_decisions: Tuple[CategoryDecision, ...]
    connection_decisions: Tuple["ConnectionDecision", ...] = ()
    special_positions: SpecialPositionSummary = _ZERO_SPECIAL_POSITION_SUMMARY

    @field_validator("plan_key")
    @classmethod
    def _validate_plan_key(cls, value: str) -> str:
        return validate_hex64(value)


def _null_first(value: Optional[str]) -> Tuple[int, str]:
    """Section 8 ordering: "sequência de code points Unicode, com null
    antes de str"."""

    return (0, "") if value is None else (1, value)


def _classify_author_id_coincidence(
    auth_asym_id: Optional[str], polymer_auth_index: Dict[str, List[Tuple[str, bool]]]
) -> AuthorIdCoincidence:
    if auth_asym_id is None:
        return "unavailable"
    nfc = unicodedata.normalize("NFC", auth_asym_id)
    matches = polymer_auth_index.get(nfc, [])
    if not matches:
        return "none"
    if len(matches) > 1:
        return "multiple"
    return "retained-polymer" if matches[0][1] else "excluded-polymer"


def build_materialization_plan(
    *,
    source_context: SourceContext,
    resolved_selection: ResolvedReceptorSelection,
    unaddressed_components: UnaddressedComponentsPolicy,
    atoms: Tuple[AtomObservation, ...],
    category_decisions: Tuple[CategoryDecision, ...] = (),
    connection_decisions: Tuple["ConnectionDecision", ...] = (),
    special_positions: SpecialPositionSummary = _ZERO_SPECIAL_POSITION_SUMMARY,
) -> MaterializationPlan:
    """The pure planner (Section 7-8): derives S/D/U/X/O and every
    `AtomDecision` from `resolved_selection` (S and D) and `atoms` (the
    in-memory M-and-beyond atom rows a future adapter will supply),
    without ever opening RAW, calling Gemmi, or reaching storage."""

    model = resolved_selection.resolved_model
    source_context_key = compute_source_context_key(source_context)

    selected_set = {chain.label_asym_id for chain in resolved_selection.resolved_chains}

    # --- MINOR-1 (corrections reconciliation 521500c): guards over the
    # received atom vector itself. The planner is the declared guard here -
    # it never infers, remaps or repairs a malformed vector (Section 6:
    # "atom_site.id é não nulo e único globalmente no arquivo"). ---
    seen_ordinals: set = set()
    seen_atom_ids: set = set()
    for atom in atoms:
        if atom.ordinal in seen_ordinals:
            raise MaterializationInputError(
                "INVALID_ID", f"duplicate atom ordinal {atom.ordinal} in the supplied atom vector"
            )
        seen_ordinals.add(atom.ordinal)
        if atom.atom_id in seen_atom_ids:
            raise MaterializationInputError(
                "INVALID_ID",
                f"duplicate atom_id {atom.atom_id!r} in the supplied atom vector; atom_site.id "
                "must be globally unique (Section 6)",
            )
        seen_atom_ids.add(atom.atom_id)

    atoms_in_model = tuple(atom for atom in atoms if atom.model == model)
    if not atoms_in_model:
        raise MaterializationInputError("EMPTY_SELECTION", "no atom rows are present in the resolved model M")

    # ARCH-02 (corrections reconciliation 521500c): a (label_asym_id, CCD)
    # disposition key whose CCD never occurs in M must have no effect on M
    # (Section 7: "chave ausente em M não tem efeito"). The asym-level
    # keep/remove aggregates used below for the *polymer* conflict check
    # must therefore be built only from keys actually present in M - unlike
    # `disposition_by_asym_ccd` itself, whose non-polymer lookups below are
    # always keyed by a CCD drawn from an atom already known to be in M.
    present_asym_ccd_pairs = {(atom.label_asym_id, atom.ccd) for atom in atoms_in_model}

    disposition_by_asym_ccd: Dict[Tuple[str, str], str] = {}
    disposition_keep_asyms: set = set()
    disposition_remove_asyms: set = set()
    for disposition in resolved_selection.resolved_component_dispositions:
        if disposition.disposition is None:
            continue
        key = (disposition.label_asym_id, disposition.ccd_code)
        disposition_by_asym_ccd[key] = disposition.disposition
        if key not in present_asym_ccd_pairs:
            continue
        if disposition.disposition == "keep":
            disposition_keep_asyms.add(disposition.label_asym_id)
        else:
            disposition_remove_asyms.add(disposition.label_asym_id)

    asym_entity_type: Dict[str, str] = {}
    asym_entity_id: Dict[str, str] = {}
    for atom in atoms_in_model:
        if atom.entity_type not in _RECOGNIZED_ENTITY_TYPES:
            raise MaterializationInputError(
                "UNSUPPORTED_REQUEST",
                f"entity_type {atom.entity_type!r} for label_asym_id {atom.label_asym_id!r} in M "
                "is not a supported receptor-materialization entity type",
            )
        previous_type = asym_entity_type.get(atom.label_asym_id)
        if previous_type is not None and previous_type != atom.entity_type:
            raise MaterializationInputError(
                "UNSUPPORTED_REQUEST",
                f"label_asym_id {atom.label_asym_id!r} links to contradictory entity types "
                f"({previous_type!r} and {atom.entity_type!r}) in M",
            )
        asym_entity_type[atom.label_asym_id] = atom.entity_type
        # MINOR-1: an asym must bind to exactly one entity_id in M - the
        # previous `setdefault` silently kept whichever entity_id was seen
        # first and accepted any later contradiction.
        previous_entity_id = asym_entity_id.get(atom.label_asym_id)
        if previous_entity_id is not None and previous_entity_id != atom.entity_id:
            raise MaterializationInputError(
                "INVALID_ID",
                f"label_asym_id {atom.label_asym_id!r} links to contradictory entity_id values "
                f"({previous_entity_id!r} and {atom.entity_id!r}) in M",
            )
        asym_entity_id[atom.label_asym_id] = atom.entity_id

    # MINOR-1: every asym named by S must actually resolve in M - a selected
    # asym absent from M is never silently accepted as a no-op.
    missing_selected = selected_set - set(asym_entity_type)
    if missing_selected:
        raise MaterializationInputError(
            "INVALID_ID",
            f"S names label_asym_id(s) {sorted(missing_selected)} that are not present in M",
        )

    polymer_asyms = {asym for asym, kind in asym_entity_type.items() if kind == "polymer"}
    nonpolymer_asyms = {asym for asym, kind in asym_entity_type.items() if kind != "polymer"}

    # --- Section 7 table, polymer rows ---
    polymer_retained: Dict[str, bool] = {}
    for asym in polymer_asyms:
        in_s = asym in selected_set
        if in_s:
            if asym in disposition_remove_asyms:
                raise MaterializationInputError(
                    "SELECTION_CONFLICT",
                    f"polymer label_asym_id {asym!r} is explicitly selected but also carries a "
                    "remove disposition; this is always refused, even for a terminal removal",
                )
            polymer_retained[asym] = True
        else:
            if asym in disposition_keep_asyms:
                raise MaterializationInputError(
                    "SELECTION_CONFLICT",
                    f"polymer label_asym_id {asym!r} is outside S but carries a keep disposition; "
                    "a polymer asym cannot be rescued by disposition alone",
                )
            polymer_retained[asym] = False

    # --- MINOR-2 (corrections reconciliation 521500c): the NFC-injective
    # correspondence must cover *every* auth_asym_id token involved in M's
    # correspondence - water/ligand included, not only polymer tokens - so
    # that a water auth id under NFD can never silently collide-and-merge
    # with a polymer auth id under NFC (Section 6). The same literal token
    # repeated across several distinct asyms remains legitimate (that is
    # the ordinary "multiple" classification, T021-037) - only *distinct*
    # raw tokens colliding under NFC are refused. ---
    all_auth_tokens_in_m = {atom.auth_asym_id for atom in atoms_in_model if atom.auth_asym_id is not None}
    _build_nfc_injective_index(all_auth_tokens_in_m)  # raises on a genuine distinct-token collision
    polymer_auth_index: Dict[str, List[Tuple[str, bool]]] = {}
    for asym in polymer_asyms:
        auth_values = {
            atom.auth_asym_id for atom in atoms_in_model
            if atom.label_asym_id == asym and atom.auth_asym_id is not None
        }
        for auth_value in auth_values:
            nfc = unicodedata.normalize("NFC", auth_value)
            polymer_auth_index.setdefault(nfc, []).append((asym, polymer_retained[asym]))

    # --- Section 7 table, non-polymer-like rows (water/non-polymer/macrolide/branched) ---
    nonpolymer_retained: Dict[str, bool] = {}
    outside_entries: List[OutsideEntry] = []

    for asym in nonpolymer_asyms:
        asym_atoms = [atom for atom in atoms_in_model if atom.label_asym_id == asym]
        groups: Dict[object, List[AtomObservation]] = {}
        order: List[object] = []
        for atom in asym_atoms:
            group_key = _occurrence_group_key(atom)
            if group_key not in groups:
                groups[group_key] = []
                order.append(group_key)
            groups[group_key].append(atom)

        in_s = asym in selected_set
        decisions_seen: set = set()
        occurrence_results = []

        for group_key in order:
            group_atoms = groups[group_key]
            ccd = group_atoms[0].ccd
            disposition = disposition_by_asym_ccd.get((asym, ccd))

            if in_s:
                if disposition == "remove":
                    raise MaterializationInputError(
                        "SELECTION_CONFLICT",
                        f"non-polymer label_asym_id {asym!r} is explicitly selected but CCD "
                        f"{ccd!r} carries a remove disposition",
                    )
                retained, decision_source = True, None
            else:
                if disposition == "keep":
                    retained, decision_source = True, "disposition-keep"
                elif disposition == "remove":
                    retained, decision_source = False, None
                elif unaddressed_components == "preserve":
                    retained, decision_source = True, "unaddressed-preserve"
                else:
                    raise MaterializationInputError(
                        "SELECTION_CONFLICT",
                        f"non-polymer label_asym_id {asym!r} CCD {ccd!r} is outside S with no "
                        "disposition, and unaddressed_components='require-disposition'",
                    )

            decisions_seen.add(retained)
            occurrence_results.append((group_atoms, retained, decision_source))

        if len(decisions_seen) > 1:
            raise MaterializationInputError(
                "NONUNIFORM_ASYM",
                f"non-polymer label_asym_id {asym!r} resolves to mixed keep/remove decisions "
                "across its CCD variants; P-3 requires a single uniform decision per asym",
            )

        asym_retained = next(iter(decisions_seen))
        nonpolymer_retained[asym] = asym_retained

        if not in_s and asym_retained:
            for group_atoms, retained, decision_source in occurrence_results:
                identity = _build_occurrence_identity(group_atoms[0])
                occurrence_key = compute_occurrence_key(identity)
                sorted_group = tuple(sorted(group_atoms, key=lambda a: a.ordinal))
                atom_ids_key = compute_atom_ids_key(sorted_group)
                author_id_coincidence = _classify_author_id_coincidence(
                    identity.auth_asym_id, polymer_auth_index
                )
                outside_entries.append(
                    OutsideEntry(
                        identity=identity,
                        occurrence_key=occurrence_key,
                        decision_source=decision_source,
                        atom_count=len(sorted_group),
                        atom_ids_key=atom_ids_key,
                        author_id_coincidence=author_id_coincidence,
                    )
                )

    # --- X: excluded polymer asyms (chain-complement) ---
    exclusions: List[XEntry] = []
    for asym in sorted(a for a in polymer_asyms if not polymer_retained[a]):
        asym_atoms = tuple(
            sorted((atom for atom in atoms_in_model if atom.label_asym_id == asym), key=lambda a: a.ordinal)
        )
        exclusions.append(
            XEntry(
                model=model,
                label_asym_id=asym,
                entity_id=asym_entity_id[asym],
                atom_count=len(asym_atoms),
                atom_ids_key=compute_atom_ids_key(asym_atoms),
            )
        )

    outside_entries.sort(
        key=lambda entry: (
            entry.identity.label_asym_id,
            entry.identity.ccd,
            _null_first(entry.identity.label_seq_id),
            _null_first(entry.identity.auth_seq_id),
            _null_first(entry.identity.insertion_token),
        )
    )

    # --- AtomDecision: every ordinal, across every model, exactly once ---
    atom_decisions: List[AtomDecision] = []
    other_models_excluded = False
    for atom in sorted(atoms, key=lambda a: a.ordinal):
        if atom.model != model:
            retained, reason = False, "model-filter"
            other_models_excluded = True
        elif atom.label_asym_id in polymer_asyms:
            retained = polymer_retained[atom.label_asym_id]
            reason = "selected-asym" if retained else "chain-complement"
        else:
            retained = nonpolymer_retained[atom.label_asym_id]
            if atom.label_asym_id in selected_set:
                reason = "selected-asym"
            else:
                disposition = disposition_by_asym_ccd.get((atom.label_asym_id, atom.ccd))
                if disposition == "keep":
                    reason = "disposition-keep"
                elif disposition == "remove":
                    reason = "disposition-remove"
                else:
                    reason = "unaddressed-preserve"

        identity = _build_occurrence_identity(atom)
        atom_decisions.append(
            AtomDecision(
                ordinal=atom.ordinal,
                atom_id=atom.atom_id,
                occurrence_key=compute_occurrence_key(identity),
                retained=retained,
                reason=reason,
            )
        )

    present_category_names = frozenset(decision.category for decision in category_decisions)
    warnings_set: set = set()
    if other_models_excluded:
        warnings_set.add("other-models-excluded")
    if outside_entries:
        warnings_set.add("outside-selected-asym-retained")
    # Section 11: mechanical warnings derived directly from already-computed
    # category/connection decisions - never fabricated, never guessed ahead
    # of the real vectors.
    if any(
        decision.action == "omit" and decision.omitted_row_ordinals
        for decision in category_decisions
    ):
        warnings_set.add("source-annotations-omitted")
    if any(
        decision.action == "omit" and decision.conn_type_token.lower() == "metalc"
        for decision in connection_decisions
    ):
        warnings_set.add("coordination-connections-omitted")
    if any(
        decision.action == "omit" and decision.conn_type_token.lower() == "hydrog"
        for decision in connection_decisions
    ):
        warnings_set.add("hydrogen-bonds-omitted")
    if "_cell" in present_category_names or "_symmetry" in present_category_names:
        warnings_set.add("source-symmetry-context-only")
    if "_struct_ncs_oper" in present_category_names:
        warnings_set.add("source-ncs-annotations-omitted")
    # R-CP2-01: the warning requires a real retained-atom intersection
    # (Section 11: "havendo interseção retida, emitir... categoria vazia
    # ou somente ocorrências removidas não exige esse aviso") - never
    # merely the category's presence, which a removed-only or empty
    # category would satisfy just as wrongly as an absent one would not.
    if special_positions.retained_annotation_count > 0:
        warnings_set.add("special-position-annotations-omitted")
    if category_decisions and "_struct_conn" not in present_category_names:
        warnings_set.add("declared-connections-unavailable")
    warnings = tuple(sorted(warnings_set))

    exclusions_key = compute_exclusions_key(source_context_key=source_context_key, model=model, entries=exclusions)
    outside_key = compute_outside_key(source_context_key=source_context_key, model=model, entries=outside_entries)
    atom_decisions_key = compute_atom_decisions_key(source_context_key=source_context_key, decisions=atom_decisions)
    connection_decisions_key = compute_connection_decisions_key(
        source_context_key=source_context_key, decisions=connection_decisions
    )
    category_decisions_key = compute_category_decisions_key(
        source_context_key=source_context_key, decisions=category_decisions
    )

    # Section 11: a non-trivial (ever filter/generate/guard_then_omit)
    # catalogue category with no `CategoryDecision` at all is one the
    # source simply never declared - tracked here, never silently dropped
    # (`_entry`/DERIVED_ENTRY is excluded: it is always generated,
    # regardless of source `_entry` presence, so its absence from source
    # is not itself an "unavailable" fact worth auditing).
    unavailable_categories = tuple(
        sorted(
            name for name, cat_def in _FROZEN_CATALOGUE.items()
            if cat_def["row_rule"] != "AUDIT_OMITTED_CATEGORY"
            and name != "_entry"
            and name not in present_category_names
        )
    )

    plan = PlanBinding(
        catalogue_sha256=CATALOGUE_SHA256,
        engine=Engine(name=ENGINE_NAME, version=ENGINE_VERSION),
        sources=source_context,
        source_context_key=source_context_key,
        selected_asym_ids=tuple(sorted(selected_set)),
        unaddressed_components=unaddressed_components,
        exclusions=tuple(exclusions),
        outside=tuple(outside_entries),
        exclusions_key=exclusions_key,
        outside_key=outside_key,
        atom_decisions_key=atom_decisions_key,
        connection_decisions_key=connection_decisions_key,
        category_decisions_key=category_decisions_key,
        unavailable_categories=unavailable_categories,
        warnings=warnings,
    )

    return MaterializationPlan(
        plan=plan,
        plan_key=compute_plan_key(plan),
        atom_decisions=tuple(atom_decisions),
        category_decisions=category_decisions,
        connection_decisions=connection_decisions,
        special_positions=special_positions,
    )


# --------------------------------------------------------------------------
# Section 12-13: PolicyRecord and the declare-side pure function
# --------------------------------------------------------------------------


class PolicyRecord(BaseModel):
    """PolicyRecord (SCHEMAS Section 6): the scientist's declared act over
    an already-computed `PlanBinding` - acknowledgements, `declared_by`,
    and `declared_at` - plus the plan's own identity for later
    cross-checking at `run` time (Section 13: "run recaptura fontes e
    plano e compara exatamente à política antes de reutilizar ou
    escrever")."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["receptor-materialization-policy"] = "receptor-materialization-policy"
    schema_version: Literal["1.0"] = "1.0"
    profile_version: Literal["1.0"] = "1.0"
    algorithm_version: Literal["1.0"] = "1.0"
    # MINOR-4: pinned exactly like PlanBinding.catalogue_sha256 above.
    catalogue_sha256: Literal[CATALOGUE_SHA256] = CATALOGUE_SHA256
    engine: Engine
    policy_id: str = Field(min_length=64, max_length=64)
    sources: SourceContext
    source_context_key: str = Field(min_length=64, max_length=64)
    plan_key: str = Field(min_length=64, max_length=64)
    unaddressed_components: UnaddressedComponentsPolicy
    exclusions_key: str = Field(min_length=64, max_length=64)
    outside_key: str = Field(min_length=64, max_length=64)
    excluded_polymer_asym_count: int = Field(ge=0)
    outside_occurrence_count: int = Field(ge=0)
    acknowledge_plan_key: str = Field(min_length=64, max_length=64)
    acknowledge_exclusions_key: Optional[str] = None
    acknowledge_outside_key: Optional[str] = None
    declared_by: str = Field(min_length=1, max_length=200)
    declared_at: str

    @field_validator(
        "policy_id", "source_context_key", "plan_key",
        "exclusions_key", "outside_key", "acknowledge_plan_key",
    )
    @classmethod
    def _validate_hex_fields(cls, value: str) -> str:
        return validate_hex64(value)

    @field_validator("acknowledge_exclusions_key", "acknowledge_outside_key")
    @classmethod
    def _validate_optional_hex_fields(cls, value: Optional[str]) -> Optional[str]:
        return value if value is None else validate_hex64(value)

    @field_validator("declared_by")
    @classmethod
    def _validate_declared_by(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("declared_by must not be blank")
        if any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in value):
            raise ValueError("declared_by must not contain C0/C1 control characters")
        return value

    @field_validator("declared_at")
    @classmethod
    def _validate_declared_at(cls, value: str) -> str:
        return validate_time_utc(value)

    @model_validator(mode="after")
    def _validate_acknowledgement_consistency(self) -> "PolicyRecord":
        if self.acknowledge_plan_key != self.plan_key:
            raise ValueError("acknowledge_plan_key must equal plan_key")
        if self.excluded_polymer_asym_count > 0:
            if self.acknowledge_exclusions_key != self.exclusions_key:
                raise ValueError("acknowledge_exclusions_key must equal exclusions_key when X is non-empty")
        elif self.acknowledge_exclusions_key is not None:
            raise ValueError("acknowledge_exclusions_key must be null when X is empty")
        if self.outside_occurrence_count > 0:
            if self.acknowledge_outside_key != self.outside_key:
                raise ValueError("acknowledge_outside_key must equal outside_key when O is non-empty")
        elif self.acknowledge_outside_key is not None:
            raise ValueError("acknowledge_outside_key must be null when O is empty")
        return self

    @model_validator(mode="after")
    def _validate_policy_id_matches_payload(self) -> "PolicyRecord":
        """MAJOR-B (corrections reconciliation 521500c): `policy_id` is
        never an arbitrary caller-supplied value - it must equal H() of
        this record's own Section 13 payload (Section 13: policy_id
        identifies the restricted payload; nothing derived "fixes" it)."""

        expected = compute_policy_id(
            plan_key=self.plan_key,
            acknowledge_plan_key=self.acknowledge_plan_key,
            acknowledge_exclusions_key=self.acknowledge_exclusions_key,
            acknowledge_outside_key=self.acknowledge_outside_key,
            declared_by=self.declared_by,
        )
        if self.policy_id != expected:
            raise ValueError("policy_id does not match H() of its own declared payload")
        return self


def compute_policy_id(
    *,
    plan_key: str,
    acknowledge_plan_key: str,
    acknowledge_exclusions_key: Optional[str],
    acknowledge_outside_key: Optional[str],
    declared_by: str,
) -> str:
    """policy_id = H({kind:'receptor-materialization-policy', version:
    '1.0', plan_key, acknowledge_plan_key, acknowledge_exclusions_key,
    acknowledge_outside_key, declared_by}) (Section 13). `declared_at` is
    never part of this payload - a repeated, semantically identical
    declaration always yields the same policy_id (T021-030/034)."""

    payload = {
        "kind": "receptor-materialization-policy",
        "version": "1.0",
        "plan_key": plan_key,
        "acknowledge_plan_key": acknowledge_plan_key,
        "acknowledge_exclusions_key": acknowledge_exclusions_key,
        "acknowledge_outside_key": acknowledge_outside_key,
        "declared_by": declared_by,
    }
    return compute_materialization_hash(payload)


def build_policy_record(
    *,
    materialization_plan: MaterializationPlan,
    acknowledge_plan_key: str,
    acknowledge_exclusions_key: Optional[str],
    acknowledge_outside_key: Optional[str],
    declared_by: str,
    declared_at: str,
    existing: Optional[PolicyRecord] = None,
) -> PolicyRecord:
    """The pure `declare` step (Section 8/13): validates the three
    acknowledgements against `materialization_plan`, computes `policy_id`,
    and - when an
    `existing` record with the same `policy_id` is supplied by the caller
    - preserves its original `declared_at` instead of the one just
    supplied (Section 13: "Declaração repetida retorna a política
    existente depois de verificar integridade e equivalência; preserva
    timestamp original").

    This function never touches a registry: `existing` is whatever the
    (future) Checkpoint 3 storage layer looked up and is handing back in
    memory - Checkpoint 1 proves only the pure identity/equivalence rule,
    never a persisted create-once guarantee (Section 16 preparation
    brief)."""

    plan = materialization_plan.plan

    if acknowledge_plan_key != materialization_plan.plan_key:
        raise MaterializationInputError(
            "ACK_MISMATCH", "acknowledge_plan_key does not match the current plan_key"
        )

    excluded_count = len(plan.exclusions)
    if excluded_count > 0:
        if acknowledge_exclusions_key is None:
            raise MaterializationInputError(
                "MISSING_ACK", "X is non-empty; acknowledge_exclusions_key is required"
            )
        if acknowledge_exclusions_key != plan.exclusions_key:
            raise MaterializationInputError(
                "ACK_MISMATCH", "acknowledge_exclusions_key does not match the current exclusions_key"
            )
    elif acknowledge_exclusions_key is not None:
        raise MaterializationInputError(
            "ACK_MISMATCH", "acknowledge_exclusions_key must be omitted when X is empty"
        )

    outside_count = len(plan.outside)
    if outside_count > 0:
        if acknowledge_outside_key is None:
            raise MaterializationInputError(
                "MISSING_ACK", "O is non-empty; acknowledge_outside_key is required"
            )
        if acknowledge_outside_key != plan.outside_key:
            raise MaterializationInputError(
                "ACK_MISMATCH", "acknowledge_outside_key does not match the current outside_key"
            )
    elif acknowledge_outside_key is not None:
        raise MaterializationInputError(
            "ACK_MISMATCH", "acknowledge_outside_key must be omitted when O is empty"
        )

    policy_id = compute_policy_id(
        plan_key=materialization_plan.plan_key,
        acknowledge_plan_key=acknowledge_plan_key,
        acknowledge_exclusions_key=acknowledge_exclusions_key,
        acknowledge_outside_key=acknowledge_outside_key,
        declared_by=declared_by,
    )

    candidate = PolicyRecord(
        catalogue_sha256=plan.catalogue_sha256,
        engine=plan.engine,
        policy_id=policy_id,
        sources=plan.sources,
        source_context_key=plan.source_context_key,
        plan_key=materialization_plan.plan_key,
        unaddressed_components=plan.unaddressed_components,
        exclusions_key=plan.exclusions_key,
        outside_key=plan.outside_key,
        excluded_polymer_asym_count=excluded_count,
        outside_occurrence_count=outside_count,
        acknowledge_plan_key=acknowledge_plan_key,
        acknowledge_exclusions_key=acknowledge_exclusions_key,
        acknowledge_outside_key=acknowledge_outside_key,
        declared_by=declared_by,
        declared_at=declared_at,
    )

    if existing is None or existing.policy_id != candidate.policy_id:
        return candidate

    # MAJOR-B (corrections reconciliation 521500c): a matching policy_id
    # alone never proves `existing` is a legitimate repeat - policy_id only
    # hashes plan_key/acks/declared_by (Section 13), so every other
    # semantic/derived field must still be verified equal before reuse
    # (Section 13: "verificar integridade e equivalência"); only
    # `declared_at` may differ. A mismatch is a typed configuration error,
    # never a silent overwrite.
    candidate_fields = candidate.model_dump(mode="json")
    existing_fields = existing.model_dump(mode="json")
    for field_name, candidate_value in candidate_fields.items():
        if field_name == "declared_at":
            continue
        if existing_fields.get(field_name) != candidate_value:
            raise MaterializationConfigurationError(
                "SOURCE_CHANGED",
                f"existing PolicyRecord field {field_name!r} does not match the value this "
                "declaration would produce for the same policy_id; refusing to reuse a "
                "tampered or divergent record",
            )
    return existing


def compute_request_key(
    *,
    policy_id: str,
    policy_file_sha256: str,
    source_context_key: str,
    plan_key: str,
    profile: str,
    algorithm: str,
    catalogue_sha256: str,
    engine: Engine,
    application_version: str,
) -> str:
    """request_key = artifact_id = H({kind:
    'receptor-materialization-request', version:'1.0', policy_id,
    policy_file_sha256, source_context_key, plan_key, profile, algorithm,
    catalogue_sha256, engine, application_version}) (Section 13).
    `application_version` participates here and nowhere else - changing
    only it changes request_key without changing plan_key/policy_id
    (T021-035/159)."""

    payload = {
        "kind": "receptor-materialization-request",
        "version": "1.0",
        "policy_id": policy_id,
        "policy_file_sha256": policy_file_sha256,
        "source_context_key": source_context_key,
        "plan_key": plan_key,
        "profile": profile,
        "algorithm": algorithm,
        "catalogue_sha256": catalogue_sha256,
        "engine": {"name": engine.name, "version": engine.version},
        "application_version": application_version,
    }
    return compute_materialization_hash(payload)


# ===========================================================================
# Increment 021, Checkpoint 2: connections, symmetry and IdentityBasis
# (frozen specification Sections 9, 10, 10.1). This section is pure
# Python - no `gemmi.*` reference anywhere; it operates on the generic
# `RawCifCategory` rows the Checkpoint 2 adapter boundary
# (`build_materialization_plan_cp2`, further below) extracts via a late,
# local import of `molecular.receptor_parsing`. The module-level import
# boundary established in Checkpoint 1 (TYPE_CHECKING-only for the 020
# types, no `receptor_parsing`/`receptor_inspection`/`receptor_resolution`
# at runtime - ARCH-01) is unchanged by this section.
# ===========================================================================


class MaterializationExecutionError(ValueError):
    """A Checkpoint 2 execution-phase error over the native DOM/catalogue
    (Section 16: INVALID_CIF, LIMIT_EXCEEDED, UNSUPPORTED_CATEGORY/TAG/
    MODEL/SYMMETRY, UNSUPPORTED_NCS, AMBIGUOUS_ENDPOINT, UNSUPPORTED_
    PARTNER/TYPE, COVALENT_CUT, INVALID_OUTPUT)."""

    _CODES = frozenset(
        {
            "INVALID_CIF", "LIMIT_EXCEEDED", "UNSUPPORTED_CATEGORY", "UNSUPPORTED_TAG",
            "UNSUPPORTED_MODEL", "UNSUPPORTED_SYMMETRY", "UNSUPPORTED_NCS", "AMBIGUOUS_ENDPOINT",
            "UNSUPPORTED_PARTNER", "UNSUPPORTED_TYPE", "COVALENT_CUT", "INVALID_OUTPUT",
        }
    )

    def __init__(self, code: str, message: str, *, phase: str = "parse") -> None:
        if code not in self._CODES:
            raise ValueError(f"unknown MaterializationExecutionError code {code!r}")
        super().__init__(message)
        self.code = code
        self.phase = phase


IdentityBasis = Literal["pdbx-1_555-convention", "source-operator", "not-applicable"]
SymmetryClass = Literal[
    "both-identity", "same-nonidentity", "different", "one-null", "both-null", "unsupported-discarded"
]
ConnectionAction = Literal["preserve", "omit"]
ConnectionReason = Literal[
    "both-retained", "both-removed", "whole-partner-removed-metalc", "whole-partner-removed-hydrog"
]

_COVALENT_CONNECTION_TYPES = frozenset(
    {"covale", "covale_base", "covale_phosphate", "covale_sugar", "modres", "disulf"}
)
_EXCEPTION_CONNECTION_TYPES = frozenset({"metalc", "hydrog"})
_RECOGNIZED_CONNECTION_TYPES = _COVALENT_CONNECTION_TYPES | _EXCEPTION_CONNECTION_TYPES

_SYMMETRY_CODE_PATTERN = re.compile(r"^([1-9][0-9]*)_([1-9])([1-9])([1-9])$")
_IDENTITY_EXPRESSION = "x,y,z"


def _raw_cif_token_to_optional(token: Optional[str]) -> Optional[str]:
    """A bare, unquoted `.`/`?` is a real CIF null; a quoted `'.'`/`'?'`
    string (already de-quoted-or-not by the caller's own raw-token
    convention) is a genuine token (Section 5). Deliberately NOT imported
    from `receptor_parsing.raw_cif_null_to_optional_str` - a module-level
    import of that name would load `receptor_parsing` (and therefore
    Gemmi) at this module's import time, which ARCH-01 forbids; this is
    the same two-line rule, duplicated locally rather than cross-module
    coupled for a trivial null check."""

    if token is None or token in (".", "?"):
        return None
    return token


def _normalize_operator_expression(expression: str) -> str:
    """Section 10: compare two operator expressions after removing ASCII
    whitespace and a single leading `+` on each of the three comma-
    separated coordinate terms - never a general symbolic simplification."""

    parts = expression.split(",")
    if len(parts) != 3:
        return "".join(ch for ch in expression if ch not in " \t")
    normalized_parts = []
    for part in parts:
        stripped = "".join(ch for ch in part if ch not in " \t")
        if stripped.startswith("+"):
            stripped = stripped[1:]
        normalized_parts.append(stripped)
    return ",".join(normalized_parts)


def _is_identity_expression(expression: Optional[str]) -> bool:
    if expression is None:
        return False
    return _normalize_operator_expression(expression) == _IDENTITY_EXPRESSION


def _parse_symmetry_code(token: Optional[str]) -> Optional[str]:
    """The `n_klm` grammar (Section 10): `n` a positive integer with no
    leading zero; `k`/`l`/`m` single digits 1-9. Returns the canonical
    `"n_klm"` string, or `None` if the token is null or does not match."""

    if token is None:
        return None
    if _SYMMETRY_CODE_PATTERN.fullmatch(token):
        return token
    return None


class SymmetryOperatorTables(NamedTuple):
    """`None` for a table means that category is entirely absent from the
    source (Section 10.1: "presença de tabelas é contexto comum do mesmo
    RAW"). A present table maps decimal id string -> expression, or ->
    `None` when that id is declared more than once within the *same*
    table (a same-table duplicate - Section 10.1: "contraditório")."""

    symmetry_equiv: Optional["dict[str, Optional[str]]"]
    space_group_symop: Optional["dict[str, Optional[str]]"]

    @property
    def any_present(self) -> bool:
        return self.symmetry_equiv is not None or self.space_group_symop is not None


def _lookup_operator(operator_id: str, tables: SymmetryOperatorTables):
    """Returns one of `("not_found",)`, `("contradictory",)`, or
    `("found", expression, every_present_table_confirms_identity)`
    (Section 10.1 points 3-4)."""

    present_tables = [t for t in (tables.symmetry_equiv, tables.space_group_symop) if t is not None]
    found_expressions = []
    for table in present_tables:
        if operator_id not in table:
            continue
        value = table[operator_id]
        if value is None:
            return ("contradictory",)
        found_expressions.append(value)

    if not found_expressions:
        return ("not_found",)

    normalized = {_normalize_operator_expression(e) for e in found_expressions}
    if len(normalized) > 1:
        return ("contradictory",)

    every_present_table_confirms_identity = all(
        (operator_id in table) and (table[operator_id] is not None) and _is_identity_expression(table[operator_id])
        for table in present_tables
    )
    return ("found", found_expressions[0], every_present_table_confirms_identity)


class _SymmetryEndpointResolution(NamedTuple):
    is_null: bool
    identity_basis: IdentityBasis
    insufficient_proof: bool  # triggers unsupported-discarded precedence at K=0 (Section 10.1)
    code: Optional[str]  # canonical "n_klm", for same/different comparison; None if null/malformed
    proven_identity: bool  # True only if confirmed identity under this profile (Section 10)


def _resolve_symmetry_endpoint(
    raw_token: Optional[str], tables: SymmetryOperatorTables
) -> _SymmetryEndpointResolution:
    """One endpoint's IdentityBasis and identity-proof status (Section
    10.1), independent of K - computed in a fixed order (ptnr1 then
    ptnr2) by the caller, never by K or category-visit order."""

    token = _raw_cif_token_to_optional(raw_token)
    if token is None:
        return _SymmetryEndpointResolution(True, "not-applicable", False, None, False)

    code = _parse_symmetry_code(token)
    if code is None:
        return _SymmetryEndpointResolution(False, "not-applicable", True, None, False)

    if not tables.any_present:
        if code == "1_555":
            return _SymmetryEndpointResolution(False, "pdbx-1_555-convention", False, code, True)
        return _SymmetryEndpointResolution(False, "not-applicable", True, code, False)

    n = code.split("_", 1)[0]
    lookup = _lookup_operator(n, tables)
    if lookup[0] != "found":
        return _SymmetryEndpointResolution(False, "not-applicable", True, code, False)

    _, _, every_present_table_confirms_identity = lookup
    if code == "1_555":
        proven_identity = every_present_table_confirms_identity
        insufficient_proof = not every_present_table_confirms_identity
    else:
        proven_identity = False
        insufficient_proof = False
    return _SymmetryEndpointResolution(False, "source-operator", insufficient_proof, code, proven_identity)


def _classify_symmetry_class(
    ptnr1: _SymmetryEndpointResolution, ptnr2: _SymmetryEndpointResolution
) -> SymmetryClass:
    """The pairwise comparison (Section 10): any non-null side with
    insufficient proof takes precedence over null-based classification
    (Section 10.1); otherwise both-null/one-null/both-identity/
    same-nonidentity/different, in that closed order."""

    if (not ptnr1.is_null and ptnr1.insufficient_proof) or (not ptnr2.is_null and ptnr2.insufficient_proof):
        return "unsupported-discarded"
    if ptnr1.is_null and ptnr2.is_null:
        return "both-null"
    if ptnr1.is_null or ptnr2.is_null:
        return "one-null"
    if ptnr1.code == ptnr2.code:
        return "both-identity" if (ptnr1.proven_identity and ptnr2.proven_identity) else "same-nonidentity"
    return "different"


_IMPOSSIBLE_IDENTITY_BASIS_PAIRS = frozenset(
    {("pdbx-1_555-convention", "source-operator"), ("source-operator", "pdbx-1_555-convention")}
)


def compute_row_key(*, source_context_key: str, category: str, ordinal: int, tokens: Sequence[Tuple[str, str]]) -> str:
    """row_key = H({kind:'receptor-materialization-source-row', version:
    '1.0', source_context_key, category, ordinal, tokens:[{tag,token},...]})
    (SCHEMAS Section 4), `tokens` already in canonical tag order."""

    payload = {
        "kind": "receptor-materialization-source-row",
        "version": "1.0",
        "source_context_key": source_context_key,
        "category": category,
        "ordinal": ordinal,
        "tokens": [{"tag": tag, "token": token} for tag, token in tokens],
    }
    return compute_materialization_hash(payload)


class ConnectionDecision(BaseModel):
    """ConnectionDecision (SCHEMAS Section 4): one accepted `struct_conn`
    row's retain/omit outcome. A row that is *rejected* (ambiguous
    endpoint, unsupported symmetry/type, covalent K=1 cut, ...) never
    produces a `ConnectionDecision` at all - the whole plan fails instead
    (Section 9: "planos recusados não têm ConnectionDecision persistida")."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    ordinal: int = Field(ge=0)
    id: str = Field(min_length=1, max_length=256)
    row_key: str = Field(min_length=64, max_length=64)
    conn_type_token: str = Field(min_length=1, max_length=256)
    endpoint_atom_ids: Tuple[Tuple[str, ...], Tuple[str, ...]]
    identity_basis: Tuple[IdentityBasis, IdentityBasis]
    retained_endpoints: int = Field(ge=0, le=2)
    symmetry_class: SymmetryClass
    action: ConnectionAction
    reason: ConnectionReason

    @field_validator("id", "conn_type_token")
    @classmethod
    def _validate_native(cls, value: str) -> str:
        return validate_native_id(value)

    @field_validator("row_key")
    @classmethod
    def _validate_hex(cls, value: str) -> str:
        return validate_hex64(value)

    @field_validator("endpoint_atom_ids")
    @classmethod
    def _validate_endpoint_sets(cls, value: Tuple[Tuple[str, ...], Tuple[str, ...]]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        for endpoint_set in value:
            if not endpoint_set:
                raise ValueError("an endpoint's atom id set must not be empty")
            if len(set(endpoint_set)) != len(endpoint_set):
                raise ValueError("an endpoint's atom id set must not repeat an atom id")
        return value

    @model_validator(mode="after")
    def _validate_identity_basis_pair(self) -> "ConnectionDecision":
        if tuple(self.identity_basis) in _IMPOSSIBLE_IDENTITY_BASIS_PAIRS:
            raise ValueError(
                "identity_basis pair is contractually impossible "
                "(pdbx-1_555-convention/source-operator in either order)"
            )
        if self.symmetry_class == "unsupported-discarded" and self.retained_endpoints != 0:
            raise ValueError("unsupported-discarded symmetry_class only ever appears at K=0")
        return self


class RawConnectionEndpointTokens(NamedTuple):
    """One `ptnr{1,2}` endpoint's raw tokens from one `_struct_conn` row -
    already null-collapsed (`.`/`?` -> `None`) by the caller, case/content
    preserved otherwise. `altloc_aliases` holds every alias column this
    profile reads for that endpoint's altloc code, in a fixed order; all
    non-null aliases must agree (Section 9)."""

    label_asym_id: Optional[str]
    label_comp_id: Optional[str]
    label_seq_id: Optional[str]
    label_atom_id: Optional[str]
    auth_seq_id: Optional[str]
    insertion_token: Optional[str]
    altloc_aliases: Tuple[Optional[str], ...]
    symmetry_token: Optional[str]


class RawConnectionRow(NamedTuple):
    ordinal: int
    id: Optional[str]
    conn_type: Optional[str]
    ptnr1: RawConnectionEndpointTokens
    ptnr2: RawConnectionEndpointTokens
    ptnr3_present: bool
    source_tokens: Tuple[Tuple[str, str], ...]


def _resolve_endpoint_altloc(tokens: RawConnectionEndpointTokens) -> Optional[str]:
    non_null_aliases = {alias for alias in tokens.altloc_aliases if alias is not None}
    if len(non_null_aliases) > 1:
        raise MaterializationExecutionError(
            "AMBIGUOUS_ENDPOINT", "endpoint altloc alias columns disagree", phase="plan"
        )
    return next(iter(non_null_aliases), None)


def _resolve_endpoint_atoms(
    tokens: RawConnectionEndpointTokens, atoms_in_model: Sequence[AtomObservation]
) -> Tuple[AtomObservation, ...]:
    """Resolves one `struct_conn` endpoint to its unique occurrence in M,
    then to the (possibly altloc-multi-valued) set of atoms named by
    `label_atom_id` within it (Section 9). Raises `AMBIGUOUS_ENDPOINT`
    if the endpoint does not exist, or exists more than once, in M - a
    missing/ambiguous endpoint is never silently treated as a no-op row."""

    if tokens.label_seq_id is not None:
        occurrence_atoms = [
            a for a in atoms_in_model
            if a.label_asym_id == tokens.label_asym_id
            and a.ccd == tokens.label_comp_id
            and a.label_seq_id == tokens.label_seq_id
        ]
    else:
        occurrence_atoms = [
            a for a in atoms_in_model
            if a.label_asym_id == tokens.label_asym_id
            and a.ccd == tokens.label_comp_id
            and a.label_seq_id is None
            and a.auth_seq_id == tokens.auth_seq_id
            and a.insertion_token == tokens.insertion_token
        ]

    occurrence_keys = {_occurrence_group_key(a) for a in occurrence_atoms}
    if len(occurrence_keys) == 0:
        raise MaterializationExecutionError(
            "AMBIGUOUS_ENDPOINT", "struct_conn endpoint does not resolve to any occurrence in M", phase="plan"
        )
    if len(occurrence_keys) > 1:
        raise MaterializationExecutionError(
            "AMBIGUOUS_ENDPOINT", "struct_conn endpoint resolves to more than one occurrence in M", phase="plan"
        )

    name_matches = tuple(a for a in occurrence_atoms if a.label_atom_id == tokens.label_atom_id)
    if not name_matches:
        raise MaterializationExecutionError(
            "AMBIGUOUS_ENDPOINT", "struct_conn endpoint names an atom absent from its resolved occurrence", phase="plan"
        )

    altloc = _resolve_endpoint_altloc(tokens)
    if altloc is not None:
        name_matches = tuple(a for a in name_matches if a.altloc == altloc)
        if not name_matches:
            raise MaterializationExecutionError(
                "AMBIGUOUS_ENDPOINT", "struct_conn endpoint names an altloc variant absent from its occurrence", phase="plan"
            )

    return tuple(sorted(name_matches, key=lambda a: a.ordinal))


def resolve_connection_decisions(
    *,
    source_context_key: str,
    atoms_in_model: Sequence[AtomObservation],
    atom_decisions_by_ordinal: "dict[int, AtomDecision]",
    raw_connections: Sequence[RawConnectionRow],
    tables: SymmetryOperatorTables,
) -> Tuple[ConnectionDecision, ...]:
    """The full Section 9/10/10.1 connection pipeline: resolve both
    endpoints in M, reject a non-null `ptnr3`, compute K (retained
    endpoint count) from the planner's own `atom_decisions`, classify
    IdentityBasis per endpoint (independent of K, in ptnr1-then-ptnr2
    order), classify `SymmetryClass`, and apply the closed 15-cell
    matrix (Section 10) together with the covalent/metalc/hydrog K=1
    exception (Section 9). A connection that fails any of these checks
    raises `MaterializationExecutionError` - the whole plan fails with
    it; a plan with any rejected connection is never produced with a
    partial `ConnectionDecision` vector."""

    decisions: List[ConnectionDecision] = []
    for row in raw_connections:
        if row.ptnr3_present:
            raise MaterializationExecutionError(
                "UNSUPPORTED_PARTNER", f"struct_conn row {row.ordinal} declares a non-null third partner", phase="plan"
            )
        if not row.id:
            raise MaterializationExecutionError(
                "INVALID_CIF", f"struct_conn row {row.ordinal} has an absent or empty id", phase="plan"
            )
        if not row.conn_type:
            raise MaterializationExecutionError(
                "INVALID_CIF", f"struct_conn row {row.ordinal} has an absent or empty conn_type_id", phase="plan"
            )

        atoms1 = _resolve_endpoint_atoms(row.ptnr1, atoms_in_model)
        atoms2 = _resolve_endpoint_atoms(row.ptnr2, atoms_in_model)

        retained1 = {atom_decisions_by_ordinal[a.ordinal].retained for a in atoms1}
        retained2 = {atom_decisions_by_ordinal[a.ordinal].retained for a in atoms2}
        if len(retained1) > 1 or len(retained2) > 1:
            raise MaterializationExecutionError(
                "AMBIGUOUS_ENDPOINT", f"struct_conn row {row.ordinal} endpoint atoms disagree on retention", phase="plan"
            )
        endpoint1_retained = next(iter(retained1))
        endpoint2_retained = next(iter(retained2))
        retained_endpoints = int(endpoint1_retained) + int(endpoint2_retained)

        ptnr1_resolution = _resolve_symmetry_endpoint(row.ptnr1.symmetry_token, tables)
        ptnr2_resolution = _resolve_symmetry_endpoint(row.ptnr2.symmetry_token, tables)
        identity_basis = (ptnr1_resolution.identity_basis, ptnr2_resolution.identity_basis)
        if identity_basis in _IMPOSSIBLE_IDENTITY_BASIS_PAIRS:
            raise MaterializationExecutionError(
                "UNSUPPORTED_SYMMETRY", f"struct_conn row {row.ordinal} has a contractually impossible identity_basis pair", phase="plan"
            )

        conn_type_lower = row.conn_type.lower()

        if retained_endpoints == 0:
            symmetry_class = _classify_symmetry_class(ptnr1_resolution, ptnr2_resolution)
            action: ConnectionAction = "omit"
            reason: ConnectionReason = "both-removed"
        else:
            if conn_type_lower not in _RECOGNIZED_CONNECTION_TYPES:
                raise MaterializationExecutionError(
                    "UNSUPPORTED_TYPE", f"struct_conn row {row.ordinal} has unrecognized conn_type {row.conn_type!r} at K>0", phase="plan"
                )
            if ptnr1_resolution.insufficient_proof or ptnr2_resolution.insufficient_proof:
                raise MaterializationExecutionError(
                    "UNSUPPORTED_SYMMETRY", f"struct_conn row {row.ordinal} has an unresolved symmetry operator at K>0", phase="plan"
                )
            symmetry_class = _classify_symmetry_class(ptnr1_resolution, ptnr2_resolution)

            if retained_endpoints == 2:
                if symmetry_class == "both-identity":
                    action, reason = "preserve", "both-retained"
                else:
                    raise MaterializationExecutionError(
                        "UNSUPPORTED_SYMMETRY",
                        f"struct_conn row {row.ordinal} K=2 requires both-identity symmetry (got {symmetry_class!r})",
                        phase="plan",
                    )
            else:  # retained_endpoints == 1
                if symmetry_class != "both-identity":
                    raise MaterializationExecutionError(
                        "UNSUPPORTED_SYMMETRY",
                        f"struct_conn row {row.ordinal} K=1 requires both-identity symmetry (got {symmetry_class!r})",
                        phase="plan",
                    )
                if conn_type_lower not in _EXCEPTION_CONNECTION_TYPES:
                    raise MaterializationExecutionError(
                        "COVALENT_CUT",
                        f"struct_conn row {row.ordinal} is a K=1 cut of a {conn_type_lower!r} connection, "
                        "never a metalc/hydrog exception",
                        phase="plan",
                    )
                removed_atoms, removed_decision_reason = (
                    (atoms1, atom_decisions_by_ordinal[atoms1[0].ordinal].reason)
                    if not endpoint1_retained
                    else (atoms2, atom_decisions_by_ordinal[atoms2[0].ordinal].reason)
                )
                if removed_decision_reason not in ("chain-complement", "disposition-remove"):
                    raise MaterializationExecutionError(
                        "COVALENT_CUT",
                        f"struct_conn row {row.ordinal} K=1 omission requires the removed partner to be a "
                        "whole recognized exclusion (chain-complement/disposition-remove), "
                        f"got reason={removed_decision_reason!r}",
                        phase="plan",
                    )
                action = "omit"
                reason = "whole-partner-removed-metalc" if conn_type_lower == "metalc" else "whole-partner-removed-hydrog"

        decisions.append(
            ConnectionDecision(
                ordinal=row.ordinal,
                id=row.id,
                row_key=compute_row_key(
                    source_context_key=source_context_key, category="struct_conn",
                    ordinal=row.ordinal, tokens=row.source_tokens,
                ),
                conn_type_token=row.conn_type,
                endpoint_atom_ids=(
                    tuple(a.atom_id for a in atoms1),
                    tuple(a.atom_id for a in atoms2),
                ),
                identity_basis=identity_basis,
                retained_endpoints=retained_endpoints,
                symmetry_class=symmetry_class,
                action=action,
                reason=reason,
            )
        )

    return tuple(decisions)


# --------------------------------------------------------------------------
# Section 11: the pinned 94-category catalogue (CATALOGUE_SHA256 above is
# the SHA-256 of the governance-committed bytes of CATALOG_021.json at the
# authorized snapshot - verified via `git show <snapshot>:<path> | sha256sum`
# before this literal was transcribed) and its row-rule engine.
#
# This constant is embedded, never read from disk at runtime (Section 11:
# "não criar catálogo configurável, lookup por diretório corrente ou
# dependência externa"). `row_rule` dispatch below treats
# RETAINED_ENTITY/RETAINED_ENTITY_ALL_SEQUENCE/RETAINED_ENTITY_ALL_BRANCH as
# one family (filter by the column whose `references` entry is
# "_entity.id", or the category's own key column when self-referential -
# i.e. `_entity` itself) and RETAINED_ASYM/RETAINED_ASYM_ALL_SEQUENCE/
# RETAINED_ASYM_ALL_BRANCH as the symmetric family keyed on
# "_struct_asym.id": the "ALL_SEQUENCE"/"ALL_BRANCH" suffix only documents
# that the kept rows are the *entire* declared sequence/branch of a
# surviving entity/asym (Section 11: "manter toda a sequência declarada...
# não afirmar que sequência declarada equivale a átomos presentes"), not a
# different membership test.
#
# C-05: the catalogue's top-level `required_categories`/`conditional_categories`
# fields (never transcribed into `_FROZEN_CATALOGUE` above, which only
# carries each category's own projection) - a retained polymer/branched
# entity or a preserved `struct_conn` row makes its companion categories
# mandatory, not merely "optional and absent", and the four categories
# below are always required outright. Transcribed from the same pinned
# `CATALOG_021.json` (`CATALOGUE_SHA256`).
_CATALOGUE_REQUIRED_CATEGORIES: Tuple[str, ...] = ("_atom_site", "_struct_asym", "_entity", "_chem_comp")
_CATALOGUE_CONDITIONAL_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "retained_polymer_entity": ("_entity_poly", "_entity_poly_seq", "_pdbx_poly_seq_scheme"),
    "retained_branched_entity": ("_pdbx_entity_branch", "_pdbx_entity_branch_list", "_pdbx_branch_scheme"),
    "retained_struct_conn": ("_struct_conn_type",),
}
_FROZEN_CATALOGUE: Dict[str, Dict[str, object]] = {'_atom_site': {'action': 'filter',
                'key': ('id',),
                'required_tags': ('id',
                                  'type_symbol',
                                  'label_atom_id',
                                  'label_alt_id',
                                  'label_comp_id',
                                  'label_asym_id',
                                  'label_entity_id',
                                  'label_seq_id',
                                  'auth_asym_id',
                                  'auth_seq_id',
                                  'pdbx_pdb_ins_code',
                                  'pdbx_pdb_model_num',
                                  'cartn_x',
                                  'cartn_y',
                                  'cartn_z',
                                  'occupancy',
                                  'b_iso_or_equiv'),
                'preserve_tags': ('group_pdb',
                                  'id',
                                  'type_symbol',
                                  'label_atom_id',
                                  'label_alt_id',
                                  'label_comp_id',
                                  'label_asym_id',
                                  'label_entity_id',
                                  'label_seq_id',
                                  'auth_atom_id',
                                  'auth_comp_id',
                                  'auth_asym_id',
                                  'auth_seq_id',
                                  'pdbx_pdb_ins_code',
                                  'pdbx_pdb_model_num',
                                  'cartn_x',
                                  'cartn_y',
                                  'cartn_z',
                                  'occupancy',
                                  'b_iso_or_equiv',
                                  'pdbx_formal_charge'),
                'omit_tags': (),
                'generate_tags': (),
                'row_rule': 'ATOM_MASK',
                'references': {'label_asym_id': '_struct_asym.id',
                               'label_entity_id': '_entity.id',
                               'label_comp_id': '_chem_comp.id',
                               'label_seq_id': 'conditional polymer: '
                                               '_entity_poly_seq(entity_id,num,mon_id) using '
                                               'label_entity_id,label_seq_id,label_comp_id; '
                                               'nonpolymer true null accepted'}},
 '_atom_site_anisotrop': {'action': 'filter',
                          'key': ('id',),
                          'required_tags': ('id',),
                          'preserve_tags': ('id',
                                            'type_symbol',
                                            'pdbx_pdb_atom_name',
                                            'pdbx_pdb_ins_code',
                                            'pdbx_pdb_model_num',
                                            'pdbx_pdb_residue_name',
                                            'pdbx_pdb_residue_no',
                                            'pdbx_pdb_strand_id',
                                            'pdbx_auth_alt_id',
                                            'pdbx_auth_asym_id',
                                            'pdbx_auth_atom_id',
                                            'pdbx_auth_atom_name',
                                            'pdbx_auth_comp_id',
                                            'pdbx_auth_seq_id',
                                            'pdbx_label_alt_id',
                                            'pdbx_label_asym_id',
                                            'pdbx_label_atom_id',
                                            'pdbx_label_comp_id',
                                            'pdbx_label_ins_code',
                                            'pdbx_label_seq_id',
                                            'pdbx_not_in_asym',
                                            'u[1][1]',
                                            'u[1][2]',
                                            'u[1][3]',
                                            'u[2][2]',
                                            'u[2][3]',
                                            'u[3][3]',
                                            'b[1][1]',
                                            'b[1][2]',
                                            'b[1][3]',
                                            'b[2][2]',
                                            'b[2][3]',
                                            'b[3][3]',
                                            'u[1][1]_esd',
                                            'u[1][2]_esd',
                                            'u[1][3]_esd',
                                            'u[2][2]_esd',
                                            'u[2][3]_esd',
                                            'u[3][3]_esd',
                                            'b[1][1]_esd',
                                            'b[1][2]_esd',
                                            'b[1][3]_esd',
                                            'b[2][2]_esd',
                                            'b[2][3]_esd',
                                            'b[3][3]_esd'),
                          'omit_tags': (),
                          'generate_tags': (),
                          'row_rule': 'ANISOTROP_ATOM_ID',
                          'references': {'id': '_atom_site.id'}},
 '_struct_asym': {'action': 'filter',
                  'key': ('id',),
                  'required_tags': ('id', 'entity_id'),
                  'preserve_tags': ('id',
                                    'entity_id',
                                    'pdbx_blank_pdb_chainid_flag',
                                    'pdbx_modified',
                                    'details'),
                  'omit_tags': (),
                  'generate_tags': (),
                  'row_rule': 'RETAINED_ASYM',
                  'references': {'entity_id': '_entity.id'}},
 '_entity': {'action': 'filter',
             'key': ('id',),
             'required_tags': ('id', 'type'),
             'preserve_tags': ('id', 'type', 'src_method', 'pdbx_description', 'details'),
             'omit_tags': ('formula_weight',
                           'pdbx_number_of_molecules',
                           'pdbx_ec',
                           'pdbx_mutation',
                           'pdbx_fragment'),
             'generate_tags': (),
             'row_rule': 'RETAINED_ENTITY',
             'references': {}},
 '_entity_poly': {'action': 'filter',
                  'key': ('entity_id',),
                  'required_tags': ('entity_id', 'type'),
                  'preserve_tags': ('entity_id',
                                    'type',
                                    'nstd_linkage',
                                    'nstd_monomer',
                                    'pdbx_seq_one_letter_code',
                                    'pdbx_seq_one_letter_code_can'),
                  'omit_tags': ('pdbx_strand_id', 'pdbx_target_identifier'),
                  'generate_tags': (),
                  'row_rule': 'RETAINED_ENTITY',
                  'references': {'entity_id': '_entity.id'}},
 '_entity_poly_seq': {'action': 'filter',
                      'key': ('entity_id', 'num', 'mon_id'),
                      'required_tags': ('entity_id', 'num', 'mon_id'),
                      'preserve_tags': ('entity_id', 'num', 'mon_id', 'hetero'),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'RETAINED_ENTITY_ALL_SEQUENCE',
                      'references': {'entity_id': '_entity.id', 'mon_id': '_chem_comp.id'}},
 '_chem_comp': {'action': 'filter',
                'key': ('id',),
                'required_tags': ('id', 'type'),
                'preserve_tags': ('id',
                                  'type',
                                  'mon_nstd_flag',
                                  'name',
                                  'pdbx_synonyms',
                                  'formula',
                                  'formula_weight',
                                  'pdbx_formal_charge',
                                  'one_letter_code',
                                  'three_letter_code',
                                  'pdbx_type',
                                  'pdbx_ambiguous_flag',
                                  'pdbx_release_status',
                                  'pdbx_modified_date',
                                  'pdbx_initial_date',
                                  'pdbx_replaced_by',
                                  'pdbx_replaces',
                                  'pdbx_processing_site'),
                'omit_tags': (),
                'generate_tags': (),
                'row_rule': 'REFERENCED_CCD_CLOSURE',
                'references': {}},
 '_pdbx_entity_nonpoly': {'action': 'filter',
                          'key': ('entity_id', 'comp_id'),
                          'required_tags': ('entity_id', 'comp_id'),
                          'preserve_tags': ('entity_id', 'comp_id', 'name'),
                          'omit_tags': (),
                          'generate_tags': (),
                          'row_rule': 'RETAINED_ENTITY',
                          'references': {'entity_id': '_entity.id', 'comp_id': '_chem_comp.id'}},
 '_pdbx_poly_seq_scheme': {'action': 'filter',
                           'key': ('asym_id', 'seq_id', 'mon_id'),
                           'required_tags': ('asym_id', 'entity_id', 'seq_id', 'mon_id'),
                           'preserve_tags': ('asym_id',
                                             'entity_id',
                                             'seq_id',
                                             'mon_id',
                                             'ndb_seq_num',
                                             'pdb_seq_num',
                                             'auth_seq_num',
                                             'pdb_mon_id',
                                             'auth_mon_id',
                                             'pdb_strand_id',
                                             'pdb_ins_code',
                                             'hetero'),
                           'omit_tags': (),
                           'generate_tags': (),
                           'row_rule': 'RETAINED_ASYM_ALL_SEQUENCE',
                           'references': {'asym_id': '_struct_asym.id',
                                          'entity_id': '_entity.id',
                                          'mon_id': '_chem_comp.id'}},
 '_pdbx_nonpoly_scheme': {'action': 'filter',
                          'key': ('asym_id', 'ndb_seq_num', 'mon_id'),
                          'required_tags': ('asym_id', 'entity_id', 'mon_id', 'ndb_seq_num'),
                          'preserve_tags': ('asym_id',
                                            'entity_id',
                                            'mon_id',
                                            'ndb_seq_num',
                                            'pdb_seq_num',
                                            'auth_seq_num',
                                            'pdb_mon_id',
                                            'auth_mon_id',
                                            'pdb_strand_id',
                                            'pdb_ins_code'),
                          'omit_tags': (),
                          'generate_tags': (),
                          'row_rule': 'RETAINED_ASYM',
                          'references': {'asym_id': '_struct_asym.id',
                                         'entity_id': '_entity.id',
                                         'mon_id': '_chem_comp.id'}},
 '_pdbx_entity_branch': {'action': 'filter',
                         'key': ('entity_id',),
                         'required_tags': ('entity_id', 'type'),
                         'preserve_tags': ('entity_id', 'type'),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'RETAINED_ENTITY',
                         'references': {'entity_id': '_entity.id'}},
 '_pdbx_entity_branch_list': {'action': 'filter',
                              'key': ('entity_id', 'num', 'comp_id'),
                              'required_tags': ('entity_id', 'num', 'comp_id'),
                              'preserve_tags': ('entity_id', 'num', 'comp_id', 'hetero'),
                              'omit_tags': (),
                              'generate_tags': (),
                              'row_rule': 'RETAINED_ENTITY_ALL_BRANCH',
                              'references': {'entity_id': '_entity.id',
                                             'comp_id': '_chem_comp.id'}},
 '_pdbx_entity_branch_link': {'action': 'filter',
                              'key': ('link_id',),
                              'required_tags': ('link_id',
                                                'entity_id',
                                                'entity_branch_list_num_1',
                                                'comp_id_1',
                                                'atom_id_1',
                                                'entity_branch_list_num_2',
                                                'comp_id_2',
                                                'atom_id_2'),
                              'preserve_tags': ('link_id',
                                                'entity_id',
                                                'entity_branch_list_num_1',
                                                'comp_id_1',
                                                'atom_id_1',
                                                'leaving_atom_id_1',
                                                'atom_stereo_config_1',
                                                'entity_branch_list_num_2',
                                                'comp_id_2',
                                                'atom_id_2',
                                                'leaving_atom_id_2',
                                                'atom_stereo_config_2',
                                                'value_order',
                                                'details'),
                              'omit_tags': (),
                              'generate_tags': (),
                              'row_rule': 'RETAINED_ENTITY_ALL_BRANCH',
                              'references': {'entity_id': '_entity.id',
                                             'entity_branch_list_num_1': '_pdbx_entity_branch_list(entity_id,num,comp_id_1)',
                                             'entity_branch_list_num_2': '_pdbx_entity_branch_list(entity_id,num,comp_id_2)'}},
 '_pdbx_branch_scheme': {'action': 'filter',
                         'key': ('asym_id', 'entity_id', 'mon_id', 'num'),
                         'required_tags': ('asym_id', 'entity_id', 'mon_id', 'num'),
                         'preserve_tags': ('asym_id',
                                           'entity_id',
                                           'mon_id',
                                           'num',
                                           'auth_asym_id',
                                           'auth_mon_id',
                                           'auth_seq_num',
                                           'hetero',
                                           'pdb_asym_id',
                                           'pdb_ins_code',
                                           'pdb_mon_id',
                                           'pdb_seq_num'),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'RETAINED_ASYM_ALL_BRANCH',
                         'references': {'asym_id': '_struct_asym.id',
                                        'entity_id': '_entity.id',
                                        'mon_id': '_chem_comp.id',
                                        'num': '_pdbx_entity_branch_list(entity_id,num,mon_id)'}},
 '_struct_conn': {'action': 'filter',
                  'key': ('id',),
                  'required_tags': ('id',
                                    'conn_type_id',
                                    'ptnr1_label_asym_id',
                                    'ptnr1_label_comp_id',
                                    'ptnr1_label_atom_id',
                                    'ptnr2_label_asym_id',
                                    'ptnr2_label_comp_id',
                                    'ptnr2_label_atom_id'),
                  'preserve_tags': ('id',
                                    'conn_type_id',
                                    'details',
                                    'pdbx_dist_value',
                                    'pdbx_value_order',
                                    'pdbx_role',
                                    'pdbx_leaving_atom_flag',
                                    'ptnr1_label_asym_id',
                                    'ptnr1_label_comp_id',
                                    'ptnr1_label_seq_id',
                                    'ptnr1_label_atom_id',
                                    'ptnr1_auth_asym_id',
                                    'ptnr1_auth_comp_id',
                                    'ptnr1_auth_seq_id',
                                    'ptnr1_auth_atom_id',
                                    'ptnr1_symmetry',
                                    'ptnr1_role',
                                    'ptnr1_label_alt_id',
                                    'pdbx_ptnr1_pdb_ins_code',
                                    'pdbx_ptnr1_label_alt_id',
                                    'pdbx_ptnr1_auth_alt_id',
                                    'ptnr2_label_asym_id',
                                    'ptnr2_label_comp_id',
                                    'ptnr2_label_seq_id',
                                    'ptnr2_label_atom_id',
                                    'ptnr2_auth_asym_id',
                                    'ptnr2_auth_comp_id',
                                    'ptnr2_auth_seq_id',
                                    'ptnr2_auth_atom_id',
                                    'ptnr2_symmetry',
                                    'ptnr2_role',
                                    'ptnr2_label_alt_id',
                                    'pdbx_ptnr2_pdb_ins_code',
                                    'pdbx_ptnr2_label_alt_id',
                                    'pdbx_ptnr2_auth_alt_id',
                                    'pdbx_ptnr3_pdb_ins_code',
                                    'pdbx_ptnr3_auth_alt_id',
                                    'pdbx_ptnr3_auth_asym_id',
                                    'pdbx_ptnr3_auth_atom_id',
                                    'pdbx_ptnr3_auth_comp_id',
                                    'pdbx_ptnr3_auth_ins_code',
                                    'pdbx_ptnr3_auth_seq_id',
                                    'pdbx_ptnr3_label_alt_id',
                                    'pdbx_ptnr3_label_asym_id',
                                    'pdbx_ptnr3_label_atom_id',
                                    'pdbx_ptnr3_label_comp_id',
                                    'pdbx_ptnr3_label_seq_id'),
                  'omit_tags': ('pdbx_pdb_id',
                                'pdbx_ptnr1_standard_comp_id',
                                'pdbx_ptnr1_atom_stereo_config',
                                'pdbx_ptnr1_leaving_atom_id',
                                'pdbx_ptnr1_mod_name',
                                'pdbx_ptnr1_replaced_atom',
                                'pdbx_ptnr1_sugar_name',
                                'pdbx_ptnr2_atom_stereo_config',
                                'pdbx_ptnr2_leaving_atom_id'),
                  'generate_tags': (),
                  'row_rule': 'CONNECTION_ACTIONS',
                  'references': {'conn_type_id': '_struct_conn_type.id'}},
 '_struct_conn_type': {'action': 'filter',
                       'key': ('id',),
                       'required_tags': ('id',),
                       'preserve_tags': ('id', 'criteria', 'reference'),
                       'omit_tags': (),
                       'generate_tags': (),
                       'row_rule': 'USED_CONNECTION_TYPES',
                       'references': {}},
 '_atom_type': {'action': 'filter',
                'key': ('symbol',),
                'required_tags': ('symbol',),
                'preserve_tags': ('symbol',),
                'omit_tags': (),
                'generate_tags': (),
                'row_rule': 'USED_ELEMENTS',
                'references': {}},
 '_cell': {'action': 'filter',
           'key': ('entry_id',),
           'required_tags': (),
           'preserve_tags': ('length_a',
                             'length_b',
                             'length_c',
                             'angle_alpha',
                             'angle_beta',
                             'angle_gamma',
                             'z_pdb',
                             'angle_alpha_esd',
                             'angle_beta_esd',
                             'angle_gamma_esd',
                             'details',
                             'formula_units_z',
                             'length_a_esd',
                             'length_b_esd',
                             'length_c_esd',
                             'volume',
                             'volume_esd',
                             'reciprocal_angle_alpha',
                             'reciprocal_angle_beta',
                             'reciprocal_angle_gamma',
                             'reciprocal_angle_alpha_esd',
                             'reciprocal_angle_beta_esd',
                             'reciprocal_angle_gamma_esd',
                             'reciprocal_length_a',
                             'reciprocal_length_b',
                             'reciprocal_length_c',
                             'reciprocal_length_a_esd',
                             'reciprocal_length_b_esd',
                             'reciprocal_length_c_esd',
                             'pdbx_esd_method'),
           'omit_tags': ('pdbx_unique_axis',),
           'generate_tags': ('entry_id',),
           'row_rule': 'SOURCE_FRAME_SINGLETON',
           'references': {}},
 '_symmetry': {'action': 'filter',
               'key': ('entry_id',),
               'required_tags': (),
               'preserve_tags': ('space_group_name_h-m',
                                 'int_tables_number',
                                 'space_group_name_hall',
                                 'cell_setting',
                                 'pdbx_full_space_group_name_h-m'),
               'omit_tags': (),
               'generate_tags': ('entry_id',),
               'row_rule': 'SOURCE_FRAME_SINGLETON',
               'references': {}},
 '_symmetry_equiv': {'action': 'filter',
                     'key': ('id',),
                     'required_tags': ('id', 'pos_as_xyz'),
                     'preserve_tags': ('id', 'pos_as_xyz'),
                     'omit_tags': (),
                     'generate_tags': (),
                     'row_rule': 'SOURCE_OPERATORS',
                     'references': {}},
 '_space_group_symop': {'action': 'filter',
                        'key': ('id',),
                        'required_tags': ('id', 'operation_xyz'),
                        'preserve_tags': ('id', 'operation_xyz'),
                        'omit_tags': ('sg_id',),
                        'generate_tags': (),
                        'row_rule': 'SOURCE_OPERATORS',
                        'references': {}},
 '_pdbx_struct_mod_residue': {'action': 'filter',
                              'key': ('id',),
                              'required_tags': ('id',),
                              'preserve_tags': ('id',
                                                'label_asym_id',
                                                'label_comp_id',
                                                'label_seq_id',
                                                'auth_asym_id',
                                                'auth_comp_id',
                                                'auth_seq_id',
                                                'pdb_ins_code',
                                                'pdb_model_num',
                                                'parent_comp_id',
                                                'details'),
                              'omit_tags': (),
                              'generate_tags': (),
                              'row_rule': 'MODIFIED_RESIDUE_CONTEXT',
                              'references': {'parent_comp_id': '_chem_comp.id'}},
 '_pdbx_unobs_or_zero_occ_residues': {'action': 'filter',
                                      'key': ('id',),
                                      'required_tags': ('id',),
                                      'preserve_tags': ('id',
                                                        'polymer_flag',
                                                        'occupancy_flag',
                                                        'pdb_model_num',
                                                        'auth_asym_id',
                                                        'auth_comp_id',
                                                        'auth_seq_id',
                                                        'pdb_ins_code',
                                                        'label_asym_id',
                                                        'label_comp_id',
                                                        'label_seq_id'),
                                      'omit_tags': (),
                                      'generate_tags': (),
                                      'row_rule': 'MISSING_RESIDUE_CONTEXT',
                                      'references': {}},
 '_pdbx_unobs_or_zero_occ_atoms': {'action': 'filter',
                                   'key': ('id',),
                                   'required_tags': ('id',),
                                   'preserve_tags': ('id',
                                                     'polymer_flag',
                                                     'occupancy_flag',
                                                     'pdb_model_num',
                                                     'auth_asym_id',
                                                     'auth_comp_id',
                                                     'auth_seq_id',
                                                     'pdb_ins_code',
                                                     'auth_atom_id',
                                                     'label_asym_id',
                                                     'label_comp_id',
                                                     'label_seq_id',
                                                     'label_atom_id',
                                                     'label_alt_id'),
                                   'omit_tags': (),
                                   'generate_tags': (),
                                   'row_rule': 'MISSING_ATOM_CONTEXT',
                                   'references': {}},
 '_entry': {'action': 'generate',
            'key': ('id',),
            'required_tags': (),
            'preserve_tags': (),
            'omit_tags': (),
            'generate_tags': ('id',),
            'row_rule': 'DERIVED_ENTRY',
            'references': {}},
 '_audit_conform': {'action': 'omit',
                    'key': (),
                    'required_tags': (),
                    'preserve_tags': (),
                    'omit_tags': (),
                    'generate_tags': (),
                    'row_rule': 'AUDIT_OMITTED_CATEGORY',
                    'references': {}},
 '_database_2': {'action': 'omit',
                 'key': (),
                 'required_tags': (),
                 'preserve_tags': (),
                 'omit_tags': (),
                 'generate_tags': (),
                 'row_rule': 'AUDIT_OMITTED_CATEGORY',
                 'references': {}},
 '_pdbx_database_status': {'action': 'omit',
                           'key': (),
                           'required_tags': (),
                           'preserve_tags': (),
                           'omit_tags': (),
                           'generate_tags': (),
                           'row_rule': 'AUDIT_OMITTED_CATEGORY',
                           'references': {}},
 '_audit_author': {'action': 'omit',
                   'key': (),
                   'required_tags': (),
                   'preserve_tags': (),
                   'omit_tags': (),
                   'generate_tags': (),
                   'row_rule': 'AUDIT_OMITTED_CATEGORY',
                   'references': {}},
 '_citation': {'action': 'omit',
               'key': (),
               'required_tags': (),
               'preserve_tags': (),
               'omit_tags': (),
               'generate_tags': (),
               'row_rule': 'AUDIT_OMITTED_CATEGORY',
               'references': {}},
 '_citation_author': {'action': 'omit',
                      'key': (),
                      'required_tags': (),
                      'preserve_tags': (),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'AUDIT_OMITTED_CATEGORY',
                      'references': {}},
 '_pdbx_entity_src_syn': {'action': 'omit',
                          'key': (),
                          'required_tags': (),
                          'preserve_tags': (),
                          'omit_tags': (),
                          'generate_tags': (),
                          'row_rule': 'AUDIT_OMITTED_CATEGORY',
                          'references': {}},
 '_entity_src_gen': {'action': 'omit',
                     'key': (),
                     'required_tags': (),
                     'preserve_tags': (),
                     'omit_tags': (),
                     'generate_tags': (),
                     'row_rule': 'AUDIT_OMITTED_CATEGORY',
                     'references': {}},
 '_entity_src_nat': {'action': 'omit',
                     'key': (),
                     'required_tags': (),
                     'preserve_tags': (),
                     'omit_tags': (),
                     'generate_tags': (),
                     'row_rule': 'AUDIT_OMITTED_CATEGORY',
                     'references': {}},
 '_struct_ref': {'action': 'omit',
                 'key': (),
                 'required_tags': (),
                 'preserve_tags': (),
                 'omit_tags': (),
                 'generate_tags': (),
                 'row_rule': 'AUDIT_OMITTED_CATEGORY',
                 'references': {}},
 '_struct_ref_seq': {'action': 'omit',
                     'key': (),
                     'required_tags': (),
                     'preserve_tags': (),
                     'omit_tags': (),
                     'generate_tags': (),
                     'row_rule': 'AUDIT_OMITTED_CATEGORY',
                     'references': {}},
 '_struct_ref_seq_dif': {'action': 'omit',
                         'key': (),
                         'required_tags': (),
                         'preserve_tags': (),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'AUDIT_OMITTED_CATEGORY',
                         'references': {}},
 '_chem_comp_atom': {'action': 'omit',
                     'key': (),
                     'required_tags': (),
                     'preserve_tags': (),
                     'omit_tags': (),
                     'generate_tags': (),
                     'row_rule': 'AUDIT_OMITTED_CATEGORY',
                     'references': {}},
 '_chem_comp_bond': {'action': 'omit',
                     'key': (),
                     'required_tags': (),
                     'preserve_tags': (),
                     'omit_tags': (),
                     'generate_tags': (),
                     'row_rule': 'AUDIT_OMITTED_CATEGORY',
                     'references': {}},
 '_exptl': {'action': 'omit',
            'key': (),
            'required_tags': (),
            'preserve_tags': (),
            'omit_tags': (),
            'generate_tags': (),
            'row_rule': 'AUDIT_OMITTED_CATEGORY',
            'references': {}},
 '_exptl_crystal': {'action': 'omit',
                    'key': (),
                    'required_tags': (),
                    'preserve_tags': (),
                    'omit_tags': (),
                    'generate_tags': (),
                    'row_rule': 'AUDIT_OMITTED_CATEGORY',
                    'references': {}},
 '_exptl_crystal_grow': {'action': 'omit',
                         'key': (),
                         'required_tags': (),
                         'preserve_tags': (),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'AUDIT_OMITTED_CATEGORY',
                         'references': {}},
 '_diffrn': {'action': 'omit',
             'key': (),
             'required_tags': (),
             'preserve_tags': (),
             'omit_tags': (),
             'generate_tags': (),
             'row_rule': 'AUDIT_OMITTED_CATEGORY',
             'references': {}},
 '_diffrn_detector': {'action': 'omit',
                      'key': (),
                      'required_tags': (),
                      'preserve_tags': (),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'AUDIT_OMITTED_CATEGORY',
                      'references': {}},
 '_diffrn_radiation': {'action': 'omit',
                       'key': (),
                       'required_tags': (),
                       'preserve_tags': (),
                       'omit_tags': (),
                       'generate_tags': (),
                       'row_rule': 'AUDIT_OMITTED_CATEGORY',
                       'references': {}},
 '_diffrn_radiation_wavelength': {'action': 'omit',
                                  'key': (),
                                  'required_tags': (),
                                  'preserve_tags': (),
                                  'omit_tags': (),
                                  'generate_tags': (),
                                  'row_rule': 'AUDIT_OMITTED_CATEGORY',
                                  'references': {}},
 '_diffrn_source': {'action': 'omit',
                    'key': (),
                    'required_tags': (),
                    'preserve_tags': (),
                    'omit_tags': (),
                    'generate_tags': (),
                    'row_rule': 'AUDIT_OMITTED_CATEGORY',
                    'references': {}},
 '_reflns': {'action': 'omit',
             'key': (),
             'required_tags': (),
             'preserve_tags': (),
             'omit_tags': (),
             'generate_tags': (),
             'row_rule': 'AUDIT_OMITTED_CATEGORY',
             'references': {}},
 '_reflns_shell': {'action': 'omit',
                   'key': (),
                   'required_tags': (),
                   'preserve_tags': (),
                   'omit_tags': (),
                   'generate_tags': (),
                   'row_rule': 'AUDIT_OMITTED_CATEGORY',
                   'references': {}},
 '_refine': {'action': 'omit',
             'key': (),
             'required_tags': (),
             'preserve_tags': (),
             'omit_tags': (),
             'generate_tags': (),
             'row_rule': 'AUDIT_OMITTED_CATEGORY',
             'references': {}},
 '_refine_hist': {'action': 'omit',
                  'key': (),
                  'required_tags': (),
                  'preserve_tags': (),
                  'omit_tags': (),
                  'generate_tags': (),
                  'row_rule': 'AUDIT_OMITTED_CATEGORY',
                  'references': {}},
 '_refine_ls_restr': {'action': 'omit',
                      'key': (),
                      'required_tags': (),
                      'preserve_tags': (),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'AUDIT_OMITTED_CATEGORY',
                      'references': {}},
 '_refine_ls_shell': {'action': 'omit',
                      'key': (),
                      'required_tags': (),
                      'preserve_tags': (),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'AUDIT_OMITTED_CATEGORY',
                      'references': {}},
 '_struct': {'action': 'omit',
             'key': (),
             'required_tags': (),
             'preserve_tags': (),
             'omit_tags': (),
             'generate_tags': (),
             'row_rule': 'AUDIT_OMITTED_CATEGORY',
             'references': {}},
 '_struct_keywords': {'action': 'omit',
                      'key': (),
                      'required_tags': (),
                      'preserve_tags': (),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'AUDIT_OMITTED_CATEGORY',
                      'references': {}},
 '_struct_conf': {'action': 'omit',
                  'key': (),
                  'required_tags': (),
                  'preserve_tags': (),
                  'omit_tags': (),
                  'generate_tags': (),
                  'row_rule': 'AUDIT_OMITTED_CATEGORY',
                  'references': {}},
 '_struct_conf_type': {'action': 'omit',
                       'key': (),
                       'required_tags': (),
                       'preserve_tags': (),
                       'omit_tags': (),
                       'generate_tags': (),
                       'row_rule': 'AUDIT_OMITTED_CATEGORY',
                       'references': {}},
 '_struct_sheet': {'action': 'omit',
                   'key': (),
                   'required_tags': (),
                   'preserve_tags': (),
                   'omit_tags': (),
                   'generate_tags': (),
                   'row_rule': 'AUDIT_OMITTED_CATEGORY',
                   'references': {}},
 '_struct_sheet_order': {'action': 'omit',
                         'key': (),
                         'required_tags': (),
                         'preserve_tags': (),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'AUDIT_OMITTED_CATEGORY',
                         'references': {}},
 '_struct_sheet_range': {'action': 'omit',
                         'key': (),
                         'required_tags': (),
                         'preserve_tags': (),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'AUDIT_OMITTED_CATEGORY',
                         'references': {}},
 '_struct_sheet_hbond': {'action': 'omit',
                         'key': (),
                         'required_tags': (),
                         'preserve_tags': (),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'AUDIT_OMITTED_CATEGORY',
                         'references': {}},
 '_struct_mon_prot_cis': {'action': 'omit',
                          'key': (),
                          'required_tags': (),
                          'preserve_tags': (),
                          'omit_tags': (),
                          'generate_tags': (),
                          'row_rule': 'AUDIT_OMITTED_CATEGORY',
                          'references': {}},
 '_struct_site': {'action': 'omit',
                  'key': (),
                  'required_tags': (),
                  'preserve_tags': (),
                  'omit_tags': (),
                  'generate_tags': (),
                  'row_rule': 'AUDIT_OMITTED_CATEGORY',
                  'references': {}},
 '_struct_site_gen': {'action': 'omit',
                      'key': (),
                      'required_tags': (),
                      'preserve_tags': (),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'AUDIT_OMITTED_CATEGORY',
                      'references': {}},
 '_atom_sites': {'action': 'omit',
                 'key': (),
                 'required_tags': (),
                 'preserve_tags': (),
                 'omit_tags': (),
                 'generate_tags': (),
                 'row_rule': 'AUDIT_OMITTED_CATEGORY',
                 'references': {}},
 '_pdbx_struct_special_symmetry': {'action': 'guard_then_omit',
                                   'key': ('id',),
                                   'required_tags': ('id',),
                                   'preserve_tags': (),
                                   'omit_tags': ('id',
                                                 'pdb_model_num',
                                                 'auth_comp_id',
                                                 'auth_asym_id',
                                                 'auth_seq_id',
                                                 'pdb_ins_code',
                                                 'label_alt_id',
                                                 'label_asym_id',
                                                 'label_comp_id',
                                                 'label_seq_id'),
                                   'generate_tags': (),
                                   'row_rule': 'SPECIAL_POSITION_GUARD_THEN_OMIT',
                                   'references': {}},
 '_pdbx_struct_assembly': {'action': 'omit',
                           'key': (),
                           'required_tags': (),
                           'preserve_tags': (),
                           'omit_tags': (),
                           'generate_tags': (),
                           'row_rule': 'AUDIT_OMITTED_CATEGORY',
                           'references': {}},
 '_pdbx_struct_assembly_gen': {'action': 'omit',
                               'key': (),
                               'required_tags': (),
                               'preserve_tags': (),
                               'omit_tags': (),
                               'generate_tags': (),
                               'row_rule': 'AUDIT_OMITTED_CATEGORY',
                               'references': {}},
 '_pdbx_struct_oper_list': {'action': 'omit',
                            'key': (),
                            'required_tags': (),
                            'preserve_tags': (),
                            'omit_tags': (),
                            'generate_tags': (),
                            'row_rule': 'AUDIT_OMITTED_CATEGORY',
                            'references': {}},
 '_struct_ncs_oper': {'action': 'guard_then_omit',
                      'key': ('id',),
                      'required_tags': ('id',
                                        'matrix[1][1]',
                                        'matrix[1][2]',
                                        'matrix[1][3]',
                                        'matrix[2][1]',
                                        'matrix[2][2]',
                                        'matrix[2][3]',
                                        'matrix[3][1]',
                                        'matrix[3][2]',
                                        'matrix[3][3]',
                                        'vector[1]',
                                        'vector[2]',
                                        'vector[3]'),
                      'preserve_tags': (),
                      'omit_tags': ('id',
                                    'code',
                                    'matrix[1][1]',
                                    'matrix[1][2]',
                                    'matrix[1][3]',
                                    'matrix[2][1]',
                                    'matrix[2][2]',
                                    'matrix[2][3]',
                                    'matrix[3][1]',
                                    'matrix[3][2]',
                                    'matrix[3][3]',
                                    'vector[1]',
                                    'vector[2]',
                                    'vector[3]',
                                    'details'),
                      'generate_tags': (),
                      'row_rule': 'NCS_GUARD_THEN_OMIT',
                      'references': {}},
 '_pdbx_audit_revision_history': {'action': 'omit',
                                  'key': (),
                                  'required_tags': (),
                                  'preserve_tags': (),
                                  'omit_tags': (),
                                  'generate_tags': (),
                                  'row_rule': 'AUDIT_OMITTED_CATEGORY',
                                  'references': {}},
 '_pdbx_audit_revision_details': {'action': 'omit',
                                  'key': (),
                                  'required_tags': (),
                                  'preserve_tags': (),
                                  'omit_tags': (),
                                  'generate_tags': (),
                                  'row_rule': 'AUDIT_OMITTED_CATEGORY',
                                  'references': {}},
 '_pdbx_audit_revision_group': {'action': 'omit',
                                'key': (),
                                'required_tags': (),
                                'preserve_tags': (),
                                'omit_tags': (),
                                'generate_tags': (),
                                'row_rule': 'AUDIT_OMITTED_CATEGORY',
                                'references': {}},
 '_software': {'action': 'omit',
               'key': (),
               'required_tags': (),
               'preserve_tags': (),
               'omit_tags': (),
               'generate_tags': (),
               'row_rule': 'AUDIT_OMITTED_CATEGORY',
               'references': {}},
 '_pdbx_audit_support': {'action': 'omit',
                         'key': (),
                         'required_tags': (),
                         'preserve_tags': (),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'AUDIT_OMITTED_CATEGORY',
                         'references': {}},
 '_pdbx_database_related': {'action': 'omit',
                            'key': (),
                            'required_tags': (),
                            'preserve_tags': (),
                            'omit_tags': (),
                            'generate_tags': (),
                            'row_rule': 'AUDIT_OMITTED_CATEGORY',
                            'references': {}},
 '_pdbx_database_pdb_obs_spr': {'action': 'omit',
                                'key': (),
                                'required_tags': (),
                                'preserve_tags': (),
                                'omit_tags': (),
                                'generate_tags': (),
                                'row_rule': 'AUDIT_OMITTED_CATEGORY',
                                'references': {}},
 '_entity_name_com': {'action': 'omit',
                      'key': (),
                      'required_tags': (),
                      'preserve_tags': (),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'AUDIT_OMITTED_CATEGORY',
                      'references': {}},
 '_exptl_crystal_grow_comp': {'action': 'omit',
                              'key': (),
                              'required_tags': (),
                              'preserve_tags': (),
                              'omit_tags': (),
                              'generate_tags': (),
                              'row_rule': 'AUDIT_OMITTED_CATEGORY',
                              'references': {}},
 '_refine_analyze': {'action': 'omit',
                     'key': (),
                     'required_tags': (),
                     'preserve_tags': (),
                     'omit_tags': (),
                     'generate_tags': (),
                     'row_rule': 'AUDIT_OMITTED_CATEGORY',
                     'references': {}},
 '_pdbx_refine': {'action': 'omit',
                  'key': (),
                  'required_tags': (),
                  'preserve_tags': (),
                  'omit_tags': (),
                  'generate_tags': (),
                  'row_rule': 'AUDIT_OMITTED_CATEGORY',
                  'references': {}},
 '_struct_biol': {'action': 'omit',
                  'key': (),
                  'required_tags': (),
                  'preserve_tags': (),
                  'omit_tags': (),
                  'generate_tags': (),
                  'row_rule': 'AUDIT_OMITTED_CATEGORY',
                  'references': {}},
 '_database_pdb_matrix': {'action': 'omit',
                          'key': (),
                          'required_tags': (),
                          'preserve_tags': (),
                          'omit_tags': (),
                          'generate_tags': (),
                          'row_rule': 'AUDIT_OMITTED_CATEGORY',
                          'references': {}},
 '_pdbx_molecule_features': {'action': 'omit',
                             'key': (),
                             'required_tags': (),
                             'preserve_tags': (),
                             'omit_tags': (),
                             'generate_tags': (),
                             'row_rule': 'AUDIT_OMITTED_CATEGORY',
                             'references': {}},
 '_pdbx_molecule': {'action': 'omit',
                    'key': (),
                    'required_tags': (),
                    'preserve_tags': (),
                    'omit_tags': (),
                    'generate_tags': (),
                    'row_rule': 'AUDIT_OMITTED_CATEGORY',
                    'references': {}},
 '_struct_site_keywords': {'action': 'omit',
                           'key': (),
                           'required_tags': (),
                           'preserve_tags': (),
                           'omit_tags': (),
                           'generate_tags': (),
                           'row_rule': 'AUDIT_OMITTED_CATEGORY',
                           'references': {}},
 '_pdbx_struct_assembly_prop': {'action': 'omit',
                                'key': (),
                                'required_tags': (),
                                'preserve_tags': (),
                                'omit_tags': (),
                                'generate_tags': (),
                                'row_rule': 'AUDIT_OMITTED_CATEGORY',
                                'references': {}},
 '_pdbx_entry_details': {'action': 'omit',
                         'key': (),
                         'required_tags': (),
                         'preserve_tags': (),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'AUDIT_OMITTED_CATEGORY',
                         'references': {}},
 '_ndb_struct_conf_na': {'action': 'omit',
                         'key': (),
                         'required_tags': (),
                         'preserve_tags': (),
                         'omit_tags': (),
                         'generate_tags': (),
                         'row_rule': 'AUDIT_OMITTED_CATEGORY',
                         'references': {}},
 '_ndb_struct_na_base_pair': {'action': 'omit',
                              'key': (),
                              'required_tags': (),
                              'preserve_tags': (),
                              'omit_tags': (),
                              'generate_tags': (),
                              'row_rule': 'AUDIT_OMITTED_CATEGORY',
                              'references': {}},
 '_ndb_struct_na_base_pair_step': {'action': 'omit',
                                   'key': (),
                                   'required_tags': (),
                                   'preserve_tags': (),
                                   'omit_tags': (),
                                   'generate_tags': (),
                                   'row_rule': 'AUDIT_OMITTED_CATEGORY',
                                   'references': {}},
 '_pdbx_refine_tls': {'action': 'omit',
                      'key': (),
                      'required_tags': (),
                      'preserve_tags': (),
                      'omit_tags': (),
                      'generate_tags': (),
                      'row_rule': 'AUDIT_OMITTED_CATEGORY',
                      'references': {}},
 '_pdbx_refine_tls_group': {'action': 'omit',
                            'key': (),
                            'required_tags': (),
                            'preserve_tags': (),
                            'omit_tags': (),
                            'generate_tags': (),
                            'row_rule': 'AUDIT_OMITTED_CATEGORY',
                            'references': {}},
 '_pdbx_struct_conn_angle': {'action': 'omit',
                             'key': (),
                             'required_tags': (),
                             'preserve_tags': (),
                             'omit_tags': (),
                             'generate_tags': (),
                             'row_rule': 'AUDIT_OMITTED_CATEGORY',
                             'references': {}}}


class RawCategoryTable(NamedTuple):
    """A plain (gemmi-free) column-major view of one CIF category's rows.

    Two separate views, per the C-02 correction (two distinct real null
    markers, `.`/`?`, must never collapse into the same hashed identity):

    - `columns[tag]` is the *decoded* view (`None`/`str`, already
      null-collapsed by the Checkpoint 2 adapter boundary via
      `receptor_parsing.raw_cif_null_to_optional_str` - never imported
      here, ARCH-01). Every lookup/membership/identity computation in
      this module reads this view.
    - `raw_columns[tag]`, when the adapter boundary supplies it, is the
      *literal* undecoded token Gemmi returned (`.`/`?`/a quoted string/
      any other token, always a real `str`, never collapsed to `None`).
      `compute_source_rows_key` hashes *this* view - the audit trail must
      distinguish "inapplicable" (`.`) from "unknown" (`?`) from an
      actually-quoted `'.'`/`'?'` string, even though all of them may
      decode to the same `None` for filtering purposes. When a caller
      omits `raw_columns` (every pure-domain test that builds a
      `RawCategoryTable` directly, never through the real adapter), the
      decoded `columns` view is used for hashing too - a deliberate,
      documented fallback, never a silent default that only happens to
      look identical.

    Every column tuple has the same length; a category with no declared
    columns has `row_count == 0`."""

    columns: Dict[str, Tuple[Optional[str], ...]]
    raw_columns: Optional[Dict[str, Tuple[str, ...]]] = None

    @property
    def row_count(self) -> int:
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

    def value(self, column: str, row_index: int) -> Optional[str]:
        return self.columns[column][row_index]

    def raw_value(self, column: str, row_index: int) -> object:
        source = self.raw_columns if self.raw_columns is not None else self.columns
        return source[column][row_index]


def compute_source_rows_key(*, source_context_key: str, category: str, table: "RawCategoryTable") -> str:
    """SCHEMAS Section 4: `source_rows_key` is the *category* envelope
    hashing the vector of already-hashed *row* envelopes - never a flat
    column/row grid of its own invention:

        row_key           = H({kind:'receptor-materialization-source-row',
                                version:'1.0', source_context_key, category,
                                ordinal, tokens:[{tag,token},...]})
        source_rows_key   = H({kind:'receptor-materialization-source-category',
                                version:'1.0', source_context_key, category,
                                rows:[row_key, ...]})

    `tokens` is built from the category's own columns in sorted (Section
    11: "ordem alfabética de... tags") order, reading the *raw* (literal,
    never null-collapsed) token via `table.raw_value` - Section 11:
    "fonte presente vazia também tem hash não nulo", computed independent
    of which rows the catalogue later keeps/omits."""

    columns_sorted = tuple(sorted(table.columns.keys()))
    row_keys = [
        compute_row_key(
            source_context_key=source_context_key, category=category, ordinal=row_index,
            tokens=tuple((column, table.raw_value(column, row_index)) for column in columns_sorted),
        )
        for row_index in range(table.row_count)
    ]
    payload = {
        "kind": "receptor-materialization-source-category",
        "version": "1.0",
        "source_context_key": source_context_key,
        "category": category,
        "rows": row_keys,
    }
    return compute_materialization_hash(payload)


def _reference_filter_column(cat_def: Dict[str, object], target_reference: str) -> str:
    references = cat_def["references"]
    for column, reference in references.items():
        if reference == target_reference:
            return column
    # Self-referential category (e.g. `_entity` filtering its own rows by
    # `_entity.id`, `_struct_asym` filtering its own rows by
    # `_struct_asym.id`): no column in `references` points at the target
    # table because this category *is* that table - its own key column is
    # the filter column.
    key = cat_def["key"]
    return key[0]


def _filter_rows_by_membership(
    *, table: "RawCategoryTable", column: str, retained_ids: FrozenSet[str],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    kept: List[int] = []
    omitted: List[int] = []
    for row_index in range(table.row_count):
        value = table.value(column, row_index)
        if value is not None and value in retained_ids:
            kept.append(row_index)
        else:
            omitted.append(row_index)
    return tuple(kept), tuple(omitted)


def _validate_key_integrity(category: str, table: "RawCategoryTable", key_columns: Tuple[str, ...]) -> None:
    """C-05: every non-omitted catalogue category has `nonnull_key: true`
    - a key column's value is never null, and a repeated key tuple is a
    profile-identity-key violation ("IDs duplicados de chaves do perfil
    recusam"), independent of whatever row_rule later filters the row."""

    if not key_columns:
        return
    seen_keys: "set[Tuple[object, ...]]" = set()
    for row_index in range(table.row_count):
        key_values = tuple(table.value(column, row_index) for column in key_columns)
        if any(value is None for value in key_values):
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"category {category!r} row {row_index} has a null key column among {key_columns!r}",
                phase="plan",
            )
        if key_values in seen_keys:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"category {category!r} declares a duplicate key {key_values!r} over columns {key_columns!r}",
                phase="plan",
            )
        seen_keys.add(key_values)


_SIMPLE_REFERENCE_PATTERN = re.compile(r"^_([a-z0-9_]+)\.([a-z0-9_]+)$")


def _validate_simple_references(
    category: str, table: "RawCategoryTable", references: Dict[str, str], categories: Mapping[str, "RawCategoryTable"],
) -> None:
    """C-05: "Validar coerência de asym↔entity↔seq/branch↔CCD... na
    fonte". A simple single-column reference (`"_category.column"`)
    must resolve to a real row in the target category, when that target
    category is itself present (its *absence* is governed by the
    required/conditional-category checks, never silently re-litigated
    here). A compound reference (`"_category(col_a,col_b,...)"`, e.g.
    branch/entity_poly_seq composite keys) is intentionally skipped here
    - it is validated by bespoke logic that knows which of the row's
    *other* columns complete the composite key, never guessed
    generically from the expression text.

    `_pdbx_struct_mod_residue.parent_comp_id` is excluded from this
    blanket per-row check - but NOT exempted from referential integrity
    outright (the governance guardrail addendum of 2026-11-24 corrected
    an earlier, overly broad reading of this rule). A row whose context
    is not retained never "enters the closure" at all (Section 11:
    "Referências que não sobrevivem... não devem ser confundidas com
    pais preservados") - checking it here, unconditionally for every
    source row regardless of retention, would wrongly demand a
    `_chem_comp` row for a parent that was never actually preserved.
    The *retained* case is instead enforced by the dedicated
    closure-completeness check below, after every closure contributor
    (atoms, sequences, branches, AND preserved parent_comp_id) is known
    - never silently accepted either way."""

    for column, target_reference in references.items():
        if (category, column) == ("_pdbx_struct_mod_residue", "parent_comp_id"):
            continue
        match = _SIMPLE_REFERENCE_PATTERN.fullmatch(target_reference)
        if match is None or column not in table.columns:
            continue
        target_category, target_column = f"_{match.group(1)}", match.group(2)
        target_table = categories.get(target_category)
        if target_table is None or target_column not in target_table.columns:
            continue
        target_values = frozenset(
            value for row_index in range(target_table.row_count)
            if (value := target_table.value(target_column, row_index)) is not None
        )
        for row_index in range(table.row_count):
            value = table.value(column, row_index)
            if value is not None and value not in target_values:
                raise MaterializationExecutionError(
                    "INVALID_CIF",
                    f"category {category!r} row {row_index} {column}={value!r} does not reference any "
                    f"{target_category}.{target_column}",
                    phase="plan",
                )


def _cat_def_for(category: str) -> Optional[Dict[str, object]]:
    cat_def = _FROZEN_CATALOGUE.get(category)
    if cat_def is not None:
        return cat_def
    if _CATALOGUE_OMISSION_FAMILY_PATTERN.fullmatch(category):
        return {
            "action": "omit", "key": (), "required_tags": (), "preserve_tags": (),
            "omit_tags": (), "generate_tags": (), "row_rule": "AUDIT_OMITTED_CATEGORY", "references": {},
        }
    return None


_NUMERIC_TOKEN_PATTERN = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?:\([0-9]+\))?$"
)
_TRAILING_SU_PATTERN = re.compile(r"\([0-9]+\)$")


def _validate_numeric_token(token: Optional[str]) -> Optional[Decimal]:
    """Section 6's closed numeric-token grammar: an su-exponent suffix is
    only ever accepted *after* the exponent ("aceitar 1.2e3(4) e recusar
    1.2(4)e3" - the regex's own `(?:[eE]...)?(?:\\(...\\))?` ordering
    already enforces this; `1.2(4)e3` fails the full match because the
    trailing `e3` sits after the su group, which the grammar never
    permits). Never Python's own numeric literal syntax (`1_000` must be
    refused, not silently accepted as 1000 the way a bare `float()` call
    would). Returns the finite central `Decimal` with the su suffix
    stripped only for *validation/comparison* - never to rewrite the
    token that gets written back out - or `None` if the token is
    absent/malformed/non-finite."""

    if token is None or not _NUMERIC_TOKEN_PATTERN.fullmatch(token):
        return None
    core = _TRAILING_SU_PATTERN.sub("", token)
    try:
        value = Decimal(core)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _finite_numeric_token(token: Optional[str]) -> bool:
    return _validate_numeric_token(token) is not None


_ANISOTROP_U_COLUMNS = ("u[1][1]", "u[1][2]", "u[1][3]", "u[2][2]", "u[2][3]", "u[3][3]")
_ANISOTROP_B_COLUMNS = ("b[1][1]", "b[1][2]", "b[1][3]", "b[2][2]", "b[2][3]", "b[3][3]")
_ANISOTROP_LABEL_AGREEMENT_COLUMNS = (
    # R-CP2-02: every `_atom_site_anisotrop` column that mirrors a real,
    # unambiguously-named `_atom_site` identity field - never a field
    # `AtomObservation`/the atom_site catalogue entry doesn't actually
    # carry (e.g. `pdbx_pdb_residue_name`/`pdbx_pdb_strand_id`, which have
    # no literal atom_site counterpart - checking them would mean
    # inventing a mapping, not reading an informed one).
    ("type_symbol", "type_symbol"),
    ("pdbx_pdb_ins_code", "pdbx_pdb_ins_code"),
    ("pdbx_pdb_model_num", "pdbx_pdb_model_num"),
    ("pdbx_label_atom_id", "label_atom_id"),
    ("pdbx_label_asym_id", "label_asym_id"),
    ("pdbx_label_comp_id", "label_comp_id"),
    ("pdbx_label_seq_id", "label_seq_id"),
    ("pdbx_label_alt_id", "label_alt_id"),
    ("pdbx_auth_asym_id", "auth_asym_id"),
    ("pdbx_auth_seq_id", "auth_seq_id"),
    ("pdbx_auth_atom_id", "auth_atom_id"),
    ("pdbx_auth_comp_id", "auth_comp_id"),
)
_NCS_MATRIX_COLUMNS = (
    "matrix[1][1]", "matrix[1][2]", "matrix[1][3]",
    "matrix[2][1]", "matrix[2][2]", "matrix[2][3]",
    "matrix[3][1]", "matrix[3][2]", "matrix[3][3]",
)
_NCS_VECTOR_COLUMNS = ("vector[1]", "vector[2]", "vector[3]")
_NCS_IDENTITY_MATRIX_DECIMAL = (
    Decimal(1), Decimal(0), Decimal(0),
    Decimal(0), Decimal(1), Decimal(0),
    Decimal(0), Decimal(0), Decimal(1),
)
_CCD_CLOSURE_SOURCE_COLUMN = {
    "_entity_poly_seq": "mon_id",
    "_pdbx_entity_branch_list": "comp_id",
    "_pdbx_nonpoly_scheme": "mon_id",
    "_pdbx_poly_seq_scheme": "mon_id",
    "_pdbx_branch_scheme": "mon_id",
}


def _tensor_family_state(
    table: "RawCategoryTable", row_index: int, columns: Sequence[str], present_columns: FrozenSet[str],
) -> str:
    # R-CP2-02: a column entirely undeclared for the whole category is
    # read as null for *this* row, exactly like a present-but-null one -
    # never short-circuited to "absent" merely because the table doesn't
    # declare every one of the six columns. A family with a genuine
    # numeric value in even one component is "partial" (and therefore
    # rejected below) regardless of whether its siblings are null because
    # the row says so or because the category never declared them.
    values = [
        table.value(column, row_index) if column in present_columns else None
        for column in columns
    ]
    if all(value is None for value in values):
        return "absent"
    if all(value is not None and _finite_numeric_token(value) for value in values):
        return "complete"
    return "partial"


def _validate_anisotrop_tensor_row(
    category: str, table: "RawCategoryTable", row_index: int, present_columns: FrozenSet[str],
) -> None:
    # Section 11: "exigir ao menos um tensor U ou B com seus seis
    # componentes numéricos finitos; família com algum valor numérico e
    # outros componentes faltantes/null é recusada."
    u_state = _tensor_family_state(table, row_index, _ANISOTROP_U_COLUMNS, present_columns)
    b_state = _tensor_family_state(table, row_index, _ANISOTROP_B_COLUMNS, present_columns)
    if u_state == "partial" or b_state == "partial":
        raise MaterializationExecutionError(
            "INVALID_CIF",
            f"category {category!r} row {row_index} has a partial anisotropic tensor "
            "(some but not all of its six numeric components present/finite)",
            phase="plan",
        )
    if u_state == "absent" and b_state == "absent":
        raise MaterializationExecutionError(
            "INVALID_CIF",
            f"category {category!r} row {row_index} has neither a complete U nor a complete B tensor",
            phase="plan",
        )


def _validate_anisotrop_field_agreement(
    category: str, table: "RawCategoryTable", row_index: int,
    atom_site_table: "RawCategoryTable", atom_site_ordinal: int, present_columns: FrozenSet[str],
) -> None:
    # Section 11: "id precisa corresponder inequivocamente a átomo fonte e
    # demais campos informados concordam."
    for anisotrop_column, atom_site_column in _ANISOTROP_LABEL_AGREEMENT_COLUMNS:
        if anisotrop_column not in present_columns or atom_site_column not in atom_site_table.columns:
            continue
        anisotrop_value = table.value(anisotrop_column, row_index)
        atom_site_value = atom_site_table.value(atom_site_column, atom_site_ordinal)
        if anisotrop_value is not None and anisotrop_value != atom_site_value:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"category {category!r} row {row_index} {anisotrop_column}={anisotrop_value!r} disagrees "
                f"with the referenced _atom_site.{atom_site_column}={atom_site_value!r}",
                phase="plan",
            )


_ATOM_SITE_COORDINATE_COLUMNS = ("cartn_x", "cartn_y", "cartn_z")


def _validate_atom_site_numeric_row(
    table: "RawCategoryTable", row_index: int, present_columns: FrozenSet[str],
) -> None:
    # Section 6: "Coordenadas presentes e finitas... Coordenadas não
    # aceitam null." - applied to every source row, regardless of
    # retention (malformed source data is a structural problem
    # independent of which atoms the plan keeps).
    for column in _ATOM_SITE_COORDINATE_COLUMNS:
        token = table.value(column, row_index) if column in present_columns else None
        if _validate_numeric_token(token) is None:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"_atom_site row {row_index} has a null/non-finite/malformed {column!r}",
                phase="plan",
            )
    # "occupancy/B_iso aceitam null real, ou esse número finito
    # (occupancy não negativa)."
    occupancy_token = table.value("occupancy", row_index) if "occupancy" in present_columns else None
    if occupancy_token is not None:
        occupancy_value = _validate_numeric_token(occupancy_token)
        if occupancy_value is None:
            raise MaterializationExecutionError(
                "INVALID_CIF", f"_atom_site row {row_index} has a non-finite/malformed occupancy", phase="plan",
            )
        if occupancy_value < 0:
            raise MaterializationExecutionError(
                "INVALID_CIF", f"_atom_site row {row_index} has a negative occupancy {occupancy_token!r}", phase="plan",
            )
    b_iso_token = table.value("b_iso_or_equiv", row_index) if "b_iso_or_equiv" in present_columns else None
    if b_iso_token is not None and _validate_numeric_token(b_iso_token) is None:
        raise MaterializationExecutionError(
            "INVALID_CIF", f"_atom_site row {row_index} has a non-finite/malformed b_iso_or_equiv", phase="plan",
        )


def _guard_ncs_row(category: str, table: "RawCategoryTable", row_index: int) -> None:
    # Section 11: "NCS não identidade com generate/null recusa
    # UNSUPPORTED_NCS antes de omitir; given/identidade podem ser omitidos
    # auditados... Código/matriz incompletos inválidos recusam INVALID_CIF."
    values: Dict[str, Decimal] = {}
    for column in _NCS_MATRIX_COLUMNS + _NCS_VECTOR_COLUMNS:
        token = table.value(column, row_index) if column in table.columns else None
        value = _validate_numeric_token(token)
        if value is None:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"category {category!r} row {row_index} has a missing/non-finite NCS operator component {column!r}",
                phase="plan",
            )
        values[column] = value

    # Exact Decimal identity - never a float tolerance (a nonzero
    # translation like 0.0000001 is never "close enough" to identity;
    # Decimal equality compares value, not representation, so "1.0" still
    # equals Decimal(1)).
    is_identity = all(
        values[column] == expected
        for column, expected in zip(_NCS_MATRIX_COLUMNS, _NCS_IDENTITY_MATRIX_DECIMAL)
    ) and all(values[column] == 0 for column in _NCS_VECTOR_COLUMNS)

    code = table.value("code", row_index) if "code" in table.columns else None
    code_normalized = code.strip().lower() if isinstance(code, str) else None
    # Section 11 (T021-154): `code` is a closed two-value mmCIF enum
    # ("given"/"generate") plus null/absent - any other literal string is
    # malformed data, never a silently-tolerated third option.
    if code_normalized not in (None, "given", "generate"):
        raise MaterializationExecutionError(
            "INVALID_CIF",
            f"category {category!r} row {row_index} declares an unrecognized NCS code {code!r} "
            "(expected 'given', 'generate', or null)",
            phase="plan",
        )
    if not is_identity and code_normalized in (None, "generate"):
        raise MaterializationExecutionError(
            "UNSUPPORTED_NCS",
            f"category {category!r} row {row_index} declares a non-identity NCS operator with "
            f"code={code!r}; this checkpoint never applies NCS transformations",
            phase="plan",
        )


def _resolve_special_position_row_atoms(
    category: str, table: "RawCategoryTable", row_index: int,
    present_columns: FrozenSet[str], auth_to_label_asym: Dict[str, "set[str]"],
    atoms_by_ordinal: Mapping[int, AtomObservation], known_label_asym_ids: FrozenSet[str],
) -> Tuple[int, ...]:
    """Section 11 (SPECIAL_POSITION_GUARD_THEN_OMIT, R-CP2-01): resolves one
    `_pdbx_struct_special_symmetry` row to the exact `_atom_site` ordinals
    its annotation refers to - filtered to the row's own declared model
    (`pdb_model_num`) and altloc (`label_alt_id`) "quando informado" when
    present, every model/altloc otherwise - never atoms it merely shares
    an asym with. "Posições especiais exigem contexto inequívoco": an
    unknown `label_asym_id`, an `auth_asym_id` not resolving to exactly
    one `label_asym_id`, no seq identity at all, or a residue identity
    that matches no `_atom_site` row in *any* model is a genuinely
    unresolvable/ambiguous context - `INVALID_CIF`, never a blind
    omission. Every *other* informed field (`label_comp_id`/
    `auth_comp_id`, and `auth_seq_id`/`pdb_ins_code` even when
    `label_seq_id` is the primary key) must also agree with the matched
    atom - a row that names a real asym/seq but a contradicting
    component or auth identity/insertion is just as unresolvable as an
    unknown asym, never silently matched on asym+seq alone. A real
    residue that simply belongs to a different model, a different
    altloc, or is not retained is instead a *resolved* context with zero
    (or partial) matches - never an error."""

    label_asym_id = table.value("label_asym_id", row_index) if "label_asym_id" in present_columns else None
    if label_asym_id is not None:
        if label_asym_id not in known_label_asym_ids:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"category {category!r} row {row_index} label_asym_id={label_asym_id!r} does not "
                "correspond to any _atom_site.label_asym_id",
                phase="plan",
            )
    else:
        auth_asym_id = table.value("auth_asym_id", row_index) if "auth_asym_id" in present_columns else None
        if auth_asym_id is None:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"category {category!r} row {row_index} has neither label_asym_id nor auth_asym_id; "
                "its context cannot be resolved unambiguously before omission",
                phase="plan",
            )
        candidates = auth_to_label_asym.get(auth_asym_id, set())
        if len(candidates) != 1:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"category {category!r} row {row_index} auth_asym_id={auth_asym_id!r} does not resolve to "
                f"exactly one label_asym_id in M ({sorted(candidates)!r})",
                phase="plan",
            )
        label_asym_id = next(iter(candidates))

    label_seq_id = table.value("label_seq_id", row_index) if "label_seq_id" in present_columns else None
    auth_seq_id = table.value("auth_seq_id", row_index) if "auth_seq_id" in present_columns else None
    insertion_token = table.value("pdb_ins_code", row_index) if "pdb_ins_code" in present_columns else None
    if label_seq_id is None and auth_seq_id is None:
        raise MaterializationExecutionError(
            "INVALID_CIF",
            f"category {category!r} row {row_index} has neither label_seq_id nor auth_seq_id; "
            "its residue context cannot be resolved unambiguously",
            phase="plan",
        )
    label_comp_id = table.value("label_comp_id", row_index) if "label_comp_id" in present_columns else None
    auth_comp_id = table.value("auth_comp_id", row_index) if "auth_comp_id" in present_columns else None

    def _same_residue(atom: AtomObservation) -> bool:
        if atom.label_asym_id != label_asym_id:
            return False
        if label_seq_id is not None:
            if atom.label_seq_id != label_seq_id:
                return False
            # R-CP2-01 residual: label_seq_id is the primary key once
            # present, but a genuinely informed auth_seq_id/insertion
            # alongside it must still agree - never silently ignored,
            # which would let a contradicting auth identity slip through
            # as if label_seq_id alone were sufficient proof.
            if auth_seq_id is not None and atom.auth_seq_id != auth_seq_id:
                return False
            if insertion_token is not None and atom.insertion_token != insertion_token:
                return False
        elif atom.auth_seq_id != auth_seq_id or atom.insertion_token != insertion_token:
            return False
        # R-CP2-01 residual: an informed label_comp_id/auth_comp_id must
        # agree with the matched atom's own ccd - chain+sequence alone is
        # never sufficient proof when the annotation itself names a
        # component that contradicts the real residue at that position.
        if label_comp_id is not None and atom.ccd != label_comp_id:
            return False
        if auth_comp_id is not None and atom.ccd != auth_comp_id:
            return False
        return True

    if not any(_same_residue(atom) for atom in atoms_by_ordinal.values()):
        raise MaterializationExecutionError(
            "INVALID_CIF",
            f"category {category!r} row {row_index} residue context (label_asym_id={label_asym_id!r}, "
            f"label_seq_id={label_seq_id!r}, auth_seq_id={auth_seq_id!r}, "
            f"insertion_token={insertion_token!r}, label_comp_id={label_comp_id!r}, "
            f"auth_comp_id={auth_comp_id!r}) does not correspond to any _atom_site row in any model - "
            "either genuinely absent or contradicting an informed field",
            phase="plan",
        )

    model_token = table.value("pdb_model_num", row_index) if "pdb_model_num" in present_columns else None
    altloc = table.value("label_alt_id", row_index) if "label_alt_id" in present_columns else None

    return tuple(
        ordinal for ordinal, atom in atoms_by_ordinal.items()
        if _same_residue(atom)
        and (model_token is None or str(atom.model) == model_token)
        and (altloc is None or atom.altloc == altloc)
    )


def compute_category_decisions(
    *,
    source_context_key: str,
    categories: Mapping[str, "RawCategoryTable"],
    retained_entity_ids: FrozenSet[str],
    retained_asym_ids: FrozenSet[str],
    resolved_model: int,
    atoms_by_ordinal: Mapping[int, AtomObservation],
    atom_decisions_by_ordinal: Mapping[int, AtomDecision],
    connection_decisions: Sequence[ConnectionDecision],
) -> Tuple[CategoryDecision, ...]:
    """Section 11: applies the pinned 94-category catalogue
    (`CATALOGUE_SHA256`) to one parsed CIF DOM (`categories` - gemmi-free
    plain tables keyed by the catalogue's own leading-underscore names,
    e.g. `"_atom_site"`; the Checkpoint 2 adapter boundary converts a real
    `receptor_parsing.ParsedCifDom021` into this shape by a late, local
    import, never at this module's top level - ARCH-01), given the
    already-decided S/D/U/X atom retention (`atom_decisions_by_ordinal`,
    keyed by the same ordinals as `atoms_by_ordinal`) and the
    already-resolved `struct_conn` connections (`connection_decisions`).

    Only catalogue categories *present* in `categories` ever produce a
    `CategoryDecision` here (an entirely absent category is not itself a
    decision to audit) - `_entry` (DERIVED_ENTRY) is the one exception,
    always generated regardless of source presence. A present category
    outside the pinned catalogue and outside the three approved
    `_pdbx_validate_`/`_pdbx_nmr_`/`_em_` omission families raises
    `UNSUPPORTED_CATEGORY`; a present, recognized category with a column
    outside its own preserve/omit/generate tags raises `UNSUPPORTED_TAG`.
    """

    resolved_model_token = str(resolved_model)
    kept_atom_ordinals = frozenset(
        ordinal for ordinal, decision in atom_decisions_by_ordinal.items() if decision.retained
    )
    retained_atom_ccds = frozenset(
        atoms_by_ordinal[ordinal].ccd for ordinal in kept_atom_ordinals if ordinal in atoms_by_ordinal
    )
    preserved_connection_type_tokens = frozenset(
        decision.conn_type_token for decision in connection_decisions if decision.action == "preserve"
    )

    auth_to_label_asym: Dict[str, "set[str]"] = {}
    for atom in atoms_by_ordinal.values():
        auth_to_label_asym.setdefault(atom.auth_asym_id, set()).add(atom.label_asym_id)
    known_label_asym_ids = frozenset(atom.label_asym_id for atom in atoms_by_ordinal.values())

    cat_defs: Dict[str, Optional[Dict[str, object]]] = {name: _cat_def_for(name) for name in categories}
    unknown_present = sorted(name for name, cat_def in cat_defs.items() if cat_def is None)
    if unknown_present:
        raise MaterializationExecutionError(
            "UNSUPPORTED_CATEGORY",
            f"source CIF declares categor(y/ies) outside the pinned catalogue and its approved "
            f"omission families: {unknown_present!r}",
            phase="plan",
        )

    # C-05: always-required categories, independent of anything retained.
    missing_required = [name for name in _CATALOGUE_REQUIRED_CATEGORIES if name not in categories]
    if missing_required:
        raise MaterializationExecutionError(
            "INVALID_CIF",
            f"source CIF is missing always-required categor(y/ies) {missing_required!r}",
            phase="plan",
        )

    # C-05: a retained polymer/branched entity, or a preserved
    # struct_conn row, makes its companion categories conditionally
    # mandatory (Section 11: "Se categoria condicional do catálogo for
    # necessária a entidade/ligação retida e estiver ausente, recusar.") -
    # never merely tracked as an absent-optional category.
    entity_table = categories["_entity"]
    entity_type_by_id: Dict[str, str] = {}
    for row_index in range(entity_table.row_count):
        entity_id = entity_table.value("id", row_index)
        entity_type = entity_table.value("type", row_index)
        if entity_id is not None and entity_type is not None:
            entity_type_by_id[entity_id] = entity_type

    retained_polymer_entities = frozenset(
        entity_id for entity_id in retained_entity_ids if entity_type_by_id.get(entity_id) == "polymer"
    )
    retained_branched_entities = frozenset(
        entity_id for entity_id in retained_entity_ids if entity_type_by_id.get(entity_id) == "branched"
    )
    any_struct_conn_preserved = any(decision.action == "preserve" for decision in connection_decisions)

    missing_conditional: "set[str]" = set()
    if retained_polymer_entities:
        missing_conditional.update(
            name for name in _CATALOGUE_CONDITIONAL_CATEGORIES["retained_polymer_entity"] if name not in categories
        )
    if retained_branched_entities:
        missing_conditional.update(
            name for name in _CATALOGUE_CONDITIONAL_CATEGORIES["retained_branched_entity"] if name not in categories
        )
    if any_struct_conn_preserved:
        missing_conditional.update(
            name for name in _CATALOGUE_CONDITIONAL_CATEGORIES["retained_struct_conn"] if name not in categories
        )
    if missing_conditional:
        raise MaterializationExecutionError(
            "INVALID_CIF",
            f"a retained polymer/branched entity or preserved connection requires conditional "
            f"categor(y/ies) {sorted(missing_conditional)!r}, which the source does not declare",
            phase="plan",
        )

    # C-05/global_rules: "Para cada átomo polimérico retido,
    # (label_entity_id,label_seq_id,label_comp_id) deve encontrar
    # (entity_id,num,mon_id) em entity_poly_seq" - a compound key the
    # generic single-column reference check below cannot express.
    if retained_polymer_entities:
        entity_poly_seq_table = categories["_entity_poly_seq"]
        valid_polymer_triples = frozenset(
            (
                entity_poly_seq_table.value("entity_id", row_index),
                entity_poly_seq_table.value("num", row_index),
                entity_poly_seq_table.value("mon_id", row_index),
            )
            for row_index in range(entity_poly_seq_table.row_count)
        )
        for ordinal in sorted(kept_atom_ordinals):
            atom = atoms_by_ordinal[ordinal]
            if atom.entity_type != "polymer":
                continue
            triple = (atom.entity_id, atom.label_seq_id, atom.ccd)
            if triple not in valid_polymer_triples:
                raise MaterializationExecutionError(
                    "INVALID_CIF",
                    f"retained polymer atom ordinal={ordinal} (entity_id={atom.entity_id!r}, "
                    f"label_seq_id={atom.label_seq_id!r}, ccd={atom.ccd!r}) has no matching "
                    "_entity_poly_seq(entity_id,num,mon_id) row",
                    phase="plan",
                )

    decisions: Dict[str, CategoryDecision] = {}
    ccd_closure_extra: "set[str]" = set()

    for category in sorted(categories):
        if category == "_chem_comp":
            continue  # REFERENCED_CCD_CLOSURE - deferred until every contributor is known.
        if category == "_entry":
            continue  # DERIVED_ENTRY - always generated below, regardless of source presence.

        cat_def = cat_defs[category]
        table = categories[category]
        present_columns = frozenset(table.columns.keys())
        row_rule = cat_def["row_rule"]

        if row_rule == "AUDIT_OMITTED_CATEGORY":
            decisions[category] = CategoryDecision(
                category=category, action="omit",
                omitted_row_ordinals=tuple(range(table.row_count)),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        allowed_tags = set(cat_def["preserve_tags"]) | set(cat_def["omit_tags"]) | set(cat_def["generate_tags"])
        unknown_tags = present_columns - allowed_tags
        if unknown_tags:
            raise MaterializationExecutionError(
                "UNSUPPORTED_TAG",
                f"category {category!r} has unrecognized tag(s) {sorted(unknown_tags)!r} outside the pinned catalogue",
                phase="plan",
            )
        missing_required = [t for t in cat_def["required_tags"] if t not in present_columns]
        if missing_required:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                f"category {category!r} is missing required tag(s) {missing_required!r}",
                phase="plan",
            )

        # C-05: key/relationship columns are never null, and a repeated
        # key tuple is a profile-identity-key violation ("IDs duplicados
        # de chaves do perfil recusam") - every non-omitted category in
        # the pinned catalogue has `nonnull_key: true`.
        # A key column that is itself a *generated* tag (e.g. _cell/
        # _symmetry's `entry_id`, synthesized fresh from `plan_key` and
        # trivially unique by construction) never appears as a real
        # source column - nothing to validate non-null/duplicate against.
        _validate_key_integrity(
            category, table, tuple(column for column in cat_def["key"] if column in present_columns),
        )
        # C-05: a simple single-column reference ("_category.column")
        # must resolve to a real row in the target category when the
        # target category is present; a compound reference
        # ("_category(col_a,col_b,...)") is handled by bespoke logic
        # elsewhere (the polymer-atom/entity_poly_seq compound key above,
        # branch references), never by this generic single-column check.
        _validate_simple_references(category, table, cat_def["references"], categories)

        if row_rule in ("RETAINED_ENTITY", "RETAINED_ENTITY_ALL_SEQUENCE", "RETAINED_ENTITY_ALL_BRANCH"):
            column = _reference_filter_column(cat_def, "_entity.id")
            kept, omitted = _filter_rows_by_membership(table=table, column=column, retained_ids=retained_entity_ids)
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=kept, omitted_row_ordinals=omitted,
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                omitted_tags=tuple(t for t in cat_def["omit_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            closure_column = _CCD_CLOSURE_SOURCE_COLUMN.get(category)
            if closure_column is not None:
                for row_index in kept:
                    value = table.value(closure_column, row_index)
                    if value is not None:
                        ccd_closure_extra.add(value)
            continue

        if row_rule in ("RETAINED_ASYM", "RETAINED_ASYM_ALL_SEQUENCE", "RETAINED_ASYM_ALL_BRANCH"):
            column = _reference_filter_column(cat_def, "_struct_asym.id")
            kept, omitted = _filter_rows_by_membership(table=table, column=column, retained_ids=retained_asym_ids)
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=kept, omitted_row_ordinals=omitted,
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                omitted_tags=tuple(t for t in cat_def["omit_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            closure_column = _CCD_CLOSURE_SOURCE_COLUMN.get(category)
            if closure_column is not None:
                for row_index in kept:
                    value = table.value(closure_column, row_index)
                    if value is not None:
                        ccd_closure_extra.add(value)
            continue

        if row_rule == "ATOM_MASK":
            if table.row_count != len(atoms_by_ordinal):
                raise MaterializationExecutionError(
                    "INVALID_CIF",
                    f"_atom_site row count {table.row_count} does not match the "
                    f"{len(atoms_by_ordinal)} AtomObservation(s) handed to the planner",
                    phase="plan",
                )
            for row_index in range(table.row_count):
                _validate_atom_site_numeric_row(table, row_index, present_columns)
            kept = tuple(sorted(kept_atom_ordinals))
            omitted = tuple(sorted(set(range(table.row_count)) - kept_atom_ordinals))
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=kept, omitted_row_ordinals=omitted,
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        if row_rule == "CONNECTION_ACTIONS":
            if table.row_count != len(connection_decisions):
                raise MaterializationExecutionError(
                    "INVALID_CIF",
                    f"_struct_conn row count {table.row_count} does not match the "
                    f"{len(connection_decisions)} ConnectionDecision(s) resolved",
                    phase="plan",
                )
            kept = tuple(sorted(d.ordinal for d in connection_decisions if d.action == "preserve"))
            omitted = tuple(sorted(d.ordinal for d in connection_decisions if d.action == "omit"))
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=kept, omitted_row_ordinals=omitted,
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                omitted_tags=tuple(t for t in cat_def["omit_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        if row_rule == "USED_CONNECTION_TYPES":
            kept, omitted = _filter_rows_by_membership(
                table=table, column="id", retained_ids=preserved_connection_type_tokens,
            )
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=kept, omitted_row_ordinals=omitted,
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        if row_rule == "USED_ELEMENTS":
            used_elements: FrozenSet[str] = frozenset()
            atom_site_table = categories.get("_atom_site")
            if atom_site_table is not None and "type_symbol" in atom_site_table.columns:
                used_elements = frozenset(
                    value for ordinal in kept_atom_ordinals
                    if (value := atom_site_table.value("type_symbol", ordinal)) is not None
                )
            kept, omitted = _filter_rows_by_membership(table=table, column="symbol", retained_ids=used_elements)
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=kept, omitted_row_ordinals=omitted,
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        if row_rule == "SOURCE_FRAME_SINGLETON":
            if table.row_count > 1:
                raise MaterializationExecutionError(
                    "INVALID_CIF",
                    f"category {category!r} (SOURCE_FRAME_SINGLETON) declares {table.row_count} rows, expected at most one",
                    phase="plan",
                )
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=tuple(range(table.row_count)), omitted_row_ordinals=(),
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                omitted_tags=tuple(t for t in cat_def["omit_tags"] if t in present_columns),
                generated_tags=tuple(cat_def["generate_tags"]),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        if row_rule == "SOURCE_OPERATORS":
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=tuple(range(table.row_count)), omitted_row_ordinals=(),
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                omitted_tags=tuple(t for t in cat_def["omit_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        if row_rule in ("MODIFIED_RESIDUE_CONTEXT", "MISSING_RESIDUE_CONTEXT", "MISSING_ATOM_CONTEXT"):
            if "label_asym_id" not in present_columns:
                raise MaterializationExecutionError(
                    "UNSUPPORTED_CATEGORY",
                    f"category {category!r} has no label_asym_id column; this checkpoint declares itself "
                    "incapable of unambiguously resolving its residue/atom context without it",
                    phase="plan",
                )
            kept_list: List[int] = []
            omitted_list: List[int] = []
            parent_comp_ids: "set[str]" = set()
            for row_index in range(table.row_count):
                label_asym_id = table.value("label_asym_id", row_index)
                if label_asym_id is None:
                    auth_asym_id = (
                        table.value("auth_asym_id", row_index) if "auth_asym_id" in present_columns else None
                    )
                    if auth_asym_id is None:
                        raise MaterializationExecutionError(
                            "INVALID_CIF",
                            f"category {category!r} row {row_index} has neither label_asym_id nor "
                            "auth_asym_id; its residue/atom context cannot be resolved",
                            phase="plan",
                        )
                    candidates = auth_to_label_asym.get(auth_asym_id, set())
                    if len(candidates) != 1:
                        raise MaterializationExecutionError(
                            "INVALID_CIF",
                            f"category {category!r} row {row_index} auth_asym_id={auth_asym_id!r} does not "
                            f"resolve to exactly one label_asym_id in M ({sorted(candidates)!r})",
                            phase="plan",
                        )
                    label_asym_id = next(iter(candidates))
                model_token = (
                    table.value("pdb_model_num", row_index) if "pdb_model_num" in present_columns else None
                )
                model_matches = model_token is None or model_token == resolved_model_token
                if model_matches and label_asym_id in retained_asym_ids:
                    kept_list.append(row_index)
                    if category == "_pdbx_struct_mod_residue" and "parent_comp_id" in present_columns:
                        parent_comp_id = table.value("parent_comp_id", row_index)
                        if parent_comp_id is not None:
                            parent_comp_ids.add(parent_comp_id)
                else:
                    omitted_list.append(row_index)
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=tuple(kept_list), omitted_row_ordinals=tuple(omitted_list),
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            ccd_closure_extra |= parent_comp_ids
            continue

        if row_rule == "ANISOTROP_ATOM_ID":
            atom_site_table = categories.get("_atom_site")
            if atom_site_table is None or "id" not in atom_site_table.columns:
                raise MaterializationExecutionError(
                    "UNSUPPORTED_CATEGORY",
                    f"category {category!r} (ANISOTROP_ATOM_ID) requires a present _atom_site.id column to resolve against",
                    phase="plan",
                )
            atom_site_id_to_ordinal: Dict[str, int] = {}
            for ordinal in range(atom_site_table.row_count):
                value = atom_site_table.value("id", ordinal)
                if value is None:
                    continue
                if value in atom_site_id_to_ordinal:
                    raise MaterializationExecutionError(
                        "INVALID_CIF", f"_atom_site.id {value!r} is not unique", phase="plan",
                    )
                atom_site_id_to_ordinal[value] = ordinal
            kept_list = []
            omitted_list = []
            for row_index in range(table.row_count):
                _validate_anisotrop_tensor_row(category, table, row_index, present_columns)
                row_id = table.value("id", row_index)
                ordinal = atom_site_id_to_ordinal.get(row_id) if row_id is not None else None
                if ordinal is None:
                    raise MaterializationExecutionError(
                        "INVALID_CIF",
                        f"category {category!r} row {row_index} id={row_id!r} does not correspond to any _atom_site.id",
                        phase="plan",
                    )
                _validate_anisotrop_field_agreement(
                    category, table, row_index, atom_site_table, ordinal, present_columns,
                )
                if ordinal in kept_atom_ordinals:
                    kept_list.append(row_index)
                else:
                    omitted_list.append(row_index)
            decisions[category] = CategoryDecision(
                category=category, action="filter",
                kept_row_ordinals=tuple(kept_list), omitted_row_ordinals=tuple(omitted_list),
                preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        if row_rule == "NCS_GUARD_THEN_OMIT":
            for row_index in range(table.row_count):
                _guard_ncs_row(category, table, row_index)
            decisions[category] = CategoryDecision(
                category=category, action="guard_then_omit",
                omitted_row_ordinals=tuple(range(table.row_count)),
                omitted_tags=tuple(t for t in cat_def["omit_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        if row_rule == "SPECIAL_POSITION_GUARD_THEN_OMIT":
            for row_index in range(table.row_count):
                _resolve_special_position_row_atoms(
                    category, table, row_index, present_columns, auth_to_label_asym,
                    atoms_by_ordinal, known_label_asym_ids,
                )
            decisions[category] = CategoryDecision(
                category=category, action="guard_then_omit",
                omitted_row_ordinals=tuple(range(table.row_count)),
                omitted_tags=tuple(t for t in cat_def["omit_tags"] if t in present_columns),
                source_rows_key=compute_source_rows_key(
                    source_context_key=source_context_key, category=category, table=table,
                ),
            )
            continue

        raise MaterializationExecutionError(
            "UNSUPPORTED_CATEGORY",
            f"category {category!r} declares row_rule {row_rule!r}, which this checkpoint does not "
            "yet execute (a declared, explicit gap - never silently approved)",
            phase="plan",
        )

    # DERIVED_ENTRY: always generated, regardless of source `_entry` presence.
    decisions["_entry"] = CategoryDecision(category="_entry", action="generate", generated_tags=("id",))

    # REFERENCED_CCD_CLOSURE (_chem_comp), deferred until every other
    # category's kept rows (and therefore every closure contributor) are
    # known.
    closure = retained_atom_ccds | ccd_closure_extra
    if "_chem_comp" in categories:
        cat_def = cat_defs["_chem_comp"]
        table = categories["_chem_comp"]
        present_columns = frozenset(table.columns.keys())
        allowed_tags = set(cat_def["preserve_tags"]) | set(cat_def["omit_tags"]) | set(cat_def["generate_tags"])
        unknown_tags = present_columns - allowed_tags
        if unknown_tags:
            raise MaterializationExecutionError(
                "UNSUPPORTED_TAG",
                f"category '_chem_comp' has unrecognized tag(s) {sorted(unknown_tags)!r} outside the pinned catalogue",
                phase="plan",
            )
        missing_required = [t for t in cat_def["required_tags"] if t not in present_columns]
        if missing_required:
            raise MaterializationExecutionError(
                "INVALID_CIF", f"category '_chem_comp' is missing required tag(s) {missing_required!r}", phase="plan",
            )
        kept, omitted = _filter_rows_by_membership(table=table, column="id", retained_ids=closure)
        # Section 11 (REFERENCED_CCD_CLOSURE, governance guardrail
        # addendum of 2026-11-24): every closure member - CCDs of
        # retained atoms, preserved sequences/branches, AND preserved
        # parent_comp_id references alike - requires a real matching
        # _chem_comp row; "sem nova busca CCD" means a missing one is
        # never fabricated/looked up, only refused.
        kept_chem_comp_ids = frozenset(table.value("id", row_index) for row_index in kept)
        missing_closure_members = closure - kept_chem_comp_ids
        if missing_closure_members:
            raise MaterializationExecutionError(
                "INVALID_CIF",
                "the referenced-CCD closure (retained atoms/preserved sequences/branches/"
                f"parent_comp_id) requires _chem_comp row(s) for {sorted(missing_closure_members)!r}, "
                "which the source does not declare",
                phase="plan",
            )
        decisions["_chem_comp"] = CategoryDecision(
            category="_chem_comp", action="filter",
            kept_row_ordinals=kept, omitted_row_ordinals=omitted,
            preserved_tags=tuple(t for t in cat_def["preserve_tags"] if t in present_columns),
            source_rows_key=compute_source_rows_key(
                source_context_key=source_context_key, category="_chem_comp", table=table,
            ),
        )
    elif closure:
        raise MaterializationExecutionError(
            "INVALID_CIF",
            "retained atoms/sequences/branches/modified-residues reference chem_comp entries but the "
            "source has no _chem_comp category",
            phase="plan",
        )

    return tuple(decisions[name] for name in sorted(decisions))


def compute_special_position_summary(
    *,
    categories: Mapping[str, "RawCategoryTable"],
    atoms_by_ordinal: Mapping[int, AtomObservation],
    atom_decisions_by_ordinal: Mapping[int, AtomDecision],
) -> SpecialPositionSummary:
    """Section 11/SCHEMAS Section 8 (R-CP2-01): the three frozen
    `_pdbx_struct_special_symmetry` counts - source annotation rows, rows
    with a real retained-atom intersection, and the unique retained atoms
    that intersection covers (never double-counted across overlapping
    annotations). Computed independently of `compute_category_decisions`'s
    own per-row guard (which already raises on an unresolvable/ambiguous
    context) using the same resolution rule, so a caller that only needs
    the three counts - never the `CategoryDecision` vector itself - does
    not have to re-derive them by hand. Zero counts when the category is
    absent, per SCHEMAS Section 8."""

    table = categories.get("_pdbx_struct_special_symmetry")
    if table is None:
        return _ZERO_SPECIAL_POSITION_SUMMARY

    present_columns = frozenset(table.columns.keys())
    kept_atom_ordinals = frozenset(
        ordinal for ordinal, decision in atom_decisions_by_ordinal.items() if decision.retained
    )
    auth_to_label_asym: Dict[str, "set[str]"] = {}
    for atom in atoms_by_ordinal.values():
        auth_to_label_asym.setdefault(atom.auth_asym_id, set()).add(atom.label_asym_id)
    known_label_asym_ids = frozenset(atom.label_asym_id for atom in atoms_by_ordinal.values())

    retained_annotation_count = 0
    retained_unique_atoms: "set[int]" = set()
    for row_index in range(table.row_count):
        matches = _resolve_special_position_row_atoms(
            "_pdbx_struct_special_symmetry", table, row_index, present_columns,
            auth_to_label_asym, atoms_by_ordinal, known_label_asym_ids,
        )
        retained_matches = tuple(ordinal for ordinal in matches if ordinal in kept_atom_ordinals)
        if retained_matches:
            retained_annotation_count += 1
            retained_unique_atoms.update(retained_matches)

    return SpecialPositionSummary(
        source_annotation_count=table.row_count,
        retained_annotation_count=retained_annotation_count,
        retained_unique_atom_count=len(retained_unique_atoms),
    )
