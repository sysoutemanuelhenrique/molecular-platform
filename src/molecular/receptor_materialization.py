"""Domain model, canonicalization and pure planner for receptor
materialization (Increment 021, Checkpoint 1 - frozen specification at
increments/021/15-final-specification/INCREMENT_021_FINAL_SPECIFICATION.md,
governance commit b0e60f6528f6fcb828aa6dd5577824e08c7c9fac).

This module is the *domain-only* slice of Increment 021: DTOs, hash
canonicalization, and a pure planner that derives the S/D/U/X/O decisions
(Sections 7-8 of the frozen spec) and the acknowledgement-checked
`PolicyRecord` (Section 12-13). It never imports `gemmi`/`rdkit`, never
opens a RAW artifact or a 020 YAML record, and never writes to storage -
those are Checkpoint 2 (adapter), Checkpoint 3 (storage/registry/service)
and Checkpoint 4 (CLI) responsibilities (Section 19 of the frozen spec).

`AtomObservation` is the explicit boundary: Checkpoint 2's future Gemmi
adapter will be the only code that ever constructs one from a real RAW
document. In this checkpoint, only domain tests construct it directly, in
memory, to exercise the planner - never a stub that fabricates a constant
"scientific success".

Deliberately not implemented here (Checkpoint 2-4, Section 19): the Gemmi
adapter, `struct_conn`/symmetry resolution (`ConnectionDecision`,
`IdentityBasis`, `SymmetryClass` - Sections 9-10), the closed catalogue's
row-level filtering (Section 11), `Summary`/`PreviewOutcome`/
`ManifestRecord`/`AuditRecord`/`ArtifactOutcome` (they need real output
byte counts this checkpoint never produces), storage/reservation/registry,
and the CLI. `PlanBinding.connection_decisions_key` is therefore always the
canonical hash of an empty connection-decision vector in this checkpoint;
`category_decisions` is accepted as an explicit, already-decided vector
(Section 11's own filter/omit/generate/guard_then_omit shape is exercised
here at the DTO level - T021-161) rather than computed from a real
catalogue lookup.

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
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple

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

    @field_validator(
        "atom_id", "label_asym_id", "entity_id", "ccd", "label_seq_id", "auth_asym_id", "auth_seq_id"
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


def compute_connection_decisions_key(*, source_context_key: str) -> str:
    """Checkpoint 1 never resolves `struct_conn` (Section 9-10 is
    Checkpoint 2): the connection-decision vector is always empty here,
    still hashed under its own envelope so the key changes with
    `source_context_key` even though the (always-empty) vector does not."""

    payload = {
        "kind": "receptor-materialization-connection-decisions",
        "version": "1.0",
        "source_context_key": source_context_key,
        "decisions": [],
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


class MaterializationPlan(BaseModel):
    """The internal, non-normative plan: `PlanBinding` plus the full
    decision vectors it was hashed from (Section 8). Never itself hashed
    or persisted - only `PlanBinding.plan_key` is."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    plan: PlanBinding
    plan_key: str = Field(min_length=64, max_length=64)
    atom_decisions: Tuple[AtomDecision, ...]
    category_decisions: Tuple[CategoryDecision, ...]

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

    warnings_set: set = set()
    if other_models_excluded:
        warnings_set.add("other-models-excluded")
    if outside_entries:
        warnings_set.add("outside-selected-asym-retained")
    warnings = tuple(sorted(warnings_set))

    exclusions_key = compute_exclusions_key(source_context_key=source_context_key, model=model, entries=exclusions)
    outside_key = compute_outside_key(source_context_key=source_context_key, model=model, entries=outside_entries)
    atom_decisions_key = compute_atom_decisions_key(source_context_key=source_context_key, decisions=atom_decisions)
    connection_decisions_key = compute_connection_decisions_key(source_context_key=source_context_key)
    category_decisions_key = compute_category_decisions_key(
        source_context_key=source_context_key, decisions=category_decisions
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
        unavailable_categories=(),
        warnings=warnings,
    )

    return MaterializationPlan(
        plan=plan,
        plan_key=compute_plan_key(plan),
        atom_decisions=tuple(atom_decisions),
        category_decisions=category_decisions,
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
