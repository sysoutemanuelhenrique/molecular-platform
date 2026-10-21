"""Molecular characterization: DTOs, closed profiles, and typed errors.

This module belongs to the "Represent" stage of the project's conceptual
pipeline. It defines the frozen, strict public contract for the
molecular-characterization feature (Increment 020-A): the ten-value
scientific profile, the optional 2D depiction profile, the closed SVG
validation/binding profile, and the full error taxonomy those profiles can
raise.

Deliberately contains NO RDKit import, NO filesystem/network I/O, and NO
CLI code - the single permitted RDKit import point for this feature is
molecular.rdkit_tautomer (Section 6.2/15 of the frozen spec). The SVG
validator defined here uses only `xml.parsers.expat` (stdlib) precisely so
it, like every other name in this module, stays importable and testable in
a base installation with no rdkit extra installed at all.

Checkpoint 1 scope only (increments/020-A/08-final-specification/
INCREMENT_020_A_FINAL_SPECIFICATION.md, Sections 4/7/8/9/10/11/13): this
module defines the complete closed-vocabulary error taxonomy and the full
public DTO shapes described by the frozen spec, including
MolecularCharacterizationResult and CharacterizationProvenance - but no
reader, service, or CLI code exists yet to populate a real Provenance from
a registered MolecularState (Checkpoints 2/3). Checkpoint 1 only
constructs these DTOs directly, from caller-supplied values, which is
exactly what tests/test_characterization.py exercises.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from molecular.inputs import validate_input_id

# --- Section 7: input format -------------------------------------------

InputFormat = Literal["sdf", "smiles"]

# --- Section 8/9/10: closed profile identifiers -------------------------

CharacterizationProfileId = Literal["molecular-characterization-v1"]
DepictionProfileId = Literal["molecular-depiction-v1"]
SvgValidationProfileId = Literal["svg-static-v1"]

CHARACTERIZATION_PROFILE_ID: CharacterizationProfileId = "molecular-characterization-v1"
DEPICTION_PROFILE_ID: DepictionProfileId = "molecular-depiction-v1"
SVG_VALIDATION_PROFILE_ID: SvgValidationProfileId = "svg-static-v1"

EngineName = Literal["rdkit"]
ENGINE_NAME: EngineName = "rdkit"

# Section 8: "versão efetivamente observada obrigatória 2026.03.6" - the
# real string rdkit.rdBase.rdkitVersion reports for rdkit==2026.3.6
# (zero-padded month), confirmed by direct execution against the pinned
# wheel, never assumed from the PyPI/pyproject version spelling.
REQUIRED_ENGINE_VERSION = "2026.03.6"

ExperimentalConditionsConsumption = Literal["not-consumed"]
EXPERIMENTAL_CONDITIONS_CONSUMPTION: ExperimentalConditionsConsumption = "not-consumed"

RotatableBondsOption = Literal["Strict"]

# --- Section 6: closed, inclusive size limits enforced by this adapter --
# (registry/representation byte limits belong to the readers of Checkpoint
# 2 and are not duplicated here - this module only owns the limits the
# adapter itself is responsible for checking.)

MAX_TOTAL_ATOM_COUNT = 4096
MAX_BOND_COUNT = 8192
MAX_SVG_UTF8_BYTES = 16_777_216

# --- Section 11: closed warnings vocabulary ------------------------------

WarningCode = Literal[
    "derived-computational-values",
    "conditions-not-consumed",
    "no-durable-query-record",
    "tpsa-no-additional-s-p",
    "generated-2d-not-conformer",
]

_RESULT_WARNINGS_BASE: tuple[WarningCode, ...] = (
    "derived-computational-values",
    "conditions-not-consumed",
    "no-durable-query-record",
    "tpsa-no-additional-s-p",
)

_ADAPTER_WARNINGS_BASE: tuple[WarningCode, ...] = (
    "derived-computational-values",
    "tpsa-no-additional-s-p",
)

_WARNING_WITH_SVG: WarningCode = "generated-2d-not-conformer"


# =========================================================================
# Section 13: error taxonomy
# =========================================================================

_INPUT_ERROR_REASONS = frozenset({"invalid-request"})

_CONFIGURATION_ERROR_REASONS = frozenset(
    {
        "unsupported-platform",
        "invalid-state-record",
        "unsupported-format",
        "invalid-reference",
        "engine-unavailable",
        "drawing-unavailable",
    }
)

_STORAGE_ERROR_REASONS = frozenset(
    {
        "missing-artifact",
        "unreadable-artifact",
        "unsafe-or-changed-artifact",
        "checksum-mismatch",
        "artifact-size-invalid",
    }
)

_EXECUTION_ERROR_REASONS = frozenset(
    {
        "invalid-chemical-input",
        "unsupported-structure",
        "molecular-size-limit",
        "descriptor-failure",
        "depiction-failure",
        "invalid-engine-result",
        "internal-failure",
    }
)


class CharacterizationError(Exception):
    """Base of the characterization error taxonomy (Section 13 of the
    frozen spec). Never raised directly - always one of the four category
    subclasses below. `reason` must belong to that subclass's own closed
    vocabulary (enforced here, not left to caller discipline);
    `input_state_id` is attached only when it was already validated by the
    caller - never a value that itself failed validation."""

    category_banner: str = "CHARACTERIZATION ERROR"
    _allowed_reasons: frozenset[str] = frozenset()

    def __init__(
        self, message: str, *, reason: str, input_state_id: str | None = None
    ) -> None:
        if reason not in self._allowed_reasons:
            raise ValueError(
                f"reason {reason!r} is not a member of {type(self).__name__}'s "
                "closed vocabulary"
            )
        super().__init__(message)
        self.reason = reason
        self.input_state_id = input_state_id


class CharacterizationInputError(CharacterizationError):
    """INPUT ERROR (Section 13): malformed request, before any file or
    engine access is attempted."""

    category_banner = "INPUT ERROR"
    _allowed_reasons = _INPUT_ERROR_REASONS


class CharacterizationConfigurationError(CharacterizationError):
    """CONFIGURATION ERROR (Section 13): the environment/engine/platform
    cannot support the request as configured - never a chemical defect of
    the input itself."""

    category_banner = "CONFIGURATION ERROR"
    _allowed_reasons = _CONFIGURATION_ERROR_REASONS


class CharacterizationStorageError(CharacterizationError):
    """STORAGE ERROR (Section 13): a filesystem/artifact-identity failure.
    Checkpoint 1's own adapter raises this only for `checksum-mismatch`
    (Section 10.2's in-memory hash cross-check) - every other reason in
    this category belongs to the Checkpoint 2 readers."""

    category_banner = "STORAGE ERROR"
    _allowed_reasons = _STORAGE_ERROR_REASONS


class CharacterizationExecutionError(CharacterizationError):
    """EXECUTION ERROR (Section 13): a chemistry/engine-execution failure
    once a verified representation buffer is already in hand."""

    category_banner = "EXECUTION ERROR"
    _allowed_reasons = _EXECUTION_ERROR_REASONS


# =========================================================================
# Checkpoint 1 corrections (MAJOR-1): actual strict-type enforcement.
#
# Section 11 requires "validação estrita" - but neither
# `ConfigDict(frozen=True, extra="forbid")` nor `ConfigDict(strict=True)`
# nor `Field(strict=True)` actually delivers that for the fields affected
# here, confirmed by direct reproduction against the exact pinned
# pydantic version installed for this checkpoint (not assumed from the
# architect's or reviewer's probes, though both independently found the
# same thing on different pydantic patch versions):
#   - a plain (non-Literal) `int` field's default (lax) validation coerces
#     a numeric string ("3"), an integral float (3.0), and a bool (True)
#     into int - all three accepted silently.
#   - a plain `float` field's default validation likewise coerces a
#     numeric string ("4.5") and a bool (True) into float.
#   - `ConfigDict(strict=True)` does NOT propagate into `Literal` field
#     validation at all: `Literal[True]` still accepts int `1`,
#     `Literal[0]` still accepts bool `False`, `Literal[-1.0]` still
#     accepts int `-1` - because Literal matching compares by `==`
#     equality, not by exact type, regardless of the model's strict mode.
#   - `Field(strict=True)` cannot even be applied to a `Literal` field:
#     pydantic raises `RuntimeError("Unable to apply constraint 'strict'
#     to schema of type 'literal'")` at class-definition time.
#
# The fix actually enforced here is a `mode="before"` validator that
# inspects `type(value)` directly, for every affected field, before
# pydantic's own (lax) coercion or Literal `==` matching ever runs - the
# only mechanism in the installed pydantic that reliably rejects these
# inputs without widening the public JSON shape or the set of valid
# native values. Valid native ints/floats/bools/strings, and their valid
# JSON serialization/round-trip, are unaffected.
# =========================================================================


def _require_strict_bool(value: object) -> object:
    """A bool-only field (including a `Literal[True]`/`Literal[False]`
    default) must reject every non-bool, including int `0`/`1` - which
    pydantic's own Literal matching otherwise accepts via `==` equality
    even under `ConfigDict(strict=True)` (confirmed by direct
    reproduction, not assumed)."""

    if type(value) is not bool:
        raise ValueError(
            f"expected a strict bool, not {type(value).__name__} (got {value!r})"
        )
    return value


def _require_strict_int(value: object) -> object:
    """An int-only field must reject bool, float (including an integral
    float like `3.0`), and numeric strings (like `"3"`) - pydantic's
    default int validation coerces all three silently (confirmed by
    direct reproduction)."""

    if type(value) is not int:
        raise ValueError(
            f"expected a strict int, not {type(value).__name__} (got {value!r})"
        )
    return value


def _require_numeric_not_bool_or_str(value: object) -> object:
    """A float-typed field that legitimately accepts a native int too
    (Section 11 does not forbid it, and RDKit itself never hands this
    adapter an int for these particular fields, but the contract does not
    rule it out either) - reject only bool and numeric strings, which
    pydantic's default float validation otherwise coerces silently."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"expected a real number (int or float), not {type(value).__name__} "
            f"(got {value!r})"
        )
    return value


def _require_strict_float(value: object) -> object:
    """A float-typed Literal constant (e.g. `bond_length=-1.0`) must
    reject bool and int equivalents too - `Literal[-1.0]` otherwise
    accepts int `-1` via `==` equality (confirmed by direct
    reproduction)."""

    if type(value) is not float:
        raise ValueError(
            f"expected a strict float, not {type(value).__name__} (got {value!r})"
        )
    return value


# =========================================================================
# Section 8: the ten scientific values
# =========================================================================


class CharacterizationValues(BaseModel):
    """The ten closed scientific fields of Section 8. Strict and frozen:
    no field may be a bool, a numeric string, or an integral float
    masquerading as an int or a float, and no float may be NaN/inf
    (Section 11's invariant, T-020A-003)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    formula: str = Field(min_length=1)
    molar_mass_g_mol: float
    exact_mass_da: float
    formal_charge_e: int
    heavy_atom_count: int = Field(ge=0)
    hbd_count: int = Field(ge=0)
    hba_count: int = Field(ge=0)
    rotatable_bond_count: int = Field(ge=0)
    tpsa_a2: float = Field(ge=0)
    clogp_wildman_crippen: float

    @field_validator(
        "molar_mass_g_mol", "exact_mass_da", "tpsa_a2", "clogp_wildman_crippen",
        mode="before",
    )
    @classmethod
    def _reject_bool_and_str_before_coercion_float(cls, value: object) -> object:
        return _require_numeric_not_bool_or_str(value)

    @field_validator(
        "molar_mass_g_mol", "exact_mass_da", "tpsa_a2", "clogp_wildman_crippen"
    )
    @classmethod
    def _reject_non_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError(f"value must be finite (got {value!r})")
        return float(value)

    @field_validator(
        "formal_charge_e",
        "heavy_atom_count",
        "hbd_count",
        "hba_count",
        "rotatable_bond_count",
        mode="before",
    )
    @classmethod
    def _reject_bool_float_and_str_as_int(cls, value: object) -> object:
        return _require_strict_int(value)

    @field_validator("molar_mass_g_mol", "exact_mass_da")
    @classmethod
    def _require_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"value must be strictly positive (got {value!r})")
        return value


# =========================================================================
# Section 11: effective_parameters - characterization blocks
# =========================================================================


class SdfParserParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["sdf"] = "sdf"
    sanitize: Literal[True] = True
    remove_hs: Literal[False] = False
    strict_parsing: Literal[True] = True

    @field_validator("sanitize", "remove_hs", "strict_parsing", mode="before")
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class SmilesParserParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["smiles"] = "smiles"
    sanitize: Literal[True] = True
    remove_hs: Literal[False] = False
    parse_name: Literal[False] = False
    allow_cxsmiles: Literal[False] = False

    @field_validator(
        "sanitize", "remove_hs", "parse_name", "allow_cxsmiles", mode="before"
    )
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class FormulaParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    separate_isotopes: Literal[True] = True
    abbreviate_h_isotopes: Literal[False] = False

    @field_validator("separate_isotopes", "abbreviate_h_isotopes", mode="before")
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class ExactMassParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    only_heavy: Literal[False] = False

    @field_validator("only_heavy", mode="before")
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class RotatableBondsParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    option: RotatableBondsOption = "Strict"


class TpsaParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    force: Literal[True] = True
    include_s_and_p: Literal[False] = False

    @field_validator("force", "include_s_and_p", mode="before")
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class CrippenParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    include_hs: Literal[True] = True
    force: Literal[True] = True

    @field_validator("include_hs", "force", mode="before")
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class CharacterizationEffectiveParameters(BaseModel):
    """effective_parameters for the characterization profile (Section 11).
    `parser` is the format-discriminated block - SDF requests carry
    SdfParserParameters, SMILES requests carry SmilesParserParameters,
    never both, never the other format's fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parser: SdfParserParameters | SmilesParserParameters
    formula: FormulaParameters = FormulaParameters()
    exact_mass: ExactMassParameters = ExactMassParameters()
    rotatable_bonds: RotatableBondsParameters = RotatableBondsParameters()
    tpsa: TpsaParameters = TpsaParameters()
    crippen: CrippenParameters = CrippenParameters()


# =========================================================================
# Section 9/10/11: depiction profile and SVG validation profile
# =========================================================================


class EmptyCoordMap(BaseModel):
    """A closed, frozen, empty mapping - the DTO form of `coordMap={}`
    (Section 9/11): never a free-form dict, always serializes as `{}`."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Compute2DParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    canon_orient: Literal[True] = True
    clear_confs: Literal[True] = True
    coord_map: EmptyCoordMap = EmptyCoordMap()
    n_flips_per_sample: Literal[0] = 0
    n_sample: Literal[0] = 0
    sample_seed: Literal[0] = 0
    permute_deg4_nodes: Literal[False] = False
    bond_length: Literal[-1.0] = -1.0
    force_rdkit: Literal[True] = True
    use_ring_templates: Literal[False] = False

    @field_validator(
        "canon_orient",
        "clear_confs",
        "permute_deg4_nodes",
        "force_rdkit",
        "use_ring_templates",
        mode="before",
    )
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)

    @field_validator(
        "n_flips_per_sample", "n_sample", "sample_seed", mode="before"
    )
    @classmethod
    def _require_strict_int_fields(cls, value: object) -> object:
        return _require_strict_int(value)

    @field_validator("bond_length", mode="before")
    @classmethod
    def _require_strict_float_fields(cls, value: object) -> object:
        return _require_strict_float(value)


class PrepareParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kekulize: Literal[False] = False
    add_chiral_hs: Literal[False] = False
    wedge_bonds: Literal[True] = True
    force_coords: Literal[False] = False
    wavy_bonds: Literal[False] = False

    @field_validator(
        "kekulize",
        "add_chiral_hs",
        "wedge_bonds",
        "force_coords",
        "wavy_bonds",
        mode="before",
    )
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class DrawerConstructorParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    width: Literal[600] = 600
    height: Literal[400] = 400
    panel_width: Literal[-1] = -1
    panel_height: Literal[-1] = -1
    no_freetype: Literal[False] = False

    @field_validator("width", "height", "panel_width", "panel_height", mode="before")
    @classmethod
    def _require_strict_int_fields(cls, value: object) -> object:
        return _require_strict_int(value)

    @field_validator("no_freetype", mode="before")
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class DrawOptionsParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prepare_mols_before_drawing: Literal[False] = False
    include_metadata: Literal[False] = False
    include_atom_tags: Literal[False] = False
    add_atom_indices: Literal[False] = False
    add_bond_indices: Literal[False] = False
    font_policy: Literal["rdkit-data-fonts-telex"] = "rdkit-data-fonts-telex"

    @field_validator(
        "prepare_mols_before_drawing",
        "include_metadata",
        "include_atom_tags",
        "add_atom_indices",
        "add_bond_indices",
        mode="before",
    )
    @classmethod
    def _require_strict_bool_fields(cls, value: object) -> object:
        return _require_strict_bool(value)


class DrawMoleculeParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    legend: Literal[""] = ""


class DepictionEffectiveParameters(BaseModel):
    """effective_parameters for the depiction profile (Section 11), split
    by drawing surface exactly as Section 9's table separates them - never
    an attribute of one surface passed as a kwarg of another."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    compute2d: Compute2DParameters = Compute2DParameters()
    prepare: PrepareParameters = PrepareParameters()
    drawer: DrawerConstructorParameters = DrawerConstructorParameters()
    draw_options: DrawOptionsParameters = DrawOptionsParameters()
    draw_molecule: DrawMoleculeParameters = DrawMoleculeParameters()
    svg_validation_profile: SvgValidationProfileId = SVG_VALIDATION_PROFILE_ID


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_hex64_lowercase(value: str) -> str:
    if not isinstance(value, str) or not _HEX64_RE.fullmatch(value):
        raise ValueError(f"expected a lowercase hex64 string (got {value!r})")
    return value


class DrawingResult(BaseModel):
    """The `drawing` block of Section 11, present only when `include_svg`
    was requested and succeeded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: DepictionProfileId = DEPICTION_PROFILE_ID
    width_px: Literal[600] = 600
    height_px: Literal[400] = 400
    font_name: str = Field(min_length=1)
    font_sha256_observed: str
    svg_sha256: str
    svg_size_bytes: int = Field(gt=0)
    xml_parser: Literal["xml.parsers.expat"] = "xml.parsers.expat"
    xml_parser_version_observed: str = Field(min_length=1)
    effective_parameters: DepictionEffectiveParameters = DepictionEffectiveParameters()

    @field_validator("width_px", "height_px", "svg_size_bytes", mode="before")
    @classmethod
    def _require_strict_int_fields(cls, value: object) -> object:
        return _require_strict_int(value)

    @field_validator("font_sha256_observed", "svg_sha256")
    @classmethod
    def _validate_hashes(cls, value: str) -> str:
        return _validate_hex64_lowercase(value)


# =========================================================================
# Section 11: provenance and the public result
# =========================================================================


class CharacterizationProvenance(BaseModel):
    """Provenance (Section 11). Checkpoint 1 defines this full shape, but
    nothing in this checkpoint populates it from a real registered
    MolecularState - that is the Checkpoint 2 service's job. Checkpoint 1
    tests construct it directly from synthetic fixture values
    (T-020A-001/002/003)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_state_id: str
    entity_id: str
    state_schema_version: Literal["1.0"] = "1.0"
    state_record_sha256_observed: str
    representation_path: str = Field(min_length=1)
    representation_format_declared: str = Field(min_length=1)
    representation_format_resolved: InputFormat
    representation_checksum_expected: str = Field(min_length=1)
    representation_checksum_observed: str
    representation_size_bytes: int = Field(gt=0)
    engine_name: EngineName = ENGINE_NAME
    engine_version: str = Field(min_length=1)
    application_version: str | None = None
    characterization_profile_id: CharacterizationProfileId = CHARACTERIZATION_PROFILE_ID
    effective_parameters: CharacterizationEffectiveParameters
    experimental_conditions_consumption: ExperimentalConditionsConsumption = (
        EXPERIMENTAL_CONDITIONS_CONSUMPTION
    )

    @field_validator("input_state_id", "entity_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)

    @field_validator("representation_size_bytes", mode="before")
    @classmethod
    def _require_strict_int_fields(cls, value: object) -> object:
        return _require_strict_int(value)

    @field_validator("state_record_sha256_observed", "representation_checksum_observed")
    @classmethod
    def _validate_observed_hash(cls, value: str) -> str:
        return _validate_hex64_lowercase(value)

    @field_validator("representation_checksum_expected")
    @classmethod
    def _validate_expected_hash_form(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9A-Fa-f]{64}", value):
            raise ValueError(
                f"expected a hex64 string, case-insensitive (got {value!r})"
            )
        return value

    @model_validator(mode="after")
    def _validate_checksum_case_insensitive_match(self) -> "CharacterizationProvenance":
        if self.representation_checksum_expected.lower() != (
            self.representation_checksum_observed.lower()
        ):
            raise ValueError(
                "representation_checksum_observed must equal "
                "representation_checksum_expected (case-insensitively)"
            )
        return self

    @model_validator(mode="after")
    def _validate_parser_block_matches_format(self) -> "CharacterizationProvenance":
        parser = self.effective_parameters.parser
        if self.representation_format_resolved == "sdf" and not isinstance(
            parser, SdfParserParameters
        ):
            raise ValueError("sdf format requires SdfParserParameters")
        if self.representation_format_resolved == "smiles" and not isinstance(
            parser, SmilesParserParameters
        ):
            raise ValueError("smiles format requires SmilesParserParameters")
        return self


class MolecularCharacterizationResult(BaseModel):
    """MolecularCharacterizationResult (Section 11): the public JSON
    contract's root object. `warnings` is strictly the fixed, ordered
    vocabulary of Section 11 - the four base codes always present, plus
    `generated-2d-not-conformer` if and only if a figure is present."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    values: CharacterizationValues
    provenance: CharacterizationProvenance
    warnings: tuple[WarningCode, ...]
    drawing: DrawingResult | None = None
    svg: str | None = None

    @model_validator(mode="after")
    def _validate_svg_drawing_consistency(self) -> "MolecularCharacterizationResult":
        if (self.drawing is None) != (self.svg is None):
            raise ValueError(
                "drawing and svg must both be present or both be absent "
                "(Section 11's cross invariant)"
            )
        if self.drawing is not None and self.svg is not None:
            svg_bytes = self.svg.encode("utf-8")
            if len(svg_bytes) != self.drawing.svg_size_bytes:
                raise ValueError(
                    "svg_size_bytes must equal len(svg.encode('utf-8')) "
                    f"(got {self.drawing.svg_size_bytes}, "
                    f"actual {len(svg_bytes)})"
                )
            import hashlib

            observed = hashlib.sha256(svg_bytes).hexdigest()
            if observed != self.drawing.svg_sha256:
                raise ValueError(
                    "svg_sha256 must equal sha256(svg.encode('utf-8')) "
                    f"(got {self.drawing.svg_sha256!r}, actual {observed!r})"
                )
        return self

    @model_validator(mode="after")
    def _validate_warnings_fixed_vocabulary(self) -> "MolecularCharacterizationResult":
        expected = _RESULT_WARNINGS_BASE + (
            (_WARNING_WITH_SVG,) if self.svg is not None else ()
        )
        if self.warnings != expected:
            raise ValueError(
                f"warnings must equal {expected!r} exactly (Section 11's fixed, "
                f"ordered vocabulary; got {self.warnings!r})"
            )
        return self


# =========================================================================
# Adapter-facing neutral outcome (Section 4/12): the un-provenanced shape
# `characterize_molecule_rdkit` actually returns. Internal structure name,
# not part of the public JSON contract (Section 4's own permission to
# adjust internal auxiliary-structure names during review).
# =========================================================================


class RDKitCharacterizationOutcome(BaseModel):
    """The RDKit adapter's neutral result (Section 4/12): never an rdkit
    object, and never itself the full public contract - the Checkpoint 2
    service combines this with the registry/reader-captured provenance to
    build MolecularCharacterizationResult. `warnings` here is only the
    subset of Section 11's vocabulary the adapter itself can determine
    from chemistry/drawing alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_name: EngineName = ENGINE_NAME
    engine_version: str = Field(min_length=1)
    representation_format_resolved: InputFormat
    values: CharacterizationValues
    effective_parameters: CharacterizationEffectiveParameters
    warnings: tuple[WarningCode, ...]
    drawing: DrawingResult | None = None
    svg: str | None = None

    @model_validator(mode="after")
    def _validate_svg_drawing_consistency(self) -> "RDKitCharacterizationOutcome":
        if (self.drawing is None) != (self.svg is None):
            raise ValueError(
                "drawing and svg must both be present or both be absent"
            )
        return self

    @model_validator(mode="after")
    def _validate_warnings_fixed_vocabulary(self) -> "RDKitCharacterizationOutcome":
        expected = _ADAPTER_WARNINGS_BASE + (
            (_WARNING_WITH_SVG,) if self.svg is not None else ()
        )
        if self.warnings != expected:
            raise ValueError(
                f"warnings must equal {expected!r} exactly (got {self.warnings!r})"
            )
        return self


# =========================================================================
# Section 10/10.1/10.2: closed SVG validation/binding profile
# =========================================================================

_SVG_NS = "http://www.w3.org/2000/svg"
_XML_NS = "http://www.w3.org/XML/1998/namespace"
_NS_SEP = " "
_XML_SPACE_ATTR = f"{_XML_NS}{_NS_SEP}space"

_SVG_ROOT_ALLOWED_ATTRS = frozenset(
    {"version", "baseProfile", "width", "height", "viewBox", _XML_SPACE_ATTR}
)
_RECT_ALLOWED_ATTRS = frozenset({"x", "y", "width", "height", "style", "fill"})
_PATH_ALLOWED_ATTRS = frozenset({"d", "class", "style", "fill"})

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_CLASS_TOKEN_RE = re.compile(r"^(?:atom|bond)-\d+$")
_VIEWBOX_RE = re.compile(
    r"^0\s+0\s+600\s+400$"
)

_D_TOKEN_RE = re.compile(
    r"[MLQ]|[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_D_ALLOWED_CHARS_RE = re.compile(r"^[MLQ0-9+\-.,eE \t]*$")
_D_COMMAND_ARITY = {"M": 2, "L": 2, "Q": 4}


class _SvgProfileViolation(Exception):
    """Internal-only signal raised by the expat callbacks and value
    validators below; always converted to CharacterizationExecutionError
    (reason=depiction-failure) at this module's public boundary - never
    allowed to escape validate_and_bind_depiction_svg."""


def _is_finite_number(token: str) -> bool:
    if not _NUMBER_RE.fullmatch(token):
        return False
    return math.isfinite(float(token))


def _validate_style_value(name: str, value: str) -> None:
    if name in ("fill", "stroke"):
        if value == "none" or _COLOR_RE.fullmatch(value):
            return
        raise _SvgProfileViolation(f"invalid color value for {name!r}: {value!r}")
    if name in ("opacity", "stroke-opacity"):
        if _is_finite_number(value) and 0.0 <= float(value) <= 1.0:
            return
        raise _SvgProfileViolation(f"invalid unit-interval value for {name!r}: {value!r}")
    if name == "fill-rule":
        if value in ("evenodd", "nonzero"):
            return
        raise _SvgProfileViolation(f"invalid fill-rule value: {value!r}")
    if name == "stroke-width":
        raw = value[:-2] if value.endswith("px") else value
        if _is_finite_number(raw) and float(raw) >= 0.0:
            return
        raise _SvgProfileViolation(f"invalid stroke-width value: {value!r}")
    if name == "stroke-linecap":
        if value in ("butt", "round", "square"):
            return
        raise _SvgProfileViolation(f"invalid stroke-linecap value: {value!r}")
    if name == "stroke-linejoin":
        if value in ("miter", "round", "bevel"):
            return
        raise _SvgProfileViolation(f"invalid stroke-linejoin value: {value!r}")
    if name == "stroke-dasharray":
        if value == "none":
            return
        parts = value.split(",")
        if not parts or any(p == "" for p in parts):
            raise _SvgProfileViolation(f"invalid stroke-dasharray value: {value!r}")
        numbers = []
        for part in parts:
            if not _is_finite_number(part):
                raise _SvgProfileViolation(
                    f"invalid stroke-dasharray value: {value!r}"
                )
            number = float(part)
            if number < 0.0:
                raise _SvgProfileViolation(
                    f"invalid stroke-dasharray value: {value!r}"
                )
            numbers.append(number)
        if not any(number > 0.0 for number in numbers):
            raise _SvgProfileViolation(
                f"stroke-dasharray must contain at least one positive number "
                f"(got {value!r})"
            )
        return
    if name == "stroke-miterlimit":
        if _is_finite_number(value) and float(value) >= 1.0:
            return
        raise _SvgProfileViolation(f"invalid stroke-miterlimit value: {value!r}")
    raise _SvgProfileViolation(f"unknown style property: {name!r}")


_STYLE_KNOWN_PROPERTIES = frozenset(
    {
        "fill",
        "stroke",
        "opacity",
        "stroke-opacity",
        "fill-rule",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-dasharray",
        "stroke-miterlimit",
    }
)


def _validate_style_attribute(value: str) -> None:
    parts = value.split(";")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    if not parts:
        raise _SvgProfileViolation("style attribute must not be empty")
    seen: set[str] = set()
    for declaration in parts:
        if declaration == "":
            raise _SvgProfileViolation("style attribute has an empty declaration")
        if ":" not in declaration:
            raise _SvgProfileViolation(
                f"malformed style declaration: {declaration!r}"
            )
        name, _, raw_value = declaration.partition(":")
        name = name.strip()
        raw_value = raw_value.strip()
        if name != name.strip() or not name:
            raise _SvgProfileViolation(f"malformed style property name: {name!r}")
        if name not in _STYLE_KNOWN_PROPERTIES:
            raise _SvgProfileViolation(f"unknown style property: {name!r}")
        if name in seen:
            raise _SvgProfileViolation(f"duplicate style property: {name!r}")
        seen.add(name)
        _validate_style_value(name, raw_value)


def _validate_class_attribute(value: str) -> None:
    tokens = value.split(" ")
    if not tokens or any(token == "" for token in tokens):
        raise _SvgProfileViolation(f"malformed class attribute: {value!r}")
    for token in tokens:
        if not _CLASS_TOKEN_RE.fullmatch(token):
            raise _SvgProfileViolation(f"disallowed class token: {token!r}")


def _validate_fill_attribute(value: str) -> None:
    if value == "none" or _COLOR_RE.fullmatch(value):
        return
    raise _SvgProfileViolation(f"invalid fill attribute value: {value!r}")


def _validate_d_attribute(value: str) -> None:
    if not value or not _D_ALLOWED_CHARS_RE.fullmatch(value):
        raise _SvgProfileViolation(f"disallowed character in d attribute: {value!r}")
    tokens = _D_TOKEN_RE.findall(value)
    if not tokens or tokens[0] != "M":
        raise _SvgProfileViolation("d attribute must start with an absolute M command")

    index = 0
    total = len(tokens)
    while index < total:
        command = tokens[index]
        if command not in _D_COMMAND_ARITY:
            raise _SvgProfileViolation(f"unsupported d command token: {command!r}")
        arity = _D_COMMAND_ARITY[command]
        index += 1
        group_count = 0
        while index < total and tokens[index] not in _D_COMMAND_ARITY:
            group = tokens[index : index + arity]
            if len(group) != arity or any(
                not _is_finite_number(value) for value in group
            ):
                raise _SvgProfileViolation(
                    f"malformed/residual argument group for {command!r} command"
                )
            index += arity
            group_count += 1
        if group_count == 0:
            raise _SvgProfileViolation(f"{command!r} command requires at least one group")


def _validate_numeric_attribute(value: str, *, allow_negative: bool = False) -> float:
    if not _is_finite_number(value):
        raise _SvgProfileViolation(f"expected a finite number (got {value!r})")
    number = float(value)
    if not allow_negative and number < 0.0:
        raise _SvgProfileViolation(f"expected a non-negative number (got {value!r})")
    return number


class _SvgNode:
    __slots__ = ("tag", "attrs")

    def __init__(self, tag: str, attrs: dict[str, str]) -> None:
        self.tag = tag
        self.attrs = attrs


class _SvgTreeBuilder:
    """Collects a validated svg/rect/path tree from expat callbacks,
    refusing anything outside the Section 10.1 closed profile as soon as
    it is seen - never a second, more permissive parse pass."""

    def __init__(self) -> None:
        self.root: _SvgNode | None = None
        self.children: list[_SvgNode] = []
        self._depth = 0
        self._current_child_closed = True

    def start_element(self, name: str, attrs: dict[str, str]) -> None:
        self._depth += 1
        if self._depth == 1:
            if name != f"{_SVG_NS}{_NS_SEP}svg":
                raise _SvgProfileViolation(f"unexpected root element: {name!r}")
            self._validate_root_attrs(attrs)
            self.root = _SvgNode("svg", attrs)
            return
        if self._depth == 2:
            local = self._require_svg_ns_local_name(name)
            if local == "rect":
                self._validate_rect_attrs(attrs)
            elif local == "path":
                self._validate_path_attrs(attrs)
            else:
                raise _SvgProfileViolation(f"unexpected child element: {name!r}")
            self.children.append(_SvgNode(local, attrs))
            return
        raise _SvgProfileViolation("rect/path elements must not have children")

    def end_element(self, name: str) -> None:
        self._depth -= 1

    def char_data(self, data: str) -> None:
        if data.strip(" \t\r\n") != "":
            raise _SvgProfileViolation("non-whitespace text content is not permitted")

    @staticmethod
    def _require_svg_ns_local_name(qualified_name: str) -> str:
        if not qualified_name.startswith(_SVG_NS + _NS_SEP):
            raise _SvgProfileViolation(
                f"element outside the SVG namespace: {qualified_name!r}"
            )
        return qualified_name[len(_SVG_NS) + 1 :]

    def _validate_root_attrs(self, attrs: dict[str, str]) -> None:
        for key in attrs:
            if key not in _SVG_ROOT_ALLOWED_ATTRS:
                raise _SvgProfileViolation(f"disallowed root attribute: {key!r}")
        if attrs.get("version") != "1.1":
            raise _SvgProfileViolation("root version must be '1.1'")
        if "baseProfile" in attrs and attrs["baseProfile"] != "full":
            raise _SvgProfileViolation("baseProfile, when present, must be 'full'")
        if _XML_SPACE_ATTR in attrs and attrs[_XML_SPACE_ATTR] != "preserve":
            raise _SvgProfileViolation("xml:space, when present, must be 'preserve'")
        if attrs.get("width") != "600px":
            raise _SvgProfileViolation("root width must be '600px'")
        if attrs.get("height") != "400px":
            raise _SvgProfileViolation("root height must be '400px'")
        view_box = attrs.get("viewBox")
        if not view_box or not _VIEWBOX_RE.fullmatch(view_box):
            raise _SvgProfileViolation("viewBox must be '0 0 600 400'")

    def _validate_rect_attrs(self, attrs: dict[str, str]) -> None:
        for key in attrs:
            if key not in _RECT_ALLOWED_ATTRS:
                raise _SvgProfileViolation(f"disallowed rect attribute: {key!r}")
        for key in ("x", "y", "width", "height"):
            if key in attrs:
                allow_negative = key in ("x", "y")
                _validate_numeric_attribute(attrs[key], allow_negative=allow_negative)
        if "style" in attrs:
            _validate_style_attribute(attrs["style"])
        if "fill" in attrs:
            _validate_fill_attribute(attrs["fill"])

    def _validate_path_attrs(self, attrs: dict[str, str]) -> None:
        for key in attrs:
            if key not in _PATH_ALLOWED_ATTRS:
                raise _SvgProfileViolation(f"disallowed path attribute: {key!r}")
        if "d" not in attrs:
            raise _SvgProfileViolation("path element is missing its d attribute")
        _validate_d_attribute(attrs["d"])
        if "class" in attrs:
            _validate_class_attribute(attrs["class"])
        if "style" in attrs:
            _validate_style_attribute(attrs["style"])
        if "fill" in attrs:
            _validate_fill_attribute(attrs["fill"])


def _parse_svg_with_closed_profile(raw_svg: str) -> _SvgTreeBuilder:
    import xml.parsers.expat as expat

    parser = expat.ParserCreate(encoding="utf-8", namespace_separator=_NS_SEP)
    parser.namespace_prefixes = False

    def _reject_doctype(*_args: object) -> None:
        raise _SvgProfileViolation("DOCTYPE declarations are not permitted")

    def _reject_entity_decl(*_args: object) -> None:
        raise _SvgProfileViolation("ENTITY declarations are not permitted")

    def _reject_external_entity_ref(*_args: object) -> int:
        raise _SvgProfileViolation("external entity references are not permitted")

    def _reject_processing_instruction(target: str, _data: str) -> None:
        raise _SvgProfileViolation(f"processing instructions are not permitted: {target!r}")

    builder = _SvgTreeBuilder()
    parser.StartDoctypeDeclHandler = _reject_doctype
    parser.EntityDeclHandler = _reject_entity_decl
    parser.ExternalEntityRefHandler = _reject_external_entity_ref
    parser.ProcessingInstructionHandler = _reject_processing_instruction
    parser.StartElementHandler = builder.start_element
    parser.EndElementHandler = builder.end_element
    parser.CharacterDataHandler = builder.char_data

    if not parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER):
        raise _SvgProfileViolation("SetParamEntityParsing(XML_PARAM_ENTITY_PARSING_NEVER) failed")

    try:
        parser.Parse(raw_svg.encode("utf-8"), True)
    except expat.ExpatError as error:
        raise _SvgProfileViolation(f"XML parsing failed: {error}") from error

    if builder.root is None:
        raise _SvgProfileViolation("no root element was parsed")
    return builder


def _escape_xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _serialize_validated_svg(
    tree: _SvgTreeBuilder,
    *,
    input_state_id: str,
    representation_sha256: str,
    engine_version: str,
) -> str:
    assert tree.root is not None

    root_attrs = dict(tree.root.attrs)
    root_attrs.pop(_XML_SPACE_ATTR, None)

    ordered_root_parts = [
        "version=\"1.1\"",
        f"baseProfile=\"{root_attrs.get('baseProfile', 'full')}\"",
        "xmlns=\"http://www.w3.org/2000/svg\"",
        "width=\"600px\"",
        "height=\"400px\"",
        "viewBox=\"0 0 600 400\"",
    ]
    if _XML_SPACE_ATTR in tree.root.attrs:
        ordered_root_parts.append("xml:space=\"preserve\"")

    ordered_root_parts.append(f"data-state-id=\"{_escape_xml_attr(input_state_id)}\"")
    ordered_root_parts.append(
        f"data-representation-sha256=\"{_escape_xml_attr(representation_sha256)}\""
    )
    ordered_root_parts.append(
        f"data-engine-version=\"{_escape_xml_attr(engine_version)}\""
    )
    ordered_root_parts.append(f"data-profile=\"{DEPICTION_PROFILE_ID}\"")

    lines = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<svg {' '.join(ordered_root_parts)}>",
    ]
    for child in tree.children:
        attr_parts = []
        for key in ("x", "y", "width", "height", "d", "class", "style", "fill"):
            if key in child.attrs:
                attr_parts.append(f"{key}=\"{_escape_xml_attr(child.attrs[key])}\"")
        lines.append(f"<{child.tag} {' '.join(attr_parts)}/>")
    lines.append("</svg>")
    return "\n".join(lines)


def validate_and_bind_depiction_svg(
    raw_svg: str,
    *,
    input_state_id: str,
    representation_sha256: str,
    engine_version: str,
) -> str:
    """Validate `raw_svg` (the still-unreleased producer output) against
    the closed profile of Section 10/10.1, then bind the four Section
    10.2 provenance attributes onto the root and return the final,
    re-serialized UTF-8 string - never a second, more permissive parse of
    untrusted SVG, and never the producer's own bytes returned verbatim.

    Raises CharacterizationExecutionError(reason="depiction-failure") for
    any violation of the closed profile, chaining the original cause.
    """

    try:
        tree = _parse_svg_with_closed_profile(raw_svg)
        final_svg = _serialize_validated_svg(
            tree,
            input_state_id=input_state_id,
            representation_sha256=representation_sha256,
            engine_version=engine_version,
        )
    except _SvgProfileViolation as error:
        raise CharacterizationExecutionError(
            f"the depiction SVG failed the closed validation profile: {error}",
            reason="depiction-failure",
            input_state_id=input_state_id,
        ) from error

    final_bytes = final_svg.encode("utf-8")
    if len(final_bytes) > MAX_SVG_UTF8_BYTES:
        raise CharacterizationExecutionError(
            "the final bound SVG exceeds the maximum UTF-8 byte size",
            reason="depiction-failure",
            input_state_id=input_state_id,
        )
    return final_svg
