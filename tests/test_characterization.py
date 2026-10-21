"""Tests for src/molecular/characterization.py: DTOs, closed profiles,
and the error taxonomy (Increment 020-A, Checkpoint 1). Deliberately
imports nothing from rdkit and nothing from molecular.rdkit_tautomer -
every name under test here must be usable in a base installation with no
rdkit extra at all (Section 15 of the frozen spec, T-020A-006)."""

import ast
import hashlib
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

import molecular.characterization as characterization_module
from molecular.characterization import (
    CharacterizationConfigurationError,
    CharacterizationEffectiveParameters,
    CharacterizationError,
    CharacterizationExecutionError,
    CharacterizationInputError,
    CharacterizationProvenance,
    CharacterizationStorageError,
    CharacterizationValues,
    Compute2DParameters,
    CrippenParameters,
    DrawerConstructorParameters,
    DrawingResult,
    DrawMoleculeParameters,
    DrawOptionsParameters,
    EmptyCoordMap,
    ExactMassParameters,
    FormulaParameters,
    MolecularCharacterizationResult,
    PrepareParameters,
    RDKitCharacterizationOutcome,
    SdfParserParameters,
    SmilesParserParameters,
    TpsaParameters,
    validate_and_bind_depiction_svg,
)


def _valid_values_kwargs():
    return dict(
        formula="C8H10N4O2",
        molar_mass_g_mol=194.194,
        exact_mass_da=194.08037556,
        formal_charge_e=0,
        heavy_atom_count=14,
        hbd_count=0,
        hba_count=3,
        rotatable_bond_count=0,
        tpsa_a2=61.82,
        clogp_wildman_crippen=-1.0293,
    )


def _valid_provenance_kwargs(**overrides):
    base = dict(
        input_state_id="caffeine-1",
        entity_id="caffeine",
        state_record_sha256_observed="a" * 64,
        representation_path="scientific/artifacts/states/caffeine-1/rep.smi",
        representation_format_declared="smiles",
        representation_format_resolved="smiles",
        representation_checksum_expected="A" * 64,
        representation_checksum_observed="a" * 64,
        representation_size_bytes=10,
        engine_version="2026.03.6",
        effective_parameters=CharacterizationEffectiveParameters(
            parser=SmilesParserParameters()
        ),
    )
    base.update(overrides)
    return base


# --- T-020A-006: importable without rdkit (static boundary check) ------


def test_module_never_imports_rdkit():
    source = Path(characterization_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("rdkit")
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                assert not node.module.startswith("rdkit")


# --- T-020A-001: extra fields rejected on every public DTO --------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CharacterizationValues(**_valid_values_kwargs(), extra_field=1),
        lambda: CharacterizationProvenance(
            **_valid_provenance_kwargs(), extra_field=1
        ),
        lambda: CharacterizationEffectiveParameters(
            parser=SmilesParserParameters(), extra_field=1
        ),
        lambda: SmilesParserParameters(extra_field=1),
        lambda: SdfParserParameters(extra_field=1),
        lambda: EmptyCoordMap(extra_field=1),
        lambda: DrawingResult(
            font_name="Telex-Regular.ttf",
            font_sha256_observed="a" * 64,
            svg_sha256="b" * 64,
            svg_size_bytes=10,
            xml_parser_version_observed="expat_2.6.0",
            extra_field=1,
        ),
    ],
)
def test_extra_field_is_rejected_on_every_dto(factory):
    with pytest.raises(ValidationError):
        factory()


def test_result_rejects_extra_field():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    with pytest.raises(ValidationError):
        MolecularCharacterizationResult(
            values=values,
            provenance=provenance,
            warnings=_base_warnings(),
            extra_field=1,
        )


def _base_warnings():
    return (
        "derived-computational-values",
        "conditions-not-consumed",
        "no-durable-query-record",
        "tpsa-no-additional-s-p",
    )


# --- T-020A-002: immutability -------------------------------------------


def test_values_are_frozen():
    values = CharacterizationValues(**_valid_values_kwargs())
    with pytest.raises(ValidationError):
        values.formal_charge_e = 1


def test_provenance_is_frozen():
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    with pytest.raises(ValidationError):
        provenance.input_state_id = "other"


def test_result_is_frozen():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    result = MolecularCharacterizationResult(
        values=values, provenance=provenance, warnings=_base_warnings()
    )
    with pytest.raises(ValidationError):
        result.schema_version = "2.0"


# --- T-020A-003: NaN/inf, bool-as-count, negative counts ----------------


@pytest.mark.parametrize("field", ["molar_mass_g_mol", "exact_mass_da", "tpsa_a2"])
def test_non_finite_floats_are_rejected(field):
    kwargs = _valid_values_kwargs()
    kwargs[field] = math.nan
    with pytest.raises(ValidationError):
        CharacterizationValues(**kwargs)
    kwargs[field] = math.inf
    with pytest.raises(ValidationError):
        CharacterizationValues(**kwargs)


def test_clogp_may_be_negative_and_formal_charge_may_be_signed():
    kwargs = _valid_values_kwargs()
    kwargs["clogp_wildman_crippen"] = -5.5
    kwargs["formal_charge_e"] = -3
    values = CharacterizationValues(**kwargs)
    assert values.clogp_wildman_crippen == -5.5
    assert values.formal_charge_e == -3


@pytest.mark.parametrize(
    "field",
    ["heavy_atom_count", "hbd_count", "hba_count", "rotatable_bond_count", "formal_charge_e"],
)
def test_bool_is_rejected_as_count(field):
    kwargs = _valid_values_kwargs()
    kwargs[field] = True
    with pytest.raises(ValidationError):
        CharacterizationValues(**kwargs)


@pytest.mark.parametrize(
    "field", ["heavy_atom_count", "hbd_count", "hba_count", "rotatable_bond_count"]
)
def test_negative_counts_are_rejected(field):
    kwargs = _valid_values_kwargs()
    kwargs[field] = -1
    with pytest.raises(ValidationError):
        CharacterizationValues(**kwargs)


def test_non_positive_mass_is_rejected():
    kwargs = _valid_values_kwargs()
    kwargs["molar_mass_g_mol"] = 0.0
    with pytest.raises(ValidationError):
        CharacterizationValues(**kwargs)


# =========================================================================
# Checkpoint 1 corrections, MAJOR-1: real strict-type enforcement.
#
# `frozen=True, extra="forbid"` alone does not reject a numeric string, an
# integral float, or a bool-for-int/float - confirmed independently by the
# adversarial review (two reproductions, two different pydantic patch
# versions, both against this exact DTO) before this fix. These tests
# reproduce the review's own exhibited cases exactly (never "fixed to
# pass" by weakening the assertion - these are the literal examples from
# evidence/claude-dto-coercion-reproduction.json, every one previously
# accepted, now reproduced here as the expected-rejection regression), and
# extend the same check across every other new DTO/profile field the
# finding applies to.
# =========================================================================


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("heavy_atom_count", "3"),  # numeric string -> int (review exhibit)
        ("heavy_atom_count", 3.0),  # integral float -> int (review exhibit)
        ("formal_charge_e", "0"),  # numeric string -> int (review exhibit)
        ("hbd_count", "1"),
        ("hba_count", 1.0),
        ("rotatable_bond_count", "0"),
    ],
)
def test_int_fields_reject_numeric_strings_and_integral_floats(field, bad_value):
    kwargs = _valid_values_kwargs()
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        CharacterizationValues(**kwargs)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("molar_mass_g_mol", "46.069"),  # numeric string -> float (review exhibit)
        ("exact_mass_da", "194.08"),
        ("tpsa_a2", "61.82"),
        ("clogp_wildman_crippen", "-1.0293"),
    ],
)
def test_float_fields_reject_numeric_strings(field, bad_value):
    kwargs = _valid_values_kwargs()
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        CharacterizationValues(**kwargs)


def test_float_fields_still_accept_a_native_int():
    # Section 11 does not forbid a native int for a float-typed field -
    # only bool and numeric strings are rejected (reconciliation's own
    # explicit scoping of this finding).
    kwargs = _valid_values_kwargs()
    kwargs["molar_mass_g_mol"] = 46
    values = CharacterizationValues(**kwargs)
    assert values.molar_mass_g_mol == 46.0
    assert isinstance(values.molar_mass_g_mol, float)


def test_sdf_parser_parameters_literal_bool_rejects_int_one():
    # the review's own exhibit: SdfParserParameters(sanitize=1) was
    # previously accepted and silently coerced to True.
    with pytest.raises(ValidationError):
        SdfParserParameters(sanitize=1)


@pytest.mark.parametrize(
    "dto_factory,field,bad_value",
    [
        (lambda **kw: SdfParserParameters(**kw), "sanitize", 1),
        (lambda **kw: SdfParserParameters(**kw), "remove_hs", 0),
        (lambda **kw: SdfParserParameters(**kw), "strict_parsing", 1),
        (lambda **kw: SmilesParserParameters(**kw), "sanitize", 1),
        (lambda **kw: SmilesParserParameters(**kw), "allow_cxsmiles", 0),
        (lambda **kw: FormulaParameters(**kw), "separate_isotopes", 1),
        (lambda **kw: FormulaParameters(**kw), "abbreviate_h_isotopes", 0),
        (lambda **kw: ExactMassParameters(**kw), "only_heavy", 0),
        (lambda **kw: TpsaParameters(**kw), "force", 1),
        (lambda **kw: TpsaParameters(**kw), "include_s_and_p", 0),
        (lambda **kw: CrippenParameters(**kw), "include_hs", 1),
        (lambda **kw: Compute2DParameters(**kw), "canon_orient", 1),
        (lambda **kw: Compute2DParameters(**kw), "permute_deg4_nodes", 0),
        (lambda **kw: Compute2DParameters(**kw), "force_rdkit", 1),
        (lambda **kw: PrepareParameters(**kw), "kekulize", 0),
        (lambda **kw: PrepareParameters(**kw), "wedge_bonds", 1),
        (lambda **kw: DrawerConstructorParameters(**kw), "no_freetype", 0),
        (lambda **kw: DrawOptionsParameters(**kw), "include_metadata", 0),
        (lambda **kw: DrawOptionsParameters(**kw), "add_atom_indices", 1),
    ],
)
def test_literal_bool_fields_reject_int_zero_or_one(dto_factory, field, bad_value):
    with pytest.raises(ValidationError):
        dto_factory(**{field: bad_value})


@pytest.mark.parametrize(
    "dto_factory,field,bad_value",
    [
        (lambda **kw: Compute2DParameters(**kw), "n_flips_per_sample", False),
        (lambda **kw: Compute2DParameters(**kw), "n_sample", 0.0),
        (lambda **kw: Compute2DParameters(**kw), "sample_seed", True),
        (lambda **kw: DrawerConstructorParameters(**kw), "width", 600.0),
        (lambda **kw: DrawerConstructorParameters(**kw), "height", True),
        (lambda **kw: DrawerConstructorParameters(**kw), "panel_width", -1.0),
    ],
)
def test_literal_int_fields_reject_bool_and_float_equivalents(dto_factory, field, bad_value):
    with pytest.raises(ValidationError):
        dto_factory(**{field: bad_value})


def test_literal_float_field_rejects_int_and_bool_equivalents():
    with pytest.raises(ValidationError):
        Compute2DParameters(bond_length=-1)
    with pytest.raises(ValidationError):
        Compute2DParameters(bond_length=False)


@pytest.mark.parametrize("field", ["width_px", "height_px", "svg_size_bytes"])
def test_drawing_result_int_fields_reject_bool_float_and_str(field):
    base = dict(
        font_name="Telex-Regular.ttf",
        font_sha256_observed="a" * 64,
        svg_sha256="b" * 64,
        svg_size_bytes=10,
        xml_parser_version_observed="expat_2.6.0",
    )
    for bad_value in (True, 10.0, "10"):
        kwargs = {**base, field: bad_value}
        with pytest.raises(ValidationError):
            DrawingResult(**kwargs)


def test_provenance_representation_size_bytes_rejects_bool_float_and_str():
    for bad_value in (True, 10.0, "10"):
        with pytest.raises(ValidationError):
            CharacterizationProvenance(
                **_valid_provenance_kwargs(representation_size_bytes=bad_value)
            )


def test_strict_fixes_preserve_valid_json_round_trip():
    values = CharacterizationValues(**_valid_values_kwargs())
    restored = CharacterizationValues.model_validate_json(values.model_dump_json())
    assert restored == values

    effective = CharacterizationEffectiveParameters(parser=SmilesParserParameters())
    restored_effective = CharacterizationEffectiveParameters.model_validate_json(
        effective.model_dump_json()
    )
    assert restored_effective == effective

    drawing = DrawMoleculeParameters()
    assert DrawMoleculeParameters.model_validate_json(drawing.model_dump_json()) == drawing


# =========================================================================
# Checkpoint 1 MAJOR-1 complement: the five numeric/literal exhibits above
# are now rejected, but plain `str` fields (IDs and hashes) still coerced
# `bytes`/`bytearray` silently into `str` on the HEAD this complement
# starts from - reproduced here exactly as the architect's isolated probe
# found it (evidence/architect-post-correction-dto-probe.json), before
# this fix. An earlier revision of the corrections report stated plain
# `str` fields already reject `bytes` natively; that statement is
# superseded by this reproduction (see the new report's errata) and is
# not repeated or relied upon here.
# =========================================================================


def test_values_formula_rejects_bytes_and_bytearray():
    for bad_value in (b"C2H6O", bytearray(b"C2H6O")):
        kwargs = _valid_values_kwargs()
        kwargs["formula"] = bad_value
        with pytest.raises(ValidationError):
            CharacterizationValues(**kwargs)


def test_values_formula_accepts_native_str_control():
    kwargs = _valid_values_kwargs()
    kwargs["formula"] = "C2H6O"
    values = CharacterizationValues(**kwargs)
    assert values.formula == "C2H6O"
    assert isinstance(values.formula, str)


@pytest.mark.parametrize(
    "field,valid_str_value",
    [
        ("input_state_id", "caffeine-1"),
        ("entity_id", "caffeine"),
        ("state_record_sha256_observed", "a" * 64),
        ("representation_checksum_expected", "A" * 64),
        ("representation_checksum_observed", "a" * 64),
    ],
)
@pytest.mark.parametrize("as_bytes", [bytes, bytearray])
def test_provenance_id_and_hash_fields_reject_bytes_and_bytearray(
    field, valid_str_value, as_bytes
):
    bad_value = as_bytes(valid_str_value.encode("ascii"))
    with pytest.raises(ValidationError):
        CharacterizationProvenance(**_valid_provenance_kwargs(**{field: bad_value}))


@pytest.mark.parametrize(
    "field,valid_str_value",
    [
        ("input_state_id", "caffeine-1"),
        ("entity_id", "caffeine"),
        ("state_record_sha256_observed", "a" * 64),
        ("representation_checksum_expected", "A" * 64),
        ("representation_checksum_observed", "a" * 64),
    ],
)
def test_provenance_id_and_hash_fields_accept_native_str_control(
    field, valid_str_value
):
    provenance = CharacterizationProvenance(
        **_valid_provenance_kwargs(**{field: valid_str_value})
    )
    assert getattr(provenance, field) == valid_str_value
    assert isinstance(getattr(provenance, field), str)


def test_major1_complement_preserves_contracted_invariants():
    # Section 11: ID case-sensitivity, the expected checksum's original
    # spelling (uppercase here), application_version nullable, and the
    # scientific values are all untouched by the bytes/bytearray fix -
    # same construction, same JSON round-trip, as before this complement.
    provenance = CharacterizationProvenance(
        **_valid_provenance_kwargs(
            input_state_id="caffeine-1",
            entity_id="caffeine",
            representation_checksum_expected="AB" * 32,
            representation_checksum_observed="ab" * 32,
            application_version=None,
        )
    )
    assert provenance.input_state_id == "caffeine-1"
    assert provenance.entity_id == "caffeine"
    assert provenance.representation_checksum_expected == "AB" * 32
    assert provenance.application_version is None
    restored = CharacterizationProvenance.model_validate_json(
        provenance.model_dump_json()
    )
    assert restored == provenance

    values = CharacterizationValues(**_valid_values_kwargs())
    assert values.formula == "C8H10N4O2"
    assert values.molar_mass_g_mol == 194.194
    restored_values = CharacterizationValues.model_validate_json(
        values.model_dump_json()
    )
    assert restored_values == values


# =========================================================================
# Checkpoint 1 MAJOR-1 string closure: the complement above fixed six of
# the 17 ordinary `str`/`str | None` fields the new DTOs define (every
# plain string field across every BaseModel this module defines, Literal/
# enum fields excluded - the same inventory the architect's probe used,
# evidence/architect-complete-string-matrix.json). The other 11 -
# DrawingResult's two hashes, its font/xml_parser version strings,
# CharacterizationProvenance.representation_path/
# representation_format_declared/engine_version/application_version, and
# the two top-level nullable `svg` fields on MolecularCharacterizationResult
# and RDKitCharacterizationOutcome - still coerced bytes/bytearray on that
# HEAD, by both the constructor and `model_validate(dict)`. This closes
# the full 17-field matrix; nothing here is a new finding beyond that one.
# =========================================================================


def _valid_drawing_kwargs(**overrides):
    base = dict(
        font_name="Telex-Regular.ttf",
        font_sha256_observed="a" * 64,
        svg_sha256=hashlib.sha256(b"<svg/>").hexdigest(),
        svg_size_bytes=len(b"<svg/>"),
        xml_parser_version_observed="expat_2.6.0",
    )
    base.update(overrides)
    return base


def _valid_result_kwargs(**overrides):
    base = dict(
        values=CharacterizationValues(**_valid_values_kwargs()),
        provenance=CharacterizationProvenance(**_valid_provenance_kwargs()),
        warnings=_base_warnings() + ("generated-2d-not-conformer",),
        drawing=DrawingResult(**_valid_drawing_kwargs()),
        svg="<svg/>",
    )
    base.update(overrides)
    return base


def _valid_outcome_kwargs(**overrides):
    base = dict(
        engine_version="2026.03.6",
        representation_format_resolved="smiles",
        values=CharacterizationValues(**_valid_values_kwargs()),
        effective_parameters=CharacterizationEffectiveParameters(
            parser=SmilesParserParameters()
        ),
        warnings=(
            "derived-computational-values",
            "tpsa-no-additional-s-p",
            "generated-2d-not-conformer",
        ),
        drawing=DrawingResult(**_valid_drawing_kwargs()),
        svg="<svg/>",
    )
    base.update(overrides)
    return base


_SEVENTEEN_FIELD_STRING_MATRIX = [
    (CharacterizationValues, "formula", _valid_values_kwargs),
    (DrawingResult, "font_name", _valid_drawing_kwargs),
    (DrawingResult, "font_sha256_observed", _valid_drawing_kwargs),
    (DrawingResult, "svg_sha256", _valid_drawing_kwargs),
    (DrawingResult, "xml_parser_version_observed", _valid_drawing_kwargs),
    (CharacterizationProvenance, "input_state_id", _valid_provenance_kwargs),
    (CharacterizationProvenance, "entity_id", _valid_provenance_kwargs),
    (
        CharacterizationProvenance,
        "state_record_sha256_observed",
        _valid_provenance_kwargs,
    ),
    (CharacterizationProvenance, "representation_path", _valid_provenance_kwargs),
    (
        CharacterizationProvenance,
        "representation_format_declared",
        _valid_provenance_kwargs,
    ),
    (
        CharacterizationProvenance,
        "representation_checksum_expected",
        _valid_provenance_kwargs,
    ),
    (
        CharacterizationProvenance,
        "representation_checksum_observed",
        _valid_provenance_kwargs,
    ),
    (CharacterizationProvenance, "engine_version", _valid_provenance_kwargs),
    (
        CharacterizationProvenance,
        "application_version",
        lambda: _valid_provenance_kwargs(application_version="0.1.0"),
    ),
    (MolecularCharacterizationResult, "svg", _valid_result_kwargs),
    (RDKitCharacterizationOutcome, "engine_version", _valid_outcome_kwargs),
    (RDKitCharacterizationOutcome, "svg", _valid_outcome_kwargs),
]

assert len(_SEVENTEEN_FIELD_STRING_MATRIX) == 17


@pytest.mark.parametrize(
    "model_cls,field,kwargs_factory",
    _SEVENTEEN_FIELD_STRING_MATRIX,
    ids=[f"{cls.__name__}.{field}" for cls, field, _ in _SEVENTEEN_FIELD_STRING_MATRIX],
)
def test_complete_17_field_string_matrix_rejects_bytes_and_bytearray_everywhere(
    model_cls, field, kwargs_factory
):
    # Mirrors the architect's own probe method exactly: the real production
    # module, the real constructor AND model_validate(dict), bytes and
    # bytearray, error localized to the target field - never a copied
    # validator, never model_construct, never a monkeypatch.
    base = kwargs_factory()
    native_value = base[field]
    assert isinstance(native_value, str)

    native = model_cls(**base)
    assert type(getattr(native, field)) is str
    assert getattr(native, field) == native_value
    assert model_cls.model_validate_json(native.model_dump_json()) == native

    for input_type in (bytes, bytearray):
        raw = input_type(native_value.encode("utf-8"))
        for use_model_validate in (False, True):
            payload = dict(base, **{field: raw})
            with pytest.raises(ValidationError) as excinfo:
                if use_model_validate:
                    model_cls.model_validate(payload)
                else:
                    model_cls(**payload)
            assert any(
                error["loc"] == (field,) for error in excinfo.value.errors()
            )


def test_string_closure_three_contracted_null_controls_still_accepted():
    # The architect's probe's own three null controls: application_version,
    # and both top-level results' svg/drawing pair, all contracted nullable
    # (Section 11) and unaffected by the bytes/bytearray fix.
    provenance = CharacterizationProvenance(
        **_valid_provenance_kwargs(application_version=None)
    )
    assert provenance.application_version is None

    result = MolecularCharacterizationResult(
        values=CharacterizationValues(**_valid_values_kwargs()),
        provenance=CharacterizationProvenance(**_valid_provenance_kwargs()),
        warnings=_base_warnings(),
    )
    assert result.svg is None and result.drawing is None

    outcome = RDKitCharacterizationOutcome(
        engine_version="2026.03.6",
        representation_format_resolved="smiles",
        values=CharacterizationValues(**_valid_values_kwargs()),
        effective_parameters=CharacterizationEffectiveParameters(
            parser=SmilesParserParameters()
        ),
        warnings=("derived-computational-values", "tpsa-no-additional-s-p"),
    )
    assert outcome.svg is None and outcome.drawing is None


def test_all_default_profile_blocks_still_construct_with_no_arguments():
    # the strict-type fix must not break the zero-argument defaults that
    # every effective_parameters block relies on.
    SdfParserParameters()
    SmilesParserParameters()
    FormulaParameters()
    ExactMassParameters()
    TpsaParameters()
    CrippenParameters()
    Compute2DParameters()
    PrepareParameters()
    DrawerConstructorParameters()
    DrawOptionsParameters()
    DrawMoleculeParameters()
    EmptyCoordMap()


# --- T-020A-004: svg/drawing consistency ---------------------------------


def _valid_drawing():
    return DrawingResult(
        font_name="Telex-Regular.ttf",
        font_sha256_observed="a" * 64,
        svg_sha256=hashlib.sha256(b"<svg/>").hexdigest(),
        svg_size_bytes=len(b"<svg/>"),
        xml_parser_version_observed="expat_2.6.0",
    )


def test_svg_without_drawing_is_rejected():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    with pytest.raises(ValidationError):
        MolecularCharacterizationResult(
            values=values,
            provenance=provenance,
            warnings=_base_warnings() + ("generated-2d-not-conformer",),
            svg="<svg/>",
        )


def test_drawing_without_svg_is_rejected():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    with pytest.raises(ValidationError):
        MolecularCharacterizationResult(
            values=values,
            provenance=provenance,
            warnings=_base_warnings() + ("generated-2d-not-conformer",),
            drawing=_valid_drawing(),
        )


def test_adapter_outcome_also_enforces_svg_drawing_consistency():
    values = CharacterizationValues(**_valid_values_kwargs())
    with pytest.raises(ValidationError):
        RDKitCharacterizationOutcome(
            engine_version="2026.03.6",
            representation_format_resolved="smiles",
            values=values,
            effective_parameters=CharacterizationEffectiveParameters(
                parser=SmilesParserParameters()
            ),
            warnings=("derived-computational-values", "tpsa-no-additional-s-p"),
            svg="<svg/>",
        )


# --- T-020A-005: SVG hash/size and checksum case-insensitivity ----------


def test_svg_hash_mismatch_is_rejected():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    drawing = _valid_drawing().model_copy(update={"svg_sha256": "f" * 64})
    with pytest.raises(ValidationError):
        MolecularCharacterizationResult(
            values=values,
            provenance=provenance,
            warnings=_base_warnings() + ("generated-2d-not-conformer",),
            drawing=drawing,
            svg="<svg/>",
        )


def test_svg_size_mismatch_is_rejected():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    drawing = _valid_drawing().model_copy(update={"svg_size_bytes": 999})
    with pytest.raises(ValidationError):
        MolecularCharacterizationResult(
            values=values,
            provenance=provenance,
            warnings=_base_warnings() + ("generated-2d-not-conformer",),
            drawing=drawing,
            svg="<svg/>",
        )


def test_valid_svg_drawing_pair_is_accepted():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    result = MolecularCharacterizationResult(
        values=values,
        provenance=provenance,
        warnings=_base_warnings() + ("generated-2d-not-conformer",),
        drawing=_valid_drawing(),
        svg="<svg/>",
    )
    assert result.svg == "<svg/>"


def test_checksum_expected_uppercase_equivalent_to_observed_lowercase():
    provenance = CharacterizationProvenance(
        **_valid_provenance_kwargs(
            representation_checksum_expected="AB" * 32,
            representation_checksum_observed="ab" * 32,
        )
    )
    assert provenance.representation_checksum_expected == "AB" * 32


def test_checksum_divergence_is_rejected():
    with pytest.raises(ValidationError):
        CharacterizationProvenance(
            **_valid_provenance_kwargs(
                representation_checksum_expected="a" * 64,
                representation_checksum_observed="b" * 64,
            )
        )


def test_malformed_expected_checksum_is_rejected():
    with pytest.raises(ValidationError):
        CharacterizationProvenance(
            **_valid_provenance_kwargs(representation_checksum_expected="not-hex")
        )


def test_parser_block_must_match_resolved_format():
    with pytest.raises(ValidationError):
        CharacterizationProvenance(
            **_valid_provenance_kwargs(
                representation_format_resolved="sdf",
                effective_parameters=CharacterizationEffectiveParameters(
                    parser=SmilesParserParameters()
                ),
            )
        )


# --- Warnings fixed vocabulary -------------------------------------------


def test_warnings_must_match_fixed_order_exactly():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    with pytest.raises(ValidationError):
        MolecularCharacterizationResult(
            values=values,
            provenance=provenance,
            warnings=("conditions-not-consumed", "derived-computational-values"),
        )


def test_generated_2d_warning_required_when_svg_present():
    values = CharacterizationValues(**_valid_values_kwargs())
    provenance = CharacterizationProvenance(**_valid_provenance_kwargs())
    with pytest.raises(ValidationError):
        MolecularCharacterizationResult(
            values=values,
            provenance=provenance,
            warnings=_base_warnings(),
            drawing=_valid_drawing(),
            svg="<svg/>",
        )


# --- Error taxonomy (Section 13) -----------------------------------------


@pytest.mark.parametrize(
    "error_type,banner",
    [
        (CharacterizationInputError, "INPUT ERROR"),
        (CharacterizationConfigurationError, "CONFIGURATION ERROR"),
        (CharacterizationStorageError, "STORAGE ERROR"),
        (CharacterizationExecutionError, "EXECUTION ERROR"),
    ],
)
def test_error_category_banner(error_type, banner):
    assert error_type.category_banner == banner
    assert issubclass(error_type, CharacterizationError)


def test_input_error_accepts_its_own_reason():
    error = CharacterizationInputError("bad", reason="invalid-request")
    assert error.reason == "invalid-request"
    assert error.input_state_id is None


def test_error_reason_outside_category_vocabulary_is_rejected():
    with pytest.raises(ValueError):
        CharacterizationInputError("bad", reason="unsupported-platform")
    with pytest.raises(ValueError):
        CharacterizationExecutionError("bad", reason="engine-unavailable")


def test_error_carries_input_state_id_only_when_given():
    error = CharacterizationExecutionError(
        "bad", reason="invalid-chemical-input", input_state_id="caffeine-1"
    )
    assert error.input_state_id == "caffeine-1"


def test_every_execution_error_reason_is_constructible():
    for reason in (
        "invalid-chemical-input",
        "unsupported-structure",
        "molecular-size-limit",
        "descriptor-failure",
        "depiction-failure",
        "invalid-engine-result",
        "internal-failure",
    ):
        error = CharacterizationExecutionError("x", reason=reason)
        assert error.reason == reason


def test_every_configuration_error_reason_is_constructible():
    for reason in (
        "unsupported-platform",
        "invalid-state-record",
        "unsupported-format",
        "invalid-reference",
        "engine-unavailable",
        "drawing-unavailable",
    ):
        error = CharacterizationConfigurationError("x", reason=reason)
        assert error.reason == reason


def test_every_storage_error_reason_is_constructible():
    for reason in (
        "missing-artifact",
        "unreadable-artifact",
        "unsafe-or-changed-artifact",
        "checksum-mismatch",
        "artifact-size-invalid",
    ):
        error = CharacterizationStorageError("x", reason=reason)
        assert error.reason == reason


def test_exception_chaining_preserves_cause():
    cause = ValueError("original")
    try:
        try:
            raise cause
        except ValueError as inner:
            raise CharacterizationExecutionError(
                "wrapped", reason="internal-failure"
            ) from inner
    except CharacterizationExecutionError as outer:
        assert outer.__cause__ is cause


# =========================================================================
# Section 10/10.1/10.2: closed SVG validation/binding profile
# =========================================================================

_VALID_ROOT = (
    "<svg version='1.1' baseProfile='full' xmlns='http://www.w3.org/2000/svg' "
    "xml:space='preserve' width='600px' height='400px' viewBox='0 0 600 400'>"
)


def _bind(svg, state_id="caffeine-1", rep_sha=None, engine_version="2026.03.6"):
    return validate_and_bind_depiction_svg(
        svg,
        input_state_id=state_id,
        representation_sha256=rep_sha or ("a" * 64),
        engine_version=engine_version,
    )


def test_minimal_valid_svg_is_accepted_and_bound():
    svg = _VALID_ROOT + (
        "<rect x='0' y='0' width='600' height='400' style='fill:#FFFFFF;stroke:none'/>"
        "<path d='M 0,0 L 1,1 Q 2,2 3,3' class='bond-0 atom-0 atom-1' "
        "style='stroke:#000000;stroke-width:2px;stroke-dasharray:6,4;"
        "stroke-miterlimit:10;stroke-linecap:butt;stroke-linejoin:miter;"
        "stroke-opacity:1;fill-rule:evenodd;opacity:0.5'/></svg>"
    )
    final = _bind(svg, state_id="caffeine-1", rep_sha="b" * 64, engine_version="2026.03.6")
    assert 'data-state-id="caffeine-1"' in final
    assert f'data-representation-sha256="{"b" * 64}"' in final
    assert 'data-engine-version="2026.03.6"' in final
    assert 'data-profile="molecular-depiction-v1"' in final
    assert final.encode("utf-8").decode("utf-8") == final


def test_binding_preserves_exact_state_id_without_normalization():
    svg = _VALID_ROOT + "<rect x='0' y='0' width='1' height='1' fill='none'/></svg>"
    final = _bind(svg, state_id="Caffeine-01_a")
    assert 'data-state-id="Caffeine-01_a"' in final


def test_binding_escapes_special_characters_in_state_id():
    # input_state_id is always pre-validated by validate_input_id upstream
    # in production, but the binder itself must still escape defensively -
    # this exercises that the serializer never concatenates raw text as
    # markup (Section 10.2).
    svg = _VALID_ROOT + "<rect x='0' y='0' width='1' height='1' fill='none'/></svg>"
    final = _bind(svg, state_id="a&b<c>d\"e")
    assert 'data-state-id="a&amp;b&lt;c&gt;d&quot;e"' in final
    assert "<c>" not in final


@pytest.mark.parametrize(
    "fragment",
    [
        "<!DOCTYPE svg>",
        "<!DOCTYPE svg [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
    ],
)
def test_doctype_is_rejected(fragment):
    svg = f"<?xml version='1.0'?>{fragment}" + _VALID_ROOT + "</svg>"
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _bind(svg)
    assert excinfo.value.reason == "depiction-failure"


def test_external_entity_reference_is_rejected():
    svg = (
        "<?xml version='1.0'?>"
        "<!DOCTYPE svg [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>"
        + _VALID_ROOT
        + "<rect x='0' y='0' width='1' height='1' fill='none'/>&xxe;</svg>"
    )
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


def test_predefined_entity_is_resolved_and_still_subject_to_grammar():
    # &amp; is a predefined entity RDKit might never emit, but the parser
    # must still resolve it and then subject the resulting value to the
    # attribute's own grammar - here it produces an invalid class token.
    svg = _VALID_ROOT + "<path d='M 0,0 L 1,1' class='atom-1&amp;x'/></svg>"
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


@pytest.mark.parametrize(
    "child",
    [
        "<foreignObject/>",
        "<image href='x.png'/>",
        "<use href='#x'/>",
        "<script>1</script>",
        "<text>hi</text>",
        "<a xmlns:xlink='http://www.w3.org/1999/xlink' xlink:href='http://evil'/>",
    ],
)
def test_disallowed_elements_are_rejected(child):
    svg = _VALID_ROOT + child + "</svg>"
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


def test_processing_instruction_is_rejected():
    svg = _VALID_ROOT + "<?evil instruction?></svg>"
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


def test_non_whitespace_text_content_is_rejected():
    svg = _VALID_ROOT + "<path d='M 0,0 L 1,1'>hello</path></svg>"
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


def test_grandchild_element_is_rejected():
    svg = _VALID_ROOT + "<rect x='0' y='0' width='1' height='1'><rect/></rect></svg>"
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


@pytest.mark.parametrize(
    "attrs",
    [
        "version='1.0'",  # wrong version
        "width='601px'",  # wrong width
        "height='399px'",  # wrong height
        "viewBox='0 0 599 400'",  # wrong viewBox
        "baseProfile='tiny'",  # wrong baseProfile
    ],
)
def test_root_attribute_violations_are_rejected(attrs):
    overridden = (
        "<svg version='1.1' baseProfile='full' xmlns='http://www.w3.org/2000/svg' "
        f"width='600px' height='400px' viewBox='0 0 600 400' {attrs}>"
    )
    with pytest.raises(CharacterizationExecutionError):
        _bind(overridden + "</svg>")


def test_unknown_root_attribute_is_rejected():
    svg = (
        "<svg version='1.1' xmlns='http://www.w3.org/2000/svg' width='600px' "
        "height='400px' viewBox='0 0 600 400' onclick='evil()'>"
        "</svg>"
    )
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


@pytest.mark.parametrize(
    "class_value",
    ["atom-0", "bond-12", "atom-0 bond-1"],
)
def test_valid_class_tokens_are_accepted(class_value):
    svg = _VALID_ROOT + f"<path d='M 0,0 L 1,1' class='{class_value}'/></svg>"
    _bind(svg)  # must not raise


@pytest.mark.parametrize(
    "class_value",
    ["evil", "atom-", "atom--1", "atom-1 ", " atom-1", "atom-1  bond-2"],
)
def test_invalid_class_tokens_are_rejected(class_value):
    svg = _VALID_ROOT + f"<path d='M 0,0 L 1,1' class='{class_value}'/></svg>"
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


@pytest.mark.parametrize(
    "style",
    [
        "color:red",  # unknown property
        "fill:#ZZZZZZ",  # malformed color
        "fill:red",  # named color not permitted
        "opacity:2",  # out of [0,1]
        "opacity:-0.1",
        "fill-rule:xor",
        "stroke-linecap:round;stroke-linecap:butt",  # duplicate property
        "stroke-width:-1px",
        "stroke-dasharray:0,0",  # no positive number
        "stroke-miterlimit:0.5",  # < 1
        "fill:url(evil.svg)",
        "",  # empty style
        "fill:none;;stroke:none",  # empty internal declaration
    ],
)
def test_invalid_style_declarations_are_rejected(style):
    svg = _VALID_ROOT + f"<rect x='0' y='0' width='1' height='1' style='{style}'/></svg>"
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


@pytest.mark.parametrize(
    "style",
    [
        "fill:#FFFFFF;stroke:none",
        "opacity:1.0;fill:#FFFFFF;stroke:none",
        "stroke-dasharray:6,4",
        "stroke-dasharray:none",
        "stroke-miterlimit:10",
        "stroke-width:2.0px",
    ],
)
def test_valid_style_declarations_are_accepted(style):
    svg = _VALID_ROOT + f"<rect x='0' y='0' width='1' height='1' style='{style}'/></svg>"
    _bind(svg)  # must not raise


@pytest.mark.parametrize(
    "d_value",
    [
        "L 0,0 1,1",  # must start with M
        "M nan,0 L 1,1",  # non-finite
        "M 0,0 L 1",  # incomplete pair (residual)
        "M 0,0 Z",  # unsupported command
        "M 0,0 L 1,1 2",  # trailing residual number
        "M 0,0 l 1,1",  # relative command not permitted
    ],
)
def test_invalid_d_attribute_is_rejected(d_value):
    svg = _VALID_ROOT + f"<path d='{d_value}'/></svg>"
    with pytest.raises(CharacterizationExecutionError):
        _bind(svg)


@pytest.mark.parametrize(
    "d_value",
    [
        "M 0,0 L 1,1",
        "M 0,0 1,1 2,2",  # implicit repeated M-arity group
        "M 0,0 Q 1,1 2,2 3,3 4,4",
        "M -1.5,2.5e1 L 3,-4",
    ],
)
def test_valid_d_attribute_is_accepted(d_value):
    svg = _VALID_ROOT + f"<path d='{d_value}'/></svg>"
    _bind(svg)  # must not raise


def test_svg_exceeding_max_bytes_is_rejected(monkeypatch):
    monkeypatch.setattr(characterization_module, "MAX_SVG_UTF8_BYTES", 10)
    svg = _VALID_ROOT + "<rect x='0' y='0' width='1' height='1' fill='none'/></svg>"
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _bind(svg)
    assert excinfo.value.reason == "depiction-failure"


def test_svg_at_exact_byte_limit_is_accepted(monkeypatch):
    svg = _VALID_ROOT + "<rect x='0' y='0' width='1' height='1' fill='none'/></svg>"
    final = _bind(svg)
    monkeypatch.setattr(
        characterization_module, "MAX_SVG_UTF8_BYTES", len(final.encode("utf-8"))
    )
    _bind(svg)  # must not raise at exactly the limit
