"""Tests for the RDKit adapter (src/molecular/rdkit_tautomer.py), the
only production module permitted to import rdkit. Protected by
pytest.importorskip at module scope, reflecting the optional nature of
this dependency (Section 6 of the Increment 016 frozen spec) - this whole
file is skipped in a base installation with no rdkit extra."""

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

import hashlib
import math
from io import BytesIO

import molecular.characterization as characterization_module
import molecular.rdkit_tautomer as rdkit_tautomer_module
from molecular.characterization import (
    CharacterizationConfigurationError,
    CharacterizationError,
    CharacterizationExecutionError,
    CharacterizationInputError,
    CharacterizationStorageError,
    MAX_BOND_COUNT,
    MAX_TOTAL_ATOM_COUNT,
    REQUIRED_ENGINE_VERSION,
)
from molecular.promotion import LigandPromotionPolicy
from molecular.rdkit_tautomer import (
    DisconnectedStructureError,
    IncompleteEnumerationError,
    InvalidPromotionSDFError,
    InvalidSDFError,
    MultipleMoleculesError,
    PromotionRecordCountError,
    RDKitEngineResolution,
    RDKitLigandInspectionResult,
    RDKitTautomerCandidate,
    RDKitTautomerEnumerationResult,
    RDKitUnavailableError,
    _classify_record_counts,
    characterize_molecule_rdkit,
    enumerate_tautomers_rdkit,
    inspect_single_sdf_for_promotion,
    resolve_rdkit_engine,
)


def _write_sdf(tmp_path, filename, smiles):
    mol = Chem.MolFromSmiles(smiles)
    path = tmp_path / filename
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def _write_multi_sdf(tmp_path, filename, smiles, count):
    mol = Chem.MolFromSmiles(smiles)
    path = tmp_path / filename
    writer = Chem.SDWriter(str(path))
    for _ in range(count):
        writer.write(mol)
    writer.close()
    return path


# --- D-016-SPIKE-08 / Section 8.3: Route A negative contract test ---


def test_route_a_absence():
    enumerator = rdMolStandardize.TautomerEnumerator()
    assert hasattr(enumerator, "SetRemoveIsotopicHs") is False
    assert hasattr(enumerator, "GetRemoveIsotopicHs") is False


# --- Route B configuration and read-back (Section 7.2/7.4/8) ---


def test_route_b_configures_and_reads_back_remove_isotopic_hs():
    params = rdMolStandardize.CleanupParameters()
    params.tautomerRemoveIsotopicHs = False
    assert params.tautomerRemoveIsotopicHs is False

    enumerator = rdMolStandardize.TautomerEnumerator(params)
    assert enumerator is not None
    # the effective value is still read from the same `params` object,
    # both before and after constructing the enumerator - no getter
    # exists on the enumerator itself (test_route_a_absence above).
    assert params.tautomerRemoveIsotopicHs is False


# --- resolve_rdkit_engine (Section 7.1) ---


def test_resolve_rdkit_engine_returns_expected_shape():
    resolution = resolve_rdkit_engine()

    assert isinstance(resolution, RDKitEngineResolution)
    assert resolution.engine_name == "rdkit"
    assert resolution.engine_version == rdkit_tautomer_module.rdBase.rdkitVersion


def test_resolve_rdkit_engine_returns_the_twelve_fixed_parameters_exactly():
    resolution = resolve_rdkit_engine()

    assert dict(resolution.parameters) == {
        "algorithm": "rdkit.MolStandardize.TautomerEnumerator",
        "input_format": "sdf",
        "output_format": "smiles",
        "max_tautomers": 1000,
        "max_transforms": 1000,
        "remove_bond_stereo": True,
        "remove_sp3_stereo": True,
        "reassign_stereo": True,
        "remove_isotopic_hs": False,
        "deduplication": "canonical-isomeric-smiles",
        "ordering": "canonical-isomeric-smiles-lexicographic",
        "selection": "complete-then-policy-limit",
    }


def test_resolve_rdkit_engine_version_has_real_observed_format():
    resolution = resolve_rdkit_engine()
    # observed format: "20XX.0X.X" (zero-padded month) - D-016-SPIKE-02
    parts = resolution.engine_version.split(".")
    assert len(parts) == 3
    assert parts[0].isdigit() and len(parts[0]) == 4


def test_resolve_rdkit_engine_missing_import_produces_rdkit_unavailable_error(monkeypatch):
    """Simulates rdkit being absent by monkeypatching this module's own
    import-boundary sentinel - never a raw ImportError propagates."""

    monkeypatch.setattr(
        rdkit_tautomer_module, "_RDKIT_IMPORT_ERROR", ImportError("simulated: no module named rdkit")
    )

    with pytest.raises(RDKitUnavailableError):
        resolve_rdkit_engine()


def test_resolve_rdkit_engine_unreadable_version_produces_rdkit_unavailable_error(monkeypatch):
    monkeypatch.setattr(rdkit_tautomer_module.rdBase, "rdkitVersion", "")

    with pytest.raises(RDKitUnavailableError):
        resolve_rdkit_engine()


# --- Chemistry: acetylacetone monodeuterated (Section 20.2/20.6) ---


def test_monodeuterated_acetylacetone_preserves_isotope_in_all_candidates(tmp_path):
    sdf_path = _write_sdf(tmp_path, "d-acac.sdf", "CC(=O)C([2H])C(=O)C")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert result.enumerated_unique_count == 5
    assert len(result.candidates) == result.new_unique_count

    for candidate in result.candidates:
        mol = Chem.MolFromSmiles(candidate.canonical_isomeric_smiles)
        assert mol is not None
        Chem.SanitizeMol(mol)  # must not raise: only sanitizable candidates are published
        assert rdMolDescriptors.CalcMolFormula(mol) == "C5H8O2"
        isotopes = [atom.GetIsotope() for atom in mol.GetAtoms() if atom.GetIsotope() != 0]
        assert isotopes == [2]


# --- Chemistry: enol-OD, corrected 5/8 count (Section 20.2/20.6) ---


def test_enol_od_produces_corrected_five_valid_eight_invalid(tmp_path):
    sdf_path = _write_sdf(tmp_path, "enol-od.sdf", "[2H]OC(C)=CC(C)=O")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    # enumerated_unique_count includes the entry itself if it survives
    # sanitization/invariants and is among the unique valid candidates -
    # the corrected spike evidence is 5 valid (of which the entry is one),
    # so new_unique_count (entry excluded) is 4.
    assert result.enumerated_unique_count == 5
    assert result.new_unique_count == 4
    assert len(result.candidates) == 4


def test_enumerated_and_new_unique_counts_are_internally_consistent(tmp_path):
    sdf_path = _write_sdf(tmp_path, "acac.sdf", "CC(=O)CC(C)=O")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert result.enumerated_unique_count - result.new_unique_count in (0, 1)
    assert len(result.candidates) == result.new_unique_count


# --- RDKitTautomerEnumerationResult._validate_counts (Section 7.3/20.2) ---


def test_validate_counts_accepts_difference_zero():
    result = RDKitTautomerEnumerationResult(
        enumerated_unique_count=3,
        new_unique_count=3,
        candidates=(
            RDKitTautomerCandidate(canonical_isomeric_smiles="C"),
            RDKitTautomerCandidate(canonical_isomeric_smiles="CC"),
            RDKitTautomerCandidate(canonical_isomeric_smiles="CCC"),
        ),
    )
    assert result.enumerated_unique_count - result.new_unique_count == 0


def test_validate_counts_accepts_difference_one():
    result = RDKitTautomerEnumerationResult(
        enumerated_unique_count=3,
        new_unique_count=2,
        candidates=(
            RDKitTautomerCandidate(canonical_isomeric_smiles="C"),
            RDKitTautomerCandidate(canonical_isomeric_smiles="CC"),
        ),
    )
    assert result.enumerated_unique_count - result.new_unique_count == 1


def test_validate_counts_rejects_negative_difference():
    with pytest.raises(Exception):
        RDKitTautomerEnumerationResult(
            enumerated_unique_count=1,
            new_unique_count=2,
            candidates=(
                RDKitTautomerCandidate(canonical_isomeric_smiles="C"),
                RDKitTautomerCandidate(canonical_isomeric_smiles="CC"),
            ),
        )


def test_validate_counts_rejects_difference_greater_than_one():
    with pytest.raises(Exception):
        RDKitTautomerEnumerationResult(
            enumerated_unique_count=5,
            new_unique_count=2,
            candidates=(
                RDKitTautomerCandidate(canonical_isomeric_smiles="C"),
                RDKitTautomerCandidate(canonical_isomeric_smiles="CC"),
            ),
        )


def test_validate_counts_rejects_candidates_length_mismatch():
    with pytest.raises(Exception):
        RDKitTautomerEnumerationResult(
            enumerated_unique_count=2,
            new_unique_count=2,
            candidates=(RDKitTautomerCandidate(canonical_isomeric_smiles="C"),),
        )


# --- TautomerEnumeratorStatus: Completed, MaxTautomersReached,
#     MaxTransformsReached, Canceled (Section 20.2, D-016-SPIKE-07) ---


def test_status_completed(tmp_path):
    sdf_path = _write_sdf(tmp_path, "guanine.sdf", "Nc1nc2[nH]cnc2c(=O)[nH]1")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert isinstance(result, RDKitTautomerEnumerationResult)


def test_status_max_tautomers_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(rdkit_tautomer_module, "_MAX_TAUTOMERS", 2)
    sdf_path = _write_sdf(tmp_path, "guanine.sdf", "Nc1nc2[nH]cnc2c(=O)[nH]1")

    with pytest.raises(IncompleteEnumerationError) as exc_info:
        enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert exc_info.value.status == "max-tautomers-reached"
    assert exc_info.value.tautomer_count == 2


def test_enumeration_logs_are_suppressed_and_restored_after_incomplete_status(
    tmp_path, monkeypatch, capfd
):
    monkeypatch.setattr(rdkit_tautomer_module, "_MAX_TAUTOMERS", 2)
    sdf_path = _write_sdf(tmp_path, "guanine.sdf", "Nc1nc2[nH]cnc2c(=O)[nH]1")

    capfd.readouterr()
    with pytest.raises(IncompleteEnumerationError) as exc_info:
        enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    captured = capfd.readouterr()
    assert exc_info.value.status == "max-tautomers-reached"
    assert exc_info.value.tautomer_count == 2
    assert "Tautomer enumeration stopped" not in captured.err

    marker = "__increment017_rdkit_logs_restored__"
    rdBase.LogWarningMsg(marker)
    restored = capfd.readouterr()
    assert marker in restored.err


def test_successful_enumeration_restores_rdkit_logs(tmp_path, capfd):
    sdf_path = _write_sdf(tmp_path, "methane.sdf", "C")

    capfd.readouterr()
    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")
    captured = capfd.readouterr()

    assert result.enumerated_unique_count == 1
    assert result.new_unique_count == 0
    assert result.candidates == ()
    assert "Tautomer enumeration stopped" not in captured.err

    marker = "__increment017_rdkit_success_logs_restored__"
    rdBase.LogWarningMsg(marker)
    restored = capfd.readouterr()
    assert marker in restored.err


def test_status_max_transforms_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(rdkit_tautomer_module, "_MAX_TRANSFORMS", 1)
    sdf_path = _write_sdf(tmp_path, "guanine.sdf", "Nc1nc2[nH]cnc2c(=O)[nH]1")

    with pytest.raises(IncompleteEnumerationError) as exc_info:
        enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert exc_info.value.status == "max-transforms-reached"


def test_status_canceled(tmp_path, monkeypatch):
    """Injects a real TautomerEnumeratorCallback via a test-only monkeypatch
    of the module's internal _build_enumerator seam - enumerate_tautomers_rdkit's
    own public contract has no callback parameter (Section 7.3), so this is
    the only way to reach a real Canceled status through the full pipeline."""

    class CancelAfterTwo(rdMolStandardize.TautomerEnumeratorCallback):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __call__(self, mol, result):
            self.calls += 1
            return self.calls <= 2

    original_build_enumerator = rdkit_tautomer_module._build_enumerator
    callback = CancelAfterTwo()

    def _build_enumerator_with_callback():
        enumerator = original_build_enumerator()
        enumerator.SetCallback(callback)
        return enumerator

    monkeypatch.setattr(rdkit_tautomer_module, "_build_enumerator", _build_enumerator_with_callback)

    sdf_path = _write_sdf(tmp_path, "guanine.sdf", "Nc1nc2[nH]cnc2c(=O)[nH]1")

    with pytest.raises(IncompleteEnumerationError) as exc_info:
        enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert exc_info.value.status == "canceled"


# --- SDF parsing (Section 9, D-016-SPIKE-03) ---


def test_single_molecule_sdf_is_accepted(tmp_path):
    sdf_path = _write_sdf(tmp_path, "single.sdf", "CC(=O)CC(C)=O")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert isinstance(result, RDKitTautomerEnumerationResult)


def test_empty_sdf_produces_invalid_sdf_error_never_raw_oserror(tmp_path):
    empty_path = tmp_path / "empty.sdf"
    empty_path.write_bytes(b"")

    with pytest.raises(InvalidSDFError):
        enumerate_tautomers_rdkit(source_path=empty_path, source_format="sdf")


def test_three_molecule_sdf_produces_multiple_molecules_error(tmp_path):
    multi_path = _write_multi_sdf(tmp_path, "multi.sdf", "CC(=O)CC(C)=O", count=3)

    with pytest.raises(MultipleMoleculesError) as exc_info:
        enumerate_tautomers_rdkit(source_path=multi_path, source_format="sdf")

    assert exc_info.value.record_count == 3


def test_unsupported_source_format_rejected_without_touching_filesystem(tmp_path):
    missing_path = tmp_path / "does-not-exist.mol2"

    with pytest.raises(ValueError):
        enumerate_tautomers_rdkit(source_path=missing_path, source_format="mol2")


# --- Charge, stereochemistry, entry exclusion, ordering (Section 20.2) ---


def test_charged_amidinium_cation_preserves_formal_charge(tmp_path):
    sdf_path = _write_sdf(tmp_path, "amidinium.sdf", "[NH2+]=CN")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    for candidate in result.candidates:
        mol = Chem.MolFromSmiles(candidate.canonical_isomeric_smiles)
        assert Chem.GetFormalCharge(mol) == 1


def test_stereocenter_spectator_descriptor_preserved(tmp_path):
    sdf_path = _write_sdf(tmp_path, "stereo.sdf", "C[C@H](F)CC(=O)CC(C)=O")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert len(result.candidates) > 0
    for candidate in result.candidates:
        assert "[C@H]" in candidate.canonical_isomeric_smiles


def test_entry_equivalent_representation_excluded_from_result(tmp_path):
    sdf_path = _write_sdf(tmp_path, "acac.sdf", "CC(=O)CC(C)=O")
    input_mol = Chem.MolFromSmiles("CC(=O)CC(C)=O")
    entry_smiles = Chem.MolToSmiles(input_mol, isomericSmiles=True)

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert all(candidate.canonical_isomeric_smiles != entry_smiles for candidate in result.candidates)


def test_candidates_are_sorted_lexicographically(tmp_path):
    sdf_path = _write_sdf(tmp_path, "acac.sdf", "CC(=O)CC(C)=O")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    smiles_list = [candidate.canonical_isomeric_smiles for candidate in result.candidates]
    assert smiles_list == sorted(smiles_list)
    assert len(smiles_list) > 1  # a non-trivial ordering check


def test_candidates_are_deduplicated_entry_excluded_and_sorted_in_that_order(tmp_path):
    """Section 15, steps 14-16 (frozen spec): deduplicate, THEN exclude
    the entry-equivalent representation, THEN sort lexicographically -
    the adapter's code must follow this exact order, not merely produce
    an observably equivalent result via a different order (MINOR-1,
    Increment 016 adversarial code review reconciliation)."""

    sdf_path = _write_sdf(tmp_path, "enol-od.sdf", "[2H]OC(C)=CC(C)=O")
    input_mol = Chem.MolFromSmiles("[2H]OC(C)=CC(C)=O")
    entry_smiles = Chem.MolToSmiles(input_mol, isomericSmiles=True)

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    smiles_list = [candidate.canonical_isomeric_smiles for candidate in result.candidates]

    # deduplication: no canonical SMILES appears more than once.
    assert len(smiles_list) == len(set(smiles_list))
    assert len(smiles_list) == result.new_unique_count

    # entry exclusion: the input's own representation never appears.
    assert entry_smiles not in smiles_list

    # final ordering: strictly lexicographic.
    assert smiles_list == sorted(smiles_list)
    assert len(smiles_list) > 1  # a non-trivial check on all three properties at once


# --- No RDKit object ever escapes a public function's return (Section 20.2) ---


def test_no_rdkit_object_escapes_resolve_rdkit_engine():
    resolution = resolve_rdkit_engine()
    assert not isinstance(resolution, Chem.rdchem.Mol)
    for value in resolution.parameters.values():
        assert isinstance(value, (str, int, float, bool))


def test_no_rdkit_object_escapes_enumerate_tautomers_rdkit(tmp_path):
    sdf_path = _write_sdf(tmp_path, "acac.sdf", "CC(=O)CC(C)=O")

    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")

    assert not isinstance(result, Chem.rdchem.Mol)
    for candidate in result.candidates:
        assert not isinstance(candidate, Chem.rdchem.Mol)
        assert isinstance(candidate.canonical_isomeric_smiles, str)


# =============================================================================
# RDKit RAW SDF chemical inspection for promotion
# (Section 9 of Increment 018's frozen final specification). Covers
# Section 19.4 (items 67-88) of the frozen spec's normative test matrix.
# =============================================================================

_PENTAVALENT_CARBON_MOLBLOCK = """pentavalent
  RDKit          2D

  6  5  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    1.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
  1  6  1  0
M  END
$$$$
"""


def _write_mol(tmp_path, filename, mol):
    path = tmp_path / filename
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def _write_raw_text(tmp_path, filename, text):
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    return path


def _default_policy(**overrides):
    return LigandPromotionPolicy(**overrides)


# --- 67-69: supplier construction / empty / missing source ---


def test_inspect_happy_path_returns_neutral_result(tmp_path):
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")

    result = inspect_single_sdf_for_promotion(source_path=sdf_path, policy=_default_policy())

    assert isinstance(result, RDKitLigandInspectionResult)
    assert result.engine_name == "rdkit"
    assert result.input_format == "sdf"
    assert result.record_count == 1
    assert result.fragment_count == 1
    assert result.formal_charge == 0
    assert result.atom_count == 3
    assert not isinstance(result, Chem.rdchem.Mol)


def test_inspect_empty_file_raises_invalid_promotion_sdf_error(tmp_path):
    empty_path = _write_raw_text(tmp_path, "empty.sdf", "")

    with pytest.raises(InvalidPromotionSDFError):
        inspect_single_sdf_for_promotion(source_path=empty_path, policy=_default_policy())


def test_inspect_missing_file_raises_invalid_promotion_sdf_error(tmp_path):
    missing_path = tmp_path / "missing.sdf"

    with pytest.raises(InvalidPromotionSDFError):
        inspect_single_sdf_for_promotion(source_path=missing_path, policy=_default_policy())


# --- 70-72: single-molecule rule ---


def test_inspect_two_records_raises_promotion_record_count_error(tmp_path):
    two_path = _write_multi_sdf(tmp_path, "two.sdf", "CCO", 2)

    with pytest.raises(PromotionRecordCountError) as exc_info:
        inspect_single_sdf_for_promotion(source_path=two_path, policy=_default_policy())

    assert exc_info.value.record_count == 2


def test_classify_record_counts_zero_records_raises_with_count_zero():
    """Item 71: total==0 was never empirically reproduced by any RD-018-01
    fixture (a supplier that constructs successfully but iterates zero
    records) - tested directly against the platform's own decision
    function rather than via an unattainable real fixture."""

    with pytest.raises(PromotionRecordCountError) as exc_info:
        _classify_record_counts(0, 0)

    assert exc_info.value.record_count == 0


def test_classify_record_counts_single_invalid_record_raises_invalid_sdf():
    with pytest.raises(InvalidPromotionSDFError):
        _classify_record_counts(1, 0)


def test_classify_record_counts_single_valid_record_accepts():
    assert _classify_record_counts(1, 1) is None


def test_classify_record_counts_two_records_one_valid_still_rejects_by_count():
    """Item 72: the single valid record is never accepted in isolation
    when the file contains more than one physical record, regardless of
    how many of them individually parsed - Section 9.2, step 6."""

    with pytest.raises(PromotionRecordCountError) as exc_info:
        _classify_record_counts(2, 1)

    assert exc_info.value.record_count == 2


# --- 73-75: RDKit's own defaults must never be relied upon implicitly ---


def test_inspect_always_passes_explicit_kwargs_to_supplier(tmp_path, monkeypatch):
    """Items 73-75: the three RDKit flags are always taken explicitly from
    `policy` and passed explicitly to `Chem.SDMolSupplier` - never RDKit's
    own native defaults (RD-018-01)."""

    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")
    captured = {}
    real_supplier = Chem.SDMolSupplier

    def _capturing_supplier(path, sanitize, removeHs, strictParsing):
        captured["sanitize"] = sanitize
        captured["removeHs"] = removeHs
        captured["strictParsing"] = strictParsing
        return real_supplier(path, sanitize=sanitize, removeHs=removeHs, strictParsing=strictParsing)

    monkeypatch.setattr(rdkit_tautomer_module.Chem, "SDMolSupplier", _capturing_supplier)

    policy = _default_policy(sanitize=True, remove_hs=False, strict_parsing=True)
    inspect_single_sdf_for_promotion(source_path=sdf_path, policy=policy)

    assert captured == {"sanitize": True, "removeHs": False, "strictParsing": True}


def test_inspect_echoes_the_policy_flags_actually_used(tmp_path):
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")
    policy = _default_policy(sanitize=True, remove_hs=True, strict_parsing=False)

    result = inspect_single_sdf_for_promotion(source_path=sdf_path, policy=policy)

    assert result.sanitize is True
    assert result.remove_hs is True
    assert result.strict_parsing is False


# --- 76-78: sanitize=True,removeHs=True masks a pentavalent-carbon error ---


def test_pentavalent_carbon_rejected_under_production_default(tmp_path):
    """Item 76: the production default (`sanitize=True, remove_hs=False,
    strict_parsing=True`) correctly rejects a molecule with an invalid
    (pentavalent) carbon valence - the single record fails to parse into a
    non-null molecule."""

    path = _write_raw_text(tmp_path, "pentavalent.sdf", _PENTAVALENT_CARBON_MOLBLOCK)

    with pytest.raises(InvalidPromotionSDFError):
        inspect_single_sdf_for_promotion(source_path=path, policy=_default_policy())


def test_pentavalent_carbon_masked_into_methane_when_remove_hs_true(tmp_path):
    """Items 77-78: `sanitize=True, removeHs=True` masks the SAME
    pentavalent-carbon valence error into a silently-accepted methane
    (RD-018-01's confirmed finding) - explicit Hs, including the ones
    responsible for the invalid valence, are stripped before the valence
    check runs. This is not a bug in this adapter: it is exactly why the
    production default is `remove_hs=False`, and this test exists to
    faithfully reproduce - not paper over - that documented RDKit
    behavior."""

    path = _write_raw_text(tmp_path, "pentavalent.sdf", _PENTAVALENT_CARBON_MOLBLOCK)
    policy = _default_policy(sanitize=True, remove_hs=True, strict_parsing=True)

    result = inspect_single_sdf_for_promotion(source_path=path, policy=policy)

    assert result.atom_count == 1
    assert result.explicit_hydrogen_count == 0


# --- 79-80: sanitize=False posterior check on a copy ---


def test_sanitize_false_pentavalent_carbon_rejected_by_posterior_check(tmp_path, capfd):
    """Items 79-80: with `sanitize=False`, the supplier itself happily
    accepts the pentavalent-carbon molecule unsanitized - the mandatory
    posterior `Chem.SanitizeMol` check on a COPY (Section 9.2, step 8)
    is what actually rejects it. Also confirms RDKit's own logger warning
    from that posterior check is suppressed and later restored."""

    path = _write_raw_text(tmp_path, "pentavalent.sdf", _PENTAVALENT_CARBON_MOLBLOCK)
    policy = _default_policy(sanitize=False, remove_hs=False, strict_parsing=True)

    capfd.readouterr()
    with pytest.raises(InvalidPromotionSDFError):
        inspect_single_sdf_for_promotion(source_path=path, policy=policy)
    captured = capfd.readouterr()
    assert "Explicit valence" not in captured.err

    marker = "__increment018_rdkit_logs_restored__"
    rdBase.LogWarningMsg(marker)
    restored = capfd.readouterr()
    assert marker in restored.err


def test_sanitize_false_valid_molecule_accepted(tmp_path):
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")
    policy = _default_policy(sanitize=False, remove_hs=False, strict_parsing=True)

    result = inspect_single_sdf_for_promotion(source_path=sdf_path, policy=policy)

    assert result.sanitize is False
    assert result.atom_count == 3


# --- 81-83: isotopic hydrogen always preserved regardless of remove_hs ---


def _write_ethanol_d1(tmp_path, filename):
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            atom.SetIsotope(2)
            break
    return _write_mol(tmp_path, filename, mol)


def test_isotopic_hydrogen_preserved_with_remove_hs_false(tmp_path):
    path = _write_ethanol_d1(tmp_path, "ethanol_d1.sdf")
    policy = _default_policy(sanitize=True, remove_hs=False, strict_parsing=True)

    result = inspect_single_sdf_for_promotion(source_path=path, policy=policy)

    assert result.isotopic_hydrogen_count == 1
    assert result.explicit_hydrogen_count == 5


def test_isotopic_hydrogen_preserved_with_remove_hs_true(tmp_path):
    """Item 82-83: even when `remove_hs=True` strips every ordinary
    explicit hydrogen, the single isotopically-labeled one survives -
    RD-018-01's confirmed finding that isotopic hydrogen is always
    preserved regardless of `remove_hs`."""

    path = _write_ethanol_d1(tmp_path, "ethanol_d1.sdf")
    policy = _default_policy(sanitize=True, remove_hs=True, strict_parsing=True)

    result = inspect_single_sdf_for_promotion(source_path=path, policy=policy)

    assert result.isotopic_hydrogen_count == 1
    assert result.explicit_hydrogen_count == 0
    assert result.atom_count == 4


# --- 84: strictParsing A/B/C - no behavioral difference demonstrated,
# but the parameter is still always passed and recorded ---


def test_strict_parsing_value_is_always_recorded_even_with_no_observed_difference(tmp_path):
    """Item 84: RD-018-01's strictParsing A/B/C epistemological rule - no
    behavioral difference was demonstrated across the corpus, and no
    universal equivalence is claimed here either; this only confirms the
    parameter is always explicitly passed and echoed back, for either
    value, on an otherwise-ordinary input."""

    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")

    result_true = inspect_single_sdf_for_promotion(
        source_path=sdf_path, policy=_default_policy(strict_parsing=True)
    )
    result_false = inspect_single_sdf_for_promotion(
        source_path=sdf_path, policy=_default_policy(strict_parsing=False)
    )

    assert result_true.strict_parsing is True
    assert result_false.strict_parsing is False


# --- 85: disconnected structure ---


def test_disconnected_structure_rejected(tmp_path):
    mol = Chem.MolFromSmiles("CCO.CC")
    path = _write_mol(tmp_path, "disconnected.sdf", mol)

    with pytest.raises(DisconnectedStructureError):
        inspect_single_sdf_for_promotion(source_path=path, policy=_default_policy())


# --- 86-88: warnings (controlled vocabulary, never a path) ---


def test_undefined_stereocenter_warning(tmp_path):
    mol = Chem.MolFromSmiles("FC(Cl)(Br)[H]")
    path = _write_mol(tmp_path, "stereo.sdf", mol)

    result = inspect_single_sdf_for_promotion(source_path=path, policy=_default_policy())

    assert "undefined-stereocenter" in result.warnings


def test_name_and_cid_absent_warnings(tmp_path):
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")

    result = inspect_single_sdf_for_promotion(source_path=sdf_path, policy=_default_policy())

    assert "sd-name-property-absent" in result.warnings
    assert "sd-cid-property-absent" in result.warnings


def test_name_and_cid_present_no_absence_warnings(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("PUBCHEM_IUPAC_NAME", "ethanol")
    mol.SetProp("PUBCHEM_COMPOUND_CID", "702")
    path = _write_mol(tmp_path, "named.sdf", mol)

    result = inspect_single_sdf_for_promotion(source_path=path, policy=_default_policy())

    assert "sd-name-property-absent" not in result.warnings
    assert "sd-cid-property-absent" not in result.warnings


def test_warnings_are_a_tuple_of_plain_strings_never_a_path(tmp_path):
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")

    result = inspect_single_sdf_for_promotion(source_path=sdf_path, policy=_default_policy())

    assert isinstance(result.warnings, tuple)
    for warning in result.warnings:
        assert isinstance(warning, str)
        assert str(tmp_path) not in warning


# --- no rdkit object escapes this function's return either ---


def test_no_rdkit_object_escapes_inspect_single_sdf_for_promotion(tmp_path):
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")

    result = inspect_single_sdf_for_promotion(source_path=sdf_path, policy=_default_policy())

    assert not isinstance(result, Chem.rdchem.Mol)
    for value in result.__dict__.values():
        assert not isinstance(value, Chem.rdchem.Mol)


# =============================================================================
# Checkpoint 3 correction - MAJOR-B1 (untyped exceptions during supplier
# iteration), MAJOR-A2/audit (path leakage in InvalidPromotionSDFError
# messages), and MINOR-B1 (`_Name` blank-value edge case).
# =============================================================================


class _ExplodingSupplier:
    """A controlled stand-in for `Chem.SDMolSupplier` whose iteration
    yields some records and then raises - reproducing the RDKit
    C++/Boost.Python mid-iteration failure mode that no real fixture in
    this corpus could trigger deterministically."""

    def __init__(self, records):
        self._records = records

    def __iter__(self):
        for item in self._records:
            if item == "BOOM":
                raise RuntimeError("simulated RDKit iteration failure")
            yield item


def test_supplier_iteration_exception_is_translated_and_chained(tmp_path, monkeypatch):
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")
    real_mol = Chem.MolFromSmiles("CCO")

    def _fake_supplier(*args, **kwargs):
        return _ExplodingSupplier([real_mol, "BOOM"])

    monkeypatch.setattr(rdkit_tautomer_module.Chem, "SDMolSupplier", _fake_supplier)

    with pytest.raises(InvalidPromotionSDFError) as exc_info:
        inspect_single_sdf_for_promotion(source_path=sdf_path, policy=_default_policy())

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "simulated RDKit iteration failure" not in str(exc_info.value)


def test_supplier_iteration_exception_never_accepts_partial_result(tmp_path, monkeypatch):
    """Even though one real, valid molecule was yielded before the
    exception, no `RDKitLigandInspectionResult` is ever produced from a
    supplier whose iteration did not complete - confirmed by the raised
    exception type itself (a returned result would mean this assertion
    never even had a chance to run)."""

    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")
    real_mol = Chem.MolFromSmiles("CCO")

    def _fake_supplier(*args, **kwargs):
        return _ExplodingSupplier([real_mol, "BOOM"])

    monkeypatch.setattr(rdkit_tautomer_module.Chem, "SDMolSupplier", _fake_supplier)

    result = None
    try:
        result = inspect_single_sdf_for_promotion(source_path=sdf_path, policy=_default_policy())
    except InvalidPromotionSDFError:
        pass

    assert result is None


def test_supplier_iteration_never_catches_keyboard_interrupt(tmp_path, monkeypatch):
    class _RaisingOnKeyboardInterrupt:
        def __iter__(self):
            raise KeyboardInterrupt()

    def _fake_supplier(*args, **kwargs):
        return _RaisingOnKeyboardInterrupt()

    monkeypatch.setattr(rdkit_tautomer_module.Chem, "SDMolSupplier", _fake_supplier)
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")

    with pytest.raises(KeyboardInterrupt):
        inspect_single_sdf_for_promotion(source_path=sdf_path, policy=_default_policy())


def test_supplier_construction_failure_message_never_leaks_path(tmp_path):
    missing_path = tmp_path / "does-not-exist.sdf"

    with pytest.raises(InvalidPromotionSDFError) as exc_info:
        inspect_single_sdf_for_promotion(source_path=missing_path, policy=_default_policy())

    assert str(missing_path) not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_posterior_sanitization_failure_message_never_leaks_path(tmp_path):
    path = _write_raw_text(tmp_path, "pentavalent.sdf", _PENTAVALENT_CARBON_MOLBLOCK)
    policy = _default_policy(sanitize=False, remove_hs=False, strict_parsing=True)

    with pytest.raises(InvalidPromotionSDFError) as exc_info:
        inspect_single_sdf_for_promotion(source_path=path, policy=policy)

    assert str(path) not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


# --- MINOR-B1 correction: `_Name` recognized, but a BLANK title line
# (RDKit auto-populates `_Name` for every molecule written via SDWriter,
# even with an empty title) must not be mistaken for a real name ---


def test_title_line_name_recognized_via_underscore_name(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("_Name", "curcumin")
    path = _write_mol(tmp_path, "titled.sdf", mol)

    result = inspect_single_sdf_for_promotion(source_path=path, policy=_default_policy())

    assert "sd-name-property-absent" not in result.warnings


def test_blank_title_line_still_emits_name_absent_warning(tmp_path):
    """RDKit's `SDWriter` always populates `_Name` from the title line,
    even when it is blank - `HasProp("_Name")` alone would therefore
    always be true and could never emit this warning at all. A blank
    value must still count as "no real name"."""

    sdf_path = _write_sdf(tmp_path, "noname.sdf", "CCO")

    result = inspect_single_sdf_for_promotion(source_path=sdf_path, policy=_default_policy())

    assert "sd-name-property-absent" in result.warnings


# =========================================================================
# Increment 020-A, Checkpoint 1: characterize_molecule_rdkit (Sections
# 4/6/7/8/9/10/12 of the frozen spec). Bytes in, neutral DTO out - every
# fixture below is either a real molecule constructed through the public
# RDKit API (never a mocked chemistry result) or a hand-crafted byte
# string exercising one exact grammar rule.
# =========================================================================

_CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
_ANTHRACENE_SMILES = "c1ccc2cc3ccccc3cc2c1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _characterize(data: bytes, *, fmt="smiles", state_id="s1", include_svg=False):
    return characterize_molecule_rdkit(
        data,
        input_format=fmt,
        input_state_id=state_id,
        representation_sha256=_sha256(data),
        include_svg=include_svg,
    )


def _sdf_bytes_from_smiles(smiles: str) -> bytes:
    mol = Chem.MolFromSmiles(smiles)
    return (Chem.MolToMolBlock(mol) + "$$$$\n").encode("ascii")


# --- Section 8: engine resolution, pinned to the exact observed version -


def test_characterization_engine_requires_exact_pinned_version():
    data = b"CCO\n"
    outcome = _characterize(data)
    assert outcome.engine_name == "rdkit"
    assert outcome.engine_version == REQUIRED_ENGINE_VERSION == "2026.03.6"


def test_characterization_engine_unavailable_on_missing_import(monkeypatch):
    monkeypatch.setattr(rdkit_tautomer_module, "_RDKIT_IMPORT_ERROR", ImportError("simulated"))

    with pytest.raises(CharacterizationConfigurationError) as excinfo:
        _characterize(b"CCO\n")
    assert excinfo.value.reason == "engine-unavailable"


def test_characterization_engine_unavailable_on_version_mismatch(monkeypatch):
    monkeypatch.setattr(rdkit_tautomer_module.rdBase, "rdkitVersion", "2099.01.1")

    with pytest.raises(CharacterizationConfigurationError) as excinfo:
        _characterize(b"CCO\n")
    assert excinfo.value.reason == "engine-unavailable"


# --- request-shape validation (defensive, before engine resolution) -----


def test_include_svg_must_be_strict_bool():
    data = b"CCO\n"
    with pytest.raises(CharacterizationInputError) as excinfo:
        characterize_molecule_rdkit(
            data,
            input_format="smiles",
            input_state_id="s1",
            representation_sha256=_sha256(data),
            include_svg=1,
        )
    assert excinfo.value.reason == "invalid-request"


@pytest.mark.parametrize(
    "bad_id", ["", ".", "..", "has space", "has/slash", "has\\backslash", "-leading-dash"]
)
def test_input_state_id_rejected_before_engine_access(bad_id):
    data = b"CCO\n"
    with pytest.raises(CharacterizationInputError) as excinfo:
        characterize_molecule_rdkit(
            data,
            input_format="smiles",
            input_state_id=bad_id,
            representation_sha256=_sha256(data),
        )
    assert excinfo.value.reason == "invalid-request"


def test_input_state_id_is_preserved_verbatim_case_sensitive():
    data = b"CCO\n"
    svg_bytes = _characterize(data, state_id="Caffeine-01_a", include_svg=True).svg
    assert 'data-state-id="Caffeine-01_a"' in svg_bytes


def test_unsupported_format_is_configuration_error():
    data = b"CCO\n"
    with pytest.raises(CharacterizationConfigurationError) as excinfo:
        _characterize(data, fmt="mol2")
    assert excinfo.value.reason == "unsupported-format"


def test_checksum_mismatch_is_storage_error():
    data = b"CCO\n"
    with pytest.raises(CharacterizationStorageError) as excinfo:
        characterize_molecule_rdkit(
            data,
            input_format="smiles",
            input_state_id="s1",
            representation_sha256="a" * 64,
        )
    assert excinfo.value.reason == "checksum-mismatch"
    assert excinfo.value.input_state_id == "s1"


# --- Section 7.1: SDF grammar --------------------------------------------


def test_sdf_real_v2000_record_is_accepted():
    data = _sdf_bytes_from_smiles("CCO")
    outcome = _characterize(data, fmt="sdf")
    assert outcome.values.formula == "C2H6O"
    assert outcome.representation_format_resolved == "sdf"


def test_sdf_two_records_is_rejected():
    data = _sdf_bytes_from_smiles("CCO") + _sdf_bytes_from_smiles("CCO")
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="sdf")
    assert excinfo.value.reason == "invalid-chemical-input"


def test_sdf_valid_plus_invalid_record_is_rejected():
    valid = _sdf_bytes_from_smiles("CCO")
    data = valid + b"garbage not a mol block\n$$$$\n"
    with pytest.raises(CharacterizationExecutionError):
        _characterize(data, fmt="sdf")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.replace(b"$$$$\n", b""),  # no delimiter
        lambda data: data + b"$$$$\n",  # two delimiters
        lambda data: data.replace(b"$$$$\n", b"$$$$ \n"),  # delimiter with trailing space
        lambda data: data + b"\n",  # whitespace after delimiter
    ],
)
def test_sdf_physical_delimiter_grammar_violations_are_rejected(mutate):
    data = mutate(_sdf_bytes_from_smiles("CCO"))
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="sdf")
    assert excinfo.value.reason == "invalid-chemical-input"


def test_sdf_lone_cr_is_rejected():
    data = _sdf_bytes_from_smiles("CCO").replace(b"\n", b"\r", 1)
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="sdf")
    assert excinfo.value.reason == "invalid-chemical-input"


def test_sdf_crlf_is_accepted():
    data = _sdf_bytes_from_smiles("CCO").replace(b"\n", b"\r\n")
    outcome = _characterize(data, fmt="sdf")
    assert outcome.values.formula == "C2H6O"


def test_sdf_dollar_substring_in_property_is_not_a_delimiter():
    base = _sdf_bytes_from_smiles("CCO")
    # insert a property block whose value contains the $$$$ substring but
    # is not an exact physical line on its own.
    hostile = base.replace(
        b"M  END\n", b"M  END\n>  <NOTE>\nvalue has $$$$ inside it\n\n"
    )
    outcome = _characterize(hostile, fmt="sdf")
    assert outcome.values.formula == "C2H6O"


def test_sdf_nul_byte_is_rejected():
    data = _sdf_bytes_from_smiles("CCO")[:10] + b"\x00" + _sdf_bytes_from_smiles("CCO")[10:]
    with pytest.raises(CharacterizationExecutionError):
        _characterize(data, fmt="sdf")


def test_sdf_bom_is_rejected():
    data = b"\xef\xbb\xbf" + _sdf_bytes_from_smiles("CCO")
    with pytest.raises(CharacterizationExecutionError):
        _characterize(data, fmt="sdf")


# --- Section 7.2: SMILES grammar -----------------------------------------


def test_smiles_enumeration_child_state_is_accepted_with_lf():
    data = (_ANTHRACENE_SMILES + "\n").encode("ascii")
    outcome = _characterize(data, fmt="smiles")
    assert outcome.values.formula == "C14H10"


def test_smiles_without_terminator_is_accepted():
    data = b"CCO"
    outcome = _characterize(data, fmt="smiles")
    assert outcome.values.formula == "C2H6O"


def test_smiles_with_single_crlf_is_accepted():
    data = b"CCO\r\n"
    outcome = _characterize(data, fmt="smiles")
    assert outcome.values.formula == "C2H6O"


@pytest.mark.parametrize(
    "data",
    [
        b"CCO\r",  # lone CR
        b"CCO\n\n",  # newline duplicated
        b"CCO\nCCO\n",  # second line
        b"CCO ethanol\n",  # name column
        b"CCO |$a;b;c$|\n",  # CXSMILES
        b"CC O\n",  # internal whitespace
        b" CCO\n",  # leading whitespace
        b"CCO \n",  # trailing whitespace before terminator
        b"\n",  # empty token
        b"",  # empty source
    ],
)
def test_smiles_grammar_violations_are_rejected(data):
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="smiles")
    assert excinfo.value.reason == "invalid-chemical-input"


def test_smiles_non_ascii_is_rejected():
    data = "CCOé\n".encode("utf-8")
    with pytest.raises(CharacterizationExecutionError):
        _characterize(data, fmt="smiles")


def test_smiles_nul_and_del_are_rejected():
    with pytest.raises(CharacterizationExecutionError):
        _characterize(b"CC\x00O\n", fmt="smiles")
    with pytest.raises(CharacterizationExecutionError):
        _characterize(b"CC\x7fO\n", fmt="smiles")


# --- Section 7.3: graph eligibility ---------------------------------------


def test_disconnected_salt_is_rejected():
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(b"[Na+].[Cl-]\n", fmt="smiles")
    assert excinfo.value.reason == "unsupported-structure"


def test_dummy_atom_is_rejected():
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(b"*CC\n", fmt="smiles")
    assert excinfo.value.reason == "unsupported-structure"


def test_sgroup_is_rejected_by_eligibility_check():
    """SGroup/polymeric markup cannot be produced by plain SMILES/SDF
    text in this corpus - this exercises the production eligibility
    function directly against a real RDKit SubstanceGroup object, the
    same object GetMolSubstanceGroups would return from any real file
    that declared one (Section 16's own allowance to test an internal
    decision function directly, mirroring _classify_record_counts
    above)."""

    mol = Chem.RWMol(Chem.MolFromSmiles("CCO"))
    sgroup = Chem.CreateMolSubstanceGroup(mol, "SRU")
    sgroup.SetAtoms([0, 1])

    with pytest.raises(CharacterizationExecutionError) as excinfo:
        rdkit_tautomer_module._validate_graph_eligibility(mol)
    assert excinfo.value.reason == "unsupported-structure"


@pytest.mark.parametrize("group_type_name", ["STEREO_AND", "STEREO_OR"])
def test_enhanced_stereo_groups_are_rejected_by_eligibility_check(group_type_name):
    mol = Chem.RWMol(Chem.MolFromSmiles("C[C@H](N)O"))
    group_type = getattr(Chem.StereoGroupType, group_type_name)
    stereo_group = Chem.CreateStereoGroup(group_type, mol, [1])
    mol.SetStereoGroups([stereo_group])

    with pytest.raises(CharacterizationExecutionError) as excinfo:
        rdkit_tautomer_module._validate_graph_eligibility(mol)
    assert excinfo.value.reason == "unsupported-structure"


def test_absolute_stereo_group_is_not_rejected_by_eligibility_check():
    mol = Chem.RWMol(Chem.MolFromSmiles("C[C@H](N)O"))
    stereo_group = Chem.CreateStereoGroup(Chem.StereoGroupType.STEREO_ABSOLUTE, mol, [1])
    mol.SetStereoGroups([stereo_group])

    rdkit_tautomer_module._validate_graph_eligibility(mol)  # must not raise


# =========================================================================
# Checkpoint 1 corrections, MINOR-2: real SDF V3000 bytes, for SGroup and
# for enhanced stereo AND/OR, through the full public adapter
# (bytes -> real ForwardSDMolSupplier -> real eligibility check) - not a
# monkeypatch of the parser or of _validate_graph_eligibility to fabricate
# the rejection, and not a hand-written V3000 text block. Each fixture is
# built the same way the architect's own research route confirmed
# (synthetic molecule -> Chem.CreateMolSubstanceGroup/CreateStereoGroup ->
# Chem.MolToMolBlock(forceV3000=True) + "$$$$\n") - reproduced here
# independently in this test file's own code, not copied from the
# evidence JSON, and verified byte-for-byte identical to it (same sizes
# and SHA-256) during this correction's own development.
# =========================================================================


def _v3000_sgroup_sdf_bytes() -> bytes:
    mol = Chem.RWMol(Chem.MolFromSmiles("CCO"))
    sgroup = Chem.CreateMolSubstanceGroup(mol, "SUP")
    sgroup.SetAtoms([0])
    sgroup.SetProp("LABEL", "Me")
    return (Chem.MolToMolBlock(mol, forceV3000=True) + "$$$$\n").encode("ascii")


def _v3000_stereo_group_sdf_bytes(group_type) -> bytes:
    mol = Chem.RWMol(Chem.MolFromSmiles("C[C@H](F)Cl"))
    stereo_group = Chem.CreateStereoGroup(group_type, mol, [1])
    mol.SetStereoGroups([stereo_group])
    return (Chem.MolToMolBlock(mol, forceV3000=True) + "$$$$\n").encode("ascii")


def test_v3000_sgroup_fixture_bytes_match_the_research_evidence_hash():
    data = _v3000_sgroup_sdf_bytes()
    assert len(data) == 428
    assert _sha256(data) == "1ec45e31c09fc6b6ed6e0b248e839e49b4c43f897e0aca9ab2ac5d6d1c2a5671"


def test_v3000_and_fixture_bytes_match_the_research_evidence_hash():
    data = _v3000_stereo_group_sdf_bytes(Chem.StereoGroupType.STEREO_AND)
    assert len(data) == 496
    assert _sha256(data) == "74040ab45b9889d7aa6ab548ed85e020b261d49891ab715b47a24a7a59a8ce16"


def test_v3000_or_fixture_bytes_match_the_research_evidence_hash():
    data = _v3000_stereo_group_sdf_bytes(Chem.StereoGroupType.STEREO_OR)
    assert len(data) == 496
    assert _sha256(data) == "8a5f0741c4f63516c4f4b787500113a8321d0e7374de15eb332cb0ab69bfabe0"


def test_v3000_sgroup_bytes_parse_with_the_group_retained_before_adapter_rejection():
    """Guards against confusing an absent group or a syntax error with
    the actual science policy (reconciliation's own instruction): parse
    the real bytes independently of the adapter first, and confirm the
    real ForwardSDMolSupplier retained exactly one SGroup on the parsed
    molecule, before ever calling the public adapter."""

    data = _v3000_sgroup_sdf_bytes()
    supplier = Chem.ForwardSDMolSupplier(
        BytesIO(data), sanitize=True, removeHs=False, strictParsing=True
    )
    items = list(supplier)
    assert len(items) == 1
    assert items[0] is not None
    assert len(Chem.GetMolSubstanceGroups(items[0])) == 1
    assert len(items[0].GetStereoGroups()) == 0


@pytest.mark.parametrize(
    "group_type,group_type_name",
    [
        (Chem.StereoGroupType.STEREO_AND, "STEREO_AND"),
        (Chem.StereoGroupType.STEREO_OR, "STEREO_OR"),
    ],
)
def test_v3000_stereo_group_bytes_parse_with_the_group_retained_before_adapter_rejection(
    group_type, group_type_name
):
    data = _v3000_stereo_group_sdf_bytes(group_type)
    supplier = Chem.ForwardSDMolSupplier(
        BytesIO(data), sanitize=True, removeHs=False, strictParsing=True
    )
    items = list(supplier)
    assert len(items) == 1
    assert items[0] is not None
    assert len(Chem.GetMolSubstanceGroups(items[0])) == 0
    retained_types = [group.GetGroupType() for group in items[0].GetStereoGroups()]
    assert retained_types == [group_type]
    assert retained_types[0].name == group_type_name


def test_v3000_sgroup_bytes_are_rejected_by_the_full_public_adapter_as_unsupported_structure():
    data = _v3000_sgroup_sdf_bytes()
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="sdf")
    assert excinfo.value.reason == "unsupported-structure"


@pytest.mark.parametrize(
    "group_type",
    [Chem.StereoGroupType.STEREO_AND, Chem.StereoGroupType.STEREO_OR],
    ids=["AND", "OR"],
)
def test_v3000_enhanced_stereo_bytes_are_rejected_by_the_full_public_adapter_as_unsupported_structure(
    group_type,
):
    data = _v3000_stereo_group_sdf_bytes(group_type)
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="sdf")
    assert excinfo.value.reason == "unsupported-structure"


def test_v3000_absolute_stereo_bytes_are_not_rejected_by_the_full_public_adapter():
    """Control case for the MINOR-2 integration fixtures: the same
    construction route, but with STEREO_ABSOLUTE (not an enhanced AND/OR
    group), must characterize normally through the full adapter - proving
    the rejection above is specific to the AND/OR policy, not an
    incidental V3000/CreateStereoGroup artifact."""

    data = _v3000_stereo_group_sdf_bytes(Chem.StereoGroupType.STEREO_ABSOLUTE)
    outcome = _characterize(data, fmt="sdf")
    assert outcome.values.formula == "C2H4ClF"


def test_empty_molecule_is_rejected():
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(b"\x20", fmt="smiles")  # single space - no token at all
    # the grammar check rejects this before eligibility is ever reached
    assert excinfo.value.reason == "invalid-chemical-input"


# --- Section 6: closed, inclusive size limits -----------------------------


def test_atom_count_over_limit_is_rejected():
    data = ("C" * (MAX_TOTAL_ATOM_COUNT + 1) + "\n").encode("ascii")
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="smiles")
    assert excinfo.value.reason == "molecular-size-limit"


def test_atom_count_at_exact_limit_is_accepted():
    data = ("C" * MAX_TOTAL_ATOM_COUNT + "\n").encode("ascii")
    outcome = _characterize(data, fmt="smiles")
    assert outcome.values.heavy_atom_count == MAX_TOTAL_ATOM_COUNT


# --- Section 8: the ten values, against the frozen reference vector ------


def test_caffeine_reference_vector():
    data = (_CAFFEINE_SMILES + "\n").encode("ascii")
    outcome = _characterize(data, fmt="smiles")
    values = outcome.values

    assert values.formula == "C8H10N4O2"
    assert values.formal_charge_e == 0
    assert values.heavy_atom_count == 14
    assert values.hbd_count == 0
    assert values.hba_count == 3
    assert values.rotatable_bond_count == 0
    assert math.isclose(values.molar_mass_g_mol, 194.194, abs_tol=1e-8)
    assert math.isclose(values.exact_mass_da, 194.08037556, abs_tol=1e-8)
    assert math.isclose(values.tpsa_a2, 61.82, abs_tol=1e-8)
    assert math.isclose(values.clogp_wildman_crippen, -1.0293, abs_tol=1e-8)


@pytest.mark.parametrize(
    "smiles,expected_formula,expected_charge",
    [
        ("CC(=O)O", "C2H4O2", 0),  # acetic acid
        ("OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "C14H11Cl2NO2", 0),  # diclofenac
        ("CCNc1nc(Cl)nc(NC(C)C)n1", "C8H14ClN5", 0),  # atrazine
        ("CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1", "C10H14NO5PS", 0),  # parathion-like structure
    ],
)
def test_real_production_fixtures_characterize_without_tautomer_enumeration(
    smiles, expected_formula, expected_charge
):
    data = (smiles + "\n").encode("ascii")
    outcome = _characterize(data, fmt="smiles")
    assert outcome.values.formula == expected_formula
    assert outcome.values.formal_charge_e == expected_charge
    # characterization never enumerates tautomers - no such field exists
    # on the outcome at all.
    assert not hasattr(outcome, "candidates")


def test_isotope_and_charge_preservation():
    outcome_13c = _characterize(b"[13CH4]\n", fmt="smiles")
    assert outcome_13c.values.formula == "[13C]H4"

    outcome_ammonium = _characterize(b"[NH4+]\n", fmt="smiles")
    assert outcome_ammonium.values.formal_charge_e == 1

    outcome_acetate = _characterize(b"CC(=O)[O-]\n", fmt="smiles")
    assert outcome_acetate.values.formal_charge_e == -1


def test_hbd_hba_use_the_pinned_api_not_nhoh_nocount():
    # methylamine=1, ammonia=1, diamine=2 (Section 16, T-020A-045)
    assert _characterize(b"CN\n", fmt="smiles").values.hbd_count == 1
    assert _characterize(b"N\n", fmt="smiles").values.hbd_count == 1
    assert _characterize(b"NCCN\n", fmt="smiles").values.hbd_count == 2


def test_rotatable_bonds_strict_discriminant_against_nonstrict():
    data = b"CC(=O)NC\n"  # N-methylacetamide
    outcome = _characterize(data, fmt="smiles")
    mol = Chem.MolFromSmiles("CC(=O)NC")
    strict = rdMolDescriptors.CalcNumRotatableBonds(
        mol, strict=rdMolDescriptors.NumRotatableBondsOptions.Strict
    )
    nonstrict = rdMolDescriptors.CalcNumRotatableBonds(
        mol, strict=rdMolDescriptors.NumRotatableBondsOptions.NonStrict
    )
    assert strict != nonstrict
    assert outcome.values.rotatable_bond_count == strict


def test_tpsa_sp_discriminant_against_include_s_and_p():
    data = b"CS(=O)(=O)C\n"  # dimethyl sulfone
    outcome = _characterize(data, fmt="smiles")
    mol = Chem.MolFromSmiles("CS(=O)(=O)C")
    no_sp = rdMolDescriptors.CalcTPSA(mol, force=True, includeSandP=False)
    with_sp = rdMolDescriptors.CalcTPSA(mol, force=True, includeSandP=True)
    assert no_sp != with_sp
    assert math.isclose(outcome.values.tpsa_a2, no_sp, abs_tol=1e-8)


def test_descriptor_failure_is_execution_error(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated descriptor failure")

    monkeypatch.setattr(rdkit_tautomer_module.rdMolDescriptors, "CalcTPSA", _raise)

    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(b"CCO\n", fmt="smiles")
    assert excinfo.value.reason == "descriptor-failure"
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_descriptor_returning_non_finite_is_rejected(monkeypatch):
    monkeypatch.setattr(
        rdkit_tautomer_module.rdMolDescriptors, "CalcTPSA", lambda *a, **k: float("nan")
    )

    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(b"CCO\n", fmt="smiles")
    assert excinfo.value.reason == "descriptor-failure"


def test_stereo_centers_are_preserved_not_enumerated():
    defined = _characterize(b"C[C@H](N)O\n", fmt="smiles")
    undefined = _characterize(b"CC(N)O\n", fmt="smiles")
    assert defined.values.formula == undefined.values.formula == "C2H7NO"
    # characterization never produces more than one candidate structure -
    # this is a single-molecule query, not an enumeration.


def test_representation_bytes_are_never_mutated():
    data = bytearray((_CAFFEINE_SMILES + "\n").encode("ascii"))
    original = bytes(data)
    _characterize(bytes(data), fmt="smiles")
    assert bytes(data) == original


# --- Sections 9/10/10.2: optional 2D depiction and SVG validator --------


@pytest.mark.parametrize(
    "smiles",
    [
        _CAFFEINE_SMILES,
        _ANTHRACENE_SMILES,  # T-020A-051's required fused-polycyclic canary
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "C[C@H](N)C(=O)O",  # alanine - a defined stereocenter drawn without
        # a filled wedge triangle in this pinned engine/font/geometry
        # combination (confirmed empirically - see the dedicated gap test
        # below for the case that DOES need one)
        "[13CH3][NH3+]",  # isotope + charge probe
    ],
)
def test_depiction_produces_a_validated_svg_for_each_probe(smiles):
    data = (smiles + "\n").encode("ascii")
    outcome = _characterize(data, fmt="smiles", include_svg=True)

    assert outcome.svg is not None
    assert outcome.drawing is not None
    assert outcome.drawing.svg_sha256 == hashlib.sha256(
        outcome.svg.encode("utf-8")
    ).hexdigest()
    assert outcome.drawing.svg_size_bytes == len(outcome.svg.encode("utf-8"))
    assert 'data-profile="molecular-depiction-v1"' in outcome.svg
    assert "generated-2d-not-conformer" in outcome.warnings


def test_depiction_preserves_explicit_hydrogens_and_isotopes():
    data = b"[2H]C([2H])([2H])CO\n"
    outcome = _characterize(data, fmt="smiles", include_svg=True)
    assert outcome.values.formula == "C2H3[2H]3O"
    assert outcome.svg is not None


def test_filled_wedge_triangle_z_command_is_an_unresolved_profile_gap():
    """CHECKPOINT 1 FINDING (for the report, not silently patched): the
    frozen spec's closed `d` command grammar (Section 10.1) is M/L/Q
    only, confronted against five bundled example drawings plus the
    mandatory antracene canary (Section 16, T-020A-051) - none of which
    happens to need a FILLED stereo wedge triangle. A real chiral center
    whose wedge bond IS drawn as a filled triangle (confirmed on this
    pinned rdkit==2026.3.6 Windows build) produces a `d` attribute
    ending in an absolute `Z` (closepath) command to close that
    triangle - a command this checkpoint's grammar does not cover.

    Per Section 10.1's own instruction ("saída que exija comando não
    coberto falha explicitamente, sem ampliar o perfil silenciosamente
    ... Qualquer incompatibilidade deve ser relatada"), this is NOT
    silently patched into the grammar here - the adapter correctly
    raises depiction-failure, and this gap is reported upward for
    explicit review rather than resolved unilaterally in this
    checkpoint."""

    data = b"C[C@@H](C(=O)O)Nc1ccc(Cl)cc1\n"
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="smiles", include_svg=True)
    assert excinfo.value.reason == "depiction-failure"


def test_force_rdkit_remains_effective_regardless_of_global_coordgen_preference():
    from rdkit.Chem import rdDepictor as rddepictor_module

    original = rddepictor_module.SetPreferCoordGen
    rddepictor_module.SetPreferCoordGen(True)
    try:
        data = (_CAFFEINE_SMILES + "\n").encode("ascii")
        outcome = _characterize(data, fmt="smiles", include_svg=True)
        assert outcome.svg is not None
    finally:
        rddepictor_module.SetPreferCoordGen(False)


def test_font_evidence_is_real_and_matches_the_documented_hash():
    data = b"CCO\n"
    outcome = _characterize(data, fmt="smiles", include_svg=True)
    assert outcome.drawing.font_name == "Telex-Regular.ttf"
    # Section 9's Linux-observed hash, now independently confirmed real
    # and byte-identical on this Windows production environment's wheel.
    assert (
        outcome.drawing.font_sha256_observed
        == "eeaa2d17d105b6b46e5368ecd990f5b19c50131ff922dbf79bfb9bb45c249871"
    )


def test_hostile_annotation_properties_are_stripped_from_the_svg():
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("_Name", "<script>evil</script>")
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", "<hostile markup>")
    data = (Chem.MolToMolBlock(mol) + "$$$$\n").encode("ascii")

    outcome = _characterize(data, fmt="sdf", include_svg=True)

    assert "<script>" not in outcome.svg
    assert "<hostile markup>" not in outcome.svg
    assert "evil" not in outcome.svg


def test_drawer_is_never_invoked_when_svg_not_requested(monkeypatch):
    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("drawing must not be attempted when include_svg=False")

    monkeypatch.setattr(rdkit_tautomer_module, "_build_drawing_result", _fail_if_called)

    data = b"CCO\n"
    outcome = _characterize(data, fmt="smiles", include_svg=False)
    assert outcome.svg is None
    assert outcome.drawing is None
    assert outcome.values.formula == "C2H6O"


def test_drawing_unavailable_does_not_affect_calculation_only_requests(monkeypatch):
    monkeypatch.setattr(
        rdkit_tautomer_module, "_RDKIT_DRAWING_IMPORT_ERROR", ImportError("simulated")
    )

    data = b"CCO\n"
    outcome = _characterize(data, fmt="smiles", include_svg=False)
    assert outcome.values.formula == "C2H6O"

    with pytest.raises(CharacterizationConfigurationError) as excinfo:
        _characterize(data, fmt="smiles", include_svg=True)
    assert excinfo.value.reason == "drawing-unavailable"


def test_depiction_failure_never_returns_a_partial_outcome(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated drawing failure")

    monkeypatch.setattr(rdkit_tautomer_module.rdMolDraw2D, "MolDraw2DSVG", _raise)

    data = b"CCO\n"
    with pytest.raises(CharacterizationExecutionError) as excinfo:
        _characterize(data, fmt="smiles", include_svg=True)
    assert excinfo.value.reason == "depiction-failure"


def test_svg_binding_links_state_id_hash_and_engine_to_the_entry():
    data = (_CAFFEINE_SMILES + "\n").encode("ascii")
    outcome = characterize_molecule_rdkit(
        data,
        input_format="smiles",
        input_state_id="caffeine-42",
        representation_sha256=_sha256(data),
        include_svg=True,
    )
    assert 'data-state-id="caffeine-42"' in outcome.svg
    assert f'data-representation-sha256="{_sha256(data)}"' in outcome.svg
    assert f'data-engine-version="{outcome.engine_version}"' in outcome.svg


# --- Section 13: RDKit native log suppression and restoration -----------


def test_native_rdkit_logs_are_suppressed_on_failure_and_restored_with_marker(capfd):
    """Checkpoint 1 corrections, MINOR-1: a second successful call is only
    indirect evidence of restoration (it would still succeed even if logs
    stayed silenced). This follows the project's own already-established
    direct-proof pattern instead (Increment 017's
    test_enumeration_logs_are_suppressed_and_restored_after_incomplete_status):
    emit rdBase.LogWarningMsg with a unique marker AFTER the operation and
    assert the marker itself reaches stderr through capfd - proof the
    BlockLogs suppression was actually lifted, not merely that a later
    unrelated call happened to work. No EnableLog call is made to force
    this; restoration relies entirely on BlockLogs's own __exit__."""

    garbage_sdf = b"garbage not a mol block\n$$$$\n"

    capfd.readouterr()
    with pytest.raises(CharacterizationExecutionError):
        _characterize(garbage_sdf, fmt="sdf")
    captured = capfd.readouterr()
    assert "ERROR" not in captured.err

    marker = "__020a_cp1_failure_logs_restored__"
    rdBase.LogWarningMsg(marker)
    restored = capfd.readouterr()
    assert marker in restored.err


def test_native_rdkit_logs_are_suppressed_on_success_and_restored_with_marker(capfd):
    capfd.readouterr()
    outcome = _characterize(b"CCO\n", fmt="smiles")
    captured = capfd.readouterr()
    assert outcome.values.formula == "C2H6O"
    assert "ERROR" not in captured.err

    marker = "__020a_cp1_success_logs_restored__"
    rdBase.LogWarningMsg(marker)
    restored = capfd.readouterr()
    assert marker in restored.err


# =========================================================================
# Checkpoint 1 corrections, MAJOR-2 + MINOR-1: KeyboardInterrupt/SystemExit
# injected at the point of use of a real production call, by adapter
# phase, plus (for every phase that runs inside `rdBase.BlockLogs()`) a
# direct LogWarningMsg-marker proof that log suppression was lifted after
# the interruption - not merely that logs were suppressed during it.
#
# Phases mapped (reconciliation's own list): entry/request validation;
# SDF parsing including iteration; SMILES parsing; value computation;
# drawing coordinates+preparation; drawing rendering; SVG validation/
# binding; result construction. CLI/service/reader phases (Checkpoints
# 2/3) are out of scope and not included.
#
# Each injection patches exactly one real production call at its point of
# use (never the whole adapter) - the rest of the production flow (real
# grammar validation, real RDKit calls on either side of the patched one,
# real BlockLogs scopes) executes exactly as in normal operation.
# =========================================================================


class _RaisingOnIteration:
    """Mirrors the project's own established pattern (see
    test_supplier_iteration_never_catches_keyboard_interrupt above) for
    the ForwardSDMolSupplier this checkpoint's adapter uses, generalized
    to carry any BaseException instance."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __iter__(self):
        raise self._exc


def _patch_entry_validation(monkeypatch, exc):
    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(rdkit_tautomer_module, "validate_input_id", _raise)
    return dict(data=b"CCO\n", fmt="smiles", include_svg=False, uses_blocklogs=False)


def _patch_sdf_parsing_iteration(monkeypatch, exc):
    def _fake_supplier(*_args, **_kwargs):
        return _RaisingOnIteration(exc)

    monkeypatch.setattr(rdkit_tautomer_module.Chem, "ForwardSDMolSupplier", _fake_supplier)
    return dict(
        data=_sdf_bytes_from_smiles("CCO"), fmt="sdf", include_svg=False, uses_blocklogs=True
    )


def _patch_smiles_parsing(monkeypatch, exc):
    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(rdkit_tautomer_module.Chem, "MolFromSmiles", _raise)
    return dict(data=b"CCO\n", fmt="smiles", include_svg=False, uses_blocklogs=True)


def _patch_value_computation(monkeypatch, exc):
    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(rdkit_tautomer_module.rdMolDescriptors, "CalcTPSA", _raise)
    return dict(data=b"CCO\n", fmt="smiles", include_svg=False, uses_blocklogs=True)


def _patch_drawing_coordinates_and_preparation(monkeypatch, exc):
    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(rdkit_tautomer_module.rdDepictor, "Compute2DCoords", _raise)
    return dict(data=b"CCO\n", fmt="smiles", include_svg=True, uses_blocklogs=True)


def _patch_drawing_rendering(monkeypatch, exc):
    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(rdkit_tautomer_module.rdMolDraw2D, "MolDraw2DSVG", _raise)
    return dict(data=b"CCO\n", fmt="smiles", include_svg=True, uses_blocklogs=True)


def _patch_svg_validation_and_binding(monkeypatch, exc):
    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(rdkit_tautomer_module, "validate_and_bind_depiction_svg", _raise)
    return dict(data=b"CCO\n", fmt="smiles", include_svg=True, uses_blocklogs=False)


def _patch_result_construction(monkeypatch, exc):
    def _raise(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(rdkit_tautomer_module, "RDKitCharacterizationOutcome", _raise)
    return dict(data=b"CCO\n", fmt="smiles", include_svg=False, uses_blocklogs=False)


_INTERRUPTION_PHASES = [
    ("entry-validation", _patch_entry_validation),
    ("sdf-parsing-iteration", _patch_sdf_parsing_iteration),
    ("smiles-parsing", _patch_smiles_parsing),
    ("value-computation", _patch_value_computation),
    ("drawing-coordinates-and-preparation", _patch_drawing_coordinates_and_preparation),
    ("drawing-rendering", _patch_drawing_rendering),
    ("svg-validation-and-binding", _patch_svg_validation_and_binding),
    ("result-construction", _patch_result_construction),
]


def _run_interruption_case(monkeypatch, capfd, apply_patch, exc):
    setup = apply_patch(monkeypatch, exc)
    data = setup["data"]

    capfd.readouterr()
    with pytest.raises(type(exc)) as excinfo:
        characterize_molecule_rdkit(
            data,
            input_format=setup["fmt"],
            input_state_id="s1",
            representation_sha256=_sha256(data),
            include_svg=setup["include_svg"],
        )
    captured = capfd.readouterr()

    assert excinfo.value is exc  # same instance, never wrapped/converted
    assert "ERROR" not in captured.err

    if setup["uses_blocklogs"]:
        marker = f"__020a_cp1_{type(exc).__name__}_{id(exc)}_logs_restored__"
        rdBase.LogWarningMsg(marker)
        restored = capfd.readouterr()
        assert marker in restored.err


@pytest.mark.parametrize(
    "phase_name,apply_patch", _INTERRUPTION_PHASES, ids=[p[0] for p in _INTERRUPTION_PHASES]
)
def test_keyboard_interrupt_propagates_unconverted_by_adapter_phase(
    phase_name, apply_patch, monkeypatch, capfd
):
    exc = KeyboardInterrupt()
    _run_interruption_case(monkeypatch, capfd, apply_patch, exc)


@pytest.mark.parametrize(
    "phase_name,apply_patch", _INTERRUPTION_PHASES, ids=[p[0] for p in _INTERRUPTION_PHASES]
)
def test_system_exit_propagates_unconverted_by_adapter_phase(
    phase_name, apply_patch, monkeypatch, capfd
):
    exc = SystemExit(42)
    _run_interruption_case(monkeypatch, capfd, apply_patch, exc)
    assert exc.code == 42


def test_adapter_never_raises_except_basexception_itself():
    """Static corroboration (not a substitute for the dynamic injection
    tests above): the adapter module contains no `except BaseException`
    clause anywhere, which is what structurally guarantees
    KeyboardInterrupt/SystemExit are never caught/converted by any
    handler added in this checkpoint."""

    import ast
    import pathlib

    source = pathlib.Path(rdkit_tautomer_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            names = (
                [alias.id for alias in node.type.elts]
                if isinstance(node.type, ast.Tuple)
                else [node.type.id if isinstance(node.type, ast.Name) else None]
            )
            assert "BaseException" not in names


# --- Section 15: legacy enumeration/imports are unaffected ---------------


def test_characterization_imports_do_not_shadow_legacy_rdkit_import_error():
    assert rdkit_tautomer_module._RDKIT_IMPORT_ERROR is None
    assert rdkit_tautomer_module._RDKIT_DRAWING_IMPORT_ERROR is None


def test_enumeration_still_works_after_characterization_additions(tmp_path):
    sdf_path = _write_sdf(tmp_path, "ethanol.sdf", "CCO")
    result = enumerate_tautomers_rdkit(source_path=sdf_path, source_format="sdf")
    assert result.enumerated_unique_count >= 1
