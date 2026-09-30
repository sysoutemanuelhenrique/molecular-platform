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

import molecular.rdkit_tautomer as rdkit_tautomer_module
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
