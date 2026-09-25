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
from molecular.rdkit_tautomer import (
    IncompleteEnumerationError,
    InvalidSDFError,
    MultipleMoleculesError,
    RDKitEngineResolution,
    RDKitTautomerCandidate,
    RDKitTautomerEnumerationResult,
    RDKitUnavailableError,
    enumerate_tautomers_rdkit,
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
