"""Tests for receptor_inspection.py - Increment 020, Checkpoint 1.

Covers T-013-042 (preserved/rewritten) and T-106-122 (new) from the frozen
final specification's normative matrix (Section 35.2). Each test is
labeled with its exact scenario id. No network access; every mmCIF/PDB
fixture is a minimal literal written under `tmp_path`, or - for T-121,
which exercises a Gemmi capability this project's pinned Gemmi 0.7.5 does
not structurally expose for legacy PDB (Section 12.4) - a directly
constructed `ParsedReceptorDocument`.
"""

import unicodedata
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from molecular.atomic import compute_sha256
from molecular.receptor_inspection import (
    NonPolymerComponent,
    StructConnEndpoint,
    build_receptor_inspection_report,
    compute_inspection_id,
)
from molecular.receptor_parsing import (
    ParsedAtom,
    ParsedMissingAtom,
    ParsedMissingResidue,
    ParsedReceptorDocument,
    ParsedResidue,
    StructConnMalformedError,
    parse_receptor_raw,
)

# --- fixture helpers ------------------------------------------------------

_ATOM_SITE_HEADER = """\
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _build_report(tmp_path: Path, name: str, cif_text: str, *, raw_format_declared="mmcif"):
    path = _write(tmp_path, name, cif_text)
    document = parse_receptor_raw(path, expected_checksum=compute_sha256(path))
    return build_receptor_inspection_report(
        document, protein_input_id="demo-protein", raw_format_declared=raw_format_declared
    )


def _build_report_pdb(tmp_path: Path, name: str, pdb_text: str):
    path = _write(tmp_path, name, pdb_text)
    document = parse_receptor_raw(path, expected_checksum=compute_sha256(path))
    return build_receptor_inspection_report(
        document, protein_input_id="demo-protein", raw_format_declared="pdb"
    )


def _entity_block(entries=(("1", "polymer", "Test protein"),)) -> str:
    lines = ["loop_", "_entity.id", "_entity.type", "_entity.pdbx_description"]
    lines += [f"{eid} '{etype}' '{desc}'" for eid, etype, desc in entries]
    lines.append("#")
    return "\n".join(lines) + "\n"


def _minimal_cif(*, atom_rows: str, entity_block: str = None, extra: str = "") -> str:
    entity_block = entity_block if entity_block is not None else _entity_block()
    return f"""\
data_TEST
#
{entity_block}\
{_ATOM_SITE_HEADER}\
{atom_rows}
#
{extra}
"""


# PDB fixed-width helpers (columns per the wwPDB format spec)


def _pdb_atom(record, serial, name, resname, chain, resseq, occ=1.00, elem=None, icode=" "):
    elem = elem or name.strip()[0]
    name_field = (" " + name).ljust(4) if len(name) < 4 else name
    return (
        f"{record:<6}{serial:>5} {name_field:<4} {resname:<3} {chain:<1}{resseq:>4}{icode:<1}   "
        f"{1.0:>8.3f}{1.0:>8.3f}{1.0:>8.3f}{occ:>6.2f}{10.0:>6.2f}          {elem:>2}"
    )


def _pdb_ssbond(chain1, resseq1, chain2, resseq2):
    return f"SSBOND   1 CYS {chain1:<1} {resseq1:>4}    CYS {chain2:<1} {resseq2:>4}"


def _pdb_link(name1, resname1, chain1, resseq1, name2, resname2, chain2, resseq2, length=None):
    line = [" "] * 80

    def put(s, start_1idx):
        for i, ch in enumerate(s):
            line[start_1idx - 1 + i] = ch

    put("LINK", 1)
    put((" " + name1).ljust(4) if len(name1) < 4 else name1, 13)
    put(resname1.rjust(3), 18)
    put(chain1, 22)
    put(str(resseq1).rjust(4), 23)
    put((" " + name2).ljust(4) if len(name2) < 4 else name2, 43)
    put(resname2.rjust(3), 48)
    put(chain2, 52)
    put(str(resseq2).rjust(4), 53)
    if length is not None:
        put(f"{length:5.2f}", 74)
    return "".join(line).rstrip()


def _pdb_conect(serial, *bonded):
    text = f"CONECT{serial:>5}"
    for b in bonded:
        text += f"{b:>5}"
    return text


# --- T-013-042 --------------------------------------------------------------


def test_multiple_models_warning(tmp_path):  # T-013
    atom_rows = (
        "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1\n"
        "ATOM 2 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 2"
    )
    report = _build_report(tmp_path, "multi_model.cif", _minimal_cif(atom_rows=atom_rows))

    assert any(w.code == "multiple-models" for w in report.warnings)


def test_multiple_assemblies_warning(tmp_path):  # T-014
    atom_rows = "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    extra = """\
loop_
_pdbx_struct_assembly.id
_pdbx_struct_assembly.details
1 'assembly one'
2 'assembly two'
#
"""
    report = _build_report(tmp_path, "multi_assembly.cif", _minimal_cif(atom_rows=atom_rows, extra=extra))

    assert any(w.code == "multiple-assemblies" for w in report.warnings)


def test_altloc_observation_and_warning(tmp_path):  # T-015
    atom_rows = (
        "ATOM 1 N N A ALA A 1 1 ? 1.0 1.0 1.0 0.60 10.0 1 A 1\n"
        "ATOM 2 N N B ALA A 1 1 ? 1.1 1.0 1.0 0.40 10.0 1 A 1"
    )
    report = _build_report(tmp_path, "altloc.cif", _minimal_cif(atom_rows=atom_rows))

    assert len(report.altloc_observations) == 1
    assert report.altloc_observations[0].altloc_ids == ("A", "B")
    assert any(w.code == "alternate-locations-present" for w in report.warnings)


def test_partial_occupancy_observation_and_warning(tmp_path):  # T-016
    atom_rows = "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 0.50 10.0 1 A 1"
    report = _build_report(tmp_path, "partial_occ.cif", _minimal_cif(atom_rows=atom_rows))

    assert len(report.occupancy_observations) == 1
    assert report.occupancy_observations[0].occupancy == Decimal("0.50")
    assert any(w.code == "partial-occupancy-present" for w in report.warnings)


def test_full_occupancy_no_observation_no_warning(tmp_path):  # T-017
    atom_rows = (
        "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1\n"
        "ATOM 2 C CA . ALA A 1 1 ? 2.0 1.0 1.0 1.00 10.0 1 A 1"
    )
    report = _build_report(tmp_path, "full_occ.cif", _minimal_cif(atom_rows=atom_rows))

    assert report.occupancy_observations == ()
    assert not any(w.code == "partial-occupancy-present" for w in report.warnings)


def test_missing_residues_observation_and_warning(tmp_path):  # T-018
    atom_rows = "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    extra = """\
loop_
_pdbx_unobs_or_zero_occ_residues.id
_pdbx_unobs_or_zero_occ_residues.PDB_model_num
_pdbx_unobs_or_zero_occ_residues.label_asym_id
_pdbx_unobs_or_zero_occ_residues.auth_asym_id
_pdbx_unobs_or_zero_occ_residues.auth_seq_id
_pdbx_unobs_or_zero_occ_residues.auth_comp_id
1 1 A A 2 GLY
#
"""
    report = _build_report(tmp_path, "missing_res.cif", _minimal_cif(atom_rows=atom_rows, extra=extra))

    assert len(report.missing_residues) == 1
    assert report.missing_residues[0].residue_name == "GLY"
    assert any(w.code == "missing-residues-reported" for w in report.warnings)


def test_missing_atoms_observation_and_warning(tmp_path):  # T-019
    atom_rows = "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    extra = """\
loop_
_pdbx_unobs_or_zero_occ_atoms.id
_pdbx_unobs_or_zero_occ_atoms.PDB_model_num
_pdbx_unobs_or_zero_occ_atoms.label_asym_id
_pdbx_unobs_or_zero_occ_atoms.auth_asym_id
_pdbx_unobs_or_zero_occ_atoms.auth_seq_id
_pdbx_unobs_or_zero_occ_atoms.auth_comp_id
_pdbx_unobs_or_zero_occ_atoms.auth_atom_id
1 1 A A 1 ALA CB
#
"""
    report = _build_report(tmp_path, "missing_atom.cif", _minimal_cif(atom_rows=atom_rows, extra=extra))

    assert len(report.missing_atoms) == 1
    assert report.missing_atoms[0].atom_name == "CB"
    assert any(w.code == "missing-atoms-reported" for w in report.warnings)


def test_ambiguous_author_chain_id_warning(tmp_path):  # T-020
    atom_rows = (
        "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1\n"
        "HETATM 2 O O . HOH B 2 . ? 2.0 1.0 1.0 1.00 10.0 101 A 1"
    )
    entity_block = _entity_block((("1", "polymer", "protein"), ("2", "water", "water")))
    report = _build_report(
        tmp_path, "ambiguous.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block)
    )

    assert any(w.code == "ambiguous-author-chain-id" for w in report.warnings)


def test_residue_and_atom_counts_match_raw_file(tmp_path):  # T-021
    atom_rows = (
        "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1\n"
        "ATOM 2 C CA . ALA A 1 1 ? 2.0 1.0 1.0 1.00 10.0 1 A 1\n"
        "ATOM 3 N N . GLY A 1 2 ? 3.0 1.0 1.0 1.00 10.0 2 A 1"
    )
    report = _build_report(tmp_path, "counts.cif", _minimal_cif(atom_rows=atom_rows))

    chain = report.chains[0]
    assert chain.residue_count == 2
    assert chain.atom_count == 3


def test_shared_label_entity_id_across_chains_no_duplicate_entity(tmp_path):  # T-022
    atom_rows = (
        "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1\n"
        "ATOM 2 N N . ALA B 1 1 ? 1.0 2.0 1.0 1.00 10.0 1 B 1"
    )
    report = _build_report(tmp_path, "shared_entity.cif", _minimal_cif(atom_rows=atom_rows))

    assert len(report.entities) == 1
    assert {chain.label_entity_id for chain in report.chains} == {"1"}


def test_legacy_pdb_unavailable_categories_but_legacy_connectivity_preserved(tmp_path):  # T-023
    content = (
        _pdb_ssbond("A", 7, "A", 20) + "\n"
        + _pdb_atom("ATOM", 1, "N", "CYS", "A", 7) + "\n"
        + _pdb_atom("ATOM", 2, "SG", "CYS", "A", 7) + "\n"
        + _pdb_atom("ATOM", 3, "N", "CYS", "A", 20) + "\n"
        + _pdb_atom("ATOM", 4, "SG", "CYS", "A", 20) + "\n"
        + "END\n"
    )
    report = _build_report_pdb(tmp_path, "legacy_conn.pdb", content)

    for category in (
        "pdbx_struct_assembly",
        "struct_conn",
        "pdbx_unobs_or_zero_occ_residues",
        "pdbx_unobs_or_zero_occ_atoms",
    ):
        assert category in report.unavailable_categories
    assert len(report.disulfide_bonds) == 1
    assert any(w.code == "legacy-pdb-information-limited" for w in report.warnings)


def test_no_missing_declared_empty_no_warning(tmp_path):  # T-024
    atom_rows = "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    report = _build_report(tmp_path, "no_missing.cif", _minimal_cif(atom_rows=atom_rows))

    assert report.missing_residues == ()
    assert report.missing_atoms == ()
    assert not any(w.code in ("missing-residues-reported", "missing-atoms-reported") for w in report.warnings)


def test_mmcif_question_mark_token_becomes_none(tmp_path):  # T-025
    atom_rows = "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    report = _build_report(tmp_path, "qmark.cif", _minimal_cif(atom_rows=atom_rows))

    # label_seq_id column in the fixture uses '1' (present); insertion code
    # column uses '?' - confirm it collapses to None on the chain's own residue.
    document_path = tmp_path / "qmark.cif"
    from molecular.receptor_parsing import parse_receptor_raw as _parse

    document = _parse(document_path, expected_checksum=compute_sha256(document_path))
    assert document.residues[0].insertion_code is None


def test_mmcif_dot_token_becomes_none(tmp_path):  # T-026
    atom_rows = "HETATM 1 ZN ZN . ZN B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1"
    entity_block = _entity_block((("2", "non-polymer", "zinc"),))
    path = _write(tmp_path, "dot_token.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))
    document = parse_receptor_raw(path, expected_checksum=compute_sha256(path))

    assert document.residues[0].label_seq_id is None


def test_water_hoh_classification(tmp_path):  # T-027
    atom_rows = "HETATM 1 O O . HOH C 3 . ? 1.0 1.0 1.0 1.00 10.0 201 A 1"
    entity_block = _entity_block((("3", "water", "water"),))
    report = _build_report(tmp_path, "hoh.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    component = report.non_polymer_components[0]
    assert component.classification == "water"
    assert component.classification_basis == "format-declared"


def test_water_dod_classification(tmp_path):  # T-028
    atom_rows = "HETATM 1 O O . DOD C 3 . ? 1.0 1.0 1.0 1.00 10.0 201 A 1"
    entity_block = _entity_block((("3", "water", "water"),))
    report = _build_report(tmp_path, "dod.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    assert report.non_polymer_components[0].classification == "water"


def test_zn_monatomic_metal_ion_classification(tmp_path):  # T-029
    atom_rows = "HETATM 1 ZN ZN . ZN B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1"
    entity_block = _entity_block((("2", "non-polymer", "zinc"),))
    report = _build_report(tmp_path, "zn.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    component = report.non_polymer_components[0]
    assert component.classification == "metal-ion"
    assert component.classification_basis == "closed-dictionary-match"


def test_polyatomic_unknown_non_polymer_classification(tmp_path):  # T-030
    atom_rows = (
        "HETATM 1 C C1 . LIG B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1\n"
        "HETATM 2 C C2 . LIG B 2 . ? 1.1 1.0 1.0 1.00 10.0 101 A 1"
    )
    entity_block = _entity_block((("2", "non-polymer", "ligand"),))
    report = _build_report(tmp_path, "ligand.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    component = report.non_polymer_components[0]
    assert component.classification == "other-non-polymer"
    assert any(w.code == "unknown-chemical-component" for w in report.warnings)


def test_modified_polymer_residue_classification(tmp_path):  # T-031
    atom_rows = "ATOM 1 N N . MSE A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    report = _build_report(tmp_path, "modres.cif", _minimal_cif(atom_rows=atom_rows))

    component = report.non_polymer_components[0]
    assert component.classification == "modified-polymer-residue"
    assert component.classification_basis == "format-declared"


def test_undetermined_entity_type_classification(tmp_path):  # T-032
    atom_rows = "HETATM 1 X X . UNK B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1"
    entity_block = _entity_block((("2", "", ""),))
    report = _build_report(tmp_path, "undetermined.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    component = report.non_polymer_components[0]
    assert component.classification == "unknown"
    assert component.classification_basis == "undetermined"


def _struct_conn_cif(conn_type_id: str) -> str:
    atom_rows = (
        "ATOM 1 N N . CYS A 1 7 ? 1.0 1.0 1.0 1.00 10.0 7 A 1\n"
        "ATOM 2 SG SG . CYS A 1 7 ? 1.5 1.0 1.0 1.00 10.0 7 A 1\n"
        "ATOM 3 N N . CYS A 1 20 ? 2.0 2.0 1.0 1.00 10.0 20 A 1\n"
        "ATOM 4 SG SG . CYS A 1 20 ? 2.5 2.0 1.0 1.00 10.0 20 A 1"
    )
    extra = f"""\
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_label_atom_id
c1 {conn_type_id} A A 7 SG A A 20 SG
#
"""
    return _minimal_cif(atom_rows=atom_rows, extra=extra)


def test_struct_conn_disulf_partitioned(tmp_path):  # T-033
    report = _build_report(tmp_path, "disulf.cif", _struct_conn_cif("disulf"))

    assert len(report.disulfide_bonds) == 1
    assert report.disulfide_bonds[0].source_category == "struct_conn"


def test_struct_conn_metalc_partitioned(tmp_path):  # T-034
    report = _build_report(tmp_path, "metalc.cif", _struct_conn_cif("metalc"))

    assert len(report.metal_coordinations) == 1
    assert any(w.code == "metal-coordination-present" for w in report.warnings)


def test_struct_conn_covale_partitioned(tmp_path):  # T-035
    report = _build_report(tmp_path, "covale.cif", _struct_conn_cif("covale"))

    assert len(report.covalent_connections) == 1
    assert any(w.code == "covalent-connection-present" for w in report.warnings)


def test_explicit_hydrogens_present(tmp_path):  # T-036
    atom_rows = (
        "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1\n"
        "ATOM 2 H H . ALA A 1 1 ? 1.1 1.0 1.0 1.00 10.0 1 A 1"
    )
    report = _build_report(tmp_path, "hydrogens.cif", _minimal_cif(atom_rows=atom_rows))

    assert report.explicit_hydrogens_present is True
    assert report.explicit_hydrogen_count == 1


def test_explicit_hydrogens_absent_warning(tmp_path):  # T-037
    atom_rows = "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    report = _build_report(tmp_path, "no_hydrogens.cif", _minimal_cif(atom_rows=atom_rows))

    assert report.explicit_hydrogens_present is False
    assert any(w.code == "explicit-hydrogens-absent" for w in report.warnings)


def test_unknown_component_never_omitted(tmp_path):  # T-038
    atom_rows = "HETATM 1 X X . UNK B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1"
    entity_block = _entity_block((("2", "", ""),))
    report = _build_report(tmp_path, "never_omitted.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    assert len(report.non_polymer_components) == 1


def test_duplicate_struct_conn_rows_deduplicated(tmp_path):  # T-039
    atom_rows = (
        "ATOM 1 N N . CYS A 1 7 ? 1.0 1.0 1.0 1.00 10.0 7 A 1\n"
        "ATOM 2 SG SG . CYS A 1 7 ? 1.5 1.0 1.0 1.00 10.0 7 A 1\n"
        "ATOM 3 N N . CYS A 1 20 ? 2.0 2.0 1.0 1.00 10.0 20 A 1\n"
        "ATOM 4 SG SG . CYS A 1 20 ? 2.5 2.0 1.0 1.00 10.0 20 A 1"
    )
    extra = """\
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_label_atom_id
c1 disulf A A 7 SG A A 20 SG
c1 disulf A A 7 SG A A 20 SG
#
"""
    report = _build_report(tmp_path, "dup_conn.cif", _minimal_cif(atom_rows=atom_rows, extra=extra))

    assert len(report.disulfide_bonds) == 1


def test_multi_atom_metal_code_never_metal_ion(tmp_path):  # T-040
    atom_rows = (
        "HETATM 1 ZN ZN1 . ZN B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1\n"
        "HETATM 2 ZN ZN2 . ZN B 2 . ? 1.1 1.0 1.0 1.00 10.0 101 A 1"
    )
    entity_block = _entity_block((("2", "non-polymer", "zinc cluster"),))
    report = _build_report(tmp_path, "zn_multi.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    component = report.non_polymer_components[0]
    assert component.classification == "other-non-polymer"


def test_classification_never_carries_role_or_disposition_fields():  # T-041
    assert "role" not in NonPolymerComponent.model_fields
    assert "disposition" not in NonPolymerComponent.model_fields


def test_extended_ccd_code_five_characters_accepted(tmp_path):  # T-042
    atom_rows = "HETATM 1 C C1 . ABCDE B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1"
    entity_block = _entity_block((("2", "non-polymer", "extended ligand"),))
    report = _build_report(tmp_path, "extended_ccd.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    assert report.non_polymer_components[0].ccd_code == "ABCDE"


# --- T-106-122 ---------------------------------------------------------------


def test_monatomic_unlisted_component_is_unknown_never_other_non_polymer(tmp_path):  # T-106
    atom_rows = "HETATM 1 XX XX . XEX B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1"
    entity_block = _entity_block((("2", "non-polymer", "mystery monatomic"),))
    report = _build_report(tmp_path, "mystery.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    component = report.non_polymer_components[0]
    assert component.classification == "unknown"
    assert component.classification_basis == "undetermined"


def test_legacy_pdb_selection_capability_inspection_only(tmp_path):  # T-107
    content = _pdb_atom("ATOM", 1, "N", "ALA", "A", 1) + "\nEND\n"
    report = _build_report_pdb(tmp_path, "inspection_only.pdb", content)

    assert report.selection_capability == "inspection-only"


def test_degenerate_mmcif_missing_label_identity_for_one_chain(tmp_path):  # T-108
    atom_rows = (
        "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1\n"
        "HETATM 2 O O . HOH ? ? . ? 2.0 1.0 1.0 1.00 10.0 201 B 1"
    )
    entity_block = _entity_block((("1", "polymer", "protein"), ("2", "water", "water")))
    report = _build_report(tmp_path, "degenerate.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block))

    degenerate_chain = next(c for c in report.chains if c.auth_asym_id == "B")
    assert degenerate_chain.label_asym_id is None
    assert degenerate_chain.label_asym_id_origin == "unavailable"
    assert report.selection_capability == "inspection-only"


def test_legacy_pdb_chain_label_asym_unavailable_auth_populated(tmp_path):  # T-109
    content = _pdb_atom("ATOM", 1, "N", "ALA", "A", 1) + "\nEND\n"
    report = _build_report_pdb(tmp_path, "chain_identity.pdb", content)

    chain = report.chains[0]
    assert chain.label_asym_id is None
    assert chain.label_asym_id_origin == "unavailable"
    assert chain.auth_asym_id == "A"


def test_legacy_pdb_entity_label_entity_id_unavailable(tmp_path):  # T-110
    content = _pdb_atom("ATOM", 1, "N", "ALA", "A", 1) + "\nEND\n"
    report = _build_report_pdb(tmp_path, "entity_identity.pdb", content)

    assert len(report.entities) >= 1
    for entity in report.entities:
        assert entity.label_entity_id is None
        assert entity.label_entity_id_origin == "unavailable"


def test_identity_origin_rejects_invalid_literal():  # T-111
    with pytest.raises(ValidationError):
        StructConnEndpoint.model_validate({"model_number": "not-an-int"})

    # IdentityOrigin itself is a Literal; verify it's enforced at a real field.
    from molecular.receptor_inspection import ChainInventoryEntry

    with pytest.raises(ValidationError):
        ChainInventoryEntry(
            label_asym_id="A",
            label_asym_id_origin="bogus-origin",
            label_entity_id="1",
            label_entity_id_origin="source",
            model_number=1,
            residue_count=1,
            atom_count=1,
        )


def test_struct_conn_label_alt_id_preserved_both_partners(tmp_path):  # T-112
    atom_rows = (
        "ATOM 1 N N . CYS A 1 7 ? 1.0 1.0 1.0 1.00 10.0 7 A 1\n"
        "ATOM 2 N N . CYS A 1 20 ? 2.0 2.0 1.0 1.00 10.0 20 A 1"
    )
    extra = """\
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.pdbx_ptnr1_label_alt_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.pdbx_ptnr2_label_alt_id
c1 disulf A A 7 N A A A 20 N B
#
"""
    report = _build_report(tmp_path, "altid.cif", _minimal_cif(atom_rows=atom_rows, extra=extra))

    connection = report.disulfide_bonds[0]
    assert connection.partner_a.label_alt_id == "A"
    assert connection.partner_b.label_alt_id == "B"


def test_struct_conn_symmetry_preserved_both_partners(tmp_path):  # T-113
    atom_rows = (
        "ATOM 1 N N . CYS A 1 7 ? 1.0 1.0 1.0 1.00 10.0 7 A 1\n"
        "ATOM 2 N N . CYS A 1 20 ? 2.0 2.0 1.0 1.00 10.0 20 A 1"
    )
    extra = """\
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr2_symmetry
c1 disulf A A 7 N 1_555 A A 20 N 2_555
#
"""
    report = _build_report(tmp_path, "symmetry.cif", _minimal_cif(atom_rows=atom_rows, extra=extra))

    connection = report.disulfide_bonds[0]
    assert connection.partner_a.symmetry == "1_555"
    assert connection.partner_b.symmetry == "2_555"


def test_struct_conn_distance_value_order_role_details_preserved_or_none(tmp_path):  # T-114
    atom_rows = (
        "ATOM 1 N N . CYS A 1 7 ? 1.0 1.0 1.0 1.00 10.0 7 A 1\n"
        "ATOM 2 N N . CYS A 1 20 ? 2.0 2.0 1.0 1.00 10.0 20 A 1\n"
        "ATOM 3 N N . CYS A 1 30 ? 3.0 2.0 1.0 1.00 10.0 30 A 1"
    )
    extra = """\
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.pdbx_dist_value
_struct_conn.pdbx_value_order
_struct_conn.pdbx_role
_struct_conn.details
c1 disulf A A 7 N A A 20 N 2.05 sing linking 'a full detail'
c2 covale A A 20 N A A 30 N . . . .
#
"""
    report = _build_report(tmp_path, "fields.cif", _minimal_cif(atom_rows=atom_rows, extra=extra))

    full = next(c for c in report.disulfide_bonds)
    assert full.distance == Decimal("2.05")
    assert full.value_order == "sing"
    assert full.role == "linking"
    assert full.details == "a full detail"

    sparse = report.covalent_connections[0]
    assert sparse.distance is None
    assert sparse.value_order is None
    assert sparse.role is None
    assert sparse.details is None


def test_struct_conn_absent_conn_type_raises_malformed(tmp_path):  # T-115
    atom_rows = (
        "ATOM 1 N N . CYS A 1 7 ? 1.0 1.0 1.0 1.00 10.0 7 A 1\n"
        "ATOM 2 N N . CYS A 1 20 ? 2.0 2.0 1.0 1.00 10.0 20 A 1"
    )
    extra = """\
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
c1 . A 7 A 20
#
"""
    path = _write(tmp_path, "malformed_conn.cif", _minimal_cif(atom_rows=atom_rows, extra=extra))

    with pytest.raises(StructConnMalformedError):
        parse_receptor_raw(path, expected_checksum=compute_sha256(path))


def test_legacy_ssbond_mapped_to_disulfide_bonds(tmp_path):  # T-116
    content = (
        _pdb_ssbond("A", 7, "A", 20) + "\n"
        + _pdb_atom("ATOM", 1, "N", "CYS", "A", 7) + "\n"
        + _pdb_atom("ATOM", 2, "SG", "CYS", "A", 7) + "\n"
        + _pdb_atom("ATOM", 3, "N", "CYS", "A", 20) + "\n"
        + _pdb_atom("ATOM", 4, "SG", "CYS", "A", 20) + "\n"
        + "END\n"
    )
    report = _build_report_pdb(tmp_path, "ssbond.pdb", content)

    connection = report.disulfide_bonds[0]
    assert connection.source_category == "legacy-pdb-ssbond"
    assert connection.partner_a.label_asym_id is None
    assert connection.partner_a.label_alt_id is None
    assert connection.partner_a.symmetry is None


def test_legacy_link_no_metal_partner_covalent(tmp_path):  # T-117
    content = (
        _pdb_link("C1", "LIG", "A", 50, "C2", "LIG", "A", 50, length=1.5) + "\n"
        + _pdb_atom("HETATM", 1, "C1", "LIG", "A", 50, elem="C") + "\n"
        + _pdb_atom("HETATM", 2, "C2", "LIG", "A", 50, elem="C") + "\n"
        + "END\n"
    )
    report = _build_report_pdb(tmp_path, "link_covalent.pdb", content)

    connection = report.covalent_connections[0]
    assert connection.source_category == "legacy-pdb-link"
    assert connection.category == "legacy-pdb-link-covalent"


def test_legacy_link_with_metal_partner_metal_coordination(tmp_path):  # T-118
    content = (
        _pdb_link("ZN", "ZN", "A", 101, "NE2", "HIS", "A", 10, length=2.1) + "\n"
        + _pdb_atom("ATOM", 1, "NE2", "HIS", "A", 10, elem="N") + "\n"
        + _pdb_atom("HETATM", 2, "ZN", "ZN", "A", 101, elem="ZN") + "\n"
        + "END\n"
    )
    report = _build_report_pdb(tmp_path, "link_metal.pdb", content)

    connection = report.metal_coordinations[0]
    assert connection.source_category == "legacy-pdb-link"
    assert connection.category == "legacy-pdb-link-metal"


def test_legacy_conect_mapped_by_metal_partner_rule(tmp_path):  # T-119
    content = (
        _pdb_atom("ATOM", 1, "NE2", "HIS", "A", 10, elem="N") + "\n"
        + _pdb_atom("HETATM", 2, "ZN", "ZN", "A", 101, elem="ZN") + "\n"
        + _pdb_conect(2, 1) + "\n"
        + "END\n"
    )
    report = _build_report_pdb(tmp_path, "conect.pdb", content)

    connection = report.metal_coordinations[0]
    assert connection.source_category == "legacy-pdb-conect"
    assert connection.category == "legacy-pdb-conect-metal"


def test_legacy_pdb_remark_465_470_not_structurally_exposed_stays_unavailable(tmp_path):  # T-120
    content = (
        "REMARK 465 MISSING RESIDUES\n"
        + _pdb_atom("ATOM", 1, "N", "ALA", "A", 1) + "\n"
        + "END\n"
    )
    report = _build_report_pdb(tmp_path, "remark465.pdb", content)

    assert "REMARK465" in report.unavailable_categories
    assert any(
        w.code == "category-unavailable" and w.category_name == "REMARK465" for w in report.warnings
    )


def test_remark_465_470_when_structurally_exposed_populates_missing(tmp_path):  # T-121
    document = ParsedReceptorDocument(
        structure_format="pdb",
        raw_checksum="a" * 64,
        parser_name="gemmi",
        parser_version="0.7.5",
        models=(),
        assemblies=(),
        entities=(),
        residues=(
            ParsedResidue(
                model_number=1,
                label_asym_id=None,
                label_entity_id=None,
                auth_asym_id="A",
                auth_seq_id="1",
                insertion_code=None,
                label_seq_id=None,
                comp_id="ALA",
                is_het=False,
                atoms=(ParsedAtom(atom_name="N", altloc=None, occupancy=Decimal("1"), element="N", serial=1),),
            ),
        ),
        connections=(),
        missing_residues=(
            ParsedMissingResidue(
                model_number=1,
                label_asym_id=None,
                auth_asym_id="A",
                residue_identifier="5",
                residue_name="GLY",
            ),
        ),
        missing_atoms=(
            ParsedMissingAtom(
                model_number=1,
                label_asym_id=None,
                auth_asym_id="A",
                residue_identifier="1",
                atom_name="CB",
            ),
        ),
        unavailable_categories=("pdbx_struct_assembly", "struct_conn"),
    )

    report = build_receptor_inspection_report(
        document, protein_input_id="demo-protein", raw_format_declared="pdb"
    )

    assert len(report.missing_residues) == 1
    assert report.missing_residues[0].residue_name == "GLY"
    assert len(report.missing_atoms) == 1
    assert report.missing_atoms[0].atom_name == "CB"
    assert "REMARK465" not in report.unavailable_categories


def test_inspection_id_golden_vector():  # T-122
    inspection_id = compute_inspection_id(
        parser_name="gemmi",
        parser_version="0.7.5",
        protein_input_id="demo-protein",
        raw_checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        raw_format="mmcif",
    )

    assert inspection_id == "73090f1a141de86e358c79f051b0245df0ca32287425935632e624d15078adf4"
    assert len(inspection_id) == 64


# --- CP1 Checkpoint 1 micro-review regression tests (non-normative) -------
# Added to close the two MAJOR findings of the Checkpoint 1 micro-review
# (CP1-MAJOR-01: water classification priority, Section 17.1; CP1-MAJOR-02:
# NFC normalization at the parser boundary, Section 33.2). These
# intentionally are not T-numbered - they do not renumber or reuse any of
# T-013-042 or T-106-122 above.


def test_cp1_major01_hoh_mmcif_without_label_entity_id_classified_water(tmp_path):  # CP1-MAJOR-01
    atom_rows = "HETATM 1 O O . HOH ? ? . ? 1.0 1.0 1.0 1.00 10.0 201 B 1"
    report = _build_report(tmp_path, "hoh_no_entity.cif", _minimal_cif(atom_rows=atom_rows))

    component = next(c for c in report.non_polymer_components if c.ccd_code == "HOH")
    assert component.classification == "water"
    assert component.classification_basis == "format-declared"


def test_cp1_major01_dod_mmcif_without_label_entity_id_classified_water(tmp_path):  # CP1-MAJOR-01
    atom_rows = "HETATM 1 O O . DOD ? ? . ? 1.0 1.0 1.0 1.00 10.0 201 B 1"
    report = _build_report(tmp_path, "dod_no_entity.cif", _minimal_cif(atom_rows=atom_rows))

    component = next(c for c in report.non_polymer_components if c.ccd_code == "DOD")
    assert component.classification == "water"


def test_cp1_major01_water_takes_priority_over_polymer_entity_association(tmp_path):  # CP1-MAJOR-01
    # HOH explicitly associated with a label_entity_id typed "polymer" by
    # _entity - water classification must still win (Section 17.1 is
    # unconditional on ccd_code, never gated by entity type association).
    atom_rows = "HETATM 1 O O . HOH A 1 1 ? 1.0 1.0 1.0 1.00 10.0 201 A 1"
    entity_block = _entity_block((("1", "polymer", "Test protein"),))
    report = _build_report(
        tmp_path, "hoh_polymer_entity.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block)
    )

    component = next(c for c in report.non_polymer_components if c.ccd_code == "HOH")
    assert component.classification == "water"


def test_cp1_major01_non_aqueous_component_without_label_entity_id_not_misclassified_water(tmp_path):  # CP1-MAJOR-01
    atom_rows = "ATOM 1 N N . MSE ? ? . ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    report = _build_report(tmp_path, "mse_no_entity.cif", _minimal_cif(atom_rows=atom_rows))

    component = next(c for c in report.non_polymer_components if c.ccd_code == "MSE")
    assert component.classification == "modified-polymer-residue"
    assert component.classification != "water"


def test_cp1_major01_legacy_pdb_water_classification_unaffected(tmp_path):  # CP1-MAJOR-01
    content = _pdb_atom("HETATM", 1, "O", "HOH", "A", 201) + "\nEND\n"
    report = _build_report_pdb(tmp_path, "hoh.pdb", content)

    component = next(c for c in report.non_polymer_components if c.ccd_code == "HOH")
    assert component.classification == "water"


def test_cp1_major01_metal_coordination_classification_unaffected(tmp_path):  # CP1-MAJOR-01
    content = (
        _pdb_link("ZN", "ZN", "A", 101, "NE2", "HIS", "A", 10, length=2.1) + "\n"
        + _pdb_atom("ATOM", 1, "NE2", "HIS", "A", 10, elem="N") + "\n"
        + _pdb_atom("HETATM", 2, "ZN", "ZN", "A", 101, elem="ZN") + "\n"
        + "END\n"
    )
    report = _build_report_pdb(tmp_path, "link_metal_regression.pdb", content)

    connection = report.metal_coordinations[0]
    assert connection.source_category == "legacy-pdb-link"
    assert connection.category == "legacy-pdb-link-metal"


def test_cp1_major02_entity_description_propagates_nfc_into_report(tmp_path):  # CP1-MAJOR-02
    nfd_description = unicodedata.normalize("NFD", "café protein")
    nfc_description = unicodedata.normalize("NFC", "café protein")
    atom_rows = "ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1"
    entity_block = _entity_block((("1", "polymer", nfd_description),))
    report = _build_report(
        tmp_path, "nfd_entity.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block)
    )

    assert report.entities[0].description == nfc_description
    assert report.entities[0].description != nfd_description


def test_cp1_major02_warning_detail_embedding_structural_value_remains_nfc(tmp_path):  # CP1-MAJOR-02
    nfd_ccd_code = unicodedata.normalize("NFD", "É")
    nfc_ccd_code = unicodedata.normalize("NFC", "É")
    atom_rows = f"HETATM 1 XX XX . '{nfd_ccd_code}' B 2 . ? 1.0 1.0 1.0 1.00 10.0 101 A 1"
    entity_block = _entity_block((("2", "non-polymer", "mystery"),))
    report = _build_report(
        tmp_path, "nfd_ccd.cif", _minimal_cif(atom_rows=atom_rows, entity_block=entity_block)
    )

    component = report.non_polymer_components[0]
    assert component.classification == "unknown"
    assert component.ccd_code == nfc_ccd_code
    assert len(component.ccd_code) == 1  # precomposed - confirms normalization actually ran

    warning = next(w for w in report.warnings if w.code == "unknown-chemical-component")
    assert nfc_ccd_code in warning.detail
    assert warning.detail == unicodedata.normalize("NFC", warning.detail)
