"""Tests for receptor_parsing.py - Increment 020, Checkpoint 1.

Covers T-001-012 (preserved/rewritten) and T-095-105 (new) from the frozen
final specification's normative matrix (Section 35.1). Each test is
labeled with its exact scenario id. No network access; every mmCIF/PDB
fixture is a minimal literal written under `tmp_path`.
"""

import gzip
import unicodedata
from pathlib import Path

import pytest

from molecular.atomic import compute_sha256
from molecular.receptor_inspection import build_receptor_inspection_report
from molecular.receptor_parsing import (
    CompressedExtensionRejectedError,
    CompressedRawContentError,
    ParsedReceptorDocument,
    RawArtifactTooLargeError,
    RawChecksumMismatchError,
    ReceptorParseFailureError,
    UnsupportedStructureFormatError,
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


def _minimal_cif(*, extra_blocks: str = "", model_num: int = 1) -> str:
    return f"""\
data_TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer 'Test protein'
#
{_ATOM_SITE_HEADER}\
ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A {model_num}
ATOM 2 C CA . ALA A 1 1 ? 2.0 1.0 1.0 1.00 10.0 1 A {model_num}
#
{extra_blocks}
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _parse(path: Path) -> ParsedReceptorDocument:
    return parse_receptor_raw(path, expected_checksum=compute_sha256(path))


# --- T-001-012 -------------------------------------------------------------


def test_mmcif_single_model_no_assembly_parses_successfully(tmp_path):  # T-001
    path = _write(tmp_path, "receptor.cif", _minimal_cif())

    document = _parse(path)

    assert document.structure_format == "mmcif"
    assert len(document.residues) == 1


def test_mmcif_multiple_models_all_listed(tmp_path):  # T-002
    content = f"""\
data_TEST
#
{_ATOM_SITE_HEADER}\
ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1
ATOM 2 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 2
#
"""
    path = _write(tmp_path, "multi_model.cif", content)

    document = _parse(path)

    assert {model.model_number for model in document.models} == {1, 2}


def test_mmcif_assembly_declared_preserves_raw_text_never_evaluated(tmp_path):  # T-003
    extra = """\
loop_
_pdbx_struct_assembly.id
_pdbx_struct_assembly.details
1 'complete icosahedral assembly'
#
loop_
_pdbx_struct_assembly_gen.assembly_id
_pdbx_struct_assembly_gen.oper_expression
_pdbx_struct_assembly_gen.asym_id_list
1 '(1-60)' A,B
#
"""
    path = _write(tmp_path, "assembly.cif", _minimal_cif(extra_blocks=extra))

    document = _parse(path)

    assert len(document.assemblies) == 1
    assembly = document.assemblies[0]
    assert assembly.assembly_id == "1"
    assert assembly.details == "complete icosahedral assembly"
    assert len(assembly.generation) == 1
    assert assembly.generation[0].expression == "(1-60)"
    assert assembly.generation[0].asym_id_list == ("A", "B")


def test_legacy_pdb_valid_identity_is_honest(tmp_path):  # T-004
    content = (
        "ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00 10.00           N\n"
        "ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00 10.00           C\n"
        "END\n"
    )
    path = _write(tmp_path, "legacy.pdb", content)

    document = _parse(path)
    report = build_receptor_inspection_report(
        document, protein_input_id="demo-protein", raw_format_declared="pdb"
    )

    assert document.structure_format == "pdb"
    assert report.selection_capability == "inspection-only"
    for chain in report.chains:
        assert chain.label_asym_id is None
        assert chain.label_asym_id_origin == "unavailable"
        assert chain.label_entity_id is None
        assert chain.label_entity_id_origin == "unavailable"
        assert chain.auth_asym_id == "A"


def test_unrecognized_format_raises(tmp_path):  # T-005
    path = _write(tmp_path, "bogus.cif", "this is not a structure file at all\njust text\n")

    with pytest.raises(UnsupportedStructureFormatError):
        _parse(path)


def test_malformed_cif_syntax_raises_with_cause(tmp_path):  # T-006
    path = _write(tmp_path, "malformed.cif", "data_X\nloop_\n_atom_site.id\n_atom_site.label\n1 2 3\n")

    with pytest.raises(ReceptorParseFailureError) as exc_info:
        _parse(path)
    assert exc_info.value.__cause__ is not None


def test_truncated_mid_category_raises(tmp_path):  # T-007
    path = _write(tmp_path, "truncated.cif", "data_X\nloop_\n_atom_site.id\n_atom_site.label\n1 ")

    with pytest.raises(ReceptorParseFailureError) as exc_info:
        _parse(path)
    assert exc_info.value.__cause__ is not None


def test_missing_assembly_category_reported_unavailable(tmp_path):  # T-008
    path = _write(tmp_path, "no_assembly.cif", _minimal_cif())

    document = _parse(path)

    assert document.assemblies == ()
    assert "pdbx_struct_assembly" in document.unavailable_categories


def test_missing_struct_conn_category_all_connection_tuples_empty(tmp_path):  # T-009
    path = _write(tmp_path, "no_struct_conn.cif", _minimal_cif())

    document = _parse(path)
    report = build_receptor_inspection_report(
        document, protein_input_id="demo-protein", raw_format_declared="mmcif"
    )

    assert report.disulfide_bonds == ()
    assert report.covalent_connections == ()
    assert report.metal_coordinations == ()
    assert "struct_conn" in document.unavailable_categories


def test_parser_version_is_runtime_observed(tmp_path):  # T-010
    import gemmi

    path = _write(tmp_path, "version.cif", _minimal_cif())

    document = _parse(path)

    assert document.parser_version == gemmi.__version__


def test_no_gemmi_type_escapes_parse_receptor_raw(tmp_path):  # T-011
    import dataclasses

    path = _write(tmp_path, "boundary.cif", _minimal_cif())
    document = _parse(path)

    def check(value, seen=None):
        seen = seen if seen is not None else set()
        if id(value) in seen:
            return
        seen.add(id(value))

        module = type(value).__module__
        assert not module.startswith("gemmi"), f"gemmi type escaped: {type(value)!r}"

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                check(getattr(value, field.name), seen)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item, seen)

    check(document)


def test_declared_vs_detected_format_divergence_is_content_authoritative(tmp_path):  # T-012
    path = _write(tmp_path, "declared_as_pdb.cif", _minimal_cif())

    document = _parse(path)
    report = build_receptor_inspection_report(
        document, protein_input_id="demo-protein", raw_format_declared="pdb"
    )

    assert report.structure_format == "mmcif"
    assert any(w.code == "declared-format-mismatch" for w in report.warnings)


# --- T-095-105 -------------------------------------------------------------


def test_setup_entities_and_mutating_methods_never_invoked():  # T-095
    import ast

    source = Path("src/molecular/receptor_parsing.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden = {
        "setup_entities",
        "add_entity_types",
        "assign_subchains",
        "ensure_entities",
        "deduplicate_entities",
        "remove_hydrogens",
        "remove_waters",
        "remove_ligands_and_waters",
        "remove_alternative_conformations",
    }
    called_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (called_names & forbidden), called_names & forbidden


def test_gz_extension_rejected_before_open(tmp_path):  # T-096
    path = _write(tmp_path, "receptor.cif.gz", "irrelevant content")

    with pytest.raises(CompressedExtensionRejectedError):
        parse_receptor_raw(path, expected_checksum="0" * 64)


def test_gzip_magic_bytes_rejected(tmp_path):  # T-097
    path = tmp_path / "receptor.cif"
    path.write_bytes(gzip.compress(b"data_X\n"))

    with pytest.raises(CompressedRawContentError):
        parse_receptor_raw(path, expected_checksum=compute_sha256(path))


def test_bzip2_magic_bytes_rejected(tmp_path):  # T-098
    path = tmp_path / "receptor.cif"
    path.write_bytes(b"BZh91AY&SY\x00\x00\x00\x00")

    with pytest.raises(CompressedRawContentError):
        parse_receptor_raw(path, expected_checksum=compute_sha256(path))


def test_xz_magic_bytes_rejected(tmp_path):  # T-099
    path = tmp_path / "receptor.cif"
    path.write_bytes(b"\xfd7zXZ\x00\x00\x00\x00\x00")

    with pytest.raises(CompressedRawContentError):
        parse_receptor_raw(path, expected_checksum=compute_sha256(path))


def test_zip_magic_bytes_rejected(tmp_path):  # T-100
    path = tmp_path / "receptor.cif"
    path.write_bytes(b"PK\x03\x04\x00\x00\x00\x00")

    with pytest.raises(CompressedRawContentError):
        parse_receptor_raw(path, expected_checksum=compute_sha256(path))


def test_cif_named_file_with_gzip_content_rejected(tmp_path):  # T-101
    path = tmp_path / "estrutura.cif"
    path.write_bytes(gzip.compress(_minimal_cif().encode("utf-8")))

    with pytest.raises(CompressedRawContentError):
        parse_receptor_raw(path, expected_checksum=compute_sha256(path))


def test_originally_gz_file_renamed_to_cif_rejected(tmp_path):  # T-102
    gz_bytes = gzip.compress(_minimal_cif().encode("utf-8"))
    renamed = tmp_path / "was_a_gz.cif"
    renamed.write_bytes(gz_bytes)

    with pytest.raises(CompressedRawContentError):
        parse_receptor_raw(renamed, expected_checksum=compute_sha256(renamed))


def test_gz_extension_rejected_even_when_content_not_compressed(tmp_path):  # T-103
    path = tmp_path / "not_actually_gz.gz"
    path.write_text(_minimal_cif(), encoding="utf-8")

    with pytest.raises(CompressedExtensionRejectedError):
        parse_receptor_raw(path, expected_checksum=compute_sha256(path))


def test_oversized_raw_artifact_rejected(tmp_path, monkeypatch):  # T-104
    path = _write(tmp_path, "receptor.cif", _minimal_cif())

    import molecular.receptor_parsing as receptor_parsing_module

    monkeypatch.setattr(receptor_parsing_module, "MAX_RAW_ARTIFACT_SIZE_BYTES", 10)

    with pytest.raises(RawArtifactTooLargeError):
        parse_receptor_raw(path, expected_checksum=compute_sha256(path))


def test_altered_raw_artifact_checksum_mismatch_rejected(tmp_path):  # T-105
    path = _write(tmp_path, "receptor.cif", _minimal_cif())
    stale_checksum = compute_sha256(path)

    path.write_text(_minimal_cif(model_num=2), encoding="utf-8")

    with pytest.raises(RawChecksumMismatchError):
        parse_receptor_raw(path, expected_checksum=stale_checksum)


# --- CP1 Checkpoint 1 micro-review regression tests (non-normative) -------
# Added to close CP1-MAJOR-02 (NFC normalization at the parser boundary,
# Section 33.2) found by the Checkpoint 1 micro-review. These intentionally
# are not T-numbered - they do not renumber or reuse any of T-001-042 or
# T-095-105 above.

_NFD_DESCRIPTION = unicodedata.normalize("NFD", "café protein")
_NFC_DESCRIPTION = unicodedata.normalize("NFC", "café protein")


def test_cp1_major02_entity_description_simple_field_normalized_to_nfc(tmp_path):  # CP1-MAJOR-02
    content = f"""\
data_TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer '{_NFD_DESCRIPTION}'
#
{_ATOM_SITE_HEADER}\
ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1
#
"""
    path = _write(tmp_path, "nfd_description.cif", content)

    document = _parse(path)

    assert document.entities[0].description == _NFC_DESCRIPTION
    assert document.entities[0].description != _NFD_DESCRIPTION
    assert unicodedata.is_normalized("NFC", document.entities[0].description)


def test_cp1_major02_assembly_details_optional_field_normalized_to_nfc(tmp_path):  # CP1-MAJOR-02
    extra = f"""\
loop_
_pdbx_struct_assembly.id
_pdbx_struct_assembly.details
1 '{_NFD_DESCRIPTION}'
#
"""
    path = _write(tmp_path, "nfd_assembly.cif", _minimal_cif(extra_blocks=extra))

    document = _parse(path)

    assert document.assemblies[0].details == _NFC_DESCRIPTION
    assert document.assemblies[0].details != _NFD_DESCRIPTION


def test_cp1_major02_struct_conn_details_nested_field_normalized_to_nfc(tmp_path):  # CP1-MAJOR-02
    content = f"""\
data_TEST
#
{_ATOM_SITE_HEADER}\
ATOM 1 N N . CYS A 1 7 ? 1.0 1.0 1.0 1.00 10.0 7 A 1
ATOM 2 SG SG . CYS A 1 7 ? 1.5 1.0 1.0 1.00 10.0 7 A 1
ATOM 3 N N . CYS A 1 20 ? 2.0 2.0 1.0 1.00 10.0 20 A 1
ATOM 4 SG SG . CYS A 1 20 ? 2.5 2.0 1.0 1.00 10.0 20 A 1
#
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
_struct_conn.details
c1 disulf A A 7 SG A A 20 SG '{_NFD_DESCRIPTION}'
#
"""
    path = _write(tmp_path, "nfd_struct_conn.cif", content)

    document = _parse(path)

    assert document.connections[0].details == _NFC_DESCRIPTION
    assert document.connections[0].details != _NFD_DESCRIPTION


def test_cp1_major02_atom_name_in_nested_atoms_tuple_normalized_to_nfc(tmp_path):  # CP1-MAJOR-02
    nfd_atom_name = unicodedata.normalize("NFD", "Nấ")
    nfc_atom_name = unicodedata.normalize("NFC", "Nấ")
    content = f"""\
data_TEST
#
{_ATOM_SITE_HEADER}\
ATOM 1 N '{nfd_atom_name}' . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1
#
"""
    path = _write(tmp_path, "nfd_atom_name.cif", content)

    document = _parse(path)

    assert document.residues[0].atoms[0].atom_name == nfc_atom_name
    assert document.residues[0].atoms[0].atom_name != nfd_atom_name


def test_cp1_major02_ascii_only_content_unaffected(tmp_path):  # CP1-MAJOR-02
    path = _write(tmp_path, "ascii.cif", _minimal_cif())

    document = _parse(path)

    assert document.entities[0].description == "Test protein"


def test_cp1_major02_none_preserved_not_converted_to_empty_string(tmp_path):  # CP1-MAJOR-02
    extra = """\
loop_
_pdbx_struct_assembly.id
_pdbx_struct_assembly.details
1 .
#
"""
    path = _write(tmp_path, "nfd_none.cif", _minimal_cif(extra_blocks=extra))

    document = _parse(path)

    assert document.assemblies[0].details is None


def test_cp1_major02_raw_checksum_and_file_bytes_unaffected_by_normalization(tmp_path):  # CP1-MAJOR-02
    content = f"""\
data_TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer '{_NFD_DESCRIPTION}'
#
{_ATOM_SITE_HEADER}\
ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1
#
"""
    path = _write(tmp_path, "nfd_checksum.cif", content)
    raw_bytes_before = path.read_bytes()
    expected_checksum = compute_sha256(path)

    document = parse_receptor_raw(path, expected_checksum=expected_checksum)

    assert document.raw_checksum == expected_checksum
    assert path.read_bytes() == raw_bytes_before
    assert document.entities[0].description == _NFC_DESCRIPTION


def test_cp1_major02_nfc_and_nfd_equivalent_inputs_produce_identical_parsed_domain_value(tmp_path):  # CP1-MAJOR-02
    def cif_with(description: str) -> str:
        return f"""\
data_TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer '{description}'
#
{_ATOM_SITE_HEADER}\
ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1
#
"""

    nfc_path = _write(tmp_path, "nfc_variant.cif", cif_with(_NFC_DESCRIPTION))
    nfd_path = _write(tmp_path, "nfd_variant.cif", cif_with(_NFD_DESCRIPTION))

    nfc_document = _parse(nfc_path)
    nfd_document = _parse(nfd_path)

    assert nfc_document.entities[0].description == nfd_document.entities[0].description
    assert nfc_document.raw_checksum != nfd_document.raw_checksum
