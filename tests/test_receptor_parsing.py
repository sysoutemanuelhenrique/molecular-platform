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


# ==========================================================================
# Increment 021, Checkpoint 2: the real-Gemmi DOM adapter
# (`parse_receptor_cif_dom021`/`serialize_cif_document021`/
# `reparse_and_extract_cif_dom021`) and the full
# `materialize_receptor_cif021` pipeline. Every test below runs against
# real Gemmi 0.7.5 - no stub/sentinel ever stands in for the native
# library. Scenario ids are the CP2 subset of CP2_SCENARIO_SCOPE.csv.
# ==========================================================================

import hashlib
from datetime import datetime, timezone

import gemmi

from molecular.inputs import AcquisitionRecord, ProteinInput, RawArtifact
from molecular.receptor_inspection import ModelDescriptor, ParserProvenance, ReceptorInspectionReport, compute_inspection_id
from molecular.receptor_materialization import RecordSnapshot, build_source_context
from molecular.receptor_materialization import CategoryDecision, MaterializationExecutionError
from molecular.receptor_resolution import ResolvedChainSelection, ResolvedComponentDisposition, ResolvedReceptorSelection
from molecular.receptor_selection import ReceptorSelectionPolicy
from molecular.receptor_parsing import (
    CifSaveFrameRejectedError,
    CifStructuralLimitExceededError,
    InvalidCifEncodingError,
    MultipleDataBlocksError,
    materialize_receptor_cif021,
    parse_receptor_cif_dom021,
    raw_cif_null_to_optional_str,
    reparse_and_extract_cif_dom021,
    serialize_cif_document021,
)

_CP2_PROTEIN_ID = "protein-cp2"
_CP2_SELECTION_ID = "sel-cp2"
# C-04: a real SHA-256/size of the actual fixture bytes below - never a
# fictitious constant - since `materialize_receptor_cif021` now checks
# the received RAW bytes against exactly these SourceContext fields
# before parsing. Computed once `_MINIMAL_CIF` is defined (Python module
# execution is top-to-bottom, so by the time any test calls
# `_cp2_source_context()` this is already the real digest/size, even
# though a few helper *definitions* above reference the name first).


def _cp2_inspection_id() -> str:
    return compute_inspection_id(
        parser_name="gemmi", parser_version="0.7.5", protein_input_id=_CP2_PROTEIN_ID,
        raw_checksum=_CP2_RAW_SHA, raw_format="mmcif",
    )


def _cp2_source_context():
    inspection_id = _cp2_inspection_id()
    protein_input = ProteinInput(
        id=_CP2_PROTEIN_ID, name="P",
        acquisition=AcquisitionRecord(
            method="uploaded", acquired_at=datetime.now(timezone.utc),
            raw_artifact=RawArtifact(path="x", format="mmcif", checksum=_CP2_RAW_SHA),
        ),
    )
    report = ReceptorInspectionReport(
        inspection_id=inspection_id, inspection_schema_version="1.0", inspection_profile_version="1.0",
        protein_input_id=_CP2_PROTEIN_ID, raw_checksum=_CP2_RAW_SHA, raw_format_declared="mmcif",
        structure_format="mmcif", parser=ParserProvenance(name="gemmi", version="0.7.5"),
        selection_capability="full", models=(ModelDescriptor(model_number=1),), assemblies=(), entities=(),
        chains=(), altloc_observations=(), occupancy_observations=(), missing_residues=(), missing_atoms=(),
        disulfide_bonds=(), covalent_connections=(), metal_coordinations=(), non_polymer_components=(),
        explicit_hydrogens_present=False, explicit_hydrogen_count=0, unavailable_categories=(), warnings=(),
    )
    policy = ReceptorSelectionPolicy(
        schema_version="1.0", selection_id=_CP2_SELECTION_ID, protein_input_id=_CP2_PROTEIN_ID,
        inspection_id=inspection_id, raw_checksum_observed=_CP2_RAW_SHA, declared_by="s",
        declared_at=datetime.now(timezone.utc),
    )
    resolution = _cp2_resolution()
    records = (
        RecordSnapshot(kind="protein", id=_CP2_PROTEIN_ID, sha256="1" * 64, size_bytes=1, schema_version=None),
        RecordSnapshot(kind="inspection", id=inspection_id, sha256="2" * 64, size_bytes=1, schema_version="1.0"),
        RecordSnapshot(kind="selection", id=_CP2_SELECTION_ID, sha256="3" * 64, size_bytes=1, schema_version="1.0"),
        RecordSnapshot(kind="resolution", id=_CP2_SELECTION_ID, sha256="4" * 64, size_bytes=1, schema_version="1.0"),
    )
    return build_source_context(
        protein_input=protein_input, inspection_report=report, policy=policy,
        resolved_selection=resolution, raw_size_bytes=_CP2_RAW_SIZE, records=records,
    )


def _cp2_resolution() -> ResolvedReceptorSelection:
    return ResolvedReceptorSelection(
        schema_version="1.0", resolved_selection_id=_CP2_SELECTION_ID, policy_selection_id=_CP2_SELECTION_ID,
        protein_input_id=_CP2_PROTEIN_ID, inspection_id=_cp2_inspection_id(), raw_checksum=_CP2_RAW_SHA,
        structure_format="mmcif", parser=ParserProvenance(name="gemmi", version="0.7.5"), resolved_model=1,
        model_resolution_basis="only-model-present", resolved_assembly=None,
        resolved_chains=(ResolvedChainSelection(label_asym_id="A", label_entity_id="E1", model_number=1),),
        resolved_component_dispositions=(), resolved_at=datetime.now(timezone.utc),
    )


_MINIMAL_CIF = """\
data_5XYZ
#
_entry.id 5XYZ
#
loop_
_entity.id
_entity.type
E1 polymer
E2 polymer
#
loop_
_struct_asym.id
_struct_asym.entity_id
A E1
B E2
#
loop_
_entity_poly.entity_id
_entity_poly.type
E1 polypeptide(L)
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
E1 1 ALA
E1 2 CYS
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
A E1 1 ALA
A E1 2 CYS
#
loop_
_chem_comp.id
_chem_comp.type
ALA 'L-peptide linking'
CYS 'L-peptide linking'
MET 'L-peptide linking'
#
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
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.pdbx_PDB_model_num
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 C CA . ALA A E1 1 A 1 ? 1 1.000 2.000 3.000 1.00 10.00
ATOM 2 S SG . CYS A E1 2 A 2 ? 1 1.500 2.500 3.500 1.00 11.00
ATOM 3 C CA . ALA A E1 1 A 1 ? 1 1.100 2.100 3.100 1.00 10.10
ATOM 4 N N  . ALA A E1 1 A 1 ? 1 1.200 2.200 3.200 1.00 10.20
ATOM 5 S SD . MET B E2 1 B 1 ? 1 9.000 9.000 9.000 1.00 12.00
#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr2_symmetry
conn1 disulf A ALA 1 CA 1_555 A ALA 1 N 1_555
conn2 metalc A CYS 2 SG 1_555 B MET 1 SD 1_555
#
loop_
_struct_conn_type.id
disulf
metalc
#
loop_
_atom_type.symbol
C
N
S
#
_cell.entry_id      5XYZ
_cell.length_a       10.000
_cell.length_b       10.000
_cell.length_c       10.000
_cell.angle_alpha    90.00
_cell.angle_beta     90.00
_cell.angle_gamma    90.00
#
_symmetry.entry_id              5XYZ
_symmetry.space_group_name_H-M  'P 1'
#
loop_
_audit_author.name
'Doe, J.'
#
"""

_CP2_RAW_BYTES = _MINIMAL_CIF.encode("utf-8")
_CP2_RAW_SHA = hashlib.sha256(_CP2_RAW_BYTES).hexdigest()
_CP2_RAW_SIZE = len(_CP2_RAW_BYTES)


def test_real_gemmi_version_pinned():
    assert gemmi.__version__ == "0.7.5"


def test_parse_cif_dom021_extracts_real_categories():  # T021-061/063/064/067
    dom = parse_receptor_cif_dom021(_MINIMAL_CIF.encode("utf-8"))
    assert dom.parser_name == "gemmi"
    assert dom.parser_version == "0.7.5"
    assert "atom_site" in dom.categories
    assert dom.categories["atom_site"].row_count == 5
    assert raw_cif_null_to_optional_str(dom.categories["atom_site"].get(0, "pdbx_pdb_ins_code")) is None
    assert dom.categories["atom_site"].get(0, "id") == "1"


def test_parse_cif_dom021_rejects_bom():
    with pytest.raises(InvalidCifEncodingError):
        parse_receptor_cif_dom021(b"\xef\xbb\xbf" + _MINIMAL_CIF.encode("utf-8"))


def test_parse_cif_dom021_rejects_save_frames():
    text = _MINIMAL_CIF + "\nsave_frame_x\n_x.y 1\nsave_\n"
    with pytest.raises(CifSaveFrameRejectedError):
        parse_receptor_cif_dom021(text.encode("utf-8"))


def test_parse_cif_dom021_rejects_multiple_blocks():
    text = _MINIMAL_CIF + "\ndata_SECOND\n_entry.id SECOND\n"
    with pytest.raises(MultipleDataBlocksError):
        parse_receptor_cif_dom021(text.encode("utf-8"))


def test_parse_cif_dom021_rejects_duplicate_canonical_tag():  # T021-087
    # Gemmi's own `read_string` parser raises a native `RuntimeError` for a
    # duplicate tag *during parsing*, before this function's own explicit
    # `document.check_for_duplicates()` call ever runs - so the rejection is
    # real and immediate, just surfaced as `ReceptorParseFailureError`
    # (the generic parse-failure wrapper) rather than the dedicated
    # `DuplicateCanonicalTagError`, which stays defense-in-depth for a
    # duplicate Gemmi's own parser does not itself catch (e.g. across two
    # distinctly-cased spellings of the same canonical tag).
    text = "data_X\n_entry.id X\n_entry.id Y\n"
    with pytest.raises(ReceptorParseFailureError):
        parse_receptor_cif_dom021(text.encode("utf-8"))


def test_parse_cif_dom021_rejects_too_many_tags_per_category():
    columns = "\n".join(f"_x.col{i}" for i in range(200))
    row = " ".join(str(i) for i in range(200))
    text = f"data_X\nloop_\n{columns}\n{row}\n"
    with pytest.raises(CifStructuralLimitExceededError):
        parse_receptor_cif_dom021(text.encode("utf-8"))


def test_serialize_and_reparse_round_trip_with_real_gemmi():
    output = serialize_cif_document021(
        block_name="tbmp_test", categories={"entry": {"id": ["TBMP_TEST"]}}, category_order=("entry",),
    )
    reparsed = reparse_and_extract_cif_dom021(output)
    assert reparsed.categories["entry"].get(0, "id") == "TBMP_TEST"


def test_materialize_receptor_cif021_real_gemmi_end_to_end():  # T021-090 (binding-real)
    raw_bytes = _MINIMAL_CIF.encode("utf-8")
    result = materialize_receptor_cif021(
        source_context=_cp2_source_context(), resolved_selection=_cp2_resolution(),
        unaddressed_components="preserve", raw_bytes=raw_bytes,
    )
    plan = result.plan

    atom_by_id = {d.atom_id: d for d in plan.atom_decisions}
    assert atom_by_id["1"].retained is True
    assert atom_by_id["2"].retained is True
    assert atom_by_id["5"].retained is False
    assert atom_by_id["5"].reason == "chain-complement"

    conn_by_id = {d.id: d for d in plan.connection_decisions}
    assert conn_by_id["conn1"].action == "preserve"
    assert conn_by_id["conn1"].symmetry_class == "both-identity"
    assert conn_by_id["conn2"].action == "omit"
    assert conn_by_id["conn2"].reason == "whole-partner-removed-metalc"

    cat_by_name = {d.category: d for d in plan.category_decisions}
    assert cat_by_name["_atom_site"].kept_row_ordinals == (0, 1, 2, 3)
    assert cat_by_name["_struct_conn"].kept_row_ordinals == (0,)
    assert cat_by_name["_chem_comp"].kept_row_ordinals == (0, 1)
    assert cat_by_name["_audit_author"].action == "omit"
    assert cat_by_name["_entry"].action == "generate"

    # Independent reparse of the real serialized bytes (Section 18) - never
    # trusting the writer/reader of the same library alone.
    reparsed_atom_site = result.reparsed.categories["atom_site"]
    assert reparsed_atom_site.row_count == 4
    reparsed_entry = result.reparsed.categories["entry"]
    assert reparsed_entry.get(0, "id") == f"tbmp_{plan.plan_key}"  # C-06: lowercase tbmp_ prefix
    assert result.output_bytes.startswith(f"data_tbmp_{plan.plan_key}".encode("utf-8"))


def test_materialize_receptor_cif021_is_deterministic_across_runs():  # Section 13: same bytes -> same plan_key
    raw_bytes = _MINIMAL_CIF.encode("utf-8")
    result_a = materialize_receptor_cif021(
        source_context=_cp2_source_context(), resolved_selection=_cp2_resolution(),
        unaddressed_components="preserve", raw_bytes=raw_bytes,
    )
    result_b = materialize_receptor_cif021(
        source_context=_cp2_source_context(), resolved_selection=_cp2_resolution(),
        unaddressed_components="preserve", raw_bytes=raw_bytes,
    )
    assert result_a.plan.plan_key == result_b.plan.plan_key
    assert result_a.output_bytes == result_b.output_bytes


# ==========================================================================
# C-07 / real fixtures (Section 23): six real, publicly deposited PDB
# entries (never a hand-authored synthetic file), downloaded verbatim and
# documented in tests/fixtures/receptor_materialization/*/PROVENANCE.json
# - origin URL, retrieval date, SHA-256 of the untouched original, and
# (where applicable) the exact, listed edit producing the `_trimmed.cif`
# derivative actually fed to Gemmi. Every checksum/size below is the real
# digest/length of those bytes, never a placeholder.
# ==========================================================================

_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "receptor_materialization"


def _real_fixture_bytes(subdir: str, filename: str) -> bytes:
    return (_FIXTURES_ROOT / subdir / filename).read_bytes()


def _real_fixture_materialize(
    *, pdbid: str, raw_bytes: bytes, resolved_model: int, resolved_chains,
    model_basis: str = "only-model-present", unaddressed: str = "preserve", dispositions=(),
):
    protein_id = f"protein-{pdbid.lower()}"
    selection_id = f"sel-{pdbid.lower()}"
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    inspection_id = compute_inspection_id(
        parser_name="gemmi", parser_version="0.7.5", protein_input_id=protein_id,
        raw_checksum=raw_sha, raw_format="mmcif",
    )
    protein_input = ProteinInput(
        id=protein_id, name=pdbid,
        acquisition=AcquisitionRecord(
            method="uploaded", acquired_at=datetime.now(timezone.utc),
            raw_artifact=RawArtifact(path=pdbid, format="mmcif", checksum=raw_sha),
        ),
    )
    report = ReceptorInspectionReport(
        inspection_id=inspection_id, inspection_schema_version="1.0", inspection_profile_version="1.0",
        protein_input_id=protein_id, raw_checksum=raw_sha, raw_format_declared="mmcif",
        structure_format="mmcif", parser=ParserProvenance(name="gemmi", version="0.7.5"),
        selection_capability="full", models=(ModelDescriptor(model_number=resolved_model),), assemblies=(),
        entities=(), chains=(), altloc_observations=(), occupancy_observations=(), missing_residues=(),
        missing_atoms=(), disulfide_bonds=(), covalent_connections=(), metal_coordinations=(),
        non_polymer_components=(), explicit_hydrogens_present=False, explicit_hydrogen_count=0,
        unavailable_categories=(), warnings=(),
    )
    policy = ReceptorSelectionPolicy(
        schema_version="1.0", selection_id=selection_id, protein_input_id=protein_id,
        inspection_id=inspection_id, raw_checksum_observed=raw_sha, declared_by="s",
        declared_at=datetime.now(timezone.utc),
    )
    resolution = ResolvedReceptorSelection(
        schema_version="1.0", resolved_selection_id=selection_id, policy_selection_id=selection_id,
        protein_input_id=protein_id, inspection_id=inspection_id, raw_checksum=raw_sha,
        structure_format="mmcif", parser=ParserProvenance(name="gemmi", version="0.7.5"),
        resolved_model=resolved_model, model_resolution_basis=model_basis, resolved_assembly=None,
        resolved_chains=resolved_chains, resolved_component_dispositions=dispositions,
        resolved_at=datetime.now(timezone.utc),
    )
    records = (
        RecordSnapshot(kind="protein", id=protein_id, sha256="1" * 64, size_bytes=1, schema_version=None),
        RecordSnapshot(kind="inspection", id=inspection_id, sha256="2" * 64, size_bytes=1, schema_version="1.0"),
        RecordSnapshot(kind="selection", id=selection_id, sha256="3" * 64, size_bytes=1, schema_version="1.0"),
        RecordSnapshot(kind="resolution", id=selection_id, sha256="4" * 64, size_bytes=1, schema_version="1.0"),
    )
    source_context = build_source_context(
        protein_input=protein_input, inspection_report=report, policy=policy,
        resolved_selection=resolution, raw_size_bytes=len(raw_bytes), records=records,
    )
    return materialize_receptor_cif021(
        source_context=source_context, resolved_selection=resolution,
        unaddressed_components=unaddressed, raw_bytes=raw_bytes,
    )


def _remove_dispositions(pairs, classification):
    return tuple(
        ResolvedComponentDisposition(label_asym_id=asym, ccd_code=ccd, classification=classification, disposition="remove")
        for asym, ccd in pairs
    )


def test_real_fixture_5i55_positive_materialization():  # T021-090/141/147, real X-ray deposition
    raw_bytes = _real_fixture_bytes("5i55", "5I55_trimmed.cif")
    assert hashlib.sha256(raw_bytes).hexdigest() == "eba9584da11b29537c0ec30f23fa01fd27ad5ccc96ce8649450eb457530b9567"
    result = _real_fixture_materialize(
        pdbid="5I55", raw_bytes=raw_bytes, resolved_model=1,
        resolved_chains=(ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=1),),
    )
    assert len(result.plan.atom_decisions) == 218
    assert all(d.retained for d in result.plan.atom_decisions)
    conn_by_id = {d.id: d for d in result.plan.connection_decisions}
    assert conn_by_id["covale1"].action == "preserve"
    # T021-141: the real deposition declares ptnr1_symmetry=ptnr2_symmetry=
    # '1_555' with no _symmetry_equiv/_space_group_symop tables present -
    # the pdbx-1_555-convention recognized without fabricating an operator.
    assert conn_by_id["covale1"].identity_basis == ("pdbx-1_555-convention", "pdbx-1_555-convention")
    assert conn_by_id["covale1"].symmetry_class == "both-identity"
    cat_by_name = {d.category: d for d in result.plan.category_decisions}
    assert cat_by_name["_chem_comp"].kept_row_ordinals
    reparsed_atom_site = result.reparsed.categories["atom_site"]
    assert reparsed_atom_site.row_count == 218


def test_real_fixture_1pfe_rejects_hydrog_nonidentity_symmetry_gate():  # T021-144/147, real negative gate
    raw_bytes = _real_fixture_bytes("1pfe", "1PFE_trimmed.cif")
    assert hashlib.sha256(raw_bytes).hexdigest() == "e0653ce8e25e4ef8cc5e37b08d6a4f6cab19eb6928e18e462a3a91f83139dc73"
    dispositions = (
        _remove_dispositions([("C", "CL")], "other-non-polymer")
        + _remove_dispositions([("D", "QUI"), ("E", "QUI")], "other-non-polymer")
        + _remove_dispositions([("F", "HOH"), ("G", "HOH")], "water")
    )
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _real_fixture_materialize(
            pdbid="1PFE", raw_bytes=raw_bytes, resolved_model=1,
            resolved_chains=(ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=1),),
            dispositions=dispositions,
        )
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"


def test_real_fixture_3dg1_positive_materialization_with_real_special_symmetry():  # T021-147/158
    raw_bytes = _real_fixture_bytes("3dg1", "3DG1_trimmed.cif")
    assert hashlib.sha256(raw_bytes).hexdigest() == "49926dfc5f524634bcc8e5c7b23aba352ce0ac32431f3621b02e022b267ebf58"
    result = _real_fixture_materialize(
        pdbid="3DG1", raw_bytes=raw_bytes, resolved_model=1,
        resolved_chains=(ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=1),),
    )
    assert len(result.plan.atom_decisions) == 41
    assert all(d.retained for d in result.plan.atom_decisions)
    cat_by_name = {d.category: d for d in result.plan.category_decisions}
    assert cat_by_name["_pdbx_struct_special_symmetry"].action == "guard_then_omit"
    assert "special-position-annotations-omitted" in result.plan.plan.warnings


def test_real_fixture_2jof_nmr_selects_exact_model_not_first():  # T021-149/162, real multi-model NMR
    raw_bytes = _real_fixture_bytes("nmr_2jof", "2JOF_trimmed.cif")
    assert hashlib.sha256(raw_bytes).hexdigest() == "837023fd1b0fc1bf5676a24fcc56f97c50346e6d5871ea65f1c6d76d98144c9b"
    result = _real_fixture_materialize(
        pdbid="2JOF", raw_bytes=raw_bytes, resolved_model=3,
        resolved_chains=(ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=3),),
        model_basis="explicitly-selected",
    )
    assert len(result.plan.atom_decisions) == 7952  # every real atom_site row, across all 28 models
    assert "other-models-excluded" in result.plan.plan.warnings
    # None of the 27 other real conformers' atoms are retained.
    retained_count = sum(1 for d in result.plan.atom_decisions if d.retained)
    assert retained_count == 284


def test_real_fixture_7dwu_em_positive_materialization():  # T021-149, real cryo-EM deposition
    raw_bytes = _real_fixture_bytes("em_7dwu", "7DWU_trimmed.cif")
    assert hashlib.sha256(raw_bytes).hexdigest() == "3da1fed528cee1e0c4db53b47f0155aceaf012bd9a334d0f8219929eefa22e8e"
    result = _real_fixture_materialize(
        pdbid="7DWU", raw_bytes=raw_bytes, resolved_model=1,
        resolved_chains=(
            ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=1),
            ResolvedChainSelection(label_asym_id="D", label_entity_id="2", model_number=1),
            ResolvedChainSelection(label_asym_id="E", label_entity_id="3", model_number=1),
        ),
    )
    retained_count = sum(1 for decision in result.plan.atom_decisions if decision.retained)
    assert retained_count > 0
    cat_by_name = {d.category: d for d in result.plan.category_decisions}
    assert "_pdbx_unobs_or_zero_occ_residues" in cat_by_name


def test_real_fixture_1a2p_ncs_given_rows_resolved_from_real_deposition():  # T021-150-154, real NCS operator values
    raw_bytes = _real_fixture_bytes("ncs_1a2p", "1A2P_trimmed.cif")
    assert hashlib.sha256(raw_bytes).hexdigest() == "7a6f3d3e0f958d6be34a4b28fe869d024b3cdadf409d821b70dcde6a68b53431"
    dom = parse_receptor_cif_dom021(raw_bytes)
    ncs = dom.categories["struct_ncs_oper"]
    assert ncs.row_count == 2
    assert [raw_cif_null_to_optional_str(ncs.get(i, "code")) for i in range(2)] == ["given", "given"]

    from molecular.receptor_materialization import RawCategoryTable, _guard_ncs_row  # pure-domain guard, real tokens

    columns = {column: tuple(ncs.get(i, column) for i in range(ncs.row_count)) for column in ncs.columns}
    table = RawCategoryTable(columns=columns)
    for row_index in range(2):
        _guard_ncs_row("_struct_ncs_oper", table, row_index)  # must not raise for either real row


def test_real_fixture_1a2p_full_pipeline_blocked_by_unrelated_real_symmetry_operator():  # honest limitation, documented in PROVENANCE.json
    raw_bytes = _real_fixture_bytes("ncs_1a2p", "1A2P_trimmed.cif")
    dispositions = (
        _remove_dispositions([("D", "ZN"), ("E", "ZN"), ("F", "ZN")], "metal-ion")
        + _remove_dispositions([("G", "HOH"), ("H", "HOH"), ("I", "HOH")], "water")
    )
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _real_fixture_materialize(
            pdbid="1A2P", raw_bytes=raw_bytes, resolved_model=1,
            resolved_chains=(ResolvedChainSelection(label_asym_id="A", label_entity_id="1", model_number=1),),
            dispositions=dispositions,
        )
    assert excinfo.value.code == "UNSUPPORTED_SYMMETRY"  # metalc2 (A-D, ptnr1_symmetry=3_675), not NCS-related


def test_raw_vs_quoted_null_marker_distinction_is_preserved():  # T021-041
    # A bare, unquoted '.'/'?' is a real CIF null; Gemmi's own raw=True
    # category access never de-quotes a value, so a *quoted* '.'/'?'
    # string (gemmi.cif.as_string() strips the quote marks but leaves the
    # content) remains the literal text "." / "?" after cooking - the
    # caller must tell these apart by whether the token was quoted in the
    # source, not merely by comparing the decoded string to '.'/'?' again.
    text = "data_x\nloop_\n_x.a\n_x.b\n. '.'\n"
    doc = gemmi.cif.read_string(text)
    block = doc.sole_block()
    raw = block.get_mmcif_category("_x.", raw=True)
    assert raw["a"][0] == "."  # unquoted null marker, raw token is the bare dot
    assert raw["b"][0] == "'.'"  # quoted literal string, raw token keeps its quotes
    # The null check must run on the *raw* (still-quoted) token, never on an
    # already as_string()-decoded one - decoding a quoted "'.'" first would
    # strip the very quote marks that distinguish it from a real null,
    # collapsing both to the same bare "." and losing the distinction.
    assert raw_cif_null_to_optional_str(raw["a"][0]) is None
    non_null = raw_cif_null_to_optional_str(raw["b"][0])
    assert non_null == "'.'"  # still quoted - correctly not treated as null
    assert gemmi.cif.as_string(non_null) == "."  # decode only after the null check confirms a real token


def test_loss_audit_raw_bytes_unchanged_and_every_omission_hashed():  # T021-088
    raw_bytes = _MINIMAL_CIF.encode("utf-8")
    raw_bytes_copy = bytes(raw_bytes)
    result = materialize_receptor_cif021(
        source_context=_cp2_source_context(), resolved_selection=_cp2_resolution(),
        unaddressed_components="preserve", raw_bytes=raw_bytes,
    )
    assert raw_bytes == raw_bytes_copy  # the adapter never mutates the RAW bytes it was handed
    for decision in result.plan.category_decisions:
        if decision.action in ("omit", "guard_then_omit", "filter"):
            assert decision.source_rows_key is not None  # every real/omitted category's rows are hashed, never a null audit


def test_output_verification_rejects_mutated_token_before_commit():  # T021-089
    decision = CategoryDecision(
        category="_atom_site", action="filter", kept_row_ordinals=(0,), omitted_row_ordinals=(),
        preserved_tags=("id", "cartn_x"), source_rows_key="1" * 64,
    )
    output_categories = {"atom_site": {"id": ["REAL_ID"], "cartn_x": ["1.0"]}}
    from molecular.receptor_parsing import ParsedCifDom021, RawCifCategory, _verify_output_reparse
    mutated = ParsedCifDom021(
        category_names=("atom_site",),
        categories={"atom_site": RawCifCategory(name="atom_site", row_count=1, columns={"id": ("CHANGED_ID",), "cartn_x": ("NaN",)})},
        parser_name="gemmi", parser_version="0.7.5",
    )
    with pytest.raises(MaterializationExecutionError) as excinfo:
        _verify_output_reparse(
            category_decisions=(decision,), entry_id="tbmp_test", output_categories=output_categories, reparsed=mutated,
        )
    assert excinfo.value.code == "INVALID_OUTPUT"


def test_real_fixture_provenance_manifests_match_actual_file_hashes():  # T021-162
    import json
    for subdir, filenames in [
        ("5i55", ["5I55.cif", "5I55_trimmed.cif"]),
        ("1pfe", ["1PFE.cif", "1PFE_trimmed.cif"]),
        ("3dg1", ["3DG1.cif", "3DG1_trimmed.cif"]),
        ("nmr_2jof", ["2JOF.cif", "2JOF_trimmed.cif"]),
        ("em_7dwu", ["7DWU.cif", "7DWU_trimmed.cif"]),
        ("ncs_1a2p", ["1A2P.cif", "1A2P_trimmed.cif"]),
    ]:
        manifest = json.loads((_FIXTURES_ROOT / subdir / "PROVENANCE.json").read_text(encoding="utf-8"))
        for filename in filenames:
            real_bytes = (_FIXTURES_ROOT / subdir / filename).read_bytes()
            recorded = manifest["files"][filename]
            assert hashlib.sha256(real_bytes).hexdigest() == recorded["sha256"], (subdir, filename)
            assert len(real_bytes) == recorded["size_bytes"], (subdir, filename)
