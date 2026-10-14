"""Gemmi adapter - the only module in this repository that imports gemmi.

Implements the Increment 020 frozen final specification's ten-step safe
read flow (Section 11.3) and translates the parsed content into a plain,
domain-vocabulary `ParsedReceptorDocument` - no `gemmi.*` type ever crosses
this module's public boundary (Section 9). `parse_receptor_raw` is the
single public entry point (Section 10.5); every other module that needs
parsed receptor content goes through it, never through `import gemmi`
directly.

`Gemmi.setup_entities()` (and its four internal operations -
`add_entity_types()`, `assign_subchains()`, `ensure_entities()`,
`deduplicate_entities()`) and the mutating convenience methods
(`remove_hydrogens()`, `remove_waters()`, `remove_ligands_and_waters()`,
`remove_alternative_conformations()`) are never called anywhere in this
module (Section 10.3) - entity/chain identity is read only from what the
source format itself declares (mmCIF's `_entity`/`_atom_site.label_*`
categories, read via gemmi's low-level, uninterpreted category access; or,
for legacy PDB, left `None`/`"unavailable"` rather than guessed).

`KeyboardInterrupt`/`SystemExit` are never caught by any translation block
below; every translated exception preserves the original via
`raise ... from error` (Section 10.3).
"""

import dataclasses
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional, Tuple

import gemmi

from molecular.atomic import compute_sha256

MAX_RAW_ARTIFACT_SIZE_BYTES = 1024**3  # 1 GiB (Section 32.2)

_COMPRESSED_EXTENSIONS = frozenset({".gz", ".bz2", ".xz", ".zip"})

_MAGIC_BYTES_BY_FORMAT = (
    ("gzip", b"\x1f\x8b"),
    ("bzip2", b"BZh"),
    ("xz", b"\xfd7zXZ\x00"),
    ("zip", b"PK\x03\x04"),
)
_MAGIC_BYTES_READ_LENGTH = 6

StructureFormat = str  # "mmcif" | "pdb" - Section 13.2's StructureFormat lives in receptor_inspection.py


class RawArtifactUnavailableError(OSError):
    """The RAW artifact could not be opened for reading (Section 30.1)."""


class RawArtifactTooLargeError(OSError):
    """The RAW artifact's size on disk exceeds the 1 GiB ceiling (Section 32.2),
    raised before any content beyond `stat()` is read."""


class CompressedExtensionRejectedError(ValueError):
    """The RAW artifact's declared extension is one of the closed,
    casefold-compared list of rejected compressed extensions (Section 11.2),
    raised before the file is ever opened - unconditionally, even when the
    content turns out not to actually be compressed (T-103)."""


class CompressedRawContentError(ValueError):
    """The RAW artifact's actual content begins with a known compression
    magic-byte signature, regardless of its declared extension (Section
    11.2)."""


class RawChecksumMismatchError(ValueError):
    """The SHA-256 computed over the RAW artifact's bytes does not match
    the checksum the caller expected (Section 11.3, step 9)."""


class UnsupportedStructureFormatError(ValueError):
    """The RAW artifact's content is not recognizable as either supported
    structure format (mmCIF or legacy PDB) - content-based, never decided
    from the declared extension alone (Section 11.1)."""


class ReceptorParseFailureError(RuntimeError):
    """Gemmi raised while opening or interpreting the RAW artifact's
    content (malformed syntax, truncated file, invalid encoding). The
    original exception is always preserved as `__cause__`; no partial
    report is ever produced on this path (Section 10.8 of the Draft,
    preserved)."""


class StructConnMalformedError(RuntimeError):
    """An `_struct_conn` row declares an absent or empty `conn_type_id`
    (Section 18.3) - never silently discarded, never guessed."""


@dataclass(frozen=True)
class ParsedModel:
    model_number: int


@dataclass(frozen=True)
class ParsedAssemblyOperation:
    expression: str
    asym_id_list: Tuple[str, ...]


@dataclass(frozen=True)
class ParsedAssembly:
    assembly_id: str
    details: Optional[str]
    generation: Tuple[ParsedAssemblyOperation, ...]


@dataclass(frozen=True)
class ParsedEntity:
    label_entity_id: Optional[str]
    entity_type_raw: Optional[str]
    description: Optional[str]


@dataclass(frozen=True)
class ParsedAtom:
    atom_name: str
    altloc: Optional[str]
    occupancy: Decimal
    element: str
    serial: Optional[int]


@dataclass(frozen=True)
class ParsedResidue:
    model_number: int
    label_asym_id: Optional[str]
    label_entity_id: Optional[str]
    auth_asym_id: Optional[str]
    auth_seq_id: Optional[str]
    insertion_code: Optional[str]
    label_seq_id: Optional[str]
    comp_id: str
    is_het: bool
    atoms: Tuple[ParsedAtom, ...]


@dataclass(frozen=True)
class ParsedConnectionEndpoint:
    label_asym_id: Optional[str]
    auth_asym_id: Optional[str]
    label_entity_id: Optional[str]
    label_seq_id: Optional[str]
    auth_seq_id: Optional[str]
    insertion_code: Optional[str]
    label_comp_id: Optional[str]
    auth_comp_id: Optional[str]
    label_atom_id: Optional[str]
    auth_atom_id: Optional[str]
    label_alt_id: Optional[str]
    symmetry: Optional[str]
    model_number: Optional[int]


@dataclass(frozen=True)
class ParsedConnection:
    struct_conn_id: Optional[str]
    conn_type: str
    source_category: str
    partner_a: ParsedConnectionEndpoint
    partner_b: ParsedConnectionEndpoint
    distance: Optional[Decimal]
    value_order: Optional[str]
    role: Optional[str]
    details: Optional[str]
    leaving_atom_flag: Optional[str]


@dataclass(frozen=True)
class ParsedMissingResidue:
    model_number: int
    label_asym_id: Optional[str]
    auth_asym_id: Optional[str]
    residue_identifier: str
    residue_name: Optional[str]


@dataclass(frozen=True)
class ParsedMissingAtom:
    model_number: int
    label_asym_id: Optional[str]
    auth_asym_id: Optional[str]
    residue_identifier: str
    atom_name: str


@dataclass(frozen=True)
class ParsedReceptorDocument:
    """Domain-vocabulary transcription of a parsed receptor RAW artifact -
    no `gemmi.*` type anywhere in this tree (Section 9)."""

    structure_format: StructureFormat
    raw_checksum: str
    parser_name: str
    parser_version: str
    models: Tuple[ParsedModel, ...]
    assemblies: Tuple[ParsedAssembly, ...]
    entities: Tuple[ParsedEntity, ...]
    residues: Tuple[ParsedResidue, ...]
    connections: Tuple[ParsedConnection, ...]
    missing_residues: Optional[Tuple[ParsedMissingResidue, ...]]
    missing_atoms: Optional[Tuple[ParsedMissingAtom, ...]]
    unavailable_categories: Tuple[str, ...]


def _reject_compressed_extension(raw_path: Path) -> None:
    suffix = raw_path.suffix.casefold()
    if suffix in _COMPRESSED_EXTENSIONS:
        raise CompressedExtensionRejectedError(
            f"RAW artifact extension {raw_path.suffix!r} is a rejected compressed "
            "extension; compressed input is never accepted, regardless of actual content"
        )


def _reject_oversized_raw(raw_path: Path) -> None:
    try:
        size = raw_path.stat().st_size
    except OSError as error:
        raise RawArtifactUnavailableError(
            f"RAW artifact could not be inspected: {error.__class__.__name__}"
        ) from error

    if size > MAX_RAW_ARTIFACT_SIZE_BYTES:
        raise RawArtifactTooLargeError(
            f"RAW artifact size ({size} bytes) exceeds the {MAX_RAW_ARTIFACT_SIZE_BYTES} "
            "byte ceiling"
        )


def _reject_compressed_content(raw_path: Path) -> None:
    try:
        with raw_path.open("rb") as file:
            header = file.read(_MAGIC_BYTES_READ_LENGTH)
    except OSError as error:
        raise RawArtifactUnavailableError(
            f"RAW artifact could not be opened: {error.__class__.__name__}"
        ) from error

    for format_name, signature in _MAGIC_BYTES_BY_FORMAT:
        if header.startswith(signature):
            raise CompressedRawContentError(
                f"RAW artifact content begins with the {format_name} magic bytes; "
                "compressed content is never accepted, regardless of declared extension"
            )


def _compute_and_verify_checksum(raw_path: Path, expected_checksum: str) -> str:
    try:
        computed = compute_sha256(raw_path)
    except OSError as error:
        raise RawArtifactUnavailableError(
            f"RAW artifact could not be read: {error.__class__.__name__}"
        ) from error

    if computed != expected_checksum:
        raise RawChecksumMismatchError(
            "the SHA-256 computed over the RAW artifact's bytes does not match "
            "the checksum already on record for it"
        )

    return computed


def _none_if_null_token(value: str) -> Optional[str]:
    """Collapse gemmi's single-character 'no value' sentinel (NUL) and the
    mmCIF `.`/`?` null tokens to `None` (Section 33.2)."""

    if value in ("\x00", "", " ", ".", "?"):
        return None
    return value


def _cif_token_to_optional_str(raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None:
        return None
    if raw_value in (".", "?"):
        return None
    return gemmi.cif.as_string(raw_value)


def _cif_token_to_decimal(raw_value: Optional[str]) -> Optional[Decimal]:
    text = _cif_token_to_optional_str(raw_value)
    if text is None:
        return None
    try:
        return Decimal(text)
    except InvalidOperation as error:
        raise ReceptorParseFailureError(
            f"a numeric mmCIF field could not be parsed as a decimal: {text!r}"
        ) from error


def _cif_token_to_optional_int(raw_value: Optional[str]) -> Optional[int]:
    text = _cif_token_to_optional_str(raw_value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as error:
        raise ReceptorParseFailureError(
            f"an integer mmCIF field could not be parsed: {text!r}"
        ) from error


class _Category:
    """Uninterpreted, column-existence-safe access to one mmCIF category's
    rows (Section 9/10.1's 'low-level, uninterpreted category access').

    `gemmi.cif.Block.find()` silently returns zero rows if even ONE of the
    requested tags is not a column of the category anywhere in the file -
    confirmed empirically against gemmi 0.7.5 - so each column is fetched
    independently via `find_values()` (which safely reports absence for
    that one column alone) and zipped by row index here instead."""

    def __init__(self, block: "gemmi.cif.Block", category: str, columns: Tuple[str, ...]):
        self._row_count = 0
        self._columns = {}
        for column in columns:
            values = block.find_values(f"{category}.{column}")
            if values:
                self._columns[column] = list(values)
                self._row_count = max(self._row_count, len(values))

    def __len__(self) -> int:
        return self._row_count

    def get(self, row_index: int, column: str) -> Optional[str]:
        values = self._columns.get(column)
        if values is None or row_index >= len(values):
            return None
        return _cif_token_to_optional_str(values[row_index])

    def get_raw(self, row_index: int, column: str) -> Optional[str]:
        """Like `get`, but without `.`/`?` collapsing or de-quoting - used
        only where the raw token itself (e.g. a bare `.` vs an absent
        column) must be told apart from a genuinely quoted value."""

        values = self._columns.get(column)
        if values is None or row_index >= len(values):
            return None
        return values[row_index]


def _category_present(block: "gemmi.cif.Block", category: str) -> bool:
    return f"{category}." in block.get_mmcif_category_names()


_ATOM_SITE_COLUMNS = (
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "auth_asym_id",
    "auth_seq_id",
    "auth_atom_id",
    "auth_comp_id",
    "pdbx_PDB_ins_code",
    "occupancy",
    "type_symbol",
    "pdbx_PDB_model_num",
    "id",
)

_STRUCT_CONN_COLUMNS = (
    "id",
    "conn_type_id",
    "ptnr1_label_asym_id",
    "ptnr1_label_entity_id",
    "ptnr1_label_seq_id",
    "ptnr1_label_atom_id",
    "ptnr1_label_comp_id",
    "ptnr1_auth_asym_id",
    "ptnr1_auth_seq_id",
    "ptnr1_auth_atom_id",
    "ptnr1_auth_comp_id",
    "ptnr1_PDB_ins_code",
    "ptnr1_symmetry",
    "pdbx_ptnr1_label_alt_id",
    "ptnr2_label_asym_id",
    "ptnr2_label_entity_id",
    "ptnr2_label_seq_id",
    "ptnr2_label_atom_id",
    "ptnr2_label_comp_id",
    "ptnr2_auth_asym_id",
    "ptnr2_auth_seq_id",
    "ptnr2_auth_atom_id",
    "ptnr2_auth_comp_id",
    "ptnr2_PDB_ins_code",
    "ptnr2_symmetry",
    "pdbx_ptnr2_label_alt_id",
    "pdbx_dist_value",
    "pdbx_value_order",
    "pdbx_role",
    "details",
    "pdbx_leaving_atom_flag",
)

_UNOBS_RESIDUES_COLUMNS = (
    "PDB_model_num",
    "label_asym_id",
    "auth_asym_id",
    "auth_seq_id",
    "auth_comp_id",
)

_UNOBS_ATOMS_COLUMNS = (
    "PDB_model_num",
    "label_asym_id",
    "auth_asym_id",
    "auth_seq_id",
    "auth_comp_id",
    "auth_atom_id",
)

_ENTITY_COLUMNS = ("id", "type", "pdbx_description")

_ASSEMBLY_COLUMNS = ("id", "details")

_ASSEMBLY_GEN_COLUMNS = ("assembly_id", "oper_expression", "asym_id_list")


def _residue_identifier(auth_seq_id: Optional[str], insertion_code: Optional[str]) -> str:
    if insertion_code:
        return f"{auth_seq_id}{insertion_code}"
    return str(auth_seq_id)


def _parse_mmcif_document(block: "gemmi.cif.Block") -> Tuple[
    Tuple[ParsedAssembly, ...],
    Tuple[ParsedEntity, ...],
    Tuple[ParsedResidue, ...],
    Tuple[ParsedConnection, ...],
    Optional[Tuple[ParsedMissingResidue, ...]],
    Optional[Tuple[ParsedMissingAtom, ...]],
    Tuple[str, ...],
]:
    unavailable_categories = []

    entities: Tuple[ParsedEntity, ...] = ()
    if _category_present(block, "_entity"):
        entity_cat = _Category(block, "_entity", _ENTITY_COLUMNS)
        entities = tuple(
            ParsedEntity(
                label_entity_id=entity_cat.get(i, "id"),
                entity_type_raw=entity_cat.get(i, "type"),
                description=entity_cat.get(i, "pdbx_description"),
            )
            for i in range(len(entity_cat))
        )

    assemblies: Tuple[ParsedAssembly, ...] = ()
    if _category_present(block, "_pdbx_struct_assembly"):
        assembly_cat = _Category(block, "_pdbx_struct_assembly", _ASSEMBLY_COLUMNS)
        gen_cat = (
            _Category(block, "_pdbx_struct_assembly_gen", _ASSEMBLY_GEN_COLUMNS)
            if _category_present(block, "_pdbx_struct_assembly_gen")
            else None
        )

        parsed_assemblies = []
        for i in range(len(assembly_cat)):
            assembly_id = assembly_cat.get(i, "id")
            generation = []
            if gen_cat is not None:
                for j in range(len(gen_cat)):
                    if gen_cat.get(j, "assembly_id") != assembly_id:
                        continue
                    expression = gen_cat.get(j, "oper_expression") or ""
                    asym_list_raw = gen_cat.get(j, "asym_id_list") or ""
                    asym_id_list = tuple(
                        token.strip() for token in asym_list_raw.split(",") if token.strip()
                    )
                    generation.append(
                        ParsedAssemblyOperation(expression=expression, asym_id_list=asym_id_list)
                    )
            parsed_assemblies.append(
                ParsedAssembly(
                    assembly_id=assembly_id or "",
                    details=assembly_cat.get(i, "details"),
                    generation=tuple(generation),
                )
            )
        assemblies = tuple(parsed_assemblies)
    else:
        unavailable_categories.append("pdbx_struct_assembly")

    if _category_present(block, "_struct_conn"):
        struct_conn_cat = _Category(block, "_struct_conn", _STRUCT_CONN_COLUMNS)
        connections = []
        for i in range(len(struct_conn_cat)):
            conn_type = struct_conn_cat.get(i, "conn_type_id")
            if not conn_type:
                raise StructConnMalformedError(
                    f"_struct_conn row {i} has an absent or empty conn_type_id"
                )
            partner_a = ParsedConnectionEndpoint(
                label_asym_id=struct_conn_cat.get(i, "ptnr1_label_asym_id"),
                auth_asym_id=struct_conn_cat.get(i, "ptnr1_auth_asym_id"),
                label_entity_id=struct_conn_cat.get(i, "ptnr1_label_entity_id"),
                label_seq_id=struct_conn_cat.get(i, "ptnr1_label_seq_id"),
                auth_seq_id=struct_conn_cat.get(i, "ptnr1_auth_seq_id"),
                insertion_code=struct_conn_cat.get(i, "ptnr1_PDB_ins_code"),
                label_comp_id=struct_conn_cat.get(i, "ptnr1_label_comp_id"),
                auth_comp_id=struct_conn_cat.get(i, "ptnr1_auth_comp_id"),
                label_atom_id=struct_conn_cat.get(i, "ptnr1_label_atom_id"),
                auth_atom_id=struct_conn_cat.get(i, "ptnr1_auth_atom_id"),
                label_alt_id=struct_conn_cat.get(i, "pdbx_ptnr1_label_alt_id"),
                symmetry=struct_conn_cat.get(i, "ptnr1_symmetry"),
                model_number=None,
            )
            partner_b = ParsedConnectionEndpoint(
                label_asym_id=struct_conn_cat.get(i, "ptnr2_label_asym_id"),
                auth_asym_id=struct_conn_cat.get(i, "ptnr2_auth_asym_id"),
                label_entity_id=struct_conn_cat.get(i, "ptnr2_label_entity_id"),
                label_seq_id=struct_conn_cat.get(i, "ptnr2_label_seq_id"),
                auth_seq_id=struct_conn_cat.get(i, "ptnr2_auth_seq_id"),
                insertion_code=struct_conn_cat.get(i, "ptnr2_PDB_ins_code"),
                label_comp_id=struct_conn_cat.get(i, "ptnr2_label_comp_id"),
                auth_comp_id=struct_conn_cat.get(i, "ptnr2_auth_comp_id"),
                label_atom_id=struct_conn_cat.get(i, "ptnr2_label_atom_id"),
                auth_atom_id=struct_conn_cat.get(i, "ptnr2_auth_atom_id"),
                label_alt_id=struct_conn_cat.get(i, "pdbx_ptnr2_label_alt_id"),
                symmetry=struct_conn_cat.get(i, "ptnr2_symmetry"),
                model_number=None,
            )
            connections.append(
                ParsedConnection(
                    struct_conn_id=struct_conn_cat.get(i, "id"),
                    conn_type=conn_type,
                    source_category="struct_conn",
                    partner_a=partner_a,
                    partner_b=partner_b,
                    distance=_cif_token_to_decimal(struct_conn_cat.get_raw(i, "pdbx_dist_value")),
                    value_order=struct_conn_cat.get(i, "pdbx_value_order"),
                    role=struct_conn_cat.get(i, "pdbx_role"),
                    details=struct_conn_cat.get(i, "details"),
                    leaving_atom_flag=struct_conn_cat.get(i, "pdbx_leaving_atom_flag"),
                )
            )
        connections = tuple(connections)
    else:
        connections = ()
        unavailable_categories.append("struct_conn")

    if _category_present(block, "_pdbx_unobs_or_zero_occ_residues"):
        unobs_res_cat = _Category(
            block, "_pdbx_unobs_or_zero_occ_residues", _UNOBS_RESIDUES_COLUMNS
        )
        missing_residues = tuple(
            ParsedMissingResidue(
                model_number=_cif_token_to_optional_int(unobs_res_cat.get_raw(i, "PDB_model_num"))
                or 1,
                label_asym_id=unobs_res_cat.get(i, "label_asym_id"),
                auth_asym_id=unobs_res_cat.get(i, "auth_asym_id"),
                residue_identifier=_residue_identifier(
                    unobs_res_cat.get(i, "auth_seq_id"), None
                ),
                residue_name=unobs_res_cat.get(i, "auth_comp_id"),
            )
            for i in range(len(unobs_res_cat))
        )
    else:
        missing_residues = None
        unavailable_categories.append("pdbx_unobs_or_zero_occ_residues")

    if _category_present(block, "_pdbx_unobs_or_zero_occ_atoms"):
        unobs_atom_cat = _Category(block, "_pdbx_unobs_or_zero_occ_atoms", _UNOBS_ATOMS_COLUMNS)
        missing_atoms = tuple(
            ParsedMissingAtom(
                model_number=_cif_token_to_optional_int(
                    unobs_atom_cat.get_raw(i, "PDB_model_num")
                )
                or 1,
                label_asym_id=unobs_atom_cat.get(i, "label_asym_id"),
                auth_asym_id=unobs_atom_cat.get(i, "auth_asym_id"),
                residue_identifier=_residue_identifier(
                    unobs_atom_cat.get(i, "auth_seq_id"), None
                ),
                atom_name=unobs_atom_cat.get(i, "auth_atom_id") or "",
            )
            for i in range(len(unobs_atom_cat))
        )
    else:
        missing_atoms = None
        unavailable_categories.append("pdbx_unobs_or_zero_occ_atoms")

    residues: Tuple[ParsedResidue, ...] = ()
    if _category_present(block, "_atom_site"):
        atom_site = _Category(block, "_atom_site", _ATOM_SITE_COLUMNS)
        residues = tuple(_group_atom_site_into_residues(atom_site))

    return (
        assemblies,
        entities,
        residues,
        connections,
        missing_residues,
        missing_atoms,
        tuple(unavailable_categories),
    )


def _group_atom_site_into_residues(atom_site: _Category) -> Tuple[ParsedResidue, ...]:
    groups: "dict[tuple, list]" = {}
    order: list = []

    for i in range(len(atom_site)):
        model_number = _cif_token_to_optional_int(atom_site.get_raw(i, "pdbx_PDB_model_num")) or 1
        label_asym_id = atom_site.get(i, "label_asym_id")
        auth_asym_id = atom_site.get(i, "auth_asym_id")
        auth_seq_id = atom_site.get(i, "auth_seq_id")
        insertion_code = atom_site.get(i, "pdbx_PDB_ins_code")

        key = (model_number, label_asym_id, auth_asym_id, auth_seq_id, insertion_code)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    residues = []
    for key in order:
        model_number, label_asym_id, auth_asym_id, auth_seq_id, insertion_code = key
        row_indices = groups[key]
        first = row_indices[0]

        atoms = tuple(
            ParsedAtom(
                atom_name=atom_site.get(idx, "label_atom_id") or "",
                altloc=atom_site.get(idx, "label_alt_id"),
                occupancy=_cif_token_to_decimal(atom_site.get_raw(idx, "occupancy"))
                or Decimal("1"),
                element=(atom_site.get(idx, "type_symbol") or "").upper(),
                serial=_cif_token_to_optional_int(atom_site.get_raw(idx, "id")),
            )
            for idx in row_indices
        )

        residues.append(
            ParsedResidue(
                model_number=model_number,
                label_asym_id=label_asym_id,
                label_entity_id=atom_site.get(first, "label_entity_id"),
                auth_asym_id=auth_asym_id,
                auth_seq_id=auth_seq_id,
                insertion_code=insertion_code,
                label_seq_id=atom_site.get(first, "label_seq_id"),
                comp_id=atom_site.get(first, "label_comp_id")
                or atom_site.get(first, "auth_comp_id")
                or "",
                is_het=False,
                atoms=atoms,
            )
        )

    return tuple(residues)


def _parse_legacy_pdb_structure(structure: "gemmi.Structure") -> Tuple[
    Tuple[ParsedResidue, ...],
    Tuple[ParsedConnection, ...],
    Tuple[str, ...],
]:
    residues = []
    atom_serial_index: "dict[int, tuple]" = {}

    for model in structure:
        model_number = model.num
        for chain in model:
            for residue in chain:
                atoms = []
                for atom in residue:
                    altloc = _none_if_null_token(atom.altloc)
                    atoms.append(
                        ParsedAtom(
                            atom_name=atom.name,
                            altloc=altloc,
                            occupancy=Decimal(f"{atom.occ:.2f}"),
                            element=atom.element.name.upper(),
                            serial=atom.serial,
                        )
                    )
                    atom_serial_index[atom.serial] = (
                        model_number,
                        chain.name,
                        residue.name,
                        str(residue.seqid.num),
                        _none_if_null_token(residue.seqid.icode),
                        atom.name,
                    )

                residues.append(
                    ParsedResidue(
                        model_number=model_number,
                        label_asym_id=None,
                        label_entity_id=None,
                        auth_asym_id=chain.name,
                        auth_seq_id=str(residue.seqid.num),
                        insertion_code=_none_if_null_token(residue.seqid.icode),
                        label_seq_id=None,
                        comp_id=residue.name,
                        is_het=(residue.het_flag == "H"),
                        atoms=tuple(atoms),
                    )
                )

    connections = []
    for connection in structure.connections:
        source_category = {
            gemmi.ConnectionType.Disulf: "legacy-pdb-ssbond",
        }.get(connection.type, "legacy-pdb-link")

        distance = Decimal(f"{connection.reported_distance:.2f}")
        if distance == Decimal("0.00"):
            distance = None

        def endpoint(partner: "gemmi.AtomAddress") -> ParsedConnectionEndpoint:
            return ParsedConnectionEndpoint(
                label_asym_id=None,
                auth_asym_id=partner.chain_name,
                label_entity_id=None,
                label_seq_id=None,
                auth_seq_id=str(partner.res_id.seqid.num),
                insertion_code=_none_if_null_token(partner.res_id.seqid.icode),
                label_comp_id=None,
                auth_comp_id=partner.res_id.name,
                label_atom_id=None,
                auth_atom_id=partner.atom_name,
                label_alt_id=None,
                symmetry=None,
                model_number=None,
            )

        connections.append(
            ParsedConnection(
                struct_conn_id=connection.name or None,
                conn_type="ssbond" if source_category == "legacy-pdb-ssbond" else "link",
                source_category=source_category,
                partner_a=endpoint(connection.partner1),
                partner_b=endpoint(connection.partner2),
                distance=distance,
                value_order=None,
                role=None,
                details=None,
                leaving_atom_flag=None,
            )
        )

    conect_map = structure.conect_map
    seen_pairs: "set[frozenset]" = set()
    for serial_a, bonded in conect_map.items():
        for serial_b in bonded:
            pair = frozenset({serial_a, serial_b})
            if pair in seen_pairs:
                continue  # CONECT is declared symmetrically; record each undirected pair once
            seen_pairs.add(pair)

            info_a = atom_serial_index.get(serial_a)
            info_b = atom_serial_index.get(serial_b)
            if info_a is None or info_b is None:
                continue

            model_a, chain_a, comp_a, seq_a, icode_a, atom_a = info_a
            model_b, chain_b, comp_b, seq_b, icode_b, atom_b = info_b

            connections.append(
                ParsedConnection(
                    struct_conn_id=None,
                    conn_type="conect",
                    source_category="legacy-pdb-conect",
                    partner_a=ParsedConnectionEndpoint(
                        label_asym_id=None,
                        auth_asym_id=chain_a,
                        label_entity_id=None,
                        label_seq_id=None,
                        auth_seq_id=seq_a,
                        insertion_code=icode_a,
                        label_comp_id=None,
                        auth_comp_id=comp_a,
                        label_atom_id=None,
                        auth_atom_id=atom_a,
                        label_alt_id=None,
                        symmetry=None,
                        model_number=None,
                    ),
                    partner_b=ParsedConnectionEndpoint(
                        label_asym_id=None,
                        auth_asym_id=chain_b,
                        label_entity_id=None,
                        label_seq_id=None,
                        auth_seq_id=seq_b,
                        insertion_code=icode_b,
                        label_comp_id=None,
                        auth_comp_id=comp_b,
                        label_atom_id=None,
                        auth_atom_id=atom_b,
                        label_alt_id=None,
                        symmetry=None,
                        model_number=None,
                    ),
                    distance=None,
                    value_order=None,
                    role=None,
                    details=None,
                    leaving_atom_flag=None,
                )
            )

    unavailable_categories = (
        "pdbx_struct_assembly",
        "struct_conn",
        "pdbx_unobs_or_zero_occ_residues",
        "pdbx_unobs_or_zero_occ_atoms",
        "REMARK465",
        "REMARK470",
    )

    return tuple(residues), tuple(connections), unavailable_categories


def _normalize_structural_strings(value):
    """Recursively normalize every `str` to NFC, anywhere it appears inside
    `value` (Section 33.2: "every string is normalized to NFC at the
    parser's boundary", unconditional and inherited unchanged).

    This is the single, central, auditable boundary for the requirement:
    called once, on the fully-assembled `ParsedReceptorDocument`, right
    before `parse_receptor_raw` returns it - before any of its content
    reaches `receptor_inspection.py`'s models or report (CP1-MAJOR-02). It
    walks every dataclass field and every tuple element generically, so it
    never needs to be updated by hand when a new string field is added to
    any `Parsed*` dataclass.

    - `None` is always preserved - absence is never turned into `""`.
    - Only `str` values are touched; `Decimal`/`int`/`bool`/`None` and
      other non-string, non-container values pass through unchanged (so
      `occupancy`/`distance` stay `Decimal`, per Section 33.2).
    - ASCII content is a no-op under NFC by definition.
    """

    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, tuple):
        return tuple(_normalize_structural_strings(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {
            field.name: _normalize_structural_strings(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        return dataclasses.replace(value, **changes)
    return value


def parse_receptor_raw(raw_path: Path, *, expected_checksum: str) -> ParsedReceptorDocument:
    """Execute Section 11.3's ten-step safe read flow and return a plain
    `ParsedReceptorDocument`. `raw_path` must already be the path derived
    from a registered `ProteinInput` (Section 11.3, step 1) - this
    function never accepts, resolves, or infers a path of its own.

    The `raw_format_declared`/`structure_format` divergence comparison
    (Section 11.5) is the inspection report builder's responsibility, not
    this function's - content detection (`structure_format`, carried in
    the returned document) always prevails over whatever format a caller
    believes the RAW artifact to be (Section 11.1).
    """

    raw_path = Path(raw_path)

    # Step 0
    _reject_compressed_extension(raw_path)
    # Step 3
    _reject_oversized_raw(raw_path)
    # Step 4/5
    _reject_compressed_content(raw_path)
    # Step 6
    raw_checksum = _compute_and_verify_checksum(raw_path, expected_checksum)

    # Step 8: hand the already-validated path to Gemmi.
    try:
        structure = gemmi.read_structure(str(raw_path), format=gemmi.CoorFormat.Detect)
    except (ValueError, RuntimeError) as error:
        raise ReceptorParseFailureError(
            f"Gemmi could not parse the RAW artifact: {error.__class__.__name__}"
        ) from error

    total_residues = sum(len(chain) for model in structure for chain in model)
    if structure.input_format not in (gemmi.CoorFormat.Mmcif, gemmi.CoorFormat.Pdb) or (
        total_residues == 0
    ):
        raise UnsupportedStructureFormatError(
            "RAW artifact content is not recognizable as either mmCIF or legacy PDB"
        )

    parser_version = gemmi.__version__

    if structure.input_format == gemmi.CoorFormat.Mmcif:
        try:
            document = gemmi.cif.read(str(raw_path))
            block = document.sole_block()
        except (ValueError, RuntimeError) as error:
            raise ReceptorParseFailureError(
                f"Gemmi could not re-read the RAW artifact's mmCIF categories: "
                f"{error.__class__.__name__}"
            ) from error

        (
            assemblies,
            entities,
            residues,
            connections,
            missing_residues,
            missing_atoms,
            unavailable_categories,
        ) = _parse_mmcif_document(block)
        structure_format = "mmcif"
    else:
        residues, connections, unavailable_categories = _parse_legacy_pdb_structure(structure)
        assemblies = ()
        entities = ()
        missing_residues = None
        missing_atoms = None
        structure_format = "pdb"

    models = tuple(
        ParsedModel(model_number=model_number)
        for model_number in sorted({residue.model_number for residue in residues})
    ) or (ParsedModel(model_number=1),)

    document = ParsedReceptorDocument(
        structure_format=structure_format,
        raw_checksum=raw_checksum,
        parser_name="gemmi",
        parser_version=parser_version,
        models=models,
        assemblies=assemblies,
        entities=entities,
        residues=residues,
        connections=connections,
        missing_residues=missing_residues,
        missing_atoms=missing_atoms,
        unavailable_categories=unavailable_categories,
    )
    return _normalize_structural_strings(document)
