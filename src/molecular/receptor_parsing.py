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
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

try:
    import gemmi
    _GEMMI_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as _gemmi_import_error:  # pragma: no cover - exercised via sys.modules sentinel in tests
    gemmi = None  # type: ignore[assignment]
    _GEMMI_IMPORT_ERROR = _gemmi_import_error

from molecular.atomic import compute_sha256
from molecular import receptor_materialization as rm


def _require_engine_available() -> None:
    """Section 3: an absent Gemmi engine must surface as a clean
    `MaterializationConfigurationError(ENGINE_UNAVAILABLE)` from every
    Checkpoint 2 entry point - never a raw `ImportError`/`AttributeError`
    from calling into a `None` `gemmi`. The 020 reader (`parse_receptor_raw`
    and its own helpers) is untouched by this guard - its historical
    behavior/API when Gemmi *is* present (the only case it has ever had to
    handle) is preserved exactly; this function is called only from the
    021 adapter's own public entry points."""

    if _GEMMI_IMPORT_ERROR is not None:
        raise rm.MaterializationConfigurationError(
            "ENGINE_UNAVAILABLE", f"Gemmi is not installed/importable: {_GEMMI_IMPORT_ERROR}",
        )


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


# ===========================================================================
# Increment 021, Checkpoint 2 additions (frozen specification Section 3:
# "entradas aditivas em receptor_parsing.py; única fronteira de Gemmi").
#
# `parse_receptor_cif_dom021` is a SEPARATE public entry point from
# `parse_receptor_raw` above - it never reuses it, never reopens a file by
# Path (Section 5/CP2 brief point 4: "parse_receptor_raw... não reutilizá-lo
# como leitor materializador"), and the 020 DTOs/functions above are
# untouched. It receives already-captured RAW bytes (checksum/size already
# verified by the 021 domain's own SourceContext boundary - Checkpoint 2
# never re-verifies those) and returns a generic, category-name-indexed raw
# token DOM: every mmCIF category actually present in the source, with its
# columns' raw tokens exactly as written (quoting preserved where gemmi's
# `raw=True` category access reports it) - never a `gemmi.*` object, never
# an interpreted/coerced value. The 021 domain's catalogue/connection/
# symmetry logic (receptor_materialization.py) decides what each category
# means; this function only captures bytes-to-tokens, once.
# ===========================================================================

MAX_CIF_RAW_ARTIFACT_SIZE_BYTES = 16 * 1024 * 1024  # 16 MiB (Section 4) - CP2's own ceiling, never the 020 reader's 1 GiB
MAX_CIF_ATOM_SITE_ROWS = 100_000
MAX_CIF_CONNECTION_ROWS = 50_000
MAX_CIF_CATEGORIES = 256
MAX_CIF_TAGS_PER_CATEGORY = 128
MAX_CIF_TOTAL_ROWS = 500_000
MAX_CIF_TOTAL_CELLS = 4_000_000
MAX_CIF_CELL_BYTES = 1024 * 1024

_SAVE_FRAME_PATTERN = re.compile(r"^save_\S", re.MULTILINE)


class InvalidCifEncodingError(ValueError):
    """The RAW bytes are not valid UTF-8, or begin with a UTF-8 BOM
    (Section 4: structural RAW must be UTF-8, with no BOM on the
    structural artifact) - raised before Gemmi ever sees the bytes."""


class CifSaveFrameRejectedError(ValueError):
    """The RAW content declares a CIF save frame (`save_...`/`save_`)
    (Section 4: "sem... save frames"). Gemmi 0.7.5 silently drops save
    frame content rather than erroring, so this module detects and
    rejects their presence itself, before parsing."""


class MultipleDataBlocksError(ValueError):
    """The RAW content declares more than one CIF data block (Section 4:
    "Um único data block")."""


class DuplicateCanonicalTagError(ValueError):
    """Two columns of the same category collide under ASCII
    case-insensitive tag comparison (Section 4/6) - detected by Gemmi's
    own `Document.check_for_duplicates()`, preserved as `__cause__`."""


class CifStructuralLimitExceededError(ValueError):
    """One of Section 4's closed scope limits (categories, tags per
    category, total rows, total cells, one cell's byte size, atom_site
    row count, struct_conn row count) was exceeded - checked before any
    further planning/serialization, never a claimed hard memory ceiling."""


@dataclass(frozen=True)
class RawCifCategory:
    """One mmCIF category's raw, uninterpreted rows - column name (always
    lowercase, trailing-dot-free, e.g. `"atom_site"`) -> tuple of raw
    tokens, one per row, in source row order. A column absent from the
    category is simply not a key of `columns`; a column present with a
    bare `.`/`?` token still appears, holding that literal token (null
    detection is the caller's job, via `raw_cif_null_to_optional_str`
    below) - `raw=True` category access never collapses or de-quotes."""

    name: str
    row_count: int
    columns: "dict[str, Tuple[str, ...]]"

    def get(self, row_index: int, column: str) -> Optional[str]:
        """The raw token at `row_index` for `column`, or `None` if that
        column is not present in this category at all (never confused
        with a present-but-null cell, which still returns its literal
        `.`/`?`/quoted token)."""

        values = self.columns.get(column)
        if values is None or row_index >= len(values):
            return None
        return values[row_index]


@dataclass(frozen=True)
class ParsedCifDom021:
    """The generic, whole-source raw DOM Checkpoint 2's domain catalogue
    and connection/symmetry resolution operate on. No `gemmi.*` type
    anywhere in this tree (Section 9, same guarantee as
    `ParsedReceptorDocument` above)."""

    category_names: Tuple[str, ...]  # canonical lowercase, e.g. "atom_site"
    categories: "dict[str, RawCifCategory]"
    parser_name: str
    parser_version: str


def raw_cif_null_to_optional_str(raw_token: Optional[str]) -> Optional[str]:
    """A bare, unquoted `.`/`?` is a real CIF null; a *quoted*
    `'.'`/`'?'`/`"."`/`"?"` string is a genuine token, not null - which is
    exactly why this must run on the still-quoted RAW token (as
    `get_mmcif_category(..., raw=True)` returns it), never on a value
    already decoded by `gemmi.cif.as_string()` (decoding first would
    strip the very quote marks that make the distinction possible,
    collapsing both to the same bare "."/"?"). Callers that need a
    *cooked* identifier for lookup/comparison - never for hashing, which
    must keep the literal raw form - decode it with `_decoded_token`
    below, strictly after this null check, never before. An absent
    column (`raw_token is None`) is also null."""

    if raw_token is None or raw_token in (".", "?"):
        return None
    return raw_token


def _decoded_token(raw_token: Optional[str]) -> Optional[str]:
    """The null-checked AND cooked (quote-stripped) value of one raw CIF
    token - for building identifiers used in lookup/comparison
    (`AtomObservation.label_asym_id`, `_entity.id`, operator expressions,
    ...), never for hashing (`compute_source_rows_key` reads
    `RawCategoryTable.raw_columns` directly, keeping the true raw form -
    the C-02 correction). A no-op for the overwhelmingly common case of
    an already-unquoted token; only a token genuinely quoted in the
    source is actually decoded here."""

    value = raw_cif_null_to_optional_str(raw_token)
    return value if value is None else gemmi.cif.as_string(value)


def _reject_invalid_cif_encoding(raw_bytes: bytes) -> str:
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        raise InvalidCifEncodingError(
            "RAW content begins with a UTF-8 BOM; the structural artifact must not carry one"
        )
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InvalidCifEncodingError("RAW content is not valid UTF-8") from error


def _reject_save_frames(text: str) -> None:
    """A real CIF save frame marker (`save_<name>`) is only syntax when it
    starts a line *outside* a semicolon-delimited multi-line text field
    (C-06): a value's literal text content may legitimately contain the
    substring `save_literal_inside_text` on a line of its own without
    that being save-frame syntax at all. A line is CIF text-field content
    whenever it falls between two lines that each start with `;` (the
    text-field delimiter) - never scanned for CIF tokens; only the lines
    outside any such span are checked against `_SAVE_FRAME_PATTERN`."""

    in_text_field = False
    for line in text.split("\n"):
        if line.startswith(";"):
            in_text_field = not in_text_field
            continue
        if in_text_field:
            continue
        if _SAVE_FRAME_PATTERN.match(line):
            raise CifSaveFrameRejectedError("RAW content declares a CIF save frame")


def _reject_cell_too_large(category_name: str, column: str, token: str) -> None:
    if len(token.encode("utf-8")) > MAX_CIF_CELL_BYTES:
        raise CifStructuralLimitExceededError(
            f"{category_name}.{column} carries a cell value larger than "
            f"{MAX_CIF_CELL_BYTES} UTF-8 bytes"
        )


def parse_receptor_cif_dom021(raw_bytes: bytes) -> ParsedCifDom021:
    """Increment 021 Checkpoint 2's single Gemmi entry point for the
    materialization domain (Section 3/19). `import gemmi` above is a
    plain, eager module-level import - this file is the one dedicated
    adapter module ARCH-01 permits to hold it - but Section 3's own
    "late import, clean configuration error when the engine is absent"
    requirement is satisfied here, not by that import statement:
    `_require_engine_available()` is the explicit guard every 021 entry
    point calls first, converting an absent/failed Gemmi import into
    `MaterializationConfigurationError(ENGINE_UNAVAILABLE)` before this
    function ever touches `gemmi.*`. Captures bytes already validated
    (checksum/size) by the 021 domain's own boundary; parses them exactly
    once into a native Gemmi DOM; extracts every present category's raw
    tokens; enforces Section 4's structural limits; and returns a plain,
    generic `ParsedCifDom021` - never a `gemmi.*` object. Never reuses
    `parse_receptor_raw`/its DTOs; never reopens a file by path."""

    _require_engine_available()
    text = _reject_invalid_cif_encoding(raw_bytes)
    _reject_save_frames(text)

    try:
        document = gemmi.cif.read_string(text)
    except (ValueError, RuntimeError) as error:
        raise ReceptorParseFailureError(
            f"Gemmi could not parse the RAW artifact's CIF syntax: {error.__class__.__name__}"
        ) from error

    if len(document) != 1:
        raise MultipleDataBlocksError(
            f"RAW content declares {len(document)} data blocks; exactly one is required"
        )

    try:
        document.check_for_duplicates()
    except RuntimeError as error:
        raise DuplicateCanonicalTagError(
            "RAW content declares a duplicate canonical tag within one category"
        ) from error

    block = document.sole_block()
    category_names_raw = block.get_mmcif_category_names()
    if len(category_names_raw) > MAX_CIF_CATEGORIES:
        raise CifStructuralLimitExceededError(
            f"RAW content declares {len(category_names_raw)} categories, "
            f"exceeding the {MAX_CIF_CATEGORIES} limit"
        )

    categories: "dict[str, RawCifCategory]" = {}
    total_rows = 0
    total_cells = 0

    for raw_name in category_names_raw:
        canonical_name = raw_name.lower().lstrip("_").rstrip(".")
        raw_columns = block.get_mmcif_category(raw_name, raw=True) or {}
        lowered_columns: "dict[str, Tuple[str, ...]]" = {}
        row_count = 0
        if len(raw_columns) > MAX_CIF_TAGS_PER_CATEGORY:
            raise CifStructuralLimitExceededError(
                f"category {canonical_name!r} declares {len(raw_columns)} tags, "
                f"exceeding the {MAX_CIF_TAGS_PER_CATEGORY} limit"
            )
        for column_name, values in raw_columns.items():
            lowered_column = column_name.lower()
            for token in values:
                _reject_cell_too_large(canonical_name, lowered_column, token)
            lowered_columns[lowered_column] = tuple(values)
            row_count = max(row_count, len(values))

        total_rows += row_count
        total_cells += row_count * len(lowered_columns)
        if canonical_name == "atom_site" and row_count > MAX_CIF_ATOM_SITE_ROWS:
            raise CifStructuralLimitExceededError(
                f"_atom_site declares {row_count} rows, exceeding the {MAX_CIF_ATOM_SITE_ROWS} limit"
            )
        if canonical_name == "struct_conn" and row_count > MAX_CIF_CONNECTION_ROWS:
            raise CifStructuralLimitExceededError(
                f"_struct_conn declares {row_count} rows, exceeding the {MAX_CIF_CONNECTION_ROWS} limit"
            )

        categories[canonical_name] = RawCifCategory(
            name=canonical_name, row_count=row_count, columns=lowered_columns
        )

    if total_rows > MAX_CIF_TOTAL_ROWS:
        raise CifStructuralLimitExceededError(
            f"RAW content declares {total_rows} total category rows, "
            f"exceeding the {MAX_CIF_TOTAL_ROWS} limit"
        )
    if total_cells > MAX_CIF_TOTAL_CELLS:
        raise CifStructuralLimitExceededError(
            f"RAW content declares {total_cells} total category cells, "
            f"exceeding the {MAX_CIF_TOTAL_CELLS} limit"
        )

    return ParsedCifDom021(
        category_names=tuple(sorted(categories)),
        categories=categories,
        parser_name="gemmi",
        parser_version=gemmi.__version__,
    )


def serialize_cif_document021(
    *, block_name: str, categories: "dict[str, dict]", category_order: Tuple[str, ...]
) -> bytes:
    """Builds and serializes a new, deterministic mmCIF document from
    already-decided categories (Section 13/18: "Serializar receptor.cif
    em bytes UTF-8 determinísticos"). `categories` maps canonical lowercase
    category name -> {column: [token, ...]} (already the exact tokens to
    write, in survivor row order); `category_order` is the exact sequence
    to emit them in (Section 11: "Tags/categorias de saída canônicas
    ordenadas"). Never reopens or reparses RAW; this is pure output
    construction from already-computed survivor tokens."""

    _require_engine_available()
    document = gemmi.cif.Document()
    block = document.add_new_block(block_name)
    for category_name in category_order:
        rows = categories.get(category_name)
        if not rows:
            continue
        block.set_mmcif_category(f"_{category_name}.", rows, raw=True)
    return document.as_string().encode("utf-8")


def reparse_and_extract_cif_dom021(output_bytes: bytes) -> ParsedCifDom021:
    """Section 18's independent reparse of the bytes just serialized -
    calls the exact same extraction path as the real RAW input
    (`parse_receptor_cif_dom021`), never a different, more lenient
    routine, so acceptance by the writer alone is never the oracle."""

    return parse_receptor_cif_dom021(output_bytes)


# --------------------------------------------------------------------------
# Increment 021, Checkpoint 2: the adapter boundary. This is the *only*
# place in the repository that both imports `gemmi` (via the functions
# above) and calls into the pure, gemmi-free `molecular.receptor_materialization`
# domain (`rm`) - ARCH-01 forbids the reverse (`rm` importing this module,
# even late/locally), never the other direction. Every `rm.*` DTO
# constructed below comes from real tokens extracted by
# `parse_receptor_cif_dom021` above; nothing here fabricates a scientific
# value.
# --------------------------------------------------------------------------

_ENTITY_TYPE_TOKENS = frozenset({"polymer", "non-polymer", "macrolide", "water", "branched", "unknown", "unrecognized"})


class MaterializationAdapterError(ValueError):
    """Wraps `rm.MaterializationExecutionError`'s closed code set for
    failures detected while building the adapter's inputs (before any
    `rm.*` function itself has a chance to raise) - e.g. a RAW document
    whose `_atom_site` rows do not resolve against `_entity`. Never a new
    vocabulary: `code` is still one of `rm.MaterializationExecutionError`'s
    codes."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MaterializationAdapterResult(NamedTuple):
    plan: "rm.MaterializationPlan"
    output_bytes: bytes
    reparsed: ParsedCifDom021


def _build_entity_type_lookup(dom: ParsedCifDom021) -> "Dict[str, str]":
    entity_table = dom.categories.get("entity")
    if entity_table is None:
        return {}
    lookup: "Dict[str, str]" = {}
    for row_index in range(entity_table.row_count):
        entity_id = _decoded_token(entity_table.get(row_index, "id"))
        entity_type_token = _decoded_token(entity_table.get(row_index, "type"))
        if entity_id is None or entity_type_token is None:
            continue
        lowered = entity_type_token.strip().lower()
        lookup[entity_id] = lowered if lowered in _ENTITY_TYPE_TOKENS else "unrecognized"
    return lookup


def _build_atom_observations(
    dom: ParsedCifDom021, entity_type_by_id: "Dict[str, str]"
) -> Tuple["rm.AtomObservation", ...]:
    table = dom.categories.get("atom_site")
    if table is None:
        raise MaterializationAdapterError("INVALID_CIF", "RAW content has no _atom_site category")

    atoms: List["rm.AtomObservation"] = []
    for row_index in range(table.row_count):
        def get(column: str) -> Optional[str]:
            return _decoded_token(table.get(row_index, column))

        atom_id = get("id")
        if atom_id is None:
            raise MaterializationAdapterError("INVALID_CIF", f"_atom_site row {row_index} has a null id")
        model_token = get("pdbx_pdb_model_num")
        if model_token is None:
            raise MaterializationAdapterError(
                "INVALID_CIF", f"_atom_site row {row_index} has a null pdbx_pdb_model_num"
            )
        try:
            model = int(model_token)
        except ValueError as error:
            raise MaterializationAdapterError(
                "INVALID_CIF",
                f"_atom_site row {row_index} pdbx_pdb_model_num={model_token!r} is not an integer",
            ) from error
        if model < 1:
            raise MaterializationAdapterError(
                "INVALID_CIF", f"_atom_site row {row_index} pdbx_pdb_model_num={model} must be positive"
            )

        label_asym_id = get("label_asym_id")
        entity_id = get("label_entity_id")
        ccd = get("label_comp_id")
        if label_asym_id is None or entity_id is None or ccd is None:
            raise MaterializationAdapterError(
                "INVALID_CIF",
                f"_atom_site row {row_index} is missing a required label_asym_id/label_entity_id/label_comp_id",
            )
        entity_type = entity_type_by_id.get(entity_id)
        if entity_type is None:
            raise MaterializationAdapterError(
                "INVALID_CIF",
                f"_atom_site row {row_index} entity_id={entity_id!r} has no matching _entity.id",
            )

        atoms.append(
            rm.AtomObservation(
                ordinal=row_index, atom_id=atom_id, model=model, label_asym_id=label_asym_id, entity_id=entity_id,
                entity_type=entity_type, ccd=ccd, label_seq_id=get("label_seq_id"), auth_asym_id=get("auth_asym_id"),
                auth_seq_id=get("auth_seq_id"), insertion_token=get("pdbx_pdb_ins_code"),
                label_atom_id=get("label_atom_id"), altloc=get("label_alt_id"),
            )
        )
    return tuple(atoms)


def _build_connection_rows(dom: ParsedCifDom021) -> Tuple["rm.RawConnectionRow", ...]:
    table = dom.categories.get("struct_conn")
    if table is None:
        return ()

    rows: List["rm.RawConnectionRow"] = []
    for row_index in range(table.row_count):
        def get(column: str) -> Optional[str]:
            return _decoded_token(table.get(row_index, column))

        def endpoint(prefix: str) -> "rm.RawConnectionEndpointTokens":
            return rm.RawConnectionEndpointTokens(
                label_asym_id=get(f"{prefix}_label_asym_id"), label_comp_id=get(f"{prefix}_label_comp_id"),
                label_seq_id=get(f"{prefix}_label_seq_id"), label_atom_id=get(f"{prefix}_label_atom_id"),
                auth_seq_id=get(f"{prefix}_auth_seq_id"), insertion_token=get(f"pdbx_{prefix}_pdb_ins_code"),
                altloc_aliases=(
                    get(f"{prefix}_label_alt_id"), get(f"pdbx_{prefix}_label_alt_id"), get(f"pdbx_{prefix}_auth_alt_id"),
                ),
                symmetry_token=get(f"{prefix}_symmetry"),
            )

        ptnr3_present = any(
            raw_cif_null_to_optional_str(table.get(row_index, column)) is not None
            for column in table.columns if column.startswith("pdbx_ptnr3_")
        )
        source_tokens = tuple(
            (column, table.get(row_index, column))
            for column in sorted(table.columns)
            if table.get(row_index, column) is not None
        )

        rows.append(
            rm.RawConnectionRow(
                ordinal=row_index, id=get("id"), conn_type=get("conn_type_id"),
                ptnr1=endpoint("ptnr1"), ptnr2=endpoint("ptnr2"), ptnr3_present=ptnr3_present,
                source_tokens=source_tokens,
            )
        )
    return tuple(rows)


def _build_symmetry_operator_tables(dom: ParsedCifDom021) -> "rm.SymmetryOperatorTables":
    def build(category_name: str, expression_column: str) -> "Optional[Dict[str, Optional[str]]]":
        table = dom.categories.get(category_name)
        if table is None:
            return None
        result: "Dict[str, Optional[str]]" = {}
        for row_index in range(table.row_count):
            operator_id = _decoded_token(table.get(row_index, "id"))
            expression = _decoded_token(table.get(row_index, expression_column))
            if operator_id is None:
                continue
            result[operator_id] = None if operator_id in result else expression
        return result

    return rm.SymmetryOperatorTables(
        symmetry_equiv=build("symmetry_equiv", "pos_as_xyz"),
        space_group_symop=build("space_group_symop", "operation_xyz"),
    )


def _build_category_tables(dom: ParsedCifDom021) -> "Dict[str, rm.RawCategoryTable]":
    """Builds both views the C-02 correction requires: the *decoded* view
    (`.`/`?` collapsed to `None`, used for every membership/lookup/
    identity computation in the domain) and the *raw* view (the literal
    token Gemmi returned, `.`/`?` kept apart from each other and from a
    quoted `'.'`/`'?'` string - used for hashing/audit, so two real
    tokens that both happen to mean "no value" never collapse into the
    same `source_rows_key`)."""

    tables: "Dict[str, rm.RawCategoryTable]" = {}
    for name, category in dom.categories.items():
        columns = {
            column: tuple(_decoded_token(value) for value in values)
            for column, values in category.columns.items()
        }
        tables[f"_{name}"] = rm.RawCategoryTable(columns=columns, raw_columns=dict(category.columns))
    return tables


def _build_output_category_tokens(
    *, dom: ParsedCifDom021, category_decisions: Tuple["rm.CategoryDecision", ...], entry_id: str,
) -> "Dict[str, Dict[str, List[str]]]":
    output: "Dict[str, Dict[str, List[str]]]" = {}
    for decision in category_decisions:
        canonical = decision.category.lstrip("_")
        if decision.category == "_entry":
            output[canonical] = {"id": [entry_id]}
            continue
        if decision.action in ("omit", "guard_then_omit") or not decision.kept_row_ordinals:
            continue
        source = dom.categories.get(canonical)
        if source is None:
            continue
        rows: "Dict[str, List[str]]" = {}
        for column in decision.preserved_tags:
            rows[column] = [source.get(row_index, column) or "?" for row_index in decision.kept_row_ordinals]
        for column in decision.generated_tags:
            if column == "entry_id":
                rows[column] = [entry_id for _ in decision.kept_row_ordinals]
        if rows:
            output[canonical] = rows
    return output


def _verify_output_reparse(
    *,
    category_decisions: Tuple["rm.CategoryDecision", ...],
    entry_id: str,
    output_categories: "Dict[str, Dict[str, List[str]]]",
    reparsed: ParsedCifDom021,
) -> None:
    """Section 18/C-03: an independent reparse of the bytes just
    serialized must reproduce exactly the *values* the plan decided to
    write (`output_categories`, the independently-derived expected token
    vector - never re-derived from the reparsed DOM itself) - cell by
    cell, in source row order, with no extra surviving column - not
    merely a matching row count and tag presence. Acceptance by the
    writer/reader of the same library alone is never sufficient proof
    (brief Section 4, point 5): a changed id, a corrupted coordinate, a
    dropped/added column, or reordered rows must all raise
    `INVALID_OUTPUT` here, even when the shape alone would look fine."""

    for decision in category_decisions:
        canonical = decision.category.lstrip("_")
        if decision.category == "_entry":
            reparsed_entry = reparsed.categories.get("entry")
            if reparsed_entry is None or reparsed_entry.row_count != 1 or reparsed_entry.get(0, "id") != entry_id:
                raise rm.MaterializationExecutionError(
                    "INVALID_OUTPUT", "reparsed _entry does not carry the expected generated id", phase="plan",
                )
            continue
        if decision.action in ("omit", "guard_then_omit") or not decision.kept_row_ordinals:
            reparsed_table = reparsed.categories.get(canonical)
            if reparsed_table is not None and reparsed_table.row_count:
                raise rm.MaterializationExecutionError(
                    "INVALID_OUTPUT",
                    f"category {decision.category!r} was decided omitted/empty but survives in the reparsed output",
                    phase="plan",
                )
            continue

        expected_columns = output_categories.get(canonical, {})
        reparsed_table = reparsed.categories.get(canonical)
        expected_rows = len(decision.kept_row_ordinals)
        if reparsed_table is None or reparsed_table.row_count != expected_rows:
            raise rm.MaterializationExecutionError(
                "INVALID_OUTPUT",
                f"category {decision.category!r} expected {expected_rows} reparsed row(s), "
                f"got {0 if reparsed_table is None else reparsed_table.row_count}",
                phase="plan",
            )

        unexpected_columns = set(reparsed_table.columns) - set(expected_columns)
        if unexpected_columns:
            raise rm.MaterializationExecutionError(
                "INVALID_OUTPUT",
                f"category {decision.category!r} reparsed output carries unexpected tag(s) {sorted(unexpected_columns)!r}",
                phase="plan",
            )

        for column, expected_values in expected_columns.items():
            if column not in reparsed_table.columns:
                raise rm.MaterializationExecutionError(
                    "INVALID_OUTPUT",
                    f"category {decision.category!r} preserved tag {column!r} is absent from the reparsed output",
                    phase="plan",
                )
            for row_index, expected_value in enumerate(expected_values):
                actual_value = reparsed_table.get(row_index, column)
                if actual_value != expected_value:
                    raise rm.MaterializationExecutionError(
                        "INVALID_OUTPUT",
                        f"category {decision.category!r} row {row_index} tag {column!r} reparsed as "
                        f"{actual_value!r}, expected {expected_value!r}",
                        phase="plan",
                    )


def materialize_receptor_cif021(
    *,
    source_context: "rm.SourceContext",
    resolved_selection: "rm.ResolvedReceptorSelection",
    unaddressed_components: "rm.UnaddressedComponentsPolicy",
    raw_bytes: bytes,
) -> MaterializationAdapterResult:
    """Increment 021 Checkpoint 2's whole adapter pipeline (brief Section
    4): confirm the received `raw_bytes` are the exact bytes
    `source_context` already describes (size/checksum) and that the
    installed Gemmi is the pinned engine (C-04 - never skip this and
    parse first), parse the RAW bytes into a DOM exactly once, build the
    pure `rm.AtomObservation`/`rm.RawConnectionRow`/
    `rm.SymmetryOperatorTables`/`rm.RawCategoryTable` inputs from real
    tokens, run the S/D/U/X/O planner, the connection/symmetry resolver
    and the catalogue engine, derive the `tbmp_<plan_key>` entry/block
    identity from the resulting `plan_key`, serialize `receptor.cif`
    bytes, and independently reparse them to verify the output. Never
    opens a file by path, never reparses the *source* RAW a second time
    - only the newly serialized bytes.

    `import gemmi` above is a plain, eager module-level import - this
    file (never `receptor_materialization.py`) is the one dedicated
    adapter module ARCH-01 permits to hold it. Section 3's "late import,
    clean configuration error when absent" requirement is satisfied by
    `_require_engine_available()` below, called first, before the
    version-pin check or anything else that would otherwise touch a
    `None` `gemmi` with a raw `AttributeError`."""

    _require_engine_available()
    if gemmi.__version__ != rm.ENGINE_VERSION:
        raise rm.MaterializationConfigurationError(
            "ENGINE_VERSION",
            f"installed Gemmi is {gemmi.__version__!r}; this checkpoint is pinned to {rm.ENGINE_VERSION!r}",
        )
    if source_context.parser.name != rm.ENGINE_NAME or source_context.parser.version != rm.ENGINE_VERSION:
        raise rm.MaterializationConfigurationError(
            "ENGINE_UNAVAILABLE",
            f"source_context.parser {source_context.parser.name}/{source_context.parser.version} "
            f"does not match the pinned engine {rm.ENGINE_NAME}/{rm.ENGINE_VERSION}",
        )
    actual_size = len(raw_bytes)
    if actual_size > MAX_CIF_RAW_ARTIFACT_SIZE_BYTES:
        raise CifStructuralLimitExceededError(
            f"RAW content is {actual_size} bytes, exceeding the {MAX_CIF_RAW_ARTIFACT_SIZE_BYTES} "
            "(16 MiB, Section 4) limit - never the 020 reader's 1 GiB ceiling"
        )
    if actual_size != source_context.raw_size_bytes:
        raise rm.MaterializationConfigurationError(
            "SOURCE_CHANGED",
            f"received {actual_size} RAW byte(s); source_context.raw_size_bytes declares "
            f"{source_context.raw_size_bytes}",
        )
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != source_context.raw_sha256:
        raise rm.MaterializationConfigurationError(
            "CHECKSUM_MISMATCH",
            f"received RAW SHA-256 {actual_sha256} does not match source_context.raw_sha256 "
            f"{source_context.raw_sha256}",
        )
    for format_name, signature in _MAGIC_BYTES_BY_FORMAT:
        if raw_bytes[: len(signature)] == signature:
            raise CompressedRawContentError(
                f"RAW content begins with the {format_name} magic bytes; "
                "compressed content is never accepted, regardless of declared format"
            )

    dom = parse_receptor_cif_dom021(raw_bytes)
    entity_type_by_id = _build_entity_type_lookup(dom)
    atoms = _build_atom_observations(dom, entity_type_by_id)

    plan_without_connections = rm.build_materialization_plan(
        source_context=source_context, resolved_selection=resolved_selection,
        unaddressed_components=unaddressed_components, atoms=atoms,
    )
    source_context_key = plan_without_connections.plan.source_context_key
    atom_decisions_by_ordinal = {d.ordinal: d for d in plan_without_connections.atom_decisions}
    atoms_by_ordinal = {a.ordinal: a for a in atoms}
    model = resolved_selection.resolved_model
    atoms_in_model = tuple(a for a in atoms if a.model == model)

    tables = _build_symmetry_operator_tables(dom)
    raw_connections = _build_connection_rows(dom)
    connection_decisions = rm.resolve_connection_decisions(
        source_context_key=source_context_key, atoms_in_model=atoms_in_model,
        atom_decisions_by_ordinal=atom_decisions_by_ordinal, raw_connections=raw_connections, tables=tables,
    )

    retained_entity_ids: "FrozenSet[str]" = frozenset(
        atoms_by_ordinal[ordinal].entity_id
        for ordinal, decision in atom_decisions_by_ordinal.items() if decision.retained
    )
    retained_asym_ids: "FrozenSet[str]" = frozenset(
        atoms_by_ordinal[ordinal].label_asym_id
        for ordinal, decision in atom_decisions_by_ordinal.items() if decision.retained
    )
    category_tables = _build_category_tables(dom)
    category_decisions = rm.compute_category_decisions(
        source_context_key=source_context_key, categories=category_tables,
        retained_entity_ids=retained_entity_ids, retained_asym_ids=retained_asym_ids, resolved_model=model,
        atoms_by_ordinal=atoms_by_ordinal, atom_decisions_by_ordinal=atom_decisions_by_ordinal,
        connection_decisions=connection_decisions,
    )

    plan = rm.build_materialization_plan(
        source_context=source_context, resolved_selection=resolved_selection,
        unaddressed_components=unaddressed_components, atoms=atoms,
        category_decisions=category_decisions, connection_decisions=connection_decisions,
    )

    # Section 11: "nome do novo block como tbmp_<plan_key>" - both the
    # generated _entry.id and the block name share the same lowercase
    # prefix (C-06: a prior "TBMP_" was a transcription slip, not a
    # distinct identity scheme).
    entry_id = f"tbmp_{plan.plan_key}"
    block_name = f"tbmp_{plan.plan_key}"
    output_categories = _build_output_category_tokens(
        dom=dom, category_decisions=category_decisions, entry_id=entry_id,
    )
    category_order = tuple(sorted(output_categories))
    output_bytes = serialize_cif_document021(
        block_name=block_name, categories=output_categories, category_order=category_order,
    )

    reparsed = reparse_and_extract_cif_dom021(output_bytes)
    _verify_output_reparse(
        category_decisions=category_decisions, entry_id=entry_id,
        output_categories=output_categories, reparsed=reparsed,
    )

    return MaterializationAdapterResult(plan=plan, output_bytes=output_bytes, reparsed=reparsed)
