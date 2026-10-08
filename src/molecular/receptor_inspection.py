"""Domain model and report builder for receptor inspection (Increment 020,
Checkpoint 1).

`ReceptorInspectionReport` is the factual, deterministic result of reading
a receptor RAW artifact - built here from the plain, gemmi-free
`ParsedReceptorDocument` that `receptor_parsing.py` produces. No `gemmi.*`
type is imported or referenced anywhere in this module.

Classification (Section 17) is mechanical and safe: it never asserts
biological function, never infers a disposition, and the closed metal-ion
CCD table (`METAL_ION_CCD_CODES_V1`) is explicitly non-exhaustive - an
unrecognized monatomic non-polymer component is always `"unknown"`, never
silently absorbed into `"other-non-polymer"` (Section 17.1).

`inspection_id` is computed over the canonical seven-key payload (Section
15.1-15.2) by reusing `promotion.py`'s `canonicalize_payload`/
`compute_semantic_key` mechanism - same algorithm, a different payload and
a different purpose (Section 15.2); it is never confused with a
`semantic_key`.
"""

from decimal import Decimal
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from molecular.promotion import compute_semantic_key, validate_semantic_key
from molecular.receptor_parsing import ParsedReceptorDocument

INSPECTION_SCHEMA_VERSION = "1.0"
INSPECTION_PROFILE_VERSION = "1.0"

# Non-exhaustive by declaration (Section 17.2) - a monatomic non-polymer
# component whose ccd_code is not listed here is always "unknown", never
# silently folded into "other-non-polymer".
METAL_ION_CCD_CODES_V1 = frozenset(
    {
        "ZN", "MG", "CA", "NA", "K", "MN", "FE", "FE2", "CU", "CU1", "NI", "CO", "CD", "HG",
        "LI", "AL", "BA", "SR", "CS", "RB", "AG", "AU", "PT", "PD", "CR", "MO", "W", "V",
        "TI", "PB", "SN", "GA", "IN", "TL", "BE", "Y", "LA", "CE", "GD", "YB", "LU", "TH",
        "U", "RU", "RH", "OS", "IR", "RE", "TC", "SC", "ZR", "HF", "NB", "TA",
    }
)

_STANDARD_AMINO_ACID_CODES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
)
_STANDARD_NUCLEOTIDE_CODES = frozenset({"A", "C", "G", "T", "U"})
STANDARD_POLYMER_RESIDUE_CODES = _STANDARD_AMINO_ACID_CODES | _STANDARD_NUCLEOTIDE_CODES

_WATER_CCD_CODES = frozenset({"HOH", "DOD"})

_RECOGNIZED_ENTITY_TYPES = frozenset({"polymer", "non-polymer", "macrolide", "water", "branched"})

_DECLARED_FORMAT_ALIASES = {"cif": "mmcif", "mmcif": "mmcif", "pdb": "pdb", "ent": "pdb"}

_MAX_MODELS = 1000
_MAX_CHAINS = 10000
_MAX_NON_POLYMER_COMPONENTS = 50000
_MAX_STRUCTURE_CONNECTIONS = 50000
_MAX_RETAINED_WARNINGS = 5000


IdentityOrigin = Literal["source", "parser-derived", "unavailable"]
SelectionCapability = Literal["full", "inspection-only"]
StructureFormat = Literal["mmcif", "pdb"]
ParserName = Literal["gemmi"]
EntityType = Literal["polymer", "non-polymer", "macrolide", "water", "branched", "unrecognized"]
NonPolymerClassification = Literal[
    "water", "metal-ion", "modified-polymer-residue", "other-non-polymer", "unknown"
]
ClassificationBasis = Literal["format-declared", "closed-dictionary-match", "undetermined"]
ConnectionCategory = Literal[
    "disulfide", "metal-coordination", "covalent",
    "legacy-pdb-ssbond", "legacy-pdb-link-covalent", "legacy-pdb-link-metal",
    "legacy-pdb-conect-covalent", "legacy-pdb-conect-metal",
]
SourceCategory = Literal["struct_conn", "legacy-pdb-ssbond", "legacy-pdb-link", "legacy-pdb-conect"]
InspectionWarningCode = Literal[
    "multiple-models", "multiple-assemblies", "alternate-locations-present",
    "partial-occupancy-present", "missing-residues-reported", "missing-atoms-reported",
    "explicit-hydrogens-absent", "unknown-chemical-component", "covalent-connection-present",
    "metal-coordination-present", "legacy-pdb-information-limited", "category-unavailable",
    "ambiguous-author-chain-id", "parser-limitation", "declared-format-mismatch",
]
InspectionReportSchemaVersion = Literal["1.0"]


class InspectionFailureError(RuntimeError):
    """Raised when the inspection report could not be built from an
    otherwise successfully parsed document (Section 30.1). `__cause__` is
    always preserved."""


class ParserProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ParserName
    version: str = Field(min_length=1)


class ModelDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_number: int = Field(ge=1)


class AssemblyOperationExpression(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expression: str = Field(min_length=1)
    asym_id_list: Tuple[str, ...]


class AssemblyDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    assembly_id: str = Field(min_length=1)
    details: Optional[str] = None
    generation: Tuple[AssemblyOperationExpression, ...] = ()


class EntityInventoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_entity_id: Optional[str] = Field(default=None, min_length=1)
    label_entity_id_origin: IdentityOrigin
    entity_type: EntityType
    description: Optional[str] = None

    @model_validator(mode="after")
    def _validate_origin_consistency(self) -> "EntityInventoryEntry":
        _require_origin_consistency(self.label_entity_id, self.label_entity_id_origin)
        return self


class ChainInventoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: Optional[str] = Field(default=None, min_length=1)
    label_asym_id_origin: IdentityOrigin
    auth_asym_id: Optional[str] = None
    label_entity_id: Optional[str] = Field(default=None, min_length=1)
    label_entity_id_origin: IdentityOrigin
    model_number: int = Field(ge=1)
    residue_count: int = Field(ge=0)
    atom_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_origin_consistency(self) -> "ChainInventoryEntry":
        _require_origin_consistency(self.label_asym_id, self.label_asym_id_origin)
        _require_origin_consistency(self.label_entity_id, self.label_entity_id_origin)
        return self


class AltlocObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: Optional[str] = None
    auth_asym_id: Optional[str] = None
    residue_identifier: str = Field(min_length=1)
    altloc_ids: Tuple[str, ...] = Field(min_length=1)


class OccupancyObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: Optional[str] = None
    auth_asym_id: Optional[str] = None
    residue_identifier: str = Field(min_length=1)
    atom_name: str = Field(min_length=1)
    occupancy: Decimal = Field(ge=0)


class MissingResidueObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: Optional[str] = None
    auth_asym_id: Optional[str] = None
    residue_identifier: str = Field(min_length=1)
    residue_name: Optional[str] = None


class MissingAtomObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: Optional[str] = None
    auth_asym_id: Optional[str] = None
    residue_identifier: str = Field(min_length=1)
    atom_name: str = Field(min_length=1)


class NonPolymerComponent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: Optional[str] = Field(default=None, min_length=1)
    label_asym_id_origin: IdentityOrigin
    auth_asym_id: Optional[str] = None
    label_entity_id: Optional[str] = Field(default=None, min_length=1)
    label_entity_id_origin: IdentityOrigin
    ccd_code: str = Field(min_length=1, max_length=10)
    atom_count: int = Field(ge=1)
    classification: NonPolymerClassification
    classification_basis: ClassificationBasis

    @model_validator(mode="after")
    def _validate_origin_consistency(self) -> "NonPolymerComponent":
        _require_origin_consistency(self.label_asym_id, self.label_asym_id_origin)
        _require_origin_consistency(self.label_entity_id, self.label_entity_id_origin)
        return self


class StructConnEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_asym_id: Optional[str] = None
    auth_asym_id: Optional[str] = None
    label_entity_id: Optional[str] = None
    label_seq_id: Optional[str] = None
    auth_seq_id: Optional[str] = None
    insertion_code: Optional[str] = None
    label_comp_id: Optional[str] = None
    auth_comp_id: Optional[str] = None
    label_atom_id: Optional[str] = None
    auth_atom_id: Optional[str] = None
    label_alt_id: Optional[str] = None
    symmetry: Optional[str] = None
    model_number: Optional[int] = Field(default=None, ge=1)


class StructureConnection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    struct_conn_id: Optional[str] = None
    conn_type: str = Field(min_length=1)
    category: ConnectionCategory
    source_category: SourceCategory
    partner_a: StructConnEndpoint
    partner_b: StructConnEndpoint
    distance: Optional[Decimal] = Field(default=None, ge=0)
    value_order: Optional[str] = None
    role: Optional[str] = None
    details: Optional[str] = None
    leaving_atom_flag: Optional[str] = None


class InspectionWarning(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: InspectionWarningCode
    model_number: Optional[int] = None
    label_asym_id: Optional[str] = None
    auth_asym_id: Optional[str] = None
    ccd_code: Optional[str] = None
    category_name: Optional[str] = None
    residue_identifier: Optional[str] = None
    detail: str = Field(min_length=1, max_length=500)


class ReceptorInspectionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inspection_id: str = Field(min_length=64, max_length=64)
    inspection_schema_version: InspectionReportSchemaVersion
    inspection_profile_version: InspectionReportSchemaVersion
    protein_input_id: str = Field(min_length=1)
    raw_checksum: str = Field(min_length=64, max_length=64)
    raw_format_declared: str = Field(min_length=1)
    structure_format: StructureFormat
    parser: ParserProvenance
    selection_capability: SelectionCapability
    models: Tuple[ModelDescriptor, ...]
    assemblies: Tuple[AssemblyDescriptor, ...]
    entities: Tuple[EntityInventoryEntry, ...]
    chains: Tuple[ChainInventoryEntry, ...]
    altloc_observations: Tuple[AltlocObservation, ...]
    occupancy_observations: Tuple[OccupancyObservation, ...]
    missing_residues: Tuple[MissingResidueObservation, ...]
    missing_atoms: Tuple[MissingAtomObservation, ...]
    disulfide_bonds: Tuple[StructureConnection, ...]
    covalent_connections: Tuple[StructureConnection, ...]
    metal_coordinations: Tuple[StructureConnection, ...]
    non_polymer_components: Tuple[NonPolymerComponent, ...]
    explicit_hydrogens_present: bool
    explicit_hydrogen_count: int = Field(ge=0)
    unavailable_categories: Tuple[str, ...]
    warnings: Tuple[InspectionWarning, ...]

    @model_validator(mode="after")
    def _validate_inspection_id(self) -> "ReceptorInspectionReport":
        validate_semantic_key(self.inspection_id)
        return self


def _require_origin_consistency(value: Optional[str], origin: IdentityOrigin) -> None:
    if (value is None) == (origin == "source"):
        raise ValueError(
            "a field must be None if and only if its *_origin is not 'source' "
            f"(got value={value!r}, origin={origin!r})"
        )


def compute_inspection_id(
    *,
    parser_name: str,
    parser_version: str,
    protein_input_id: str,
    raw_checksum: str,
    raw_format: str,
) -> str:
    """Compute the deterministic, content-derived `inspection_id` (Section
    15.1-15.2): SHA-256 over the canonical JSON of the seven-key payload,
    using the same canonicalization mechanism `promotion.py` already uses
    for `semantic_key` - a different payload, a different purpose, never
    the same identity concept (Section 15.7/23.1)."""

    payload = {
        "inspection_schema_version": INSPECTION_SCHEMA_VERSION,
        "inspection_profile_version": INSPECTION_PROFILE_VERSION,
        "parser_name": parser_name,
        "parser_version": parser_version,
        "protein_input_id": protein_input_id,
        "raw_checksum": raw_checksum,
        "raw_format": raw_format,
    }
    return compute_semantic_key(payload)


def _classify_entity_type_raw(raw_type: Optional[str]) -> EntityType:
    if raw_type is None:
        return "unrecognized"
    normalized = raw_type.strip().lower()
    if normalized in _RECOGNIZED_ENTITY_TYPES:
        return normalized  # type: ignore[return-value]
    return "unrecognized"


def _build_entities_mmcif(parsed: ParsedReceptorDocument) -> Tuple[EntityInventoryEntry, ...]:
    entities = []
    for entity in parsed.entities:
        entities.append(
            EntityInventoryEntry(
                label_entity_id=entity.label_entity_id,
                label_entity_id_origin="source" if entity.label_entity_id else "unavailable",
                entity_type=_classify_entity_type_raw(entity.entity_type_raw),
                description=entity.description,
            )
        )
    return tuple(entities)


def _build_entities_legacy_pdb(parsed: ParsedReceptorDocument) -> Tuple[EntityInventoryEntry, ...]:
    """Legacy PDB declares no `_entity` category at all (Section 12.2) - the
    closest honest proxy available is one entity per distinct auth chain,
    typed from whether that chain carries any ATOM-flagged (polymer-like)
    residue; `label_entity_id` is always unavailable."""

    seen_chains: dict = {}
    order = []
    for residue in parsed.residues:
        key = residue.auth_asym_id
        if key not in seen_chains:
            seen_chains[key] = False
            order.append(key)
        if not residue.is_het:
            seen_chains[key] = True

    return tuple(
        EntityInventoryEntry(
            label_entity_id=None,
            label_entity_id_origin="unavailable",
            entity_type="polymer" if seen_chains[auth_asym_id] else "non-polymer",
            description=None,
        )
        for auth_asym_id in order
    )


def _build_chains(
    parsed: ParsedReceptorDocument,
) -> Tuple[ChainInventoryEntry, ...]:
    groups: dict = {}
    order = []
    for residue in parsed.residues:
        key = (residue.model_number, residue.label_asym_id, residue.auth_asym_id)
        if key not in groups:
            groups[key] = {"residue_count": 0, "atom_count": 0, "label_entity_id": residue.label_entity_id}
            order.append(key)
        groups[key]["residue_count"] += 1
        groups[key]["atom_count"] += len(residue.atoms)

    chains = []
    for key in order:
        model_number, label_asym_id, auth_asym_id = key
        data = groups[key]
        label_entity_id = data["label_entity_id"]
        chains.append(
            ChainInventoryEntry(
                label_asym_id=label_asym_id,
                label_asym_id_origin="source" if label_asym_id else "unavailable",
                auth_asym_id=auth_asym_id,
                label_entity_id=label_entity_id,
                label_entity_id_origin="source" if label_entity_id else "unavailable",
                model_number=model_number,
                residue_count=data["residue_count"],
                atom_count=data["atom_count"],
            )
        )
    return tuple(chains)


def _classify_non_polymer(
    *, ccd_code: str, atom_count: int
) -> Tuple[NonPolymerClassification, ClassificationBasis]:
    if ccd_code in _WATER_CCD_CODES:
        return "water", "format-declared"
    if atom_count == 1:
        if ccd_code in METAL_ION_CCD_CODES_V1:
            return "metal-ion", "closed-dictionary-match"
        return "unknown", "undetermined"
    return "other-non-polymer", "closed-dictionary-match"


def _build_non_polymer_components(
    parsed: ParsedReceptorDocument,
) -> Tuple[NonPolymerComponent, ...]:
    components = []
    for residue in parsed.residues:
        comp_id = residue.comp_id
        atom_count = max(len(residue.atoms), 1)

        if residue.label_entity_id is not None:
            entity_type_raw = next(
                (
                    entity.entity_type_raw
                    for entity in parsed.entities
                    if entity.label_entity_id == residue.label_entity_id
                ),
                None,
            )
            is_polymer_entity = _classify_entity_type_raw(entity_type_raw) == "polymer"
        else:
            is_polymer_entity = not residue.is_het

        if is_polymer_entity:
            if comp_id in STANDARD_POLYMER_RESIDUE_CODES:
                continue  # an ordinary, unmodified backbone residue is never reported
            classification: NonPolymerClassification = "modified-polymer-residue"
            basis: ClassificationBasis = "format-declared"
        else:
            classification, basis = _classify_non_polymer(ccd_code=comp_id, atom_count=atom_count)

        components.append(
            NonPolymerComponent(
                label_asym_id=residue.label_asym_id,
                label_asym_id_origin="source" if residue.label_asym_id else "unavailable",
                auth_asym_id=residue.auth_asym_id,
                label_entity_id=residue.label_entity_id,
                label_entity_id_origin="source" if residue.label_entity_id else "unavailable",
                ccd_code=comp_id,
                atom_count=atom_count,
                classification=classification,
                classification_basis=basis,
            )
        )
    return tuple(components)


def _build_altloc_and_occupancy_observations(
    parsed: ParsedReceptorDocument,
) -> Tuple[Tuple[AltlocObservation, ...], Tuple[OccupancyObservation, ...]]:
    altloc_observations = []
    occupancy_observations = []

    for residue in parsed.residues:
        residue_identifier = _residue_identifier(residue.auth_seq_id, residue.insertion_code)
        altlocs = sorted({atom.altloc for atom in residue.atoms if atom.altloc is not None})
        if altlocs:
            altloc_observations.append(
                AltlocObservation(
                    label_asym_id=residue.label_asym_id,
                    auth_asym_id=residue.auth_asym_id,
                    residue_identifier=residue_identifier,
                    altloc_ids=tuple(altlocs),
                )
            )

        for atom in residue.atoms:
            if atom.occupancy != Decimal("1"):
                occupancy_observations.append(
                    OccupancyObservation(
                        label_asym_id=residue.label_asym_id,
                        auth_asym_id=residue.auth_asym_id,
                        residue_identifier=residue_identifier,
                        atom_name=atom.atom_name,
                        occupancy=atom.occupancy,
                    )
                )

    return tuple(altloc_observations), tuple(occupancy_observations)


def _residue_identifier(auth_seq_id: Optional[str], insertion_code: Optional[str]) -> str:
    base = auth_seq_id if auth_seq_id is not None else "?"
    return f"{base}{insertion_code}" if insertion_code else base


def _classify_connection_partner(
    endpoint, non_polymer_lookup: dict
) -> Optional[NonPolymerClassification]:
    key = (endpoint.label_asym_id or endpoint.auth_asym_id, endpoint.auth_seq_id)
    return non_polymer_lookup.get(key)


def _build_connections(
    parsed: ParsedReceptorDocument, non_polymer_lookup: dict
) -> Tuple[StructureConnection, ...]:
    connections = []
    for parsed_conn in parsed.connections:
        endpoint_a = _build_struct_conn_endpoint(parsed_conn.partner_a)
        endpoint_b = _build_struct_conn_endpoint(parsed_conn.partner_b)

        if parsed_conn.source_category == "struct_conn":
            conn_type_lower = parsed_conn.conn_type.lower()
            if conn_type_lower == "disulf":
                category: ConnectionCategory = "disulfide"
            elif conn_type_lower == "metalc":
                category = "metal-coordination"
            else:
                category = "covalent"
        elif parsed_conn.source_category == "legacy-pdb-ssbond":
            category = "disulfide"
        else:
            partner_classifications = {
                _classify_connection_partner(parsed_conn.partner_a, non_polymer_lookup),
                _classify_connection_partner(parsed_conn.partner_b, non_polymer_lookup),
            }
            is_metal = "metal-ion" in partner_classifications
            if parsed_conn.source_category == "legacy-pdb-link":
                category = "legacy-pdb-link-metal" if is_metal else "legacy-pdb-link-covalent"
            else:
                category = "legacy-pdb-conect-metal" if is_metal else "legacy-pdb-conect-covalent"

        connections.append(
            StructureConnection(
                struct_conn_id=parsed_conn.struct_conn_id,
                conn_type=parsed_conn.conn_type,
                category=category,
                source_category=parsed_conn.source_category,  # type: ignore[arg-type]
                partner_a=endpoint_a,
                partner_b=endpoint_b,
                distance=parsed_conn.distance,
                value_order=parsed_conn.value_order,
                role=parsed_conn.role,
                details=parsed_conn.details,
                leaving_atom_flag=parsed_conn.leaving_atom_flag,
            )
        )
    return tuple(connections)


def _build_struct_conn_endpoint(endpoint) -> StructConnEndpoint:
    return StructConnEndpoint(
        label_asym_id=endpoint.label_asym_id,
        auth_asym_id=endpoint.auth_asym_id,
        label_entity_id=endpoint.label_entity_id,
        label_seq_id=endpoint.label_seq_id,
        auth_seq_id=endpoint.auth_seq_id,
        insertion_code=endpoint.insertion_code,
        label_comp_id=endpoint.label_comp_id,
        auth_comp_id=endpoint.auth_comp_id,
        label_atom_id=endpoint.label_atom_id,
        auth_atom_id=endpoint.auth_atom_id,
        label_alt_id=endpoint.label_alt_id,
        symmetry=endpoint.symmetry,
        model_number=endpoint.model_number,
    )


def _asym_sort_key(label_asym_id: Optional[str], auth_asym_id: Optional[str]) -> str:
    value = label_asym_id if label_asym_id is not None else auth_asym_id
    return value if value is not None else ""


def _dedupe_preserving_first(items: Tuple) -> Tuple:
    return tuple(dict.fromkeys(items))


def _truncate_with_limit_warning(
    items: list, limit: int, *, category_name: str, warnings: list
) -> list:
    if len(items) > limit:
        warnings.append(
            InspectionWarning(
                code="parser-limitation",
                category_name=category_name,
                detail=f"{category_name} enumeration exceeded {limit}; stopped at the limit",
            )
        )
        return items[:limit]
    return items


def build_receptor_inspection_report(
    parsed: ParsedReceptorDocument,
    *,
    protein_input_id: str,
    raw_format_declared: str,
) -> ReceptorInspectionReport:
    """Build a `ReceptorInspectionReport` from a gemmi-free
    `ParsedReceptorDocument` (Section 14-19). Pure and deterministic: given
    the same document, protein_input_id, and declared format, this always
    produces the same report (modulo nothing - there is no non-deterministic
    input left at this point, Section 33)."""

    warnings: list = []

    is_legacy_pdb = parsed.structure_format == "pdb"
    entities = (
        _build_entities_legacy_pdb(parsed) if is_legacy_pdb else _build_entities_mmcif(parsed)
    )
    chains = _build_chains(parsed)
    non_polymer_components = _build_non_polymer_components(parsed)

    non_polymer_lookup = _build_non_polymer_lookup(parsed, non_polymer_components)
    connections = _build_connections(parsed, non_polymer_lookup)

    altloc_observations, occupancy_observations = _build_altloc_and_occupancy_observations(parsed)

    missing_residues = tuple(
        MissingResidueObservation(
            label_asym_id=item.label_asym_id,
            auth_asym_id=item.auth_asym_id,
            residue_identifier=item.residue_identifier,
            residue_name=item.residue_name,
        )
        for item in (parsed.missing_residues or ())
    )
    missing_atoms = tuple(
        MissingAtomObservation(
            label_asym_id=item.label_asym_id,
            auth_asym_id=item.auth_asym_id,
            residue_identifier=item.residue_identifier,
            atom_name=item.atom_name,
        )
        for item in (parsed.missing_atoms or ())
    )

    explicit_hydrogen_count = sum(
        1
        for residue in parsed.residues
        for atom in residue.atoms
        if atom.element in ("H", "D")
    )
    explicit_hydrogens_present = explicit_hydrogen_count > 0

    disulfide_bonds = tuple(c for c in connections if c.category == "disulfide")
    metal_coordinations = tuple(
        c for c in connections if c.category in ("metal-coordination", "legacy-pdb-link-metal", "legacy-pdb-conect-metal")
    )
    covalent_connections = tuple(
        c
        for c in connections
        if c.category in ("covalent", "legacy-pdb-link-covalent", "legacy-pdb-conect-covalent")
    )

    unavailable_categories = parsed.unavailable_categories

    # --- warnings ---
    if len(parsed.models) > 1:
        warnings.append(
            InspectionWarning(code="multiple-models", detail=f"{len(parsed.models)} models present")
        )

    assembly_ids = {assembly.assembly_id for assembly in parsed.assemblies}
    if len(assembly_ids) > 1:
        warnings.append(
            InspectionWarning(
                code="multiple-assemblies", detail=f"{len(assembly_ids)} assemblies declared"
            )
        )

    for observation in altloc_observations:
        warnings.append(
            InspectionWarning(
                code="alternate-locations-present",
                label_asym_id=observation.label_asym_id,
                auth_asym_id=observation.auth_asym_id,
                residue_identifier=observation.residue_identifier,
                detail=f"residue {observation.residue_identifier} has alternate locations",
            )
        )

    for observation in occupancy_observations:
        warnings.append(
            InspectionWarning(
                code="partial-occupancy-present",
                label_asym_id=observation.label_asym_id,
                auth_asym_id=observation.auth_asym_id,
                residue_identifier=observation.residue_identifier,
                detail=f"atom {observation.atom_name} has partial occupancy {observation.occupancy}",
            )
        )

    for item in missing_residues:
        warnings.append(
            InspectionWarning(
                code="missing-residues-reported",
                label_asym_id=item.label_asym_id,
                auth_asym_id=item.auth_asym_id,
                residue_identifier=item.residue_identifier,
                detail=f"residue {item.residue_identifier} was not observed",
            )
        )

    for item in missing_atoms:
        warnings.append(
            InspectionWarning(
                code="missing-atoms-reported",
                label_asym_id=item.label_asym_id,
                auth_asym_id=item.auth_asym_id,
                residue_identifier=item.residue_identifier,
                detail=f"atom {item.atom_name} of residue {item.residue_identifier} was not observed",
            )
        )

    if not explicit_hydrogens_present:
        warnings.append(
            InspectionWarning(
                code="explicit-hydrogens-absent", detail="no explicit hydrogen atoms were observed"
            )
        )

    for component in non_polymer_components:
        if component.classification in ("other-non-polymer", "unknown"):
            warnings.append(
                InspectionWarning(
                    code="unknown-chemical-component",
                    label_asym_id=component.label_asym_id,
                    auth_asym_id=component.auth_asym_id,
                    ccd_code=component.ccd_code,
                    detail=f"component {component.ccd_code} could not be classified",
                )
            )

    if covalent_connections:
        warnings.append(
            InspectionWarning(
                code="covalent-connection-present",
                detail=f"{len(covalent_connections)} covalent connection(s) present",
            )
        )
    if metal_coordinations:
        warnings.append(
            InspectionWarning(
                code="metal-coordination-present",
                detail=f"{len(metal_coordinations)} metal coordination(s) present",
            )
        )

    if is_legacy_pdb:
        warnings.append(
            InspectionWarning(
                code="legacy-pdb-information-limited",
                detail="legacy PDB input carries no label_asym_id/label_entity_id identity",
            )
        )

    for category_name in unavailable_categories:
        warnings.append(
            InspectionWarning(
                code="category-unavailable",
                category_name=category_name,
                detail=f"category {category_name} is not available from this RAW artifact",
            )
        )

    ambiguous_auth_ids = _find_ambiguous_author_chain_ids(chains)
    for model_number, auth_asym_id in ambiguous_auth_ids:
        warnings.append(
            InspectionWarning(
                code="ambiguous-author-chain-id",
                model_number=model_number,
                auth_asym_id=auth_asym_id,
                detail=f"auth_asym_id {auth_asym_id} maps to multiple label_asym_id values",
            )
        )

    normalized_declared = _DECLARED_FORMAT_ALIASES.get(
        raw_format_declared.strip().lower(), raw_format_declared.strip().lower()
    )
    if normalized_declared != parsed.structure_format:
        warnings.append(
            InspectionWarning(
                code="declared-format-mismatch",
                detail=(
                    f"declared format {raw_format_declared!r} differs from the "
                    f"content-detected format {parsed.structure_format!r}"
                ),
            )
        )

    models_list = list(parsed.models)
    models_list = _truncate_with_limit_warning(models_list, _MAX_MODELS, category_name="models", warnings=warnings)
    chains_list = _truncate_with_limit_warning(
        list(chains), _MAX_CHAINS, category_name="chains", warnings=warnings
    )
    non_polymer_list = _truncate_with_limit_warning(
        list(non_polymer_components),
        _MAX_NON_POLYMER_COMPONENTS,
        category_name="non_polymer_components",
        warnings=warnings,
    )
    connections_list = _truncate_with_limit_warning(
        list(connections),
        _MAX_STRUCTURE_CONNECTIONS,
        category_name="structure_connections",
        warnings=warnings,
    )

    warnings = _dedupe_preserving_first(tuple(warnings))
    warnings = _truncate_with_limit_warning(
        list(warnings), _MAX_RETAINED_WARNINGS, category_name="warnings", warnings=[]
    )

    selection_capability: SelectionCapability = (
        "full"
        if all(entry.label_asym_id_origin == "source" for entry in chains_list)
        and all(entry.label_entity_id_origin == "source" for entry in entities)
        and not is_legacy_pdb
        else "inspection-only"
    )

    inspection_id = compute_inspection_id(
        parser_name=parsed.parser_name,
        parser_version=parsed.parser_version,
        protein_input_id=protein_input_id,
        raw_checksum=parsed.raw_checksum,
        raw_format=parsed.structure_format,
    )

    sorted_entities = _sort_entities(entities)
    sorted_chains = sorted(
        chains_list, key=lambda c: (c.model_number, _asym_sort_key(c.label_asym_id, c.auth_asym_id))
    )
    sorted_non_polymer = sorted(
        non_polymer_list,
        key=lambda c: (_asym_sort_key(c.label_asym_id, c.auth_asym_id), c.ccd_code),
    )
    sorted_missing_residues = tuple(
        sorted(
            _dedupe_preserving_first(missing_residues),
            key=lambda m: (_asym_sort_key(m.label_asym_id, m.auth_asym_id), m.residue_identifier),
        )
    )
    sorted_missing_atoms = tuple(
        sorted(
            _dedupe_preserving_first(missing_atoms),
            key=lambda m: (
                _asym_sort_key(m.label_asym_id, m.auth_asym_id),
                m.residue_identifier,
                m.atom_name,
            ),
        )
    )
    sorted_altloc = tuple(
        sorted(
            _dedupe_preserving_first(altloc_observations),
            key=lambda o: (_asym_sort_key(o.label_asym_id, o.auth_asym_id), o.residue_identifier),
        )
    )
    sorted_occupancy = tuple(
        sorted(
            _dedupe_preserving_first(occupancy_observations),
            key=lambda o: (
                _asym_sort_key(o.label_asym_id, o.auth_asym_id),
                o.residue_identifier,
                o.atom_name,
            ),
        )
    )

    def connection_sort_key(c: StructureConnection):
        return (
            _asym_sort_key(c.partner_a.label_asym_id, c.partner_a.auth_asym_id),
            c.partner_a.auth_seq_id or "",
            _asym_sort_key(c.partner_b.label_asym_id, c.partner_b.auth_asym_id),
            c.partner_b.auth_seq_id or "",
        )

    sorted_connections = sorted(_dedupe_preserving_first(tuple(connections_list)), key=connection_sort_key)
    sorted_disulfide = tuple(c for c in sorted_connections if c.category == "disulfide")
    sorted_metal = tuple(
        c
        for c in sorted_connections
        if c.category in ("metal-coordination", "legacy-pdb-link-metal", "legacy-pdb-conect-metal")
    )
    sorted_covalent = tuple(
        c
        for c in sorted_connections
        if c.category in ("covalent", "legacy-pdb-link-covalent", "legacy-pdb-conect-covalent")
    )

    def warning_sort_key(w: InspectionWarning):
        return (
            w.code,
            w.model_number if w.model_number is not None else -1,
            w.label_asym_id or "",
            w.auth_asym_id or "",
            w.ccd_code or "",
            w.category_name or "",
            w.residue_identifier or "",
        )

    sorted_warnings = tuple(sorted(warnings, key=warning_sort_key))

    try:
        return ReceptorInspectionReport(
            inspection_id=inspection_id,
            inspection_schema_version=INSPECTION_SCHEMA_VERSION,
            inspection_profile_version=INSPECTION_PROFILE_VERSION,
            protein_input_id=protein_input_id,
            raw_checksum=parsed.raw_checksum,
            raw_format_declared=raw_format_declared,
            structure_format=parsed.structure_format,
            parser=ParserProvenance(name=parsed.parser_name, version=parsed.parser_version),
            selection_capability=selection_capability,
            models=tuple(ModelDescriptor(model_number=m.model_number) for m in models_list),
            assemblies=tuple(
                AssemblyDescriptor(
                    assembly_id=assembly.assembly_id,
                    details=assembly.details,
                    generation=tuple(
                        AssemblyOperationExpression(
                            expression=op.expression, asym_id_list=op.asym_id_list
                        )
                        for op in assembly.generation
                    ),
                )
                for assembly in parsed.assemblies
            ),
            entities=sorted_entities,
            chains=tuple(sorted_chains),
            altloc_observations=sorted_altloc,
            occupancy_observations=sorted_occupancy,
            missing_residues=sorted_missing_residues,
            missing_atoms=sorted_missing_atoms,
            disulfide_bonds=sorted_disulfide,
            covalent_connections=sorted_covalent,
            metal_coordinations=sorted_metal,
            non_polymer_components=tuple(sorted_non_polymer),
            explicit_hydrogens_present=explicit_hydrogens_present,
            explicit_hydrogen_count=explicit_hydrogen_count,
            unavailable_categories=unavailable_categories,
            warnings=sorted_warnings,
        )
    except ValueError as error:
        raise InspectionFailureError(
            f"receptor inspection report could not be built: {error.__class__.__name__}"
        ) from error


def _build_non_polymer_lookup(parsed: ParsedReceptorDocument, components: Tuple[NonPolymerComponent, ...]) -> dict:
    lookup = {}
    component_by_key = {}
    for component in components:
        key = (
            component.label_asym_id or component.auth_asym_id,
        )
        component_by_key.setdefault(key, []).append(component)

    for residue in parsed.residues:
        asym = residue.label_asym_id or residue.auth_asym_id
        candidates = component_by_key.get((asym,), [])
        for component in candidates:
            if component.ccd_code == residue.comp_id:
                lookup[(asym, residue.auth_seq_id)] = component.classification
                break
    return lookup


def _find_ambiguous_author_chain_ids(chains: Tuple[ChainInventoryEntry, ...]):
    by_model: dict = {}
    for chain in chains:
        if chain.auth_asym_id is None:
            continue
        key = (chain.model_number, chain.auth_asym_id)
        by_model.setdefault(key, set()).add(chain.label_asym_id)

    return sorted(
        key for key, label_ids in by_model.items() if len(label_ids) > 1
    )


def _sort_entities(entities: Tuple[EntityInventoryEntry, ...]) -> Tuple[EntityInventoryEntry, ...]:
    source_entities = [e for e in entities if e.label_entity_id_origin == "source"]
    unavailable_entities = [e for e in entities if e.label_entity_id_origin != "source"]
    source_entities = sorted(_dedupe_preserving_first(tuple(source_entities)), key=lambda e: e.label_entity_id)
    unavailable_entities = list(_dedupe_preserving_first(tuple(unavailable_entities)))
    return tuple(source_entities) + tuple(unavailable_entities)
