"""RDKit adapter: the only production module permitted to import rdkit.

Route B (rdMolStandardize.CleanupParameters passed to the
TautomerEnumerator constructor) is the only production route for
remove_isotopic_hs in rdkit==2026.3.6, confirmed by the Increment 016
technical spike (dual introspection via dir() and help() of the real,
installed API). Route A (a setter/getter on TautomerEnumerator itself)
does not exist in this version and MUST NOT be attempted, imitated, or
simulated. See test_route_a_absence in tests/test_rdkit_tautomer.py for
the negative contract test guarding this fact against a silent future
change.

The `import rdkit...` block below is the single point of entry for RDKit
into this codebase - no other production module may import rdkit in any
form (Section 6.2 of the frozen spec). It is attempted once, at this
module's own import time (not inside each function), and the resulting
ImportError, if any, is caught right here rather than left to propagate:
this lets molecular.application import this module's names
(resolve_rdkit_engine, enumerate_tautomers_rdkit, and the two public
result models) as ordinary default-argument values, even in a base
installation with no rdkit extra installed - and defers the actual
"rdkit is unavailable" failure to the first point a caller can reach: a
call to resolve_rdkit_engine() (Section 7.1).
"""

from collections import Counter
from pathlib import Path
from typing import Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from molecular.promotion import LigandPromotionPolicy

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError as _rdkit_import_error:  # pragma: no cover - exercised by packaging tests
    Chem = None
    rdBase = None
    rdMolDescriptors = None
    rdMolStandardize = None
    _RDKIT_IMPORT_ERROR: Optional[ImportError] = _rdkit_import_error
else:
    _RDKIT_IMPORT_ERROR = None


EngineName = Literal["rdkit"]
InputFormat = Literal["sdf"]
EnumerationStatus = Literal[
    "completed", "max-tautomers-reached", "max-transforms-reached", "canceled"
]

_ENGINE_NAME: EngineName = "rdkit"
_ALGORITHM = "rdkit.MolStandardize.TautomerEnumerator"
_INPUT_FORMAT: InputFormat = "sdf"
_OUTPUT_FORMAT = "smiles"
_MAX_TAUTOMERS = 1000
_MAX_TRANSFORMS = 1000
_REASSIGN_STEREO = True
_REMOVE_BOND_STEREO = True
_REMOVE_SP3_STEREO = True
_REMOVE_ISOTOPIC_HS = False
_DEDUPLICATION = "canonical-isomeric-smiles"
_ORDERING = "canonical-isomeric-smiles-lexicographic"
_SELECTION = "complete-then-policy-limit"

# D-016-SPIKE-07: the four real TautomerEnumeratorStatus values, confirmed
# by execution (Completed=0, MaxTautomersReached=1, MaxTransformsReached=2,
# Canceled=3), mapped to this adapter's own closed-vocabulary status names.
_STATUS_NAME_MAP: Mapping[str, EnumerationStatus] = {
    "Completed": "completed",
    "MaxTautomersReached": "max-tautomers-reached",
    "MaxTransformsReached": "max-transforms-reached",
    "Canceled": "canceled",
}


class RDKitUnavailableError(RuntimeError):
    """Raised when rdkit cannot be imported, or its version cannot be
    read programmatically. Always a resolution-phase error."""


class InvalidSDFError(ValueError):
    """Raised when the SDF source file is empty, missing, or otherwise
    fails to parse into at least one record. Wraps the OSError
    Chem.SDMolSupplier raises at construction time for an empty/invalid
    file (D-016-SPIKE-03). Always an execution-phase error."""


class MultipleMoleculesError(ValueError):
    """Raised when the SDF source file contains a record count other
    than exactly 1. RDKit does not reject multi-molecule SDF natively;
    this is the platform's own business rule (D-016-SPIKE-03), enforced
    here by checking len(supplier). Always an execution-phase error."""

    def __init__(self, record_count: int) -> None:
        super().__init__(
            f"expected exactly 1 molecule in the SDF source, found {record_count}"
        )
        self.record_count = record_count


class IncompleteEnumerationError(RuntimeError):
    """Raised when TautomerEnumeratorResult.status is not Completed.
    Carries the real status name and the (possibly truncated) tautomer
    count RDKit returned before stopping (D-016-SPIKE-07). Always an
    execution-phase error; the caller MUST NOT publish any candidate from
    an incomplete enumeration (D-016-SPIKE-04's sibling rule)."""

    def __init__(self, status: EnumerationStatus, tautomer_count: int) -> None:
        super().__init__(
            f"tautomer enumeration did not complete: status={status!r}, "
            f"tautomer_count={tautomer_count}"
        )
        self.status = status
        self.tautomer_count = tautomer_count


class RDKitEngineResolution(BaseModel):
    """The fixed, explicit engine configuration this adapter will use,
    resolved BEFORE any molecule is loaded - so an Application Service can
    persist it into ResolvedExecutionSnapshot ahead of invoking
    enumerate_tautomers_rdkit. Every value is fixed by this module's own
    module-level constants (D-016-SPIKE-06); none is read from the caller
    or from any per-molecule state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_name: EngineName
    engine_version: str = Field(min_length=1)
    parameters: Mapping[str, str | int | float | bool]


def resolve_rdkit_engine() -> RDKitEngineResolution:
    """Read rdkit's real version via rdkit.rdBase.rdkitVersion
    (D-016-SPIKE-02 - never pip metadata), and return the fixed, explicit
    parameter set this adapter uses for every enumeration call. rdkit
    itself is imported lazily relative to this module's own import
    (already attempted at module scope above, not inside this function -
    see this module's own docstring and Section 6.2 of the frozen spec
    for why the ImportError boundary still lives here, at the first point
    a caller can reach).

    Raises RDKitUnavailableError if rdkit is not installed, or if
    rdBase.rdkitVersion is missing or unreadable. Never raises anything
    else. Never touches a molecule, a file, or a TautomerEnumerator
    instance - this function is safe to call during an ExecutionAttempt's
    resolution phase, before any execution-phase error becomes possible.
    """

    if _RDKIT_IMPORT_ERROR is not None:
        raise RDKitUnavailableError(
            f"rdkit is not installed or failed to import: {_RDKIT_IMPORT_ERROR}"
        ) from _RDKIT_IMPORT_ERROR

    engine_version = getattr(rdBase, "rdkitVersion", None)
    if not engine_version:
        raise RDKitUnavailableError("rdkit.rdBase.rdkitVersion is missing or unreadable")

    return RDKitEngineResolution(
        engine_name=_ENGINE_NAME,
        engine_version=engine_version,
        parameters={
            "algorithm": _ALGORITHM,
            "input_format": _INPUT_FORMAT,
            "output_format": _OUTPUT_FORMAT,
            "max_tautomers": _MAX_TAUTOMERS,
            "max_transforms": _MAX_TRANSFORMS,
            "remove_bond_stereo": _REMOVE_BOND_STEREO,
            "remove_sp3_stereo": _REMOVE_SP3_STEREO,
            "reassign_stereo": _REASSIGN_STEREO,
            "remove_isotopic_hs": _REMOVE_ISOTOPIC_HS,
            "deduplication": _DEDUPLICATION,
            "ordering": _ORDERING,
            "selection": _SELECTION,
        },
    )


def _build_enumerator():
    """Construct a fresh Route B TautomerEnumerator, with a new
    CleanupParameters object per call (Section 7.2/7.4 of the frozen
    spec) - never reused across calls, never a conditional path toward a
    Route A that does not exist in this rdkit version (Section 8)."""

    params = rdMolStandardize.CleanupParameters()
    params.maxTautomers = _MAX_TAUTOMERS
    params.maxTransforms = _MAX_TRANSFORMS
    params.tautomerReassignStereo = _REASSIGN_STEREO
    params.tautomerRemoveBondStereo = _REMOVE_BOND_STEREO
    params.tautomerRemoveSp3Stereo = _REMOVE_SP3_STEREO
    params.tautomerRemoveIsotopicHs = _REMOVE_ISOTOPIC_HS
    enumerator = rdMolStandardize.TautomerEnumerator(params)
    # Audit: the six effective values are read from the SAME `params`
    # object used at construction, never from a getter on the enumerator -
    # no getter exists for tautomerRemoveIsotopicHs (Section 8.1) and, for
    # a single source of truth (D-016-SPIKE-06), the other five are never
    # re-read from enumerator.GetMaxTautomers() and equivalents either.
    assert params.tautomerRemoveIsotopicHs is False
    return enumerator


class RDKitTautomerCandidate(BaseModel):
    """One tautomer candidate that survived sanitization and every
    chemical invariant check (Section 15.3), expressed as neutral data -
    never an rdkit Mol object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_isomeric_smiles: str = Field(min_length=1)


class RDKitTautomerEnumerationResult(BaseModel):
    """The adapter's neutral output: already sanitized, invariant-
    checked, deduplicated, entry-excluded, and lexicographically sorted.
    Never carries an rdkit object. `candidates` is only ever the SET OF
    VALID NEW candidates - status is not part of this shape at all: a
    non-Completed status is raised as IncompleteEnumerationError before
    any RDKitTautomerEnumerationResult is ever constructed (Section 15,
    step 9), so every instance of this model that exists implicitly
    corresponds to status=Completed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enumerated_unique_count: int = Field(ge=0)
    new_unique_count: int = Field(ge=0)
    candidates: tuple[RDKitTautomerCandidate, ...]

    @model_validator(mode="after")
    def _validate_counts(self) -> "RDKitTautomerEnumerationResult":
        if len(self.candidates) != self.new_unique_count:
            raise ValueError(
                "len(candidates) must equal new_unique_count "
                f"(got {len(self.candidates)} candidates, "
                f"new_unique_count={self.new_unique_count})"
            )
        if self.new_unique_count > self.enumerated_unique_count:
            raise ValueError(
                "new_unique_count must not exceed enumerated_unique_count "
                f"(got new_unique_count={self.new_unique_count}, "
                f"enumerated_unique_count={self.enumerated_unique_count})"
            )
        if self.enumerated_unique_count - self.new_unique_count not in (0, 1):
            raise ValueError(
                "enumerated_unique_count - new_unique_count must be 0 or 1 - "
                "the only physically possible divergence between the two "
                "counts is excluding the input-equivalent candidate zero or "
                "one time (Section 15, step 15); deduplication (step 14) "
                "already guarantees at most one occurrence of the input's "
                "own canonical isomeric SMILES among the unique candidates "
                f"(got enumerated_unique_count={self.enumerated_unique_count}, "
                f"new_unique_count={self.new_unique_count})"
            )
        return self


class InvalidPromotionSDFError(ValueError):
    """`invalid-sdf` (Section 17 of Increment 018's frozen final
    specification). Raised when the SDF source is empty, not SDF at all
    (supplier construction itself fails - Section 9.2, step 3), when the
    single physical record fails to parse into a non-null molecule
    (Section 9.4), or when a molecule accepted under `sanitize=False` fails
    the mandatory posterior sanitization check on a copy (Section 9.2,
    step 8; Section 9.6, layer 5). Distinct from `InvalidSDFError` above:
    that one is the tautomer enumerator's own error for a different call
    site and a different caller contract (Section 9's docstring rule:
    never reuse a tautomer-enumeration decision for the RAW inspection
    contract when the two differ)."""


class PromotionRecordCountError(ValueError):
    """`record-count` (Section 17). Raised whenever the physical record
    count is anything other than exactly one non-null record: zero
    records (supplier constructed successfully but iterated none - never
    empirically reproduced by any RD-018-01 fixture, Section 9.4/9.6, but
    logically required), or two or more records regardless of how many of
    them are individually valid (Section 9.2, step 6 - the single valid
    record is never accepted in isolation when the file contains more than
    one record, in either order)."""

    def __init__(self, record_count: int) -> None:
        super().__init__(
            f"expected exactly 1 physical SDF record, found {record_count}"
        )
        self.record_count = record_count


class DisconnectedStructureError(ValueError):
    """`disconnected-structure` (Section 17). Raised when the single
    accepted molecule has more than one connected fragment (Section 9.4,
    "um registro desconectado"; `LigandPromotionPolicy.component_policy`
    is always `"reject-disconnected"`, Section 6.1)."""


# Controlled vocabulary for warnings (Section 9.2, step 9; Section 6.3's
# invariant that warnings use controlled vocabulary, never a path). Only
# what is genuinely derivable from `source_path` + `policy` alone: an
# expected entity name or PubChem CID to compare against is not part of
# this adapter's frozen signature (Section 9.1) and is therefore never
# compared here - see this module's own docstring/report for that scope
# note.
_WARNING_UNDEFINED_STEREOCENTER = "undefined-stereocenter"
_WARNING_SD_NAME_PROPERTY_ABSENT = "sd-name-property-absent"
_WARNING_SD_CID_PROPERTY_ABSENT = "sd-cid-property-absent"
_SD_NAME_PROPERTY_KEYS = ("PUBCHEM_IUPAC_NAME", "NAME", "Name", "name")
_SD_CID_PROPERTY_KEY = "PUBCHEM_COMPOUND_CID"


class RDKitLigandInspectionResult(BaseModel):
    """The RDKit adapter's neutral inspection result for RAW ligand
    promotion (Section 9.1/9.2 of the frozen spec). Deliberately mirrors
    the RDKit-derived subset of `molecular.promotion.
    ResolvedLigandPromotionSnapshot` field-for-field (everything except
    `raw_checksum_persisted`/`snapshot_checksum`/`snapshot_size_bytes`,
    which come from Route B, not from this adapter) - a future Application
    Service combines the two. No `Mol` object crosses this boundary
    (Section 9.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_name: EngineName
    engine_version: str = Field(min_length=1)
    input_format: InputFormat
    sanitize: bool
    remove_hs: bool
    strict_parsing: bool
    record_count: Literal[1]
    fragment_count: Literal[1]
    formal_charge: int
    atom_count: int = Field(ge=1)
    explicit_hydrogen_count: int = Field(ge=0)
    isotopic_hydrogen_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()


def _classify_record_counts(total: int, valid: int) -> None:
    """The single-molecule rule RDKit does not itself enforce (Section
    9.2, steps 4-6; Section 9.4's table) - a pure decision function over
    the platform's own hand-counted tallies, factored out so the
    total==0 branch (never empirically reproduced by any RD-018-01
    fixture) is directly unit-testable without depending on a real RDKit
    parse producing that exact outcome. Returns None (accept) only when
    exactly one physical record exists and it parsed into a non-null
    molecule."""

    if total == 0:
        raise PromotionRecordCountError(0)
    if total == 1:
        if valid == 1:
            return
        raise InvalidPromotionSDFError(
            "the single record in the SDF source failed to parse into a valid molecule"
        )
    raise PromotionRecordCountError(total)


def inspect_single_sdf_for_promotion(
    *, source_path: Path, policy: LigandPromotionPolicy
) -> RDKitLigandInspectionResult:
    """Inspect a single-molecule SDF snapshot for RAW ligand promotion
    (Section 9.1/9.2 of the frozen spec). `source_path` MUST already be a
    verified private snapshot (Section 8) - this function never validates
    or re-reads the original RAW path, and never writes to `source_path`.

    The three RDKit flags are always taken explicitly from `policy` and
    passed explicitly to `Chem.SDMolSupplier` - RD-018-01 confirmed the
    library's own defaults (`sanitize=True, removeHs=True,
    strictParsing=True`) differ from the production policy
    (`removeHs=False`) and must never be relied upon implicitly (Section
    9.2, step 2; Section 7.1).

    Raises (all resolution-phase, Section 17):
    - RDKitUnavailableError if rdkit cannot be resolved at all.
    - InvalidPromotionSDFError for an empty/non-SDF source (supplier
      construction itself fails), a single record that fails to parse, or
      a molecule that fails the mandatory posterior-sanitization check
      when `sanitize=False` was requested.
    - PromotionRecordCountError for zero or two-or-more physical records
      (Section 9.2, step 6 - never accepts the one valid record in
      isolation when more than one record is present).
    - DisconnectedStructureError for more than one connected fragment.

    Never treats `Chem.MolToSmiles` succeeding as proof of eligibility
    (Section 9.6) - no SMILES is even generated here, precisely because
    this adapter never needs it for anything this contract requires.
    """

    engine = resolve_rdkit_engine()

    try:
        supplier = Chem.SDMolSupplier(
            str(source_path),
            sanitize=policy.sanitize,
            removeHs=policy.remove_hs,
            strictParsing=policy.strict_parsing,
        )
    except OSError as error:
        raise InvalidPromotionSDFError(str(error)) from error

    total = 0
    valid = 0
    sole_mol = None
    with rdBase.BlockLogs():
        for record in supplier:
            total += 1
            if record is not None:
                valid += 1
                sole_mol = record

    _classify_record_counts(total, valid)
    mol = sole_mol
    if mol is None:  # pragma: no cover - guaranteed unreachable by _classify_record_counts
        raise InvalidPromotionSDFError("no valid molecule was produced")

    fragments = Chem.GetMolFrags(mol, sanitizeFrags=False)
    if len(fragments) != 1:
        raise DisconnectedStructureError(
            f"expected exactly 1 connected fragment, found {len(fragments)}"
        )

    # Section 9.2, step 7: measurements are taken on the ORIGINAL mol,
    # strictly before any posterior sanitization (step 8) - never on the
    # sanitized copy, and never reordered relative to step 8 below.
    formal_charge = Chem.GetFormalCharge(mol)
    atom_count = mol.GetNumAtoms()
    explicit_hydrogen_count = sum(
        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 0
    )
    isotopic_hydrogen_count = sum(
        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1 and atom.GetIsotope() != 0
    )

    warnings: list[str] = []
    try:
        chiral_centers = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
    except Exception:
        # Stereo perception can require sanitized ring/aromaticity
        # information this module does not force onto the original mol
        # (Section 9.2 forbids sanitizing the original at all) - an
        # inability to determine stereo here is not itself grounds to
        # reject the molecule; it simply means this specific warning
        # cannot be produced for this input.
        chiral_centers = ()
    if any(tag == "?" for _, tag in chiral_centers):
        warnings.append(_WARNING_UNDEFINED_STEREOCENTER)

    if not any(mol.HasProp(key) for key in _SD_NAME_PROPERTY_KEYS):
        warnings.append(_WARNING_SD_NAME_PROPERTY_ABSENT)
    if not mol.HasProp(_SD_CID_PROPERTY_KEY):
        warnings.append(_WARNING_SD_CID_PROPERTY_ABSENT)

    # Section 9.2, step 8: only when sanitize=False was requested, verify
    # chemical usability via Chem.SanitizeMol on a COPY - never the
    # original object, never the source file. This runs strictly AFTER
    # the measurements above, exactly matching the algorithm's own order.
    if not policy.sanitize:
        mol_copy = Chem.Mol(mol)
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol_copy)
        except Chem.MolSanitizeException as error:
            raise InvalidPromotionSDFError(
                f"molecule failed sanitization on a copy: {error}"
            ) from error

    return RDKitLigandInspectionResult(
        engine_name=engine.engine_name,
        engine_version=engine.engine_version,
        input_format=_INPUT_FORMAT,
        sanitize=policy.sanitize,
        remove_hs=policy.remove_hs,
        strict_parsing=policy.strict_parsing,
        record_count=1,
        fragment_count=1,
        formal_charge=formal_charge,
        atom_count=atom_count,
        explicit_hydrogen_count=explicit_hydrogen_count,
        isotopic_hydrogen_count=isotopic_hydrogen_count,
        warnings=tuple(warnings),
    )


def _formula(mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


def _isotope_counts(mol) -> "Counter[int]":
    return Counter(atom.GetIsotope() for atom in mol.GetAtoms() if atom.GetIsotope() != 0)


def enumerate_tautomers_rdkit(
    *, source_path: Path, source_format: InputFormat
) -> RDKitTautomerEnumerationResult:
    """Enumerate tautomers of the single molecule in the SDF file at
    source_path, using Route B exclusively (Section 7.2), and return only
    the sanitized, invariant-checked, deduplicated, entry-excluded,
    lexicographically-sorted set of new candidates.

    source_format is currently only ever "sdf" - the parameter exists so
    a future format addition is a signature-compatible extension, not a
    breaking change; this function raises ValueError immediately for any
    other value, without touching the filesystem.

    Raises (all execution-phase; see Section 18):
    - InvalidSDFError if the file is empty, missing, or otherwise fails
      to construct a Chem.SDMolSupplier (D-016-SPIKE-03).
    - MultipleMoleculesError if the SDF does not contain exactly one
      record.
    - IncompleteEnumerationError if TautomerEnumeratorResult.status is
      not Completed.

    Never raises RDKitUnavailableError - that is resolve_rdkit_engine's
    exclusive responsibility, and MUST already have succeeded before this
    function is ever called (Section 15, resolution phase).

    Performs, in this exact order (Section 15, execution-phase steps):
    load the SDF via Chem.SDMolSupplier; require exactly one record;
    build the Route B enumerator (Section 7.2); call Enumerate(mol);
    require status Completed; sanitize every candidate independently via
    Chem.SanitizeMol, discarding (never raising for) any that fails;
    verify element counts, isotope counts, and formal charge against the
    input molecule for every candidate that did sanitize, discarding any
    that diverges; serialize the surviving candidates to canonical
    isomeric SMILES; deduplicate by that string; exclude the string equal
    to the input molecule's own canonical isomeric SMILES; sort the
    remainder lexicographically.
    """

    if source_format != "sdf":
        raise ValueError(
            f"unsupported source_format: {source_format!r} (only 'sdf' is supported)"
        )

    try:
        supplier = Chem.SDMolSupplier(str(source_path))
    except OSError as error:
        raise InvalidSDFError(str(error)) from error

    record_count = len(supplier)
    if record_count != 1:
        raise MultipleMoleculesError(record_count)

    input_mol = supplier[0]
    if input_mol is None:
        raise InvalidSDFError(
            f"the single record in {source_path} failed to parse into a valid molecule"
        )

    entry_formula = _formula(input_mol)
    entry_charge = Chem.GetFormalCharge(input_mol)
    entry_isotopes = _isotope_counts(input_mol)
    entry_smiles = Chem.MolToSmiles(input_mol, isomericSmiles=True)

    enumerator = _build_enumerator()
    with rdBase.BlockLogs():
        result = enumerator.Enumerate(input_mol)

    if result.status != rdMolStandardize.TautomerEnumeratorStatus.Completed:
        raise IncompleteEnumerationError(
            status=_STATUS_NAME_MAP[result.status.name],
            tautomer_count=len(result.tautomers),
        )

    valid_smiles: list[str] = []
    for candidate in result.tautomers:
        try:
            Chem.SanitizeMol(candidate)
        except Chem.MolSanitizeException:
            continue

        if _formula(candidate) != entry_formula:
            continue
        if Chem.GetFormalCharge(candidate) != entry_charge:
            continue
        if _isotope_counts(candidate) != entry_isotopes:
            continue

        valid_smiles.append(Chem.MolToSmiles(candidate, isomericSmiles=True))

    # Section 15, steps 14-16, in this exact order: deduplicate, THEN
    # exclude the entry-equivalent representation, THEN sort the
    # remainder lexicographically - the code must follow the same order
    # this function's own docstring describes, not merely produce an
    # equivalent result via a different order.
    enumerated_unique = set(valid_smiles)
    new_unique = {smiles for smiles in enumerated_unique if smiles != entry_smiles}
    sorted_new_unique = sorted(new_unique)

    return RDKitTautomerEnumerationResult(
        enumerated_unique_count=len(enumerated_unique),
        new_unique_count=len(sorted_new_unique),
        candidates=tuple(
            RDKitTautomerCandidate(canonical_isomeric_smiles=smiles)
            for smiles in sorted_new_unique
        ),
    )
