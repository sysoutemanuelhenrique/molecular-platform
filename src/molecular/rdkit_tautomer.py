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

import math
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from molecular.characterization import (
    CharacterizationConfigurationError,
    CharacterizationEffectiveParameters,
    CharacterizationError,
    CharacterizationExecutionError,
    CharacterizationInputError,
    CharacterizationStorageError,
    CharacterizationValues,
    Compute2DParameters,
    DepictionEffectiveParameters,
    DrawingResult,
    MAX_BOND_COUNT,
    MAX_TOTAL_ATOM_COUNT,
    RDKitCharacterizationOutcome,
    REQUIRED_ENGINE_VERSION,
    SdfParserParameters,
    SmilesParserParameters,
    validate_and_bind_depiction_svg,
)
from molecular.inputs import validate_input_id
from molecular.promotion import LigandPromotionPolicy

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError as _rdkit_import_error:  # pragma: no cover - exercised by packaging tests
    Chem = None
    rdBase = None
    Descriptors = None
    rdMolDescriptors = None
    rdMolStandardize = None
    _RDKIT_IMPORT_ERROR: Optional[ImportError] = _rdkit_import_error
else:
    _RDKIT_IMPORT_ERROR = None

# Section 15 of the Increment 020-A frozen spec: drawing availability is
# deliberately separate from calculation availability - a missing Draw/
# RDConfig import (or, later, a missing font asset) must never shadow the
# legacy `_RDKIT_IMPORT_ERROR` sentinel above, and must never make
# calculation-only characterization (include_svg=False) or the existing
# tautomer enumeration unavailable.
try:
    from rdkit import RDConfig
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError as _rdkit_drawing_import_error:  # pragma: no cover - exercised by packaging tests
    RDConfig = None
    rdDepictor = None
    rdMolDraw2D = None
    _RDKIT_DRAWING_IMPORT_ERROR: Optional[ImportError] = _rdkit_drawing_import_error
else:
    _RDKIT_DRAWING_IMPORT_ERROR = None


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
# Checkpoint 3 correction (MINOR-B1): `_Name` is RDKit's own special
# property, auto-populated from the MOL block's title line (distinct from
# an explicit SD data field such as `>  <NAME>`) - a real, common pattern
# (e.g. ChemDraw/OpenBabel exports that only ever populate the title line).
# Recognized here so such a file is never incorrectly flagged as nameless.
_SD_NAME_PROPERTY_KEYS = ("PUBCHEM_IUPAC_NAME", "NAME", "Name", "name", "_Name")
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
        raise InvalidPromotionSDFError(
            f"the SDF source could not be parsed by the supplier: {error.__class__.__name__}"
        ) from error

    total = 0
    valid = 0
    sole_mol = None
    try:
        with rdBase.BlockLogs():
            for record in supplier:
                total += 1
                if record is not None:
                    valid += 1
                    sole_mol = record
    except Exception as error:
        # Checkpoint 3 correction (MAJOR-B1): construction succeeding is no
        # guarantee that iterating every record will not itself raise (the
        # RDKit C++/Boost.Python layer can raise mid-iteration on a
        # malformed record, distinct from simply yielding `None`). Never
        # let a raw framework exception escape this adapter's closed error
        # taxonomy (Section 17), and never accept the partial `total`/
        # `valid` tally accumulated so far as if iteration had completed.
        # `except Exception` deliberately excludes KeyboardInterrupt/
        # SystemExit/GeneratorExit (direct BaseException subclasses).
        raise InvalidPromotionSDFError(
            f"iterating the SDF source raised {error.__class__.__name__}"
        ) from error

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

    # `_Name` is RDKit's own auto-populated title-line property: it is
    # ALWAYS present (HasProp is True) once a molecule has gone through
    # Chem.SDWriter, even when the title line is blank - so presence alone
    # is not "has a real name" for this one key. A blank value carries no
    # more name information than no value at all, so it is treated as
    # absent for every key checked here, not only `_Name`.
    if not any(mol.HasProp(key) and mol.GetProp(key).strip() for key in _SD_NAME_PROPERTY_KEYS):
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
                f"molecule failed sanitization on a copy: {error.__class__.__name__}"
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


# =========================================================================
# Increment 020-A, Checkpoint 1: molecular characterization RDKit adapter
# (Sections 4/6/7/8/9/10/12 of the frozen spec). Bytes in, neutral DTO
# out - no filesystem I/O, no TautomerEnumerator reuse, and every DTO/
# error name used below comes from molecular.characterization, never a
# new ad hoc shape defined in this module.
# =========================================================================

_SD_DELIMITER_RE = re.compile(r"\r\n|\n")
_SMILES_ASCII_MAX_ORDINAL = 0x7E


def _resolve_characterization_engine() -> tuple[str, str]:
    """Section 8: resolve+pin the RDKit engine for characterization,
    independent of enumerate_tautomers_rdkit's own resolve_rdkit_engine
    (a different profile with its own parameter set). Requires the exact
    observed version string 2026.03.6 - no other engine or version is
    substituted."""

    if _RDKIT_IMPORT_ERROR is not None:
        raise CharacterizationConfigurationError(
            "the rdkit engine is not installed or failed to import",
            reason="engine-unavailable",
        )
    engine_version = getattr(rdBase, "rdkitVersion", None)
    if not engine_version or engine_version != REQUIRED_ENGINE_VERSION:
        raise CharacterizationConfigurationError(
            f"rdkit engine version must be {REQUIRED_ENGINE_VERSION!r} "
            f"(observed {engine_version!r})",
            reason="engine-unavailable",
        )
    return _ENGINE_NAME, engine_version


def _validate_sdf_grammar(data: bytes) -> None:
    """Section 7.1: the exact SDF physical grammar, checked on the raw
    bytes before RDKit ever sees them. Never sniffs, never strips, never
    tolerates a lone CR or anything after the single terminal `$$$$`
    line."""

    if b"\x00" in data:
        raise CharacterizationExecutionError(
            "SDF source must not contain a NUL byte", reason="invalid-chemical-input"
        )
    if data.startswith(b"\xef\xbb\xbf"):
        raise CharacterizationExecutionError(
            "SDF source must not have a UTF-8 BOM", reason="invalid-chemical-input"
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CharacterizationExecutionError(
            "SDF source is not strict UTF-8", reason="invalid-chemical-input"
        ) from error

    index = 0
    while True:
        index = text.find("\r", index)
        if index == -1:
            break
        if index + 1 >= len(text) or text[index + 1] != "\n":
            raise CharacterizationExecutionError(
                "SDF source must not contain a lone CR", reason="invalid-chemical-input"
            )
        index += 1

    lines = _SD_DELIMITER_RE.split(text)
    delimiter_count = lines.count("$$$$")
    if delimiter_count != 1:
        raise CharacterizationExecutionError(
            "SDF source must contain exactly one physical record, terminated "
            f"by exactly one '$$$$' line (found {delimiter_count})",
            reason="invalid-chemical-input",
        )
    delimiter_index = lines.index("$$$$")
    remainder = lines[delimiter_index + 1 :]
    if remainder not in ([], [""]):
        raise CharacterizationExecutionError(
            "nothing may follow the terminal '$$$$' line except a single "
            "trailing line terminator",
            reason="invalid-chemical-input",
        )


def _validate_smiles_grammar(data: bytes) -> str:
    """Section 7.2: the exact SMILES physical grammar. Returns the bare
    token (never including the trailing terminator) for RDKit to parse."""

    try:
        text = data.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise CharacterizationExecutionError(
            "SMILES source must be strict ASCII", reason="invalid-chemical-input"
        ) from error

    if "\x00" in text or "\x7f" in text:
        raise CharacterizationExecutionError(
            "SMILES source must not contain NUL or DEL", reason="invalid-chemical-input"
        )

    carriage_return_count = text.count("\r")
    if carriage_return_count:
        if carriage_return_count != 1 or not text.endswith("\r\n"):
            raise CharacterizationExecutionError(
                "SMILES source must not contain a lone CR", reason="invalid-chemical-input"
            )

    newline_count = text.count("\n")
    if newline_count:
        if newline_count != 1 or not text.endswith("\n"):
            raise CharacterizationExecutionError(
                "SMILES source must have at most one trailing LF/CRLF, nothing else",
                reason="invalid-chemical-input",
            )

    if text.endswith("\r\n"):
        token = text[:-2]
    elif text.endswith("\n"):
        token = text[:-1]
    else:
        token = text

    if not token:
        raise CharacterizationExecutionError(
            "SMILES source must contain a non-empty token", reason="invalid-chemical-input"
        )
    for character in token:
        if ord(character) < 0x20 or ord(character) == 0x7F:
            raise CharacterizationExecutionError(
                "SMILES token must not contain control characters",
                reason="invalid-chemical-input",
            )
        if character.isspace():
            raise CharacterizationExecutionError(
                "SMILES token must not contain whitespace",
                reason="invalid-chemical-input",
            )
    return token


def _parse_molecule_rdkit(representation_bytes: bytes, *, input_format: str):
    """Sections 7.1/7.2/12 step 5: grammar validation, then exactly one
    RDKit parse call, with native logs suppressed for the call's own
    minimal scope (Section 13's logging rule)."""

    if input_format == "sdf":
        _validate_sdf_grammar(representation_bytes)
        try:
            with rdBase.BlockLogs():
                supplier = Chem.ForwardSDMolSupplier(
                    BytesIO(representation_bytes),
                    sanitize=True,
                    removeHs=False,
                    strictParsing=True,
                )
                items = list(supplier)
        except Exception as error:
            raise CharacterizationExecutionError(
                f"the SDF source could not be parsed: {error.__class__.__name__}",
                reason="invalid-chemical-input",
            ) from error
        if len(items) != 1 or items[0] is None:
            raise CharacterizationExecutionError(
                "the SDF source must contain exactly one valid molecule",
                reason="invalid-chemical-input",
            )
        return items[0]

    if input_format == "smiles":
        token = _validate_smiles_grammar(representation_bytes)
        params = Chem.SmilesParserParams()
        params.sanitize = True
        params.removeHs = False
        params.parseName = False
        params.allowCXSMILES = False
        try:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(token, params)
        except Exception as error:
            raise CharacterizationExecutionError(
                f"the SMILES source could not be parsed: {error.__class__.__name__}",
                reason="invalid-chemical-input",
            ) from error
        if mol is None:
            raise CharacterizationExecutionError(
                "the SMILES token did not parse into a valid molecule",
                reason="invalid-chemical-input",
            )
        return mol

    raise CharacterizationConfigurationError(
        f"unsupported representation format: {input_format!r}",
        reason="unsupported-format",
    )


def _validate_graph_eligibility(mol) -> None:
    """Section 7.3: graph eligibility, checked before size limits and
    before any value is computed. Never fragments, neutralizes, or
    otherwise edits the molecule to make it eligible."""

    if mol.GetNumAtoms() == 0:
        raise CharacterizationExecutionError(
            "the molecule is empty", reason="unsupported-structure"
        )

    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise CharacterizationExecutionError(
            "dummy atoms (atomic number 0) are not permitted",
            reason="unsupported-structure",
        )
    if any(atom.HasQuery() for atom in mol.GetAtoms()) or any(
        bond.HasQuery() for bond in mol.GetBonds()
    ):
        raise CharacterizationExecutionError(
            "query atoms/bonds are not permitted", reason="unsupported-structure"
        )
    if len(Chem.GetMolSubstanceGroups(mol)) > 0:
        raise CharacterizationExecutionError(
            "SGroup/polymeric structures are not permitted",
            reason="unsupported-structure",
        )
    for group in mol.GetStereoGroups():
        if group.GetGroupType() in (
            Chem.StereoGroupType.STEREO_AND,
            Chem.StereoGroupType.STEREO_OR,
        ):
            raise CharacterizationExecutionError(
                "enhanced stereo AND/OR groups are not permitted",
                reason="unsupported-structure",
            )

    fragments = Chem.GetMolFrags(mol, sanitizeFrags=False)
    if len(fragments) != 1:
        raise CharacterizationExecutionError(
            "exactly one connected component is required "
            f"(found {len(fragments)}; no component was selected or removed)",
            reason="unsupported-structure",
        )


def _validate_size_limits(mol) -> None:
    """Section 6: the inclusive atom/bond ceilings, checked after parsing
    and after eligibility, before any descriptor is computed."""

    atom_count = mol.GetNumAtoms()
    if atom_count > MAX_TOTAL_ATOM_COUNT:
        raise CharacterizationExecutionError(
            f"atom count {atom_count} exceeds the limit of {MAX_TOTAL_ATOM_COUNT}",
            reason="molecular-size-limit",
        )
    bond_count = mol.GetNumBonds()
    if bond_count > MAX_BOND_COUNT:
        raise CharacterizationExecutionError(
            f"bond count {bond_count} exceeds the limit of {MAX_BOND_COUNT}",
            reason="molecular-size-limit",
        )


def _compute_characterization_values(mol) -> CharacterizationValues:
    """Section 8: the ten values, each via the explicit method/kwargs the
    frozen spec pins - never a different overload, and never a value
    substituted by zero/null on failure (Section 11's invariant)."""

    try:
        with rdBase.BlockLogs():
            formula = rdMolDescriptors.CalcMolFormula(
                mol, separateIsotopes=True, abbreviateHIsotopes=False
            )
            molar_mass = Descriptors.MolWt(mol)
            exact_mass = rdMolDescriptors.CalcExactMolWt(mol, onlyHeavy=False)
            formal_charge = Chem.GetFormalCharge(mol)
            heavy_atom_count = mol.GetNumHeavyAtoms()
            hbd_count = rdMolDescriptors.CalcNumHBD(mol)
            hba_count = rdMolDescriptors.CalcNumHBA(mol)
            rotatable_bond_count = rdMolDescriptors.CalcNumRotatableBonds(
                mol, strict=rdMolDescriptors.NumRotatableBondsOptions.Strict
            )
            tpsa = rdMolDescriptors.CalcTPSA(mol, force=True, includeSandP=False)
            clogp = rdMolDescriptors.CalcCrippenDescriptors(
                mol, includeHs=True, force=True
            )[0]
    except Exception as error:
        raise CharacterizationExecutionError(
            f"a descriptor method failed: {error.__class__.__name__}",
            reason="descriptor-failure",
        ) from error

    if not formula:
        raise CharacterizationExecutionError(
            "formula must not be empty", reason="descriptor-failure"
        )
    for value in (molar_mass, exact_mass, tpsa, clogp):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
            value
        ):
            raise CharacterizationExecutionError(
                "a descriptor returned a non-finite or mistyped value",
                reason="descriptor-failure",
            )
    for value in (formal_charge, heavy_atom_count, hbd_count, hba_count, rotatable_bond_count):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CharacterizationExecutionError(
                "a descriptor returned a non-integer or mistyped value",
                reason="descriptor-failure",
            )

    try:
        return CharacterizationValues(
            formula=formula,
            molar_mass_g_mol=molar_mass,
            exact_mass_da=exact_mass,
            formal_charge_e=formal_charge,
            heavy_atom_count=heavy_atom_count,
            hbd_count=hbd_count,
            hba_count=hba_count,
            rotatable_bond_count=rotatable_bond_count,
            tpsa_a2=tpsa,
            clogp_wildman_crippen=clogp,
        )
    except Exception as error:
        raise CharacterizationExecutionError(
            f"the computed values failed the closed DTO contract: {error}",
            reason="descriptor-failure",
        ) from error


# Section 9: the exact annotation properties omitted from the drawing
# copy before any coordinate/preparation step - never a chemical field.
_DRAWING_ANNOTATION_PROPERTIES_TO_CLEAR = (
    "atomLabel",
    "_displayLabel",
    "_displayLabelW",
    "molFileAlias",
    "atomNote",
    "bondNote",
    "_MolFileRLabel",
)


def _clear_drawing_annotation_properties(mol) -> None:
    for atom in mol.GetAtoms():
        for key in _DRAWING_ANNOTATION_PROPERTIES_TO_CLEAR:
            if atom.HasProp(key):
                atom.ClearProp(key)
    for bond in mol.GetBonds():
        for key in _DRAWING_ANNOTATION_PROPERTIES_TO_CLEAR:
            if bond.HasProp(key):
                bond.ClearProp(key)


def _graph_signature(mol) -> tuple:
    """Section 9's before/after comparison: per-index atomic numbers,
    isotopes, formal charges, explicit/implicit H counts, and bond
    orders/aromaticity - never conformer/wedge direction, which are
    presentation, not chemistry."""

    atoms = tuple(
        (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            atom.GetNumExplicitHs(),
            atom.GetNumImplicitHs(),
        )
        for atom in mol.GetAtoms()
    )
    # PrepareMolForDrawing may flip a bond's begin/end atom order to orient
    # a wedge outward from its stereocenter (a presentation convention,
    # never a connectivity change) - the endpoints are therefore compared
    # as an unordered pair, never a directional (begin, end) tuple.
    bonds = frozenset(
        (
            frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())),
            bond.GetBondType(),
            bond.GetIsAromatic(),
        )
        for bond in mol.GetBonds()
    )
    return atoms, bonds


def _resolve_font_path_and_hash() -> tuple[Path, str]:
    import hashlib

    font_path = Path(RDConfig.RDDataDir) / "Fonts" / "Telex-Regular.ttf"
    if not font_path.is_file():
        raise CharacterizationConfigurationError(
            "the RDKit Telex-Regular.ttf font asset is unavailable",
            reason="drawing-unavailable",
        )
    font_bytes = font_path.read_bytes()
    return font_path, hashlib.sha256(font_bytes).hexdigest()


def _build_drawing_result(
    mol,
    *,
    input_state_id: str,
    representation_sha256: str,
    engine_version: str,
) -> tuple[DrawingResult, str]:
    """Sections 9/10/10.2: produce, validate, and bind a 2D depiction SVG
    for `mol`. Raises CharacterizationConfigurationError(reason=
    "drawing-unavailable") if the drawing modules/font are unavailable,
    or CharacterizationExecutionError(reason="depiction-failure") for any
    graph-invariant or SVG-profile violation. `mol` itself is never
    mutated - only a fresh copy is prepared for drawing."""

    if _RDKIT_DRAWING_IMPORT_ERROR is not None:
        raise CharacterizationConfigurationError(
            "the rdkit drawing modules (Draw/RDConfig) are unavailable",
            reason="drawing-unavailable",
        )

    font_path, font_sha256 = _resolve_font_path_and_hash()

    draw_mol = Chem.Mol(mol)
    _clear_drawing_annotation_properties(draw_mol)
    before_signature = _graph_signature(draw_mol)
    before_isomeric_smiles = Chem.MolToSmiles(draw_mol, isomericSmiles=True)

    try:
        with rdBase.BlockLogs():
            rdDepictor.Compute2DCoords(
                draw_mol,
                canonOrient=True,
                clearConfs=True,
                coordMap={},
                nFlipsPerSample=0,
                nSample=0,
                sampleSeed=0,
                permuteDeg4Nodes=False,
                bondLength=-1.0,
                forceRDKit=True,
                useRingTemplates=False,
            )
            prepared = rdMolDraw2D.PrepareMolForDrawing(
                draw_mol,
                kekulize=False,
                addChiralHs=False,
                wedgeBonds=True,
                forceCoords=False,
                wavyBonds=False,
            )
    except Exception as error:
        raise CharacterizationExecutionError(
            f"2D coordinate generation/preparation failed: {error.__class__.__name__}",
            reason="depiction-failure",
        ) from error

    after_signature = _graph_signature(prepared)
    if before_signature != after_signature:
        raise CharacterizationExecutionError(
            "the drawing preparation altered the molecule's chemical invariants",
            reason="depiction-failure",
        )
    try:
        with rdBase.BlockLogs():
            after_isomeric_smiles = Chem.MolToSmiles(prepared, isomericSmiles=True)
    except Exception as error:
        raise CharacterizationExecutionError(
            f"post-preparation stereochemistry check failed: {error.__class__.__name__}",
            reason="depiction-failure",
        ) from error
    if before_isomeric_smiles != after_isomeric_smiles:
        raise CharacterizationExecutionError(
            "the drawing preparation altered stereochemistry",
            reason="depiction-failure",
        )

    try:
        with rdBase.BlockLogs():
            drawer = rdMolDraw2D.MolDraw2DSVG(600, 400, -1, -1, False)
            options = drawer.drawOptions()
            options.prepareMolsBeforeDrawing = False
            options.includeMetadata = False
            options.includeAtomTags = False
            options.addAtomIndices = False
            options.addBondIndices = False
            options.fontFile = str(font_path)
            drawer.DrawMolecule(prepared, legend="")
            drawer.FinishDrawing()
            raw_svg = drawer.GetDrawingText()
    except Exception as error:
        raise CharacterizationExecutionError(
            f"SVG rendering failed: {error.__class__.__name__}",
            reason="depiction-failure",
        ) from error

    final_svg = validate_and_bind_depiction_svg(
        raw_svg,
        input_state_id=input_state_id,
        representation_sha256=representation_sha256,
        engine_version=engine_version,
    )
    final_bytes = final_svg.encode("utf-8")

    import hashlib
    import xml.parsers.expat as expat

    drawing = DrawingResult(
        font_name=font_path.name,
        font_sha256_observed=font_sha256,
        svg_sha256=hashlib.sha256(final_bytes).hexdigest(),
        svg_size_bytes=len(final_bytes),
        xml_parser_version_observed=expat.EXPAT_VERSION,
        effective_parameters=DepictionEffectiveParameters(
            compute2d=Compute2DParameters(),
        ),
    )
    return drawing, final_svg


def characterize_molecule_rdkit(
    representation_bytes: bytes,
    *,
    input_format: str,
    input_state_id: str,
    representation_sha256: str,
    include_svg: bool = False,
) -> RDKitCharacterizationOutcome:
    """Section 4/12: the RDKit adapter's single public entry point.
    `representation_bytes` must already be a verified buffer (Section 6's
    reader protocol is the caller's responsibility, never this
    function's) - this function performs NO filesystem I/O of its own,
    never reopens anything, and calls into RDKit exactly once per
    molecule for parsing (Section 12, step 4/5: "uma vez").

    Raises (never anything outside this closed taxonomy):
    - CharacterizationInputError(reason="invalid-request") for a
      malformed include_svg or input_state_id.
    - CharacterizationConfigurationError for an unavailable/mismatched
      engine, an unsupported format, or unavailable drawing.
    - CharacterizationStorageError(reason="checksum-mismatch") if
      representation_sha256 does not match representation_bytes.
    - CharacterizationExecutionError for any chemistry/drawing failure.
    """

    if not isinstance(include_svg, bool):
        raise CharacterizationInputError(
            "include_svg must be a strict bool", reason="invalid-request"
        )
    if not isinstance(input_state_id, str):
        raise CharacterizationInputError(
            "input_state_id must be a str", reason="invalid-request"
        )
    try:
        validate_input_id(input_state_id)
    except ValueError as error:
        raise CharacterizationInputError(
            f"input_state_id is not a valid identifier: {error}",
            reason="invalid-request",
        ) from error

    engine_name, engine_version = _resolve_characterization_engine()

    if not isinstance(input_format, str):
        raise CharacterizationInputError(
            "input_format must be a str", reason="invalid-request"
        )
    if input_format not in ("sdf", "smiles"):
        raise CharacterizationConfigurationError(
            f"unsupported representation format: {input_format!r}",
            reason="unsupported-format",
        )

    if not isinstance(representation_sha256, str):
        raise CharacterizationInputError(
            "representation_sha256 must be a str", reason="invalid-request"
        )
    import hashlib

    observed_sha256 = hashlib.sha256(representation_bytes).hexdigest()
    if observed_sha256 != representation_sha256.lower():
        raise CharacterizationStorageError(
            "representation_sha256 does not match representation_bytes",
            reason="checksum-mismatch",
            input_state_id=input_state_id,
        )

    try:
        mol = _parse_molecule_rdkit(
            representation_bytes, input_format=input_format
        )
        _validate_graph_eligibility(mol)
        _validate_size_limits(mol)
        values = _compute_characterization_values(mol)
    except CharacterizationExecutionError as error:
        if error.input_state_id is None:
            error.input_state_id = input_state_id
        raise

    if input_format == "sdf":
        parser_block = SdfParserParameters()
    else:
        parser_block = SmilesParserParameters()
    effective_parameters = CharacterizationEffectiveParameters(parser=parser_block)

    drawing: Optional[DrawingResult] = None
    svg: Optional[str] = None
    warnings: tuple[str, ...] = ("derived-computational-values", "tpsa-no-additional-s-p")

    if include_svg:
        try:
            drawing, svg = _build_drawing_result(
                mol,
                input_state_id=input_state_id,
                representation_sha256=representation_sha256,
                engine_version=engine_version,
            )
        except CharacterizationError as error:
            if error.input_state_id is None:
                error.input_state_id = input_state_id
            raise
        warnings = warnings + ("generated-2d-not-conformer",)

    try:
        return RDKitCharacterizationOutcome(
            engine_name=engine_name,
            engine_version=engine_version,
            representation_format_resolved=input_format,
            values=values,
            effective_parameters=effective_parameters,
            warnings=warnings,
            drawing=drawing,
            svg=svg,
        )
    except Exception as error:
        raise CharacterizationExecutionError(
            f"the adapter outcome failed the closed DTO contract: {error}",
            reason="invalid-engine-result",
            input_state_id=input_state_id,
        ) from error
