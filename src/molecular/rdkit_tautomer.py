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

    enumerated_unique = sorted(set(valid_smiles))
    new_unique = [smiles for smiles in enumerated_unique if smiles != entry_smiles]

    return RDKitTautomerEnumerationResult(
        enumerated_unique_count=len(enumerated_unique),
        new_unique_count=len(new_unique),
        candidates=tuple(
            RDKitTautomerCandidate(canonical_isomeric_smiles=smiles) for smiles in new_unique
        ),
    )
