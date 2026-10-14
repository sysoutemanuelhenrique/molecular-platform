"""Application Service for receptor inspection/selection/resolution
(Increment 020, Checkpoint 3).

This module is the only orchestrator of `inspect_receptor`/
`declare_receptor_selection`/`resolve_receptor_selection` (Section 25 of
the frozen spec): it invokes the Checkpoint 1/2 domain and registry layers
(`receptor_parsing.py`/`receptor_inspection.py`/`receptor_selection.py`/
`receptor_resolution.py`/`receptor_registry.py`), enforces
`selection_capability == "full"` before any chain/component field is even
examined, decides retry-vs-conflict under a colliding id, resolves
scientist-declared identifiers against a loaded inspection report, and
owns best-effort `started`/`success`/`error` activity logging for all
three operations (Section 31.1) - none of that orchestration lives in the
Checkpoint 1/2 layers themselves, exactly as their own docstrings already
state.

This module also re-exports thin, non-logging wrappers for the four pure
reads already built by `receptor_registry.py`
(`load_receptor_inspection`/`load_receptor_selection`/
`load_resolved_receptor_selection`/`list_receptor_selections`) so that
every one of Section 25.1's seven public functions is reachable from this
one module - mirroring `conditions_service.py`'s own
`load_conditions`/`list_conditions` wrappers.

Named `component_dispositions` resolution (Section 25.3/25.5 do not name
an edge case for "a scientist names a component absent from the
inspection") reuses `ChainNotFoundError` for that absence - a deliberate,
minimal, documented completion of an edge case the frozen matrix does not
test, in the same spirit as Checkpoint 2's documented
`model_resolution_basis="explicitly-selected"` naming decision - never a
new, un-frozen taxonomy class.

Every `except` clause below names a specific Exception subtype, never a
bare `except:`/`except Exception:` - `KeyboardInterrupt`/`SystemExit`
always propagate unmodified (Section 10.3).
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import yaml
from pydantic import ValidationError

from molecular.activity import record_activity
from molecular.atomic import compute_sha256
from molecular.inputs import ProteinInput, validate_input_id
from molecular.receptor_inspection import (
    InspectionFailureError,
    ReceptorInspectionReport,
    build_receptor_inspection_report,
    compute_inspection_id,
)
from molecular.receptor_parsing import (
    CompressedExtensionRejectedError,
    CompressedRawContentError,
    RawArtifactTooLargeError,
    RawArtifactUnavailableError,
    RawChecksumMismatchError,
    ReceptorParseFailureError,
    StructConnMalformedError,
    UnsupportedStructureFormatError,
    parse_receptor_raw,
)
from molecular.receptor_registry import (
    ReceptorInspectionIdentityConflictError,
    ReceptorInspectionNotFoundError,
    ReceptorInspectionRecordInvalidError,
    ReceptorResolutionIdentityConflictError,
    ReceptorResolutionRecordInvalidError,
    ReceptorSelectionIdentityConflictError,
    ReceptorSelectionNotFoundError,
    ReceptorSelectionRecordInvalidError,
    list_receptor_selections as _registry_list_receptor_selections,
    load_receptor_inspection as _registry_load_receptor_inspection,
    load_receptor_selection as _registry_load_receptor_selection,
    load_resolved_receptor_selection as _registry_load_resolved_receptor_selection,
    register_receptor_inspection,
    register_receptor_selection,
    register_resolved_receptor_selection,
)
from molecular.receptor_resolution import (
    ModelResolutionBasis,
    ResolvedChainSelection,
    ResolvedComponentDisposition,
    ResolvedReceptorSelection,
)
from molecular.receptor_selection import (
    ChainSelector,
    ComponentDisposition,
    Disposition,
    ReceptorSelectionPolicy,
)
from molecular.registry import load_protein

_INSPECT_OPERATION = "receptor-inspect"
_DECLARE_OPERATION = "receptor-selection-declare"
_RESOLVE_OPERATION = "receptor-selection-resolve"

_SELECTION_COMPARED_FIELDS: Tuple[str, ...] = tuple(
    name for name in ReceptorSelectionPolicy.model_fields if name != "declared_at"
)


class ReceptorInputNotFoundError(FileNotFoundError):
    """The named `protein_input_id` has no persisted `ProteinInput` at all
    (Section 30.1). Distinct from `ReceptorInputRecordInvalidError`, which
    signals a registration that exists but is unreadable/invalid."""


class ReceptorInputRecordInvalidError(RuntimeError):
    """A persisted `ProteinInput` exists on disk but cannot be read back:
    its YAML fails to parse, or its content fails schema validation
    (corrects F-07). The original `yaml.YAMLError`/`pydantic.ValidationError`
    is always preserved as `__cause__`."""


class InspectionSelectionUnsupportedError(RuntimeError):
    """`declare_receptor_selection` was called against an inspection whose
    `selection_capability` is `"inspection-only"` (corrects F-01) - raised
    before any field of `selected_chains`/`component_dispositions` is even
    examined (Section 21.4)."""


class ReceptorSelectionValidationError(ValueError):
    """An invalid `selection_id`/`protein_input_id` format, or a
    `pydantic.ValidationError` raised while constructing the candidate
    `ReceptorSelectionPolicy`. The original `ValueError`/`ValidationError`
    is always preserved as `__cause__`."""


class ReceptorSelectionConflictError(RuntimeError):
    """A `declare_receptor_selection` request under an already-existing
    `selection_id` diverges from the persisted policy on at least one
    compared field (every field except `declared_at`). Never raised when
    the content is equivalent - that case is a recognized, safe retry
    instead."""


class ReceptorPersistenceError(OSError):
    """An unexpected filesystem failure during a create-once write itself,
    distinct from an identity conflict (caught separately, and first,
    since every `*IdentityConflictError` here is itself an `OSError`
    subtype). The original `OSError` is always preserved as `__cause__`."""


class ModelNotFoundError(RuntimeError):
    """A scientist-declared `selected_model` is absent from the referenced
    inspection report's `models` (Section 25.5)."""


class AmbiguousModelSelectionError(RuntimeError):
    """`selected_model=None` against a report with more than one model
    (Section 25.5) - the scientist must declare one explicitly."""


class AssemblyNotFoundError(RuntimeError):
    """A scientist-declared `selected_assembly` is absent from the
    referenced inspection report's `assemblies` (Section 25.5)."""


class ChainNotFoundError(RuntimeError):
    """A scientist-declared chain (`selected_chains`), or a named
    component (`component_dispositions`) absent from the inspected
    report's non-polymer inventory, was not confirmed against the
    referenced inspection report (Section 25.4-25.5; T-061). Tudo-ou-nada:
    no partial `ResolvedReceptorSelection` is ever persisted when this is
    raised."""


_INSPECT_PROPAGATED_ERRORS: Tuple[type, ...] = (
    ReceptorInputNotFoundError,
    ReceptorInputRecordInvalidError,
    CompressedExtensionRejectedError,
    CompressedRawContentError,
    RawArtifactTooLargeError,
    RawArtifactUnavailableError,
    RawChecksumMismatchError,
    UnsupportedStructureFormatError,
    ReceptorParseFailureError,
    StructConnMalformedError,
    InspectionFailureError,
    ReceptorInspectionRecordInvalidError,
    ReceptorPersistenceError,
)


@dataclass(frozen=True)
class ReceptorInspectionOutcome:
    """The factual result of one `inspect_receptor` call. `created` is
    never inferred by a caller from a timestamp."""

    report: ReceptorInspectionReport
    created: bool


@dataclass(frozen=True)
class ReceptorSelectionOutcome:
    """The factual result of one `declare_receptor_selection` call."""

    policy: ReceptorSelectionPolicy
    reused: bool


@dataclass(frozen=True)
class ReceptorResolutionOutcome:
    """The factual result of one `resolve_receptor_selection` call."""

    resolution: ResolvedReceptorSelection
    reused: bool


@dataclass(frozen=True)
class ReceptorInspectActivityDetails:
    """Typed activity-log context for `inspect_receptor` (Section 31.1).
    `inspection_id=""` is a deliberate sentinel for the `started` entry,
    and for any `error` entry raised before step 5 computes the real id
    (Section 25.2) - no caller ever observes this placeholder as a real
    id, since it is only ever written to the best-effort activity log."""

    protein_input_id: str
    inspection_id: str
    created: bool = False


@dataclass(frozen=True)
class ReceptorSelectionActivityDetails:
    """Typed activity-log context for `declare_receptor_selection`.
    `inspection_id=""` is the same started-entry sentinel described on
    `ReceptorInspectActivityDetails`."""

    selection_id: str
    protein_input_id: str
    inspection_id: str
    declared_by: str
    reused: bool = False


@dataclass(frozen=True)
class ReceptorResolutionActivityDetails:
    """Typed activity-log context for `resolve_receptor_selection`."""

    selection_id: str
    inspection_id: str
    reused: bool = False


def _record_activity_safely(project_dir: Path, *, command: str, outcome: str, details) -> None:
    """Best-effort activity logging - a failure to write this line can
    never affect the caller's own outcome or returned value. Catches
    Exception, not BaseException: a KeyboardInterrupt/SystemExit reaching
    this helper must still propagate."""

    try:
        record_activity(project_dir, command=command, outcome=outcome, details=details)
    except Exception:
        pass


def _load_protein_input_or_raise(project_dir: Path, protein_input_id: str) -> ProteinInput:
    try:
        return load_protein(project_dir, protein_input_id)
    except FileNotFoundError as error:
        raise ReceptorInputNotFoundError(
            f"no protein input registered under '{protein_input_id}'"
        ) from error
    except (yaml.YAMLError, ValidationError) as error:
        raise ReceptorInputRecordInvalidError(
            f"protein input '{protein_input_id}' exists but is not readable/valid"
        ) from error


def inspect_receptor(project_dir: Path, *, protein_input_id: str) -> ReceptorInspectionOutcome:
    """Create-or-reuse a `ReceptorInspectionReport` for `protein_input_id`
    (Section 25.2, corrects F-04). A malformed `protein_input_id` raises a
    bare `ValueError` with no observable side effect at all - the same
    "fails before started is ever logged" discipline already used by
    `declare_conditions` (`conditions_service.py`); every CLI caller
    already pre-validates id format before reaching this function, so
    this path is reachable only from a direct, non-CLI caller.
    """

    validate_input_id(protein_input_id)

    details = ReceptorInspectActivityDetails(
        protein_input_id=protein_input_id, inspection_id="", created=False
    )
    _record_activity_safely(project_dir, command=_INSPECT_OPERATION, outcome="started", details=details)

    try:
        protein_input = _load_protein_input_or_raise(project_dir, protein_input_id)
    except (ReceptorInputNotFoundError, ReceptorInputRecordInvalidError):
        _record_activity_safely(project_dir, command=_INSPECT_OPERATION, outcome="error", details=details)
        raise

    raw_path = project_dir / protein_input.acquisition.raw_artifact.path
    expected_checksum = protein_input.acquisition.raw_artifact.checksum

    try:
        parsed = parse_receptor_raw(raw_path, expected_checksum=expected_checksum)
    except (
        CompressedExtensionRejectedError,
        CompressedRawContentError,
        RawArtifactTooLargeError,
        RawArtifactUnavailableError,
        RawChecksumMismatchError,
        UnsupportedStructureFormatError,
        ReceptorParseFailureError,
        StructConnMalformedError,
    ):
        _record_activity_safely(project_dir, command=_INSPECT_OPERATION, outcome="error", details=details)
        raise

    inspection_id = compute_inspection_id(
        parser_name=parsed.parser_name,
        parser_version=parsed.parser_version,
        protein_input_id=protein_input_id,
        raw_checksum=parsed.raw_checksum,
        raw_format=parsed.structure_format,
    )
    details = replace(details, inspection_id=inspection_id)

    existing_report: Optional[ReceptorInspectionReport] = None
    try:
        existing_report = _registry_load_receptor_inspection(project_dir, inspection_id)
    except ReceptorInspectionNotFoundError:
        pass
    except ReceptorInspectionRecordInvalidError:
        _record_activity_safely(project_dir, command=_INSPECT_OPERATION, outcome="error", details=details)
        raise

    if existing_report is not None:
        _record_activity_safely(
            project_dir, command=_INSPECT_OPERATION, outcome="success",
            details=replace(details, created=False),
        )
        return ReceptorInspectionOutcome(report=existing_report, created=False)

    try:
        report = build_receptor_inspection_report(
            parsed,
            protein_input_id=protein_input_id,
            raw_format_declared=protein_input.acquisition.raw_artifact.format,
        )
    except InspectionFailureError:
        _record_activity_safely(project_dir, command=_INSPECT_OPERATION, outcome="error", details=details)
        raise

    try:
        register_receptor_inspection(project_dir, report)
    except ReceptorInspectionIdentityConflictError:
        # inspection_id is a deterministic hash of report content (Section
        # 15); a collision here can only mean the identical payload was
        # already persisted by a concurrent caller - a create-once hit,
        # never an error (Section 24.2). The reload itself is protected the
        # same way declare_receptor_selection already protects its own
        # recovery reload: a RecordInvalidError hit in the narrow TOCTOU
        # window (a concurrent writer leaving the colliding file corrupted/
        # partially written) logs one error entry and re-raises with
        # __cause__ intact, never a silent success.
        try:
            report = _registry_load_receptor_inspection(project_dir, inspection_id)
        except ReceptorInspectionRecordInvalidError:
            _record_activity_safely(project_dir, command=_INSPECT_OPERATION, outcome="error", details=details)
            raise
        _record_activity_safely(
            project_dir, command=_INSPECT_OPERATION, outcome="success",
            details=replace(details, created=False),
        )
        return ReceptorInspectionOutcome(report=report, created=False)
    except OSError as error:
        _record_activity_safely(project_dir, command=_INSPECT_OPERATION, outcome="error", details=details)
        raise ReceptorPersistenceError(
            f"failed to persist receptor inspection report '{inspection_id}'"
        ) from error

    _record_activity_safely(
        project_dir, command=_INSPECT_OPERATION, outcome="success",
        details=replace(details, created=True),
    )
    return ReceptorInspectionOutcome(report=report, created=True)


def _diverging_selection_fields(
    existing: ReceptorSelectionPolicy, candidate: ReceptorSelectionPolicy
) -> Tuple[str, ...]:
    return tuple(
        name
        for name in _SELECTION_COMPARED_FIELDS
        if getattr(existing, name) != getattr(candidate, name)
    )


def declare_receptor_selection(
    project_dir: Path,
    *,
    selection_id: str,
    protein_input_id: str,
    selected_model: Optional[int],
    selected_assembly: Optional[str],
    selected_chains: Tuple[ChainSelector, ...],
    water_disposition: Optional[Disposition],
    metal_disposition: Optional[Disposition],
    component_dispositions: Tuple[ComponentDisposition, ...],
    declared_by: str,
    label: Optional[str],
    rationale: Optional[str],
) -> ReceptorSelectionOutcome:
    """Declare a new `ReceptorSelectionPolicy`, or recognize a safe retry
    of an already-declared one (Section 25.3). Invokes `inspect_receptor`
    internally - never reimplements that flow separately - and copies
    `inspection_id`/`raw_checksum_observed` from its factual result, never
    from a caller-supplied value (the signature above does not even accept
    one).
    """

    try:
        validate_input_id(selection_id)
        validate_input_id(protein_input_id)
    except ValueError as error:
        raise ReceptorSelectionValidationError(str(error)) from error

    details = ReceptorSelectionActivityDetails(
        selection_id=selection_id,
        protein_input_id=protein_input_id,
        inspection_id="",
        declared_by=declared_by,
    )
    _record_activity_safely(project_dir, command=_DECLARE_OPERATION, outcome="started", details=details)

    try:
        inspect_outcome = inspect_receptor(project_dir, protein_input_id=protein_input_id)
    except _INSPECT_PROPAGATED_ERRORS:
        _record_activity_safely(project_dir, command=_DECLARE_OPERATION, outcome="error", details=details)
        raise

    details = replace(details, inspection_id=inspect_outcome.report.inspection_id)

    if inspect_outcome.report.selection_capability != "full":
        _record_activity_safely(project_dir, command=_DECLARE_OPERATION, outcome="error", details=details)
        raise InspectionSelectionUnsupportedError(
            f"protein input '{protein_input_id}' has selection_capability="
            f"{inspect_outcome.report.selection_capability!r}; a receptor "
            "selection requires 'full'"
        )

    declared_at = datetime.now(timezone.utc)
    try:
        candidate = ReceptorSelectionPolicy(
            schema_version="1.0",
            selection_id=selection_id,
            protein_input_id=protein_input_id,
            inspection_id=inspect_outcome.report.inspection_id,
            raw_checksum_observed=inspect_outcome.report.raw_checksum,
            selected_model=selected_model,
            selected_assembly=selected_assembly,
            selected_chains=tuple(selected_chains),
            water_disposition=water_disposition,
            metal_disposition=metal_disposition,
            component_dispositions=tuple(component_dispositions),
            declared_by=declared_by,
            label=label,
            rationale=rationale,
            declared_at=declared_at,
        )
    except ValidationError as error:
        _record_activity_safely(project_dir, command=_DECLARE_OPERATION, outcome="error", details=details)
        raise ReceptorSelectionValidationError(str(error)) from error

    try:
        register_receptor_selection(project_dir, candidate)
    except ReceptorSelectionIdentityConflictError:
        try:
            existing = _registry_load_receptor_selection(project_dir, selection_id)
        except ReceptorSelectionRecordInvalidError:
            _record_activity_safely(project_dir, command=_DECLARE_OPERATION, outcome="error", details=details)
            raise

        diverging = _diverging_selection_fields(existing, candidate)
        if not diverging:
            _record_activity_safely(
                project_dir, command=_DECLARE_OPERATION, outcome="success",
                details=replace(details, reused=True),
            )
            return ReceptorSelectionOutcome(policy=existing, reused=True)

        _record_activity_safely(project_dir, command=_DECLARE_OPERATION, outcome="error", details=details)
        raise ReceptorSelectionConflictError(
            f"selection_id '{selection_id}' already exists with divergent "
            f"{', '.join(diverging)}"
        )
    except OSError as error:
        _record_activity_safely(project_dir, command=_DECLARE_OPERATION, outcome="error", details=details)
        raise ReceptorPersistenceError(
            f"failed to persist receptor selection policy '{selection_id}'"
        ) from error

    _record_activity_safely(project_dir, command=_DECLARE_OPERATION, outcome="success", details=details)
    return ReceptorSelectionOutcome(policy=candidate, reused=False)


def _resolve_model(
    selected_model: Optional[int], report: ReceptorInspectionReport
) -> Tuple[int, ModelResolutionBasis]:
    available = {model.model_number for model in report.models}
    if selected_model is None:
        if len(available) == 1:
            return next(iter(available)), "only-model-present"
        raise AmbiguousModelSelectionError(
            f"report has {len(available)} models; selected_model must be declared explicitly"
        )
    if selected_model not in available:
        raise ModelNotFoundError(f"model {selected_model} is not present in the inspected report")
    return selected_model, "explicitly-selected"


def _resolve_assembly(
    selected_assembly: Optional[str], report: ReceptorInspectionReport
) -> Optional[str]:
    if selected_assembly is None:
        return None
    available = {assembly.assembly_id for assembly in report.assemblies}
    if selected_assembly not in available:
        raise AssemblyNotFoundError(
            f"assembly '{selected_assembly}' is not present in the inspected report"
        )
    return selected_assembly


def _resolve_chains(
    selected_chains: Tuple[ChainSelector, ...],
    report: ReceptorInspectionReport,
    resolved_model: int,
) -> Tuple[ResolvedChainSelection, ...]:
    by_label = {
        chain.label_asym_id: chain
        for chain in report.chains
        if chain.model_number == resolved_model
    }

    resolved = []
    for selector in selected_chains:
        entry = by_label.get(selector.label_asym_id)
        if entry is None:
            raise ChainNotFoundError(
                f"chain '{selector.label_asym_id}' is not present in model "
                f"{resolved_model} of the inspected report"
            )
        resolved.append(
            ResolvedChainSelection(
                label_asym_id=entry.label_asym_id,
                auth_asym_id=entry.auth_asym_id,
                label_entity_id=entry.label_entity_id,
                model_number=resolved_model,
            )
        )
    return tuple(resolved)


def _resolve_component_dispositions(
    policy: ReceptorSelectionPolicy, report: ReceptorInspectionReport
) -> Tuple[ResolvedComponentDisposition, ...]:
    """Build the bounded, resolved subset of non-polymer dispositions
    (Section 14.3 rule 5): every water/metal-ion component is auto-
    included when the corresponding bucket disposition is declared
    (T-066), and every individually-named `component_dispositions` entry
    is resolved against the inspected inventory by `(label_asym_id,
    ccd_code)`.

    A named component absent from the inventory raises `ChainNotFoundError`
    - see the module docstring for why this reuses that class rather than
    introducing a new, un-frozen taxonomy entry for an edge case the
    frozen matrix (T-081-094, T-131-137) does not itself test.
    """

    resolved = []
    seen = set()

    def _add_bucket(classification: str, disposition: Optional[Disposition], source: str) -> None:
        if disposition is None:
            return
        for component in report.non_polymer_components:
            if component.classification != classification:
                continue
            key = (component.label_asym_id, component.ccd_code)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(
                ResolvedComponentDisposition(
                    label_asym_id=component.label_asym_id,
                    ccd_code=component.ccd_code,
                    classification=classification,
                    disposition=disposition,
                    disposition_source=source,
                )
            )

    _add_bucket("water", policy.water_disposition, "water_disposition")
    _add_bucket("metal-ion", policy.metal_disposition, "metal_disposition")

    inventory_by_key = {
        (component.label_asym_id, component.ccd_code): component
        for component in report.non_polymer_components
    }
    for item in policy.component_dispositions:
        key = (item.label_asym_id, item.ccd_code)
        component = inventory_by_key.get(key)
        if component is None:
            raise ChainNotFoundError(
                f"named component (label_asym_id={item.label_asym_id!r}, "
                f"ccd_code={item.ccd_code!r}) is not present in the inspected report"
            )
        if key in seen:
            continue
        seen.add(key)
        resolved.append(
            ResolvedComponentDisposition(
                label_asym_id=component.label_asym_id,
                ccd_code=component.ccd_code,
                classification=component.classification,
                disposition=item.disposition,
                disposition_source="component_dispositions",
            )
        )

    return tuple(resolved)


def resolve_receptor_selection(project_dir: Path, *, selection_id: str) -> ReceptorResolutionOutcome:
    """Resolve a persisted `ReceptorSelectionPolicy` against the RAW and
    against the `ReceptorInspectionReport` it references by `inspection_id`
    - never recomputed, never substituted by a newer inspection (Section
    25.4, corrects F-04). Tudo-ou-nada: no `ResolvedReceptorSelection` is
    ever persisted unless every declared identifier is confirmed.
    """

    validate_input_id(selection_id)

    details = ReceptorResolutionActivityDetails(
        selection_id=selection_id, inspection_id="", reused=False
    )
    _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="started", details=details)

    try:
        policy = _registry_load_receptor_selection(project_dir, selection_id)
    except (ReceptorSelectionNotFoundError, ReceptorSelectionRecordInvalidError):
        _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="error", details=details)
        raise

    details = replace(details, inspection_id=policy.inspection_id)

    try:
        report = _registry_load_receptor_inspection(project_dir, policy.inspection_id)
    except (ReceptorInspectionNotFoundError, ReceptorInspectionRecordInvalidError):
        _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="error", details=details)
        raise

    try:
        protein_input = _load_protein_input_or_raise(project_dir, policy.protein_input_id)
    except (ReceptorInputNotFoundError, ReceptorInputRecordInvalidError):
        _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="error", details=details)
        raise

    raw_path = project_dir / protein_input.acquisition.raw_artifact.path
    try:
        current_checksum = compute_sha256(raw_path)
    except OSError as error:
        _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="error", details=details)
        raise RawArtifactUnavailableError(
            f"RAW artifact for '{policy.protein_input_id}' could not be read "
            "for checksum revalidation"
        ) from error

    if current_checksum != report.raw_checksum:
        _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="error", details=details)
        raise RawChecksumMismatchError(
            "the RAW artifact's current SHA-256 no longer matches the checksum "
            f"recorded in inspection report '{report.inspection_id}'"
        )

    try:
        resolved_model, basis = _resolve_model(policy.selected_model, report)
        resolved_assembly = _resolve_assembly(policy.selected_assembly, report)
        resolved_chains = _resolve_chains(policy.selected_chains, report, resolved_model)
        resolved_component_dispositions = _resolve_component_dispositions(policy, report)
    except (ModelNotFoundError, AmbiguousModelSelectionError, AssemblyNotFoundError, ChainNotFoundError):
        _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="error", details=details)
        raise

    resolved_at = datetime.now(timezone.utc)
    candidate = ResolvedReceptorSelection(
        schema_version="1.0",
        resolved_selection_id=selection_id,
        policy_selection_id=selection_id,
        protein_input_id=policy.protein_input_id,
        inspection_id=policy.inspection_id,
        raw_checksum=report.raw_checksum,
        structure_format=report.structure_format,
        parser=report.parser,
        resolved_model=resolved_model,
        model_resolution_basis=basis,
        resolved_assembly=resolved_assembly,
        resolved_chains=resolved_chains,
        resolved_component_dispositions=resolved_component_dispositions,
        resolved_at=resolved_at,
    )

    try:
        register_resolved_receptor_selection(project_dir, candidate)
    except ReceptorResolutionIdentityConflictError:
        # Fully deterministic given an immutable policy + a pinned,
        # immutable inspection: a genuine content divergence under the
        # same selection_id is structurally impossible here, unlike
        # declare_receptor_selection's own (scientist-authored) retry
        # comparison - so any collision is always treated as a reused hit.
        # The reload itself is protected the same way
        # declare_receptor_selection already protects its own recovery
        # reload: a RecordInvalidError hit in the narrow TOCTOU window logs
        # one error entry and re-raises with __cause__ intact, never a
        # silent success.
        try:
            existing = _registry_load_resolved_receptor_selection(project_dir, selection_id)
        except ReceptorResolutionRecordInvalidError:
            _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="error", details=details)
            raise
        _record_activity_safely(
            project_dir, command=_RESOLVE_OPERATION, outcome="success",
            details=replace(details, reused=True),
        )
        return ReceptorResolutionOutcome(resolution=existing, reused=True)
    except OSError as error:
        _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="error", details=details)
        raise ReceptorPersistenceError(
            f"failed to persist resolved receptor selection '{selection_id}'"
        ) from error

    _record_activity_safely(project_dir, command=_RESOLVE_OPERATION, outcome="success", details=details)
    return ReceptorResolutionOutcome(resolution=candidate, reused=False)


def load_receptor_inspection(project_dir: Path, inspection_id: str) -> ReceptorInspectionReport:
    """Thin wrapper over the registry's own load. No activity logging - a
    read performs no state-changing operation to log."""

    return _registry_load_receptor_inspection(project_dir, inspection_id)


def load_receptor_selection(project_dir: Path, selection_id: str) -> ReceptorSelectionPolicy:
    """Thin wrapper over the registry's own load. No activity logging."""

    return _registry_load_receptor_selection(project_dir, selection_id)


def load_resolved_receptor_selection(
    project_dir: Path, selection_id: str
) -> ResolvedReceptorSelection:
    """Thin wrapper over the registry's own load. No activity logging."""

    return _registry_load_resolved_receptor_selection(project_dir, selection_id)


def list_receptor_selections(project_dir: Path) -> Tuple[ReceptorSelectionPolicy, ...]:
    """Thin wrapper over the registry's own listing. No activity logging."""

    return _registry_list_receptor_selections(project_dir)
