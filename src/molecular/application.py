"""PubChem ligand acquisition application service.

Orchestrates the `ligand fetch-pubchem` use case: resolve the experiment,
validate the PubChem CID, acquire the 2D SDF, register the ligand, and
record the outcome to the experiment's activity log. This module is the
seam between the CLI (and any future client) and the existing domain and
persistence modules - it owns no persistence or acquisition logic of its
own, only the ordering and outcome bookkeeping.

CLI-agnostic by design: no typer import, no console formatting, no default
paths. Every domain/persistence exception raised by the modules this calls
propagates unchanged - callers decide how to present them.
"""

from pathlib import Path

from molecular.activity import record_activity
from molecular.config import load_experiment
from molecular.context import ExperimentContext, resolve_experiment
from molecular.inputs import LigandInput
from molecular.pubchem import (
    build_pubchem_sdf_url,
    cleanup_temp_artifact,
    fetch_pubchem_sdf_2d,
)
from molecular.registry import register_ligand

_OPERATION = "ligand-fetch-pubchem"


class InvalidPubChemCidError(ValueError):
    """Raised when a PubChem CID is not a positive integer."""


def _record_activity_safely(project_dir: Path, *, outcome: str) -> None:
    """Best-effort activity logging that must never affect the caller's own outcome."""

    try:
        record_activity(project_dir, command=_OPERATION, outcome=outcome)
    except OSError:
        pass


def register_ligand_from_pubchem(
    experiment_id: str,
    input_id: str,
    cid: int,
    name: str,
    *,
    base_dir: Path,
) -> LigandInput:
    """Resolve `experiment_id` under `base_dir`, acquire ligand `input_id`
    from PubChem CID `cid`, and register it under the resolved experiment.

    Order is significant and preserved exactly: the experiment is resolved
    *before* the CID is validated, so an unresolvable experiment_id always
    surfaces as an experiment-resolution error (ExperimentNotFoundError,
    InvalidExperimentConfigError, AmbiguousExperimentError), even when the
    CID is also invalid.

    Once the experiment is resolved, every subsequent failure - an invalid
    CID (InvalidPubChemCidError), a PubChem acquisition failure
    (PubChemNotFoundError, PubChemAcquisitionError), or a registration
    failure (InputIdentityConflictError/FileExistsError, FileNotFoundError,
    pydantic ValidationError, OSError) - is recorded as exactly one
    outcome=error activity entry before the original exception propagates
    unchanged. A resolution failure records no activity entry at all, since
    there is no resolved project_dir to log against. On success, exactly
    one outcome=success entry is recorded. Activity logging is always
    best-effort: a failure to write the log entry can never mask the
    underlying success or error, and never substitutes a different
    exception for the one that actually occurred.

    The downloaded temporary SDF is always cleaned up once fetched,
    regardless of whether registration succeeds, conflicts, or raises.
    """

    context: ExperimentContext = resolve_experiment(
        experiment_id, base_dir=base_dir, loader=load_experiment
    )

    try:
        if cid <= 0:
            raise InvalidPubChemCidError(f"'{cid}' is not a valid PubChem CID.")

        temp_path = fetch_pubchem_sdf_2d(cid)

        try:
            ligand = register_ligand(
                context.project_dir,
                input_id=input_id,
                name=name,
                method="downloaded",
                raw_file=temp_path,
                external_id=f"CID {cid}",
                source_name="PubChem",
                source_identifier=f"CID {cid}",
                source_uri=build_pubchem_sdf_url(cid),
                file_format="sdf",
            )
        finally:
            cleanup_temp_artifact(temp_path)

    except Exception:
        _record_activity_safely(context.project_dir, outcome="error")
        raise

    _record_activity_safely(context.project_dir, outcome="success")
    return ligand
