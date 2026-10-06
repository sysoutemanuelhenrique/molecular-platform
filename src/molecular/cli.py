from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import List, Optional, Tuple

import typer
import yaml

from pydantic import ValidationError

from molecular.activity import record_activity
from molecular.application import (
    InvalidPubChemCidError,
    TautomerEnumerationAttemptFailedError,
    TautomerEnumerationAttemptPersistenceError,
    enumerate_state_tautomers,
    register_derived_state,
    register_ligand_from_pubchem,
)
from molecular.atomic import (
    RawArtifactIdentityUnstableError,
    RawArtifactIntegrityError,
    RawArtifactNoFollowUnavailableError,
    RawArtifactSharingViolationError,
    RawArtifactTypeError,
    UnsafeRawArtifactPathError,
    atomic_write_text,
)
from molecular.conditions import PhDeclaration
from molecular.conditions_registry import ConditionsRecordInvalidError
from molecular.conditions_service import (
    ConditionsDeclarationConflictError,
    ConditionsPersistenceError,
    ConditionsPredecessorNotFoundError,
    ConditionsValidationError,
    declare_conditions_from_values,
    list_conditions,
    load_conditions,
)
from molecular.config import load_experiment
from molecular.context import (
    AmbiguousExperimentError,
    ExperimentContext,
    ExperimentNotFoundError,
    InvalidExperimentConfigError,
    resolve_experiment,
    scan_experiments,
)
from molecular.execution import TautomerEnumerationPolicy
from molecular.execution_registry import load_execution_attempt
from molecular.inputs import validate_input_id
from molecular.models import ExperimentConfig
from molecular.promotion import LigandPromotionPolicy
from molecular.promotion_registry import (
    PromotionPublicationUnavailableError,
    load_promotion_attempt,
)
from molecular.promotion_service import (
    LigandPromotionAttemptFailedError,
    LigandPromotionAttemptPersistenceError,
    PromotionConsistencyError,
    PromotionIdentityConflictError,
    PromotionReservationConflictError,
    PromotionRevisionRequiredError,
    UnsupportedPromotionFormatError,
    promote_ligand_input,
)
from molecular.pubchem import PubChemAcquisitionError, PubChemNotFoundError
from molecular.rdkit_tautomer import (
    DisconnectedStructureError,
    IncompleteEnumerationError,
    InvalidPromotionSDFError,
    InvalidSDFError,
    MultipleMoleculesError,
    PromotionRecordCountError,
    RDKitUnavailableError,
)
from molecular.registry import load_ligand, load_protein, register_ligand, register_protein
from molecular.scientific import ScientificEntity
from molecular.scientific_registry import (
    list_entity_states,
    list_state_conformers,
    load_conformer,
    load_entity,
    load_state,
    register_conformer,
    register_entity,
    register_state,
)
from molecular.transformation import TransformationParameter, TransformationProducer
from molecular.transformation_registry import (
    list_state_transformations,
    load_state_transformation,
)

app = typer.Typer(
    help="Molecular Platform - scientific molecular research platform."
)

ligand_app = typer.Typer(help="Manage ligand inputs.")
protein_app = typer.Typer(help="Manage protein inputs.")
experiment_app = typer.Typer(help="Query experiment metadata.")
entity_app = typer.Typer(help="Manage scientific entities.")
state_app = typer.Typer(help="Manage molecular states.")
conformer_app = typer.Typer(help="Manage conformers.")
provenance_app = typer.Typer(help="Query state-transformation provenance.")
conditions_app = typer.Typer(help="Manage experimental conditions declarations.")
app.add_typer(ligand_app, name="ligand")
app.add_typer(protein_app, name="protein")
app.add_typer(experiment_app, name="experiment")
app.add_typer(entity_app, name="entity")
app.add_typer(state_app, name="state")
app.add_typer(conformer_app, name="conformer")
app.add_typer(provenance_app, name="provenance")
app.add_typer(conditions_app, name="conditions")

@app.callback()
def main():
    """Molecular Platform - scientific molecular research platform."""
    pass


@app.command()
def init(
    name: str,
    title: Optional[str] = typer.Option(None, "--title"),
    responsible: Optional[str] = typer.Option(None, "--responsible"),
    summary: Optional[str] = typer.Option(None, "--summary"),
):
    """Create a new molecular experiment."""

    project_dir = Path(name)

    directories = [
        project_dir / "inputs" / "receptor",
        project_dir / "inputs" / "ligands",
        project_dir / "inputs" / "proteins",
        project_dir / "artifacts",
        project_dir / "results",
        project_dir / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    experiment_file = project_dir / "experiment.yaml"

    if not experiment_file.exists():
        try:
            config = ExperimentConfig(
                schema_version="1.1",
                experiment_id=name,
                title=title,
                responsible=responsible,
                summary=summary,
                created_at=datetime.now(timezone.utc),
            )
        except ValidationError as error:
            typer.echo("CONFIGURATION ERROR")
            typer.echo(f"Experiment '{name}' metadata is invalid.")

            for item in error.errors():
                field = ".".join(str(part) for part in item["loc"])
                typer.echo(f"- {field}: {item['msg']}")

            raise typer.Exit(code=1)

        atomic_write_text(
            experiment_file,
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        )

    typer.echo(f"Experiment '{name}' created successfully.")

def _record_activity_safely(project_dir: Path, command: str, outcome: str) -> None:
    """Best-effort activity logging that must never affect the caller's own outcome."""

    try:
        record_activity(project_dir, command=command, outcome=outcome)
    except OSError:
        pass


def _resolve_experiment_or_exit(
    identifier: str, *, error_sanitizer=None
) -> ExperimentContext:
    """Resolve an experiment by its logical experiment_id, or exit with a
    friendly error.

    There is no activity log entry on failure here: without a resolved
    context there is no project_dir to log against. error_sanitizer is
    opt-in presentation only; existing commands preserve their exact
    output by leaving it as None.
    """

    def _present(value: object) -> str:
        text = str(value)
        return error_sanitizer(text) if error_sanitizer is not None else text

    try:
        return resolve_experiment(identifier, loader=load_experiment)

    except ExperimentNotFoundError:
        typer.echo("INPUT ERROR")
        typer.echo(_present(f"Experiment '{identifier}' was not found."))
        raise typer.Exit(code=1)

    except InvalidExperimentConfigError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(_present(f"Experiment '{identifier}' is invalid."))

        for item in error.validation_error.errors():
            field = ".".join(str(part) for part in item["loc"])
            typer.echo(f"- {field}: {_present(item['msg'])}")

        raise typer.Exit(code=1)

    except AmbiguousExperimentError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(_present(error))
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(_present(f"Could not read the current directory: {error}"))
        raise typer.Exit(code=1)


@app.command()
def validate(name: str):
    """Validate a molecular experiment."""

    context = _resolve_experiment_or_exit(name)

    typer.echo(
        f"Experiment '{context.config.experiment_id}' is valid "
        f"(schema {context.config.schema_version})."
    )

    _record_activity_safely(context.project_dir, command="validate", outcome="success")


@app.command()
def plan(name: str):
    """Show the execution plan for a molecular experiment."""

    context = _resolve_experiment_or_exit(name)

    typer.echo("EXPERIMENT PLAN")
    typer.echo()
    typer.echo(f"Experiment: {context.config.experiment_id}")
    typer.echo(f"Schema: {context.config.schema_version}")
    typer.echo("Status: VALID")
    typer.echo()
    typer.echo("No execution steps are defined yet.")

    _record_activity_safely(context.project_dir, command="plan", outcome="success")


def _add_scientific_input(
    *,
    register_fn,
    label: str,
    command: str,
    context: ExperimentContext,
    input_id: str,
    name: str,
    acquisition_method: str,
    raw_file: str,
    external_id: Optional[str],
    source_name: Optional[str],
    source_identifier: Optional[str],
    source_uri: Optional[str],
    file_format: Optional[str],
) -> None:
    """Shared registration flow for `ligand add` / `protein add`."""

    try:
        registered = register_fn(
            context.project_dir,
            input_id=input_id,
            name=name,
            method=acquisition_method,
            raw_file=Path(raw_file),
            external_id=external_id,
            source_name=source_name,
            source_identifier=source_identifier,
            source_uri=source_uri,
            file_format=file_format,
        )

    except (FileNotFoundError, FileExistsError) as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"{label} '{input_id}' is invalid.")

        for item in error.errors():
            field = ".".join(str(part) for part in item["loc"])
            typer.echo(f"- {field}: {item['msg']}")

        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    except ValueError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    typer.echo(f"{label} '{registered.id}' registered successfully.")
    _record_activity_safely(context.project_dir, command=command, outcome="success")


def _echo_validation_errors(error: ValidationError) -> None:
    """Print one line per pydantic validation error, in the shape already
    used throughout this module."""

    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"])
        typer.echo(f"- {field}: {item['msg']}")


def _validate_id_or_exit(
    value: str,
    *,
    project_dir: Optional[Path] = None,
    command: Optional[str] = None,
    error_sanitizer=None,
) -> str:
    """Validate a scientific/input id at the CLI boundary before calling any
    load/list function that does not validate id format internally itself.

    A malformed id never reaches the underlying function: this fails first,
    with a clean INPUT ERROR banner and a nonzero exit, never a Python
    traceback. project_dir/command are optional - single-object read
    commands pass them to log outcome="error" on this failure, mirroring
    how they already log a later not-found failure; collection-listing
    commands omit them, since listings never write activity log entries at
    all (mirroring `experiment list`).
    """

    try:
        return validate_input_id(value)
    except ValueError as error:
        typer.echo("INPUT ERROR")
        message = str(error)
        if error_sanitizer is not None:
            message = error_sanitizer(message)
        typer.echo(message)
        if project_dir is not None and command is not None:
            _record_activity_safely(project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)


def _call_or_exit(
    fn,
    *args,
    project_dir: Optional[Path] = None,
    command: Optional[str] = None,
    **kwargs,
):
    """Call a read-only registry/list function and translate every failure
    mode it can realistically raise into a clean CLI banner, never letting
    OSError, yaml.YAMLError, or pydantic.ValidationError escape as an
    unhandled traceback - the same INPUT/CONFIGURATION/STORAGE ERROR
    vocabulary already used by the write-side commands.

    project_dir/command are optional: single-object read commands pass
    them to log outcome="error" on this failure, mirroring how they log a
    not-found failure; collection-listing commands omit them, since
    listings never write activity log entries at all, even on failure
    (mirroring `experiment list`).
    """

    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        if project_dir is not None and command is not None:
            _record_activity_safely(project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)
    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        _echo_validation_errors(error)
        if project_dir is not None and command is not None:
            _record_activity_safely(project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)
    except yaml.YAMLError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(str(error))
        if project_dir is not None and command is not None:
            _record_activity_safely(project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)
    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        if project_dir is not None and command is not None:
            _record_activity_safely(project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)


def _echo_scientific_input(item, *, label: str, project_dir: Path) -> None:
    """Print a registered ScientificInput's (Ligand/Protein) real fields -
    never inventing a field the model does not have.

    Prints both the stored relative path (as persisted in the metadata
    YAML) and a resolved absolute path, so the artifact can be used
    directly - by copy-paste - as a future --source-file argument,
    without the scientist having to resolve it against the project
    directory by hand.
    """

    typer.echo(label)
    typer.echo()
    typer.echo(f"ID: {item.id}")
    typer.echo(f"Name: {item.name}")

    if item.external_id:
        typer.echo(f"External ID: {item.external_id}")

    typer.echo(f"Acquisition method: {item.acquisition.method}")

    if item.acquisition.source_name:
        typer.echo(f"Source name: {item.acquisition.source_name}")
    if item.acquisition.source_identifier:
        typer.echo(f"Source identifier: {item.acquisition.source_identifier}")
    if item.acquisition.source_uri:
        typer.echo(f"Source URI: {item.acquisition.source_uri}")

    typer.echo(f"Acquired at: {item.acquisition.acquired_at.isoformat()}")
    typer.echo()
    typer.echo("Raw artifact:")
    typer.echo(f"Stored path: {item.acquisition.raw_artifact.path}")
    resolved_path = (Path(project_dir) / item.acquisition.raw_artifact.path).resolve()
    typer.echo(f"Resolved source path: {resolved_path}")
    typer.echo(f"Format: {item.acquisition.raw_artifact.format}")
    typer.echo(f"Checksum: {item.acquisition.raw_artifact.checksum}")


@ligand_app.command("add")
def ligand_add(
    experiment: str,
    input_id: str,
    name: str = typer.Option(..., "--name"),
    acquisition_method: str = typer.Option(..., "--acquisition-method"),
    raw_file: str = typer.Option(..., "--raw-file"),
    external_id: Optional[str] = typer.Option(None, "--external-id"),
    source_name: Optional[str] = typer.Option(None, "--source-name"),
    source_identifier: Optional[str] = typer.Option(None, "--source-identifier"),
    source_uri: Optional[str] = typer.Option(None, "--source-uri"),
    file_format: Optional[str] = typer.Option(None, "--format"),
):
    """Register a ligand input and its acquisition provenance."""

    context = _resolve_experiment_or_exit(experiment)

    _add_scientific_input(
        register_fn=register_ligand,
        label="Ligand",
        command="ligand-add",
        context=context,
        input_id=input_id,
        name=name,
        acquisition_method=acquisition_method,
        raw_file=raw_file,
        external_id=external_id,
        source_name=source_name,
        source_identifier=source_identifier,
        source_uri=source_uri,
        file_format=file_format,
    )


@ligand_app.command("show")
def ligand_show(experiment: str, input_id: str):
    """Show a registered ligand's acquisition record and stored artifact
    path."""

    context = _resolve_experiment_or_exit(experiment)
    validated_id = _validate_id_or_exit(
        input_id, project_dir=context.project_dir, command="ligand-show"
    )

    ligand = _call_or_exit(
        load_ligand,
        context.project_dir,
        validated_id,
        project_dir=context.project_dir,
        command="ligand-show",
    )

    _echo_scientific_input(ligand, label="LIGAND", project_dir=context.project_dir)
    _record_activity_safely(context.project_dir, command="ligand-show", outcome="success")


@ligand_app.command("fetch-pubchem")
def ligand_fetch_pubchem(
    experiment: str,
    input_id: str,
    cid: int = typer.Option(..., "--cid"),
    name: str = typer.Option(..., "--name"),
):
    """Download a ligand's 2D SDF structure from PubChem by CID and register it."""

    try:
        ligand = register_ligand_from_pubchem(
            experiment, input_id, cid, name, base_dir=Path(".")
        )

    except (
        ExperimentNotFoundError,
        PubChemNotFoundError,
        InvalidPubChemCidError,
        FileNotFoundError,
        FileExistsError,
    ) as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    except InvalidExperimentConfigError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"Experiment '{experiment}' is invalid.")

        for item in error.validation_error.errors():
            field = ".".join(str(part) for part in item["loc"])
            typer.echo(f"- {field}: {item['msg']}")

        raise typer.Exit(code=1)

    except AmbiguousExperimentError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"Ligand '{input_id}' is invalid.")

        for item in error.errors():
            field = ".".join(str(part) for part in item["loc"])
            typer.echo(f"- {field}: {item['msg']}")

        raise typer.Exit(code=1)

    except PubChemAcquisitionError as error:
        typer.echo("ACQUISITION ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    typer.echo(f"Ligand '{ligand.id}' registered successfully.")


def _promotion_failure_banner(error: BaseException) -> str:
    """Map the structured original scientific/storage cause to the frozen
    CLI vocabulary (Section 15.4/17 of Increment 018's frozen spec) -
    mirrors `_tautomer_failure_banner`'s own structure exactly, for the
    promotion workflow's own closed set of possible `original_error`
    types. `promote_ligand_input` never lets anything else reach this
    function as `original_error` once resolution/execution has begun."""

    if isinstance(
        error,
        (
            UnsafeRawArtifactPathError,
            RawArtifactTypeError,
            RawArtifactIntegrityError,
            RawArtifactNoFollowUnavailableError,
            RawArtifactIdentityUnstableError,
            RawArtifactSharingViolationError,
            PromotionReservationConflictError,
            PromotionPublicationUnavailableError,
        ),
    ):
        return "STORAGE ERROR"
    if isinstance(error, (InvalidPromotionSDFError, DisconnectedStructureError, PromotionRecordCountError)):
        return "EXECUTION ERROR"
    if isinstance(error, (RDKitUnavailableError, PromotionConsistencyError)):
        return "CONFIGURATION ERROR"
    if isinstance(error, FileNotFoundError):
        return "INPUT ERROR"
    if isinstance(
        error,
        (UnsupportedPromotionFormatError, PromotionRevisionRequiredError, PromotionIdentityConflictError),
    ):
        return "INPUT ERROR"
    if isinstance(error, ValueError):
        return "INPUT ERROR"
    if isinstance(error, (yaml.YAMLError, ValidationError)):
        return "CONFIGURATION ERROR"
    if isinstance(error, OSError):
        return "STORAGE ERROR"
    return "EXECUTION ERROR"


def _echo_promotion_unconfirmed_removals(
    residual_entity_ids: tuple,
    residual_state_ids: tuple,
    residual_reservation_keys: tuple,
) -> None:
    """Render only removals/releases whose cleanup could not be confirmed -
    mirrors `_echo_unconfirmed_removals` exactly in spirit, one field wider
    (reservations, Section 6.6 of the frozen spec)."""

    if not residual_entity_ids and not residual_state_ids and not residual_reservation_keys:
        return

    typer.echo()
    typer.echo(
        "Unconfirmed removals: "
        f"{len(residual_entity_ids)} entity(ies), "
        f"{len(residual_state_ids)} state(s), "
        f"{len(residual_reservation_keys)} reservation(s)"
    )
    for entity_id in residual_entity_ids:
        typer.echo(f"  Entity '{entity_id}'")
    for state_id in residual_state_ids:
        typer.echo(f"  State '{state_id}'")
    for reservation_key in residual_reservation_keys:
        typer.echo(f"  Reservation '{reservation_key}'")


@ligand_app.command("promote")
def ligand_promote(
    experiment: str,
    input_id: str,
    entity: str = typer.Option(..., "--entity"),
    state: str = typer.Option(..., "--state"),
    name: str = typer.Option(..., "--name"),
    revision: Optional[str] = typer.Option(None, "--revision"),
):
    """Promote a registered ligand RAW artifact into an auditable
    ScientificEntity/MolecularState pair (Section 15.1 of Increment 018's
    frozen final specification)."""

    context = _resolve_experiment_or_exit(experiment, error_sanitizer=_sanitize_cli_error_text)
    validated_input_id = _validate_id_or_exit(
        input_id,
        project_dir=context.project_dir,
        command="ligand-promote",
        error_sanitizer=_sanitize_cli_error_text,
    )
    validated_entity_id = _validate_id_or_exit(
        entity,
        project_dir=context.project_dir,
        command="ligand-promote",
        error_sanitizer=_sanitize_cli_error_text,
    )
    validated_state_id = _validate_id_or_exit(
        state,
        project_dir=context.project_dir,
        command="ligand-promote",
        error_sanitizer=_sanitize_cli_error_text,
    )
    validated_revision: Optional[str] = None
    if revision is not None:
        validated_revision = _validate_id_or_exit(
            revision,
            project_dir=context.project_dir,
            command="ligand-promote",
            error_sanitizer=_sanitize_cli_error_text,
        )

    # Section 15.1: the CLI constructs exclusively the one profile RD-018-01
    # confirmed correct - no individual RDKit flag is ever exposed here.
    policy = LigandPromotionPolicy(sanitize=True, remove_hs=False, strict_parsing=True)

    # No `_record_activity_safely` call surrounds this: `promote_ligand_input`
    # itself already logs started/success/error for command="ligand-promote"
    # once it is reached (Section 14.5) - adding another entry here for the
    # same outcome would duplicate the activity log, which this command
    # never does. Only the pre-call `_validate_id_or_exit` failures above
    # log their own "error" entry, since the service is never reached then.
    try:
        attempt = promote_ligand_input(
            context.project_dir,
            input_id=validated_input_id,
            entity_id=validated_entity_id,
            state_id=validated_state_id,
            entity_name=name,
            policy=policy,
            intentional_revision=validated_revision,
        )

    except LigandPromotionAttemptPersistenceError as error:
        # Checkpoint 4 correction (BLOCKER-01): when intended_status is
        # "succeeded", the promotion record/entity/state may already be
        # durably published - this branch must never claim the promotion
        # itself failed, only that this attempt's own bookkeeping status
        # could not be written.
        typer.echo("STORAGE ERROR")
        typer.echo()
        typer.echo("LIGAND PROMOTION STATUS NOT PERSISTED")
        typer.echo()
        typer.echo(f"Attempt ID: {error.attempt_id}")
        typer.echo(f"Input: {error.input_id}")
        typer.echo(f"Phase: {error.phase}")
        typer.echo(f"Intended status: {error.intended_status}")
        if error.intended_status == "succeeded":
            typer.echo("Scientific commit: may already be published")
            typer.echo("Operational status persisted: no")
        else:
            typer.echo("Terminal status persisted: no")
        typer.echo(f"Original error: {_sanitize_cli_error_text(error.original_error)}")
        typer.echo(f"Persistence error: {_sanitize_cli_error_text(error.persistence_error)}")
        _echo_promotion_unconfirmed_removals(
            error.residual_entity_ids,
            error.residual_state_ids,
            error.residual_reservation_keys,
        )
        raise typer.Exit(code=1)

    except LigandPromotionAttemptFailedError as error:
        typer.echo(_promotion_failure_banner(error.original_error))
        typer.echo()

        try:
            failed_attempt = load_promotion_attempt(context.project_dir, error.attempt_id)
        except Exception:
            typer.echo("LIGAND PROMOTION FAILED")
            typer.echo()
            typer.echo(f"Attempt ID: {error.attempt_id}")
            typer.echo(f"Input: {error.input_id}")
            typer.echo(f"Status: {error.status}")
            typer.echo(f"Error: {_sanitize_cli_error_text(error.original_error)}")
            typer.echo()
            typer.echo("(additional attempt details could not be loaded)")
            raise typer.Exit(code=1)

        typer.echo("LIGAND PROMOTION FAILED")
        typer.echo()
        typer.echo(f"Attempt ID: {error.attempt_id}")
        typer.echo(f"Input: {error.input_id}")
        typer.echo(f"Status: {error.status}")
        typer.echo(f"Phase: {failed_attempt.error.phase}")
        typer.echo(f"Error: {_sanitize_cli_error_text(failed_attempt.error.message)}")
        _echo_promotion_unconfirmed_removals(
            failed_attempt.error.residual_entity_ids,
            failed_attempt.error.residual_state_ids,
            failed_attempt.error.residual_reservation_keys,
        )
        raise typer.Exit(code=1)

    except ValueError as error:
        typer.echo("INPUT ERROR")
        typer.echo(_sanitize_cli_error_text(error))
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(_sanitize_cli_error_text(error))
        raise typer.Exit(code=1)

    # Section 15.5: nothing is echoed until every value the banner below
    # needs is already sitting on the validated `attempt` object returned
    # by the Application Service - there is no further I/O between here
    # and the first `typer.echo` call, so no banner can ever be printed
    # ahead of a subsequent failure to load/format something.
    if attempt.result.reused_existing_promotion:
        typer.echo("LIGAND PROMOTION")
        typer.echo()
        typer.echo(f"Attempt ID: {attempt.id}")
        typer.echo(f"Promotion ID: {attempt.result.promotion_id}")
        typer.echo("Result: existing promotion reused")
        typer.echo(f"Entity: {attempt.result.entity_id}")
        typer.echo(f"Initial state: {attempt.result.state_id}")
        return

    entity_annotation = "created" if attempt.result.entity_created else "reused"
    state_annotation = "created" if attempt.result.state_created else "reused"

    typer.echo("LIGAND PROMOTION")
    typer.echo()
    typer.echo(f"Attempt ID: {attempt.id}")
    typer.echo(f"Promotion ID: {attempt.result.promotion_id}")
    typer.echo(f"Input: {validated_input_id}")
    typer.echo(f"Entity: {attempt.result.entity_id} ({entity_annotation})")
    typer.echo(f"Initial state: {attempt.result.state_id} ({state_annotation})")
    typer.echo(f"Representation: {attempt.result.representation.path}")
    typer.echo(f"RAW checksum verified: {attempt.resolved.snapshot_checksum}")
    typer.echo(f"Engine: {attempt.resolved.engine_name} {attempt.resolved.engine_version}")

    if attempt.resolved.warnings:
        typer.echo()
        typer.echo("Warnings:")
        for warning in attempt.resolved.warnings:
            typer.echo(f"  {warning}")


@protein_app.command("add")
def protein_add(
    experiment: str,
    input_id: str,
    name: str = typer.Option(..., "--name"),
    acquisition_method: str = typer.Option(..., "--acquisition-method"),
    raw_file: str = typer.Option(..., "--raw-file"),
    external_id: Optional[str] = typer.Option(None, "--external-id"),
    source_name: Optional[str] = typer.Option(None, "--source-name"),
    source_identifier: Optional[str] = typer.Option(None, "--source-identifier"),
    source_uri: Optional[str] = typer.Option(None, "--source-uri"),
    file_format: Optional[str] = typer.Option(None, "--format"),
):
    """Register a protein input and its acquisition provenance."""

    context = _resolve_experiment_or_exit(experiment)

    _add_scientific_input(
        register_fn=register_protein,
        label="Protein",
        command="protein-add",
        context=context,
        input_id=input_id,
        name=name,
        acquisition_method=acquisition_method,
        raw_file=raw_file,
        external_id=external_id,
        source_name=source_name,
        source_identifier=source_identifier,
        source_uri=source_uri,
        file_format=file_format,
    )


@protein_app.command("show")
def protein_show(experiment: str, input_id: str):
    """Show a registered protein's acquisition record and stored artifact
    path."""

    context = _resolve_experiment_or_exit(experiment)
    validated_id = _validate_id_or_exit(
        input_id, project_dir=context.project_dir, command="protein-show"
    )

    protein = _call_or_exit(
        load_protein,
        context.project_dir,
        validated_id,
        project_dir=context.project_dir,
        command="protein-show",
    )

    _echo_scientific_input(protein, label="PROTEIN", project_dir=context.project_dir)
    _record_activity_safely(context.project_dir, command="protein-show", outcome="success")


def _count_registered_inputs(project_dir: Path, kind: str) -> int:
    """Count registered <kind> inputs: one .yaml file per input, ignoring
    the RAW artifact subdirectories that sit alongside them."""

    kind_dir = project_dir / "inputs" / kind

    if not kind_dir.is_dir():
        return 0

    return sum(1 for entry in kind_dir.iterdir() if entry.is_file() and entry.suffix == ".yaml")


@experiment_app.command("get")
def experiment_get(identifier: str):
    """Show metadata and input counts for a single experiment."""

    context = _resolve_experiment_or_exit(identifier)

    typer.echo("EXPERIMENT")
    typer.echo()
    typer.echo(f"ID: {context.config.experiment_id}")
    typer.echo(f"Directory: {context.project_dir.name}")
    typer.echo(f"Title: {context.config.title or ''}")
    typer.echo(f"Responsible: {context.config.responsible or ''}")
    typer.echo(
        f"Created: {context.config.created_at.isoformat() if context.config.created_at else ''}"
    )
    typer.echo("Status: VALID")

    if context.config.summary:
        typer.echo()
        typer.echo("Summary:")
        typer.echo(context.config.summary)

    typer.echo()
    typer.echo("Inputs:")
    typer.echo(f"Ligands: {_count_registered_inputs(context.project_dir, 'ligands')}")
    typer.echo(f"Proteins: {_count_registered_inputs(context.project_dir, 'proteins')}")

    _record_activity_safely(context.project_dir, command="experiment-get", outcome="success")


@experiment_app.command("list")
def experiment_list():
    """List all valid experiments found in the current directory."""

    try:
        valid, skipped = scan_experiments(loader=load_experiment)
    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(f"Could not read the current directory: {error}")
        raise typer.Exit(code=1)

    typer.echo("EXPERIMENTS")
    typer.echo()

    if not valid:
        typer.echo("No experiments found in the current directory.")
    else:
        id_width = max([len("ID")] + [len(context.config.experiment_id) for context in valid])
        title_width = max([len("TITLE")] + [len(context.config.title or "") for context in valid])
        responsible_width = max(
            [len("RESPONSIBLE")] + [len(context.config.responsible or "") for context in valid]
        )

        typer.echo(
            f"{'ID'.ljust(id_width)}   {'TITLE'.ljust(title_width)}   "
            f"{'RESPONSIBLE'.ljust(responsible_width)}   DIRECTORY"
        )

        for context in valid:
            typer.echo(
                f"{context.config.experiment_id.ljust(id_width)}   "
                f"{(context.config.title or '').ljust(title_width)}   "
                f"{(context.config.responsible or '').ljust(responsible_width)}   "
                f"{context.project_dir.name}"
            )

    if skipped:
        typer.echo()
        typer.echo(
            f"{len(skipped)} directories skipped (invalid experiment.yaml): "
            f"{', '.join(skipped)}"
        )


# --- entity add/show ---


@entity_app.command("add")
def entity_add(
    experiment: str,
    entity_id: str,
    name: str = typer.Option(..., "--name"),
):
    """Register a ScientificEntity: the stable scientific identity of a
    molecular object followed within this experiment, independent of any
    particular molecular state, conformation, or file format."""

    context = _resolve_experiment_or_exit(experiment)

    try:
        entity = ScientificEntity(schema_version="1.0", id=entity_id, name=name)
    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"Entity '{entity_id}' is invalid.")
        _echo_validation_errors(error)
        _record_activity_safely(context.project_dir, command="entity-add", outcome="error")
        raise typer.Exit(code=1)

    try:
        registered = register_entity(context.project_dir, entity)
    except FileExistsError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="entity-add", outcome="error")
        raise typer.Exit(code=1)
    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="entity-add", outcome="error")
        raise typer.Exit(code=1)

    typer.echo(f"Entity '{registered.id}' registered successfully.")
    _record_activity_safely(context.project_dir, command="entity-add", outcome="success")


@entity_app.command("show")
def entity_show(experiment: str, entity_id: str):
    """Show a registered ScientificEntity."""

    context = _resolve_experiment_or_exit(experiment)
    validated_id = _validate_id_or_exit(
        entity_id, project_dir=context.project_dir, command="entity-show"
    )

    entity = _call_or_exit(
        load_entity,
        context.project_dir,
        validated_id,
        project_dir=context.project_dir,
        command="entity-show",
    )

    typer.echo("ENTITY")
    typer.echo()
    typer.echo(f"Schema version: {entity.schema_version}")
    typer.echo(f"ID: {entity.id}")
    typer.echo(f"Name: {entity.name}")

    _record_activity_safely(context.project_dir, command="entity-show", outcome="success")


# --- state add/show/list/derive ---


@state_app.command("add")
def state_add(
    experiment: str,
    state_id: str,
    entity_id: str = typer.Option(..., "--entity"),
    source_file: str = typer.Option(..., "--source-file"),
    format: str = typer.Option(..., "--format"),
):
    """Register a MolecularState and its representation artifact, under an
    already-registered ScientificEntity."""

    context = _resolve_experiment_or_exit(experiment)

    try:
        state = register_state(
            context.project_dir,
            state_id=state_id,
            entity_id=entity_id,
            source_file=Path(source_file),
            format=format,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="state-add", outcome="error")
        raise typer.Exit(code=1)
    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="state-add", outcome="error")
        raise typer.Exit(code=1)

    typer.echo(f"State '{state.id}' registered successfully.")
    _record_activity_safely(context.project_dir, command="state-add", outcome="success")


@state_app.command("show")
def state_show(experiment: str, state_id: str):
    """Show a registered MolecularState."""

    context = _resolve_experiment_or_exit(experiment)
    validated_id = _validate_id_or_exit(
        state_id, project_dir=context.project_dir, command="state-show"
    )

    state = _call_or_exit(
        load_state,
        context.project_dir,
        validated_id,
        project_dir=context.project_dir,
        command="state-show",
    )

    typer.echo("STATE")
    typer.echo()
    typer.echo(f"ID: {state.id}")
    typer.echo(f"Entity: {state.entity_id}")
    typer.echo(f"Representation path: {state.representation.path}")
    typer.echo(f"Representation format: {state.representation.format}")
    typer.echo(f"Representation checksum: {state.representation.checksum}")

    _record_activity_safely(context.project_dir, command="state-show", outcome="success")


@state_app.command("list")
def state_list(
    experiment: str,
    entity_id: str = typer.Option(..., "--entity"),
):
    """List every registered MolecularState for a given entity.

    A malformed --entity id is rejected before any lookup is attempted. A
    well-formed but unregistered --entity id is not an error - it simply
    yields no results, since this is a read-only filter over already-
    registered states, not a check that the entity itself exists. This
    command never writes an activity log entry, mirroring `experiment
    list`.
    """

    context = _resolve_experiment_or_exit(experiment)
    validated_entity_id = _validate_id_or_exit(entity_id)

    states = _call_or_exit(list_entity_states, context.project_dir, validated_entity_id)

    typer.echo("STATES")
    typer.echo()

    if not states:
        typer.echo(f"No states found for entity '{validated_entity_id}'.")
        return

    for state in states:
        typer.echo(f"{state.id}  (entity={state.entity_id})")


def _echo_unconfirmed_removals(
    residual_state_ids: tuple[str, ...],
    residual_operation_ids: tuple[str, ...],
) -> None:
    """Render only removals whose cleanup could not be confirmed."""

    if not residual_state_ids and not residual_operation_ids:
        return

    typer.echo()
    typer.echo(
        "Unconfirmed removals: "
        f"{len(residual_state_ids)} state(s), "
        f"{len(residual_operation_ids)} transformation(s)"
    )
    for state_id in residual_state_ids:
        typer.echo(f"  State '{state_id}'")
    for operation_id in residual_operation_ids:
        typer.echo(f"  Transformation '{operation_id}'")


_TAUTOMER_URL_RE = re.compile(r"(?i)\bhttps?://[^\s'\"<>]+")
_TAUTOMER_FILE_SUFFIX_RE = re.compile(
    r"\.[A-Za-z0-9]{1,12}(?:\.[A-Za-z0-9]{1,12})*"
)
_TAUTOMER_PATH_CONNECTORS = (" and ", " then ")


def _tautomer_path_start(text: str, index: int):
    """Return (kind, content_start) for an unambiguous absolute-path prefix."""

    if (
        index + 2 < len(text)
        and text[index].isalpha()
        and text[index + 1] == ":"
        and text[index + 2] in "\\/"
        and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_"))
    ):
        return "windows", index + 2

    if text.startswith("\\\\", index):
        cursor = index
        while cursor < len(text) and text[cursor] == "\\":
            cursor += 1
        if (
            cursor - index >= 2
            and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_"))
        ):
            return "unc", cursor

    if text[index] == "/":
        previous = text[index - 1] if index else ""
        if (
            not text.startswith("//", index)
            and (index == 0 or not (previous.isalnum() or previous in "_+)]"))
        ):
            return "posix", index + 1

    return None


def _tautomer_is_path_separator(character: str, kind: str) -> bool:
    if kind == "windows":
        return character in "\\/"
    if kind == "unc":
        return character == "\\"
    return character == "/"


def _tautomer_path_end(
    text: str,
    *,
    content_start: int,
    kind: str,
):
    """Find one path's first file-like endpoint without crossing text boundaries."""

    cursor = content_start
    while cursor < len(text):
        character = text[cursor]

        if character in "\r\n'\"<>":
            return None

        if character in ",;)}]":
            return None

        connector = next(
            (
                candidate
                for candidate in _TAUTOMER_PATH_CONNECTORS
                if text.startswith(candidate, cursor)
            ),
            None,
        )
        if connector is not None:
            after_connector = cursor + len(connector)
            # A connector word only separates two distinct paths when a new
            # absolute path actually starts right after it; otherwise it is
            # part of this path's own directory name (e.g. "Jane and Joe"),
            # so keep scanning through it for this same path's real endpoint.
            if (
                after_connector < len(text)
                and _tautomer_path_start(text, after_connector) is not None
            ):
                return None
            cursor = after_connector
            continue

        if _tautomer_is_path_separator(character, kind):
            while (
                cursor < len(text)
                and _tautomer_is_path_separator(text[cursor], kind)
            ):
                cursor += 1
            continue

        if cursor > content_start and _tautomer_path_start(text, cursor) is not None:
            return None

        suffix = _TAUTOMER_FILE_SUFFIX_RE.match(text, cursor)
        if suffix is None:
            cursor += 1
            continue

        endpoint = suffix.end()
        if endpoint < len(text) and _tautomer_is_path_separator(text[endpoint], kind):
            cursor = endpoint
            continue

        following = text[endpoint] if endpoint < len(text) else ""
        if (
            not following
            or following.isspace()
            or following in "'\",;:)]}!?#."
        ):
            return endpoint

        cursor = endpoint

    return None


def _redact_tautomer_paths_outside_urls(text: str) -> str:
    pieces = []
    cursor = 0

    while cursor < len(text):
        path = _tautomer_path_start(text, cursor)
        if path is None:
            pieces.append(text[cursor])
            cursor += 1
            continue

        kind, content_start = path
        endpoint = _tautomer_path_end(
            text,
            content_start=content_start,
            kind=kind,
        )
        if endpoint is None:
            pieces.append(text[cursor])
            cursor += 1
            continue

        pieces.append("[redacted-path]")
        cursor = endpoint

    return "".join(pieces)


def _sanitize_tautomer_error_text(value: object) -> str:
    """Redact absolute paths while preserving URLs and chemistry verbatim."""

    text = str(value)
    pieces = []
    cursor = 0

    for url in _TAUTOMER_URL_RE.finditer(text):
        pieces.append(_redact_tautomer_paths_outside_urls(text[cursor : url.start()]))
        pieces.append(url.group(0))
        cursor = url.end()

    pieces.append(_redact_tautomer_paths_outside_urls(text[cursor:]))
    return "".join(pieces)


def _sanitize_cli_error_text(value: object) -> str:
    """Generic alias for `_sanitize_tautomer_error_text` (Increment 018,
    Checkpoint 5) - the redaction logic is not actually specific to
    tautomer enumeration (it only ever inspected the TEXT of an error
    message, never anything about tautomers), so it is reused verbatim,
    under a name that does not misleadingly imply a different command, for
    every other CLI command needing the same path-redaction/URL-and-
    chemistry-preservation guarantee (e.g. `ligand promote`). The
    Increment 016/017 tautomer command keeps calling
    `_sanitize_tautomer_error_text` directly, byte-for-byte unchanged, so
    its own approved behavior is never at risk of this alias's existence."""

    return _sanitize_tautomer_error_text(value)


def _echo_tautomer_validation_errors(error: ValidationError) -> None:
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"])
        typer.echo(f"- {field}: {_sanitize_tautomer_error_text(item['msg'])}")


def _tautomer_failure_banner(error: BaseException) -> str:
    """Map the structured original cause to the frozen CLI vocabulary."""

    if isinstance(error, (InvalidSDFError, MultipleMoleculesError, IncompleteEnumerationError)):
        return "EXECUTION ERROR"
    if isinstance(error, RDKitUnavailableError):
        return "CONFIGURATION ERROR"
    if isinstance(error, (FileNotFoundError, FileExistsError)):
        return "INPUT ERROR"
    if isinstance(error, ValueError):
        return "INPUT ERROR"
    if isinstance(error, OSError):
        return "STORAGE ERROR"
    return "EXECUTION ERROR"


def _exit_tautomer_result_presentation_error(
    *,
    banner: str,
    attempt_id: str,
    input_state_id: str,
    operation_id: str,
    error: Exception,
) -> None:
    typer.echo(banner)
    typer.echo()
    typer.echo("TAUTOMER ENUMERATION RESULT ERROR")
    typer.echo()
    typer.echo(f"Attempt ID: {attempt_id}")
    typer.echo(f"Input state: {input_state_id}")
    typer.echo("Scientific enumeration completed: yes")
    typer.echo("Stored result presentation: failed")
    typer.echo(f"Transformation: '{operation_id}'")
    typer.echo(f"Error: {_sanitize_tautomer_error_text(error)}")
    raise typer.Exit(code=1)


@state_app.command("enumerate-tautomers")
def state_enumerate_tautomers(
    experiment: str,
    input_state: str = typer.Option(..., "--input-state"),
    max_states: Optional[int] = typer.Option(None, "--max-states"),
):
    """Enumerate and publish tautomeric states from an existing MolecularState."""

    context = _resolve_experiment_or_exit(
        experiment, error_sanitizer=_sanitize_tautomer_error_text
    )
    validated_input_state = _validate_id_or_exit(
        input_state,
        project_dir=context.project_dir,
        command="state-enumerate-tautomers",
        error_sanitizer=_sanitize_tautomer_error_text,
    )

    try:
        policy = TautomerEnumerationPolicy(
            schema_version="1.0",
            max_states=max_states,
            source="scientist",
        )
    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        _echo_tautomer_validation_errors(error)
        raise typer.Exit(code=1)

    try:
        attempt = enumerate_state_tautomers(
            context.project_dir,
            input_state_id=validated_input_state,
            policy=policy,
        )

    except TautomerEnumerationAttemptPersistenceError as error:
        typer.echo("STORAGE ERROR")
        typer.echo()
        typer.echo(f"Attempt ID: {error.attempt_id}")
        typer.echo(f"Input state: {error.input_state_id}")
        typer.echo(f"Phase: {error.phase}")
        typer.echo(f"Intended status: {error.intended_status}")
        typer.echo("Terminal status persisted: no")
        typer.echo(
            f"Original error: {_sanitize_tautomer_error_text(error.original_error)}"
        )
        typer.echo(
            f"Persistence error: {_sanitize_tautomer_error_text(error.persistence_error)}"
        )
        _echo_unconfirmed_removals(
            error.residual_state_ids,
            error.residual_operation_ids,
        )
        raise typer.Exit(code=1)

    except TautomerEnumerationAttemptFailedError as error:
        typer.echo(_tautomer_failure_banner(error.original_error))
        typer.echo()

        try:
            failed_attempt = load_execution_attempt(context.project_dir, error.attempt_id)
        except Exception:
            typer.echo("TAUTOMER ENUMERATION FAILED")
            typer.echo()
            typer.echo(f"Attempt ID: {error.attempt_id}")
            typer.echo(f"Input state: {error.input_state_id}")
            typer.echo(f"Status: {error.status}")
            typer.echo(f"Error: {_sanitize_tautomer_error_text(error.original_error)}")
            typer.echo()
            typer.echo("(additional attempt details could not be loaded)")
            raise typer.Exit(code=1)

        typer.echo("TAUTOMER ENUMERATION FAILED")
        typer.echo()
        typer.echo(f"Attempt ID: {error.attempt_id}")
        typer.echo(f"Input state: {error.input_state_id}")
        typer.echo(f"Status: {error.status}")
        typer.echo(f"Phase: {failed_attempt.error.phase}")
        typer.echo(f"Error: {_sanitize_tautomer_error_text(failed_attempt.error.message)}")
        _echo_unconfirmed_removals(
            failed_attempt.error.residual_state_ids,
            failed_attempt.error.residual_operation_ids,
        )
        raise typer.Exit(code=1)

    except FileNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(_sanitize_tautomer_error_text(error))
        raise typer.Exit(code=1)

    except FileExistsError as error:
        typer.echo("INPUT ERROR")
        typer.echo(_sanitize_tautomer_error_text(error))
        raise typer.Exit(code=1)

    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        _echo_tautomer_validation_errors(error)
        raise typer.Exit(code=1)

    except ValueError as error:
        typer.echo("INPUT ERROR")
        typer.echo(_sanitize_tautomer_error_text(error))
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(_sanitize_tautomer_error_text(error))
        raise typer.Exit(code=1)

    operation_ids = tuple(attempt.operation_ids or ())
    resolved_results: list[tuple[str, str]] = []
    for operation_id in operation_ids:
        try:
            transformation = load_state_transformation(
                context.project_dir, operation_id
            )
        except (FileNotFoundError, OSError) as error:
            _exit_tautomer_result_presentation_error(
                banner="STORAGE ERROR",
                attempt_id=attempt.id,
                input_state_id=validated_input_state,
                operation_id=operation_id,
                error=error,
            )
        except (ValidationError, yaml.YAMLError) as error:
            _exit_tautomer_result_presentation_error(
                banner="CONFIGURATION ERROR",
                attempt_id=attempt.id,
                input_state_id=validated_input_state,
                operation_id=operation_id,
                error=error,
            )
        resolved_results.append((operation_id, transformation.child_state_id))

    typer.echo("TAUTOMER ENUMERATION")
    typer.echo()
    typer.echo(f"Attempt ID: {attempt.id}")
    typer.echo(f"Input state: {validated_input_state}")
    typer.echo("Status: succeeded")
    typer.echo(f"Engine: {attempt.resolved.engine_name} {attempt.resolved.engine_version}")
    typer.echo(f"New states published: {len(operation_ids)}")
    typer.echo()

    for operation_id, child_state_id in resolved_results:
        typer.echo(
            f"  State '{child_state_id}'  "
            f"(transformation '{operation_id}')"
        )


def _split_param(entry: str) -> Tuple[str, str]:
    if "=" not in entry:
        raise ValueError(f"--param must be in the form name=value (got {entry!r})")

    name, _, raw_value = entry.partition("=")
    return name, raw_value


def _coerce_parameter_value(raw_value: str):
    """Coerce a raw --param value string into the str/int/float/bool shape
    TransformationParameter.value accepts, in this fixed order: bool
    literal, then int, then float, then fall back to the original string."""

    lowered = raw_value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"

    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        pass

    return raw_value


@state_app.command("derive")
def state_derive(
    experiment: str,
    operation: str = typer.Option(..., "--operation"),
    parent_state: str = typer.Option(..., "--parent-state"),
    child_state: str = typer.Option(..., "--child-state"),
    transformation_id: str = typer.Option(..., "--transformation-id"),
    source_file: str = typer.Option(..., "--source-file"),
    format: str = typer.Option(..., "--format"),
    producer_name: Optional[str] = typer.Option(None, "--producer-name"),
    producer_version: Optional[str] = typer.Option(None, "--producer-version"),
    method_description: Optional[str] = typer.Option(None, "--method-description"),
    param: List[str] = typer.Option([], "--param"),
):
    """Register a new molecular state and a StateTransformationRecord
    showing that it was derived from an already-registered parent state
    through the named operation, using the supplied artifact as the
    child's representation.

    This command does not perform protonation or tautomerization itself -
    it records that the transformation the supplied artifact already
    represents took place, and how (a named producer tool, a manual
    method description, or both).
    """

    context = _resolve_experiment_or_exit(experiment)

    def _preflight_error(message: str) -> None:
        """Every CLI-side pre-flight check below happens strictly before
        register_derived_state is ever called, so - unlike the failures
        below that call - it is always safe to log a command="state-derive"
        error entry here: register_derived_state's own started/error
        logging (under command="register-derived-state") never happens for
        these, so nothing is duplicated."""

        typer.echo("INPUT ERROR")
        typer.echo(message)
        _record_activity_safely(context.project_dir, command="state-derive", outcome="error")
        raise typer.Exit(code=1)

    if operation not in ("protonation", "tautomerization"):
        _preflight_error(
            f"'{operation}' is not a supported operation. "
            "Supported operations: protonation, tautomerization."
        )

    if producer_version is not None and producer_name is None:
        _preflight_error("--producer-version requires --producer-name.")

    if producer_name is None and method_description is None:
        _preflight_error(
            "Either --producer-name or --method-description is required - "
            "a transformation must carry some usable provenance."
        )

    try:
        parameter_pairs = [_split_param(entry) for entry in param]
    except ValueError as error:
        _preflight_error(str(error))

    try:
        producer = (
            TransformationProducer(name=producer_name, version=producer_version)
            if producer_name is not None
            else None
        )

        parameters = tuple(
            TransformationParameter(name=name, value=_coerce_parameter_value(raw_value))
            for name, raw_value in parameter_pairs
        )
    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo("Transformation provenance is invalid.")
        _echo_validation_errors(error)
        _record_activity_safely(context.project_dir, command="state-derive", outcome="error")
        raise typer.Exit(code=1)

    try:
        child, transformation = register_derived_state(
            context.project_dir,
            parent_state_id=parent_state,
            child_state_id=child_state,
            source_file=Path(source_file),
            format=format,
            transformation_id=transformation_id,
            operation=operation,
            producer=producer,
            method_description=method_description,
            parameters=parameters,
        )
    except ValidationError as error:
        # register_derived_state itself already performs its own
        # started/success/error activity logging under
        # command="register-derived-state" - no CLI-level log is added
        # for any failure from this call onward.
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"Transformation '{transformation_id}' is invalid.")
        _echo_validation_errors(error)
        raise typer.Exit(code=1)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)
    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    typer.echo(f"State '{child.id}' registered successfully.")
    typer.echo(f"Transformation '{transformation.id}' recorded successfully.")


# --- conformer add/show/list ---


@conformer_app.command("add")
def conformer_add(
    experiment: str,
    conformer_id: str,
    state_id: str = typer.Option(..., "--state"),
    source_file: str = typer.Option(..., "--source-file"),
    format: str = typer.Option(..., "--format"),
):
    """Register an existing geometry file as a Conformer of an
    already-registered MolecularState.

    This command does not generate a conformer - it records a geometry the
    scientist already has. Conformer generation is not implemented in this
    increment.
    """

    context = _resolve_experiment_or_exit(experiment)

    try:
        conformer = register_conformer(
            context.project_dir,
            conformer_id=conformer_id,
            state_id=state_id,
            source_file=Path(source_file),
            format=format,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="conformer-add", outcome="error")
        raise typer.Exit(code=1)
    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="conformer-add", outcome="error")
        raise typer.Exit(code=1)

    typer.echo(f"Conformer '{conformer.id}' registered successfully.")
    _record_activity_safely(context.project_dir, command="conformer-add", outcome="success")


@conformer_app.command("show")
def conformer_show(experiment: str, conformer_id: str):
    """Show a registered Conformer."""

    context = _resolve_experiment_or_exit(experiment)
    validated_id = _validate_id_or_exit(
        conformer_id, project_dir=context.project_dir, command="conformer-show"
    )

    conformer = _call_or_exit(
        load_conformer,
        context.project_dir,
        validated_id,
        project_dir=context.project_dir,
        command="conformer-show",
    )

    typer.echo("CONFORMER")
    typer.echo()
    typer.echo(f"ID: {conformer.id}")
    typer.echo(f"State: {conformer.state_id}")
    typer.echo(f"Representation path: {conformer.representation.path}")
    typer.echo(f"Representation format: {conformer.representation.format}")
    typer.echo(f"Representation checksum: {conformer.representation.checksum}")

    _record_activity_safely(context.project_dir, command="conformer-show", outcome="success")


@conformer_app.command("list")
def conformer_list(
    experiment: str,
    state_id: str = typer.Option(..., "--state"),
):
    """List every registered Conformer for a given state.

    A malformed --state id is rejected before any lookup is attempted. A
    well-formed but unregistered --state id simply yields no results. This
    command never writes an activity log entry, mirroring `experiment
    list`.
    """

    context = _resolve_experiment_or_exit(experiment)
    validated_state_id = _validate_id_or_exit(state_id)

    conformers = _call_or_exit(list_state_conformers, context.project_dir, validated_state_id)

    typer.echo("CONFORMERS")
    typer.echo()

    if not conformers:
        typer.echo(f"No conformers found for state '{validated_state_id}'.")
        return

    for conformer in conformers:
        typer.echo(f"{conformer.id}  (state={conformer.state_id})")


# --- provenance show ---


@provenance_app.command("show")
def provenance_show(
    experiment: str,
    state_id: str = typer.Option(..., "--state"),
):
    """Show StateTransformationRecord provenance for a MolecularState.

    This represents only recorded state-transformation provenance - never
    a complete provenance trail for the molecule, and never the activity
    log. Acquisition provenance (see `ligand show`/`protein show`) is
    tracked separately and is not linked to this MolecularState in the
    current model.
    """

    context = _resolve_experiment_or_exit(experiment)
    validated_state_id = _validate_id_or_exit(
        state_id, project_dir=context.project_dir, command="provenance-show"
    )

    transformations = _call_or_exit(
        list_state_transformations,
        context.project_dir,
        child_state_id=validated_state_id,
        project_dir=context.project_dir,
        command="provenance-show",
    )

    typer.echo("PROVENANCE")
    typer.echo()

    if not transformations:
        typer.echo(
            f"No registered state-transformation provenance exists for "
            f"'{validated_state_id}'. Acquisition provenance is tracked "
            "separately and is not linked to this MolecularState in the "
            "current model."
        )
        _record_activity_safely(context.project_dir, command="provenance-show", outcome="success")
        return

    for transformation in transformations:
        typer.echo(f"Transformation: {transformation.id}")
        typer.echo(f"Parent state: {transformation.parent_state_id}")
        typer.echo(f"Child state: {transformation.child_state_id}")
        typer.echo(f"Operation: {transformation.operation}")

        if transformation.producer is not None:
            typer.echo(f"Producer: {transformation.producer.name}")
            if transformation.producer.version is not None:
                typer.echo(f"Producer version: {transformation.producer.version}")

        if transformation.method_description is not None:
            typer.echo(f"Method description: {transformation.method_description}")

        for parameter in transformation.parameters:
            typer.echo(f"Parameter: {parameter.name}={parameter.value}")

        typer.echo(f"Created at: {transformation.created_at.isoformat()}")
        typer.echo()

    _record_activity_safely(context.project_dir, command="provenance-show", outcome="success")


# --- conditions add/revise/show/list ---
#
# declare_conditions_from_values already performs its own best-effort
# started/success/error activity logging, AND already decides the
# factual created/reused outcome itself (Section 22.6 of the frozen
# spec, extended by the Checkpoint 3 correction) - no command below
# calls record_activity, directly or via _record_activity_safely, at any
# point, including on a pre-flight failure that never reaches the
# Application Service. This is deliberately stricter than some earlier
# commands in this module (e.g. ligand_show, state_derive's own
# pre-flight checks): for this feature, the CLI layer never owns any
# activity-log entry of its own, full stop.
#
# Checkpoint 3 correction (CP3-M1/CP3-M2): this block no longer
# constructs PhPoint/PhRange/PhDeclaration itself, and no longer infers
# created-vs-reused from a clock reading. Both decisions now belong
# entirely to the Application Service (conditions_service.py) - see
# _parse_ph_values_or_exit and _declare_conditions_or_exit below.


def _parse_ph_values_or_exit(
    ph: Optional[str], ph_min: Optional[str], ph_max: Optional[str]
) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
    """Parse --ph / --ph-min + --ph-max into raw Decimal value(s) only -
    never into PhPoint/PhRange/PhDeclaration. Returns
    (ph_point, ph_min, ph_max), exactly one of which (point) xor (both
    range bounds) is non-None, ready to pass straight into
    declare_conditions_from_values.

    This function's own responsibility ends at Section 22.1 steps 1-3 of
    the frozen spec: which CLI flags were given (an exclusivity/syntax
    concern about the options themselves, never a scientific one), the
    64-character length pre-check, and Decimal(...) conversion - never
    float(...) - rejecting only a syntactically invalid string
    (decimal.InvalidOperation). Everything about whether the resulting
    Decimal(s) are scientifically/structurally valid - the three
    decimal_* field constraints, an inverted range, or the point/range
    exclusivity rule itself - is Application Service territory
    (declare_conditions_from_values), never re-implemented here.
    """

    has_ph = ph is not None
    has_min = ph_min is not None
    has_max = ph_max is not None

    if has_ph and (has_min or has_max):
        typer.echo("INPUT ERROR")
        typer.echo("Provide either --ph or both --ph-min/--ph-max, never both forms.")
        raise typer.Exit(code=1)

    if not has_ph and not (has_min and has_max):
        typer.echo("INPUT ERROR")
        typer.echo("Provide either --ph or both --ph-min and --ph-max.")
        raise typer.Exit(code=1)

    raw_fields = [("--ph", ph)] if has_ph else [("--ph-min", ph_min), ("--ph-max", ph_max)]

    for option_name, raw_value in raw_fields:
        if len(raw_value) > 64:
            typer.echo("INPUT ERROR")
            typer.echo(f"{option_name} must not exceed 64 characters.")
            raise typer.Exit(code=1)

    try:
        decimals = [Decimal(raw_value) for _, raw_value in raw_fields]
    except InvalidOperation:
        typer.echo("INPUT ERROR")
        typer.echo("pH value is not a valid decimal number.")
        raise typer.Exit(code=1)

    if has_ph:
        return decimals[0], None, None
    return None, decimals[0], decimals[1]


def _format_ph(ph: PhDeclaration) -> str:
    if ph.point is not None:
        return str(ph.point.value)
    return f"{ph.range.minimum} - {ph.range.maximum}"


def _echo_conditions_declaration(declaration, *, reused: bool) -> None:
    """Print a ScientificConditionsDeclaration's real fields - never
    inventing a field the model does not have, never hiding label/
    rationale/supersedes_id when present. `reused` is always the
    Application Service's own factual determination (never a clock-based
    inference) - callers that have no such fact (show) pass False
    explicitly, since reuse recognition does not apply to reading an
    already-persisted record."""

    typer.echo("SCIENTIFIC CONDITIONS DECLARED")
    typer.echo()
    typer.echo(f"Conditions ID: {declaration.id}")
    typer.echo(f"pH: {_format_ph(declaration.ph)}")
    typer.echo(f"Purpose: {declaration.purpose}")
    typer.echo(f"Declared by: {declaration.declared_by}")
    typer.echo(f"Declared at: {declaration.declared_at.isoformat()}")

    if declaration.label is not None:
        typer.echo(f"Label: {declaration.label}")
    if declaration.rationale is not None:
        typer.echo(f"Rationale: {declaration.rationale}")
    if declaration.supersedes_id is not None:
        typer.echo(f"Supersedes: {declaration.supersedes_id}")
    if reused:
        typer.echo("Result: existing declaration reused")


def _declare_conditions_or_exit(
    project_dir: Path,
    *,
    conditions_id: str,
    ph_point: Optional[Decimal],
    ph_min: Optional[Decimal],
    ph_max: Optional[Decimal],
    purpose: str,
    declared_by: str,
    label: Optional[str],
    rationale: Optional[str],
    supersedes_id: Optional[str],
) -> None:
    """Shared declare/revise flow: exactly one call into
    declare_conditions_from_values, no second persistence path of any
    kind, and no CLI-level activity logging - the Application Service
    already owns its own started/success/error logging (Section 22.6)
    and already decides the factual created/reused outcome itself.

    No clock reading, no activity-log inspection, no pre-existing-file
    check, and no pre-load of any kind happens here or anywhere else in
    this function: `outcome.reused` is used exactly as returned, with no
    further inference.
    """

    try:
        outcome = declare_conditions_from_values(
            project_dir,
            conditions_id=conditions_id,
            ph_point=ph_point,
            ph_min=ph_min,
            ph_max=ph_max,
            purpose=purpose,
            declared_by=declared_by,
            label=label,
            rationale=rationale,
            supersedes_id=supersedes_id,
        )
    except ConditionsValidationError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)
    except ConditionsPredecessorNotFoundError as error:
        typer.echo("PREDECESSOR NOT FOUND")
        typer.echo(str(error))
        raise typer.Exit(code=1)
    except ConditionsRecordInvalidError as error:
        # Reachable via two distinct paths when supersedes_id is given
        # (Section 22.0/22.2's correction): the named predecessor itself
        # unreadable, or the colliding existing record at conditions_id
        # unreadable (the same single path `add` always takes). The
        # command does not need to, and does not, distinguish which -
        # str(error) already names whichever id actually is unreadable
        # (never any field read from the unreadable file itself, since by
        # definition it could not be parsed).
        typer.echo("RECORD INVALID")
        typer.echo(str(error))
        raise typer.Exit(code=1)
    except ConditionsDeclarationConflictError as error:
        typer.echo("CONFLICT")
        typer.echo(str(error))
        raise typer.Exit(code=1)
    except ConditionsPersistenceError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    _echo_conditions_declaration(outcome.declaration, reused=outcome.reused)


@conditions_app.command("add")
def conditions_add(
    experiment: str,
    conditions_id: str,
    ph: Optional[str] = typer.Option(None, "--ph"),
    ph_min: Optional[str] = typer.Option(None, "--ph-min"),
    ph_max: Optional[str] = typer.Option(None, "--ph-max"),
    declared_by: str = typer.Option(..., "--declared-by"),
    purpose: str = typer.Option("simulation_input", "--purpose"),
    label: Optional[str] = typer.Option(None, "--label"),
    rationale: Optional[str] = typer.Option(None, "--rationale"),
):
    """Declare a new scientist-authored experimental conditions record.

    Never infers, defaults, recommends, or computes a pH - the scientist
    must supply either --ph or both --ph-min/--ph-max explicitly. This
    command performs no persistence of its own: every write, every
    idempotency/conflict decision, and the pH value's own scientific
    validity are delegated entirely to the Application Service
    (declare_conditions_from_values).
    """

    context = _resolve_experiment_or_exit(experiment)
    ph_point, ph_min_value, ph_max_value = _parse_ph_values_or_exit(ph, ph_min, ph_max)

    _declare_conditions_or_exit(
        context.project_dir,
        conditions_id=conditions_id,
        ph_point=ph_point,
        ph_min=ph_min_value,
        ph_max=ph_max_value,
        purpose=purpose,
        declared_by=declared_by,
        label=label,
        rationale=rationale,
        supersedes_id=None,
    )


@conditions_app.command("revise")
def conditions_revise(
    experiment: str,
    conditions_id: str,
    supersedes: str = typer.Option(..., "--supersedes"),
    ph: Optional[str] = typer.Option(None, "--ph"),
    ph_min: Optional[str] = typer.Option(None, "--ph-min"),
    ph_max: Optional[str] = typer.Option(None, "--ph-max"),
    declared_by: str = typer.Option(..., "--declared-by"),
    purpose: str = typer.Option("simulation_input", "--purpose"),
    label: Optional[str] = typer.Option(None, "--label"),
    rationale: Optional[str] = typer.Option(None, "--rationale"),
):
    """Declare a new conditions record that explicitly revises --supersedes.

    A revision is a brand-new record under its own new conditions_id -
    never an in-place edit of the predecessor, which is never modified by
    this command. No "latest"/"active" declaration is ever selected
    automatically anywhere in this command or in the Application Service.
    """

    context = _resolve_experiment_or_exit(experiment)
    ph_point, ph_min_value, ph_max_value = _parse_ph_values_or_exit(ph, ph_min, ph_max)

    _declare_conditions_or_exit(
        context.project_dir,
        conditions_id=conditions_id,
        ph_point=ph_point,
        ph_min=ph_min_value,
        ph_max=ph_max_value,
        purpose=purpose,
        declared_by=declared_by,
        label=label,
        rationale=rationale,
        supersedes_id=supersedes,
    )


@conditions_app.command("show")
def conditions_show(experiment: str, conditions_id: str):
    """Show a declared experimental conditions record."""

    context = _resolve_experiment_or_exit(experiment)
    validated_id = _validate_id_or_exit(conditions_id)

    try:
        declaration = load_conditions(context.project_dir, validated_id)
    except FileNotFoundError as error:
        typer.echo("NOT FOUND")
        typer.echo(str(error))
        raise typer.Exit(code=1)
    except ConditionsRecordInvalidError as error:
        typer.echo("RECORD INVALID")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    _echo_conditions_declaration(declaration, reused=False)


@conditions_app.command("list")
def conditions_list(experiment: str):
    """List every declared experimental conditions record for an experiment.

    The order is exactly the (declared_at, id) order list_conditions
    itself returns, unmodified - no re-sorting, no filter, no "current"/
    "active"/"latest" ranking of any kind.
    """

    context = _resolve_experiment_or_exit(experiment)

    try:
        declarations = list_conditions(context.project_dir)
    except ConditionsRecordInvalidError as error:
        typer.echo("RECORD INVALID")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    typer.echo("CONDITIONS")
    typer.echo()

    if not declarations:
        typer.echo(
            "No experimental conditions declared for experiment "
            f"'{context.config.experiment_id}'."
        )
        return

    for declaration in declarations:
        fields = [
            f"ph={_format_ph(declaration.ph)}",
            f"purpose={declaration.purpose}",
            f"declared_by={declaration.declared_by}",
            f"declared_at={declaration.declared_at.isoformat()}",
        ]
        if declaration.supersedes_id is not None:
            fields.append(f"supersedes={declaration.supersedes_id}")
        typer.echo(f"{declaration.id}  ({', '.join(fields)})")
