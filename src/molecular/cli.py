from datetime import datetime, timezone
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
from molecular.atomic import atomic_write_text
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
from molecular.pubchem import PubChemAcquisitionError, PubChemNotFoundError
from molecular.rdkit_tautomer import (
    IncompleteEnumerationError,
    InvalidSDFError,
    MultipleMoleculesError,
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
app.add_typer(ligand_app, name="ligand")
app.add_typer(protein_app, name="protein")
app.add_typer(experiment_app, name="experiment")
app.add_typer(entity_app, name="entity")
app.add_typer(state_app, name="state")
app.add_typer(conformer_app, name="conformer")
app.add_typer(provenance_app, name="provenance")

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


_TAUTOMER_WINDOWS_DRIVE_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])"
    r"[A-Z]:(?:\\+|/)"
    r"(?:[^\\/\r\n'\"<>|?*]+(?:\\+|/))*"
    r"[^\\/\r\n'\"<>|?*]*?\.[A-Za-z0-9]{1,12}"
    r"(?:\.[A-Za-z0-9]{1,12})*"
)
_TAUTOMER_UNC_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])"
    r"\\{2,}"
    r"[^\\/\r\n'\"<>|?*]+\\+"
    r"(?:[^\\/\r\n'\"<>|?*]+\\+)*"
    r"[^\\/\r\n'\"<>|?*]*?\.[A-Za-z0-9]{1,12}"
    r"(?:\.[A-Za-z0-9]{1,12})*"
)
_TAUTOMER_POSIX_FILE_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_+)\]])"
    r"/(?!/)"
    r"(?:[^/\r\n'\"<>]+/)+"
    r"[^/\r\n'\"<>]*?\.[A-Za-z0-9]{1,12}"
    r"(?:\.[A-Za-z0-9]{1,12})*"
)


def _sanitize_tautomer_error_text(value: object) -> str:
    """Redact only lexically unambiguous absolute paths for this command."""

    text = str(value)
    text = _TAUTOMER_WINDOWS_DRIVE_PATH_RE.sub("[redacted-path]", text)
    text = _TAUTOMER_UNC_PATH_RE.sub("[redacted-path]", text)
    return _TAUTOMER_POSIX_FILE_PATH_RE.sub("[redacted-path]", text)


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
