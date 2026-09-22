from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import typer
import yaml

from pydantic import ValidationError

from molecular.activity import record_activity
from molecular.application import (
    InvalidPubChemCidError,
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
from molecular.inputs import validate_input_id
from molecular.models import ExperimentConfig
from molecular.pubchem import PubChemAcquisitionError, PubChemNotFoundError
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
from molecular.transformation_registry import list_state_transformations

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


def _resolve_experiment_or_exit(identifier: str) -> ExperimentContext:
    """Resolve an experiment by its logical experiment_id, or exit with a
    friendly error.

    There is no activity log entry on failure here: without a resolved
    context there is no project_dir to log against.
    """

    try:
        return resolve_experiment(identifier, loader=load_experiment)

    except ExperimentNotFoundError:
        typer.echo("INPUT ERROR")
        typer.echo(f"Experiment '{identifier}' was not found.")
        raise typer.Exit(code=1)

    except InvalidExperimentConfigError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"Experiment '{identifier}' is invalid.")

        for item in error.validation_error.errors():
            field = ".".join(str(part) for part in item["loc"])
            typer.echo(f"- {field}: {item['msg']}")

        raise typer.Exit(code=1)

    except AmbiguousExperimentError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(f"Could not read the current directory: {error}")
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
    value: str, *, project_dir: Optional[Path] = None, command: Optional[str] = None
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
        typer.echo(str(error))
        if project_dir is not None and command is not None:
            _record_activity_safely(project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)


def _echo_scientific_input(item, *, label: str) -> None:
    """Print a registered ScientificInput's (Ligand/Protein) real fields -
    never inventing a field the model does not have."""

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
    typer.echo(f"Path: {item.acquisition.raw_artifact.path}")
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

    try:
        ligand = load_ligand(context.project_dir, validated_id)
    except FileNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="ligand-show", outcome="error")
        raise typer.Exit(code=1)

    _echo_scientific_input(ligand, label="LIGAND")
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

    try:
        protein = load_protein(context.project_dir, validated_id)
    except FileNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="protein-show", outcome="error")
        raise typer.Exit(code=1)

    _echo_scientific_input(protein, label="PROTEIN")
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

    try:
        entity = load_entity(context.project_dir, validated_id)
    except FileNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="entity-show", outcome="error")
        raise typer.Exit(code=1)

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

    try:
        state = load_state(context.project_dir, validated_id)
    except FileNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="state-show", outcome="error")
        raise typer.Exit(code=1)

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

    states = list_entity_states(context.project_dir, validated_entity_id)

    typer.echo("STATES")
    typer.echo()

    if not states:
        typer.echo(f"No states found for entity '{validated_entity_id}'.")
        return

    for state in states:
        typer.echo(f"{state.id}  (entity={state.entity_id})")


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

    if operation not in ("protonation", "tautomerization"):
        typer.echo("INPUT ERROR")
        typer.echo(
            f"'{operation}' is not a supported operation. "
            "Supported operations: protonation, tautomerization."
        )
        raise typer.Exit(code=1)

    if producer_version is not None and producer_name is None:
        typer.echo("INPUT ERROR")
        typer.echo("--producer-version requires --producer-name.")
        raise typer.Exit(code=1)

    if producer_name is None and method_description is None:
        typer.echo("INPUT ERROR")
        typer.echo(
            "Either --producer-name or --method-description is required - "
            "a transformation must carry some usable provenance."
        )
        raise typer.Exit(code=1)

    try:
        parameter_pairs = [_split_param(entry) for entry in param]
    except ValueError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        raise typer.Exit(code=1)

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

    try:
        conformer = load_conformer(context.project_dir, validated_id)
    except FileNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command="conformer-show", outcome="error")
        raise typer.Exit(code=1)

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

    conformers = list_state_conformers(context.project_dir, validated_state_id)

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

    transformations = list_state_transformations(
        context.project_dir, child_state_id=validated_state_id
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
