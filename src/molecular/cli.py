from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import yaml

from pydantic import ValidationError

from molecular.activity import record_activity
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
from molecular.models import ExperimentConfig
from molecular.pubchem import (
    PubChemAcquisitionError,
    PubChemNotFoundError,
    build_pubchem_sdf_url,
    cleanup_temp_artifact,
    fetch_pubchem_sdf_2d,
)
from molecular.registry import register_ligand, register_protein

app = typer.Typer(
    help="Molecular Platform - scientific molecular research platform."
)

ligand_app = typer.Typer(help="Manage ligand inputs.")
protein_app = typer.Typer(help="Manage protein inputs.")
experiment_app = typer.Typer(help="Query experiment metadata.")
app.add_typer(ligand_app, name="ligand")
app.add_typer(protein_app, name="protein")
app.add_typer(experiment_app, name="experiment")

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

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    typer.echo(f"{label} '{registered.id}' registered successfully.")
    _record_activity_safely(context.project_dir, command=command, outcome="success")


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


@ligand_app.command("fetch-pubchem")
def ligand_fetch_pubchem(
    experiment: str,
    input_id: str,
    cid: int = typer.Option(..., "--cid"),
    name: str = typer.Option(..., "--name"),
):
    """Download a ligand's 2D SDF structure from PubChem by CID and register it."""

    command = "ligand-fetch-pubchem"

    context = _resolve_experiment_or_exit(experiment)

    if cid <= 0:
        typer.echo("INPUT ERROR")
        typer.echo(f"'{cid}' is not a valid PubChem CID.")
        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    try:
        temp_path = fetch_pubchem_sdf_2d(cid)

    except PubChemNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    except PubChemAcquisitionError as error:
        typer.echo("ACQUISITION ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        _record_activity_safely(context.project_dir, command=command, outcome="error")
        raise typer.Exit(code=1)

    try:
        _add_scientific_input(
            register_fn=register_ligand,
            label="Ligand",
            command=command,
            context=context,
            input_id=input_id,
            name=name,
            acquisition_method="downloaded",
            raw_file=str(temp_path),
            external_id=f"CID {cid}",
            source_name="PubChem",
            source_identifier=f"CID {cid}",
            source_uri=build_pubchem_sdf_url(cid),
            file_format="sdf",
        )
    finally:
        cleanup_temp_artifact(temp_path)


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
