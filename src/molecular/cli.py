from pathlib import Path
from typing import Optional

import typer
import yaml

from pydantic import ValidationError

from molecular.activity import record_activity
from molecular.config import load_experiment
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
app.add_typer(ligand_app, name="ligand")
app.add_typer(protein_app, name="protein")

@app.callback()
def main():
    """Molecular Platform - scientific molecular research platform."""
    pass


@app.command()
def init(name: str):
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
        experiment_file.write_text(
            'schema_version: "1.0"\n'
            f'experiment_id: "{name}"\n',
            encoding="utf-8",
        )

    typer.echo(f"Experiment '{name}' created successfully.")

def _record_activity_safely(name: str, command: str, outcome: str) -> None:
    """Best-effort activity logging that must never affect the caller's own outcome."""

    try:
        record_activity(Path(name), command=command, outcome=outcome)
    except OSError:
        pass


def _load_experiment_or_exit(name: str, command: str):
    """Load and validate an experiment, exiting with a friendly error on failure."""

    experiment_file = Path(name) / "experiment.yaml"

    try:
        return load_experiment(experiment_file)

    except FileNotFoundError:
        typer.echo("INPUT ERROR")
        typer.echo(f"Experiment '{name}' was not found.")
        _record_activity_safely(name, command=command, outcome="error")
        raise typer.Exit(code=1)

    except yaml.YAMLError:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"Experiment '{name}' has malformed YAML.")
        _record_activity_safely(name, command=command, outcome="error")
        raise typer.Exit(code=1)

    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"Experiment '{name}' is invalid.")

        for item in error.errors():
            field = ".".join(str(part) for part in item["loc"])
            typer.echo(f"- {field}: {item['msg']}")

        _record_activity_safely(name, command=command, outcome="error")
        raise typer.Exit(code=1)


@app.command()
def validate(name: str):
    """Validate a molecular experiment."""

    config = _load_experiment_or_exit(name, command="validate")

    typer.echo(
        f"Experiment '{config.experiment_id}' is valid "
        f"(schema {config.schema_version})."
    )

    _record_activity_safely(name, command="validate", outcome="success")


@app.command()
def plan(name: str):
    """Show the execution plan for a molecular experiment."""

    config = _load_experiment_or_exit(name, command="plan")

    typer.echo("EXPERIMENT PLAN")
    typer.echo()
    typer.echo(f"Experiment: {config.experiment_id}")
    typer.echo(f"Schema: {config.schema_version}")
    typer.echo("Status: VALID")
    typer.echo()
    typer.echo("No execution steps are defined yet.")

    _record_activity_safely(name, command="plan", outcome="success")


def _add_scientific_input(
    *,
    register_fn,
    label: str,
    command: str,
    experiment: str,
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

    _load_experiment_or_exit(experiment, command=command)

    try:
        registered = register_fn(
            Path(experiment),
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
        _record_activity_safely(experiment, command=command, outcome="error")
        raise typer.Exit(code=1)

    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"{label} '{input_id}' is invalid.")

        for item in error.errors():
            field = ".".join(str(part) for part in item["loc"])
            typer.echo(f"- {field}: {item['msg']}")

        _record_activity_safely(experiment, command=command, outcome="error")
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        _record_activity_safely(experiment, command=command, outcome="error")
        raise typer.Exit(code=1)

    typer.echo(f"{label} '{registered.id}' registered successfully.")
    _record_activity_safely(experiment, command=command, outcome="success")


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

    _add_scientific_input(
        register_fn=register_ligand,
        label="Ligand",
        command="ligand-add",
        experiment=experiment,
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

    _load_experiment_or_exit(experiment, command=command)

    if cid <= 0:
        typer.echo("INPUT ERROR")
        typer.echo(f"'{cid}' is not a valid PubChem CID.")
        _record_activity_safely(experiment, command=command, outcome="error")
        raise typer.Exit(code=1)

    try:
        temp_path = fetch_pubchem_sdf_2d(cid)

    except PubChemNotFoundError as error:
        typer.echo("INPUT ERROR")
        typer.echo(str(error))
        _record_activity_safely(experiment, command=command, outcome="error")
        raise typer.Exit(code=1)

    except PubChemAcquisitionError as error:
        typer.echo("ACQUISITION ERROR")
        typer.echo(str(error))
        _record_activity_safely(experiment, command=command, outcome="error")
        raise typer.Exit(code=1)

    except OSError as error:
        typer.echo("STORAGE ERROR")
        typer.echo(str(error))
        _record_activity_safely(experiment, command=command, outcome="error")
        raise typer.Exit(code=1)

    try:
        _add_scientific_input(
            register_fn=register_ligand,
            label="Ligand",
            command=command,
            experiment=experiment,
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

    _add_scientific_input(
        register_fn=register_protein,
        label="Protein",
        command="protein-add",
        experiment=experiment,
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