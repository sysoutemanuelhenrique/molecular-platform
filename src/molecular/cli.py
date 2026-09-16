from pathlib import Path

import typer

from pydantic import ValidationError

from molecular.config import load_experiment

app = typer.Typer(
    help="Molecular Platform - scientific molecular research platform."
)

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

@app.command()
def validate(name: str):
    """Validate a molecular experiment."""

    experiment_file = Path(name) / "experiment.yaml"

    try:
        config = load_experiment(experiment_file)

    except FileNotFoundError:
        typer.echo("INPUT ERROR")
        typer.echo(f"Experiment '{name}' was not found.")
        raise typer.Exit(code=1)

    except ValidationError as error:
        typer.echo("CONFIGURATION ERROR")
        typer.echo(f"Experiment '{name}' is invalid.")

        for item in error.errors():
            field = ".".join(str(part) for part in item["loc"])
            typer.echo(f"- {field}: {item['msg']}")

        raise typer.Exit(code=1)

    typer.echo(
        f"Experiment '{config.experiment_id}' is valid "
        f"(schema {config.schema_version})."
    )