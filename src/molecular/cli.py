from pathlib import Path

import typer


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