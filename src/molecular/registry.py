import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Type, TypeVar

import yaml

from molecular.inputs import (
    AcquisitionMethod,
    AcquisitionRecord,
    LigandInput,
    ProteinInput,
    RawArtifact,
    ScientificInput,
)

ScientificInputT = TypeVar("ScientificInputT", bound=ScientificInput)


def _compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _build_scientific_input(
    model_cls: Type[ScientificInputT],
    input_id: str,
    name: str,
    external_id: str | None,
    method: AcquisitionMethod,
    source_name: str | None,
    source_identifier: str | None,
    source_uri: str | None,
    relative_path: Path,
    file_format_value: str,
    checksum: str,
    acquired_at: datetime,
) -> ScientificInputT:
    raw_artifact = RawArtifact(
        path=relative_path.as_posix(),
        format=file_format_value,
        checksum=checksum,
    )

    acquisition = AcquisitionRecord(
        method=method,
        source_name=source_name,
        source_identifier=source_identifier,
        source_uri=source_uri,
        acquired_at=acquired_at,
        raw_artifact=raw_artifact,
    )

    return model_cls(
        id=input_id,
        name=name,
        external_id=external_id,
        acquisition=acquisition,
    )


def _register_input(
    project_dir: Path,
    kind: str,
    model_cls: Type[ScientificInputT],
    input_id: str,
    name: str,
    method: AcquisitionMethod,
    raw_file: Path,
    external_id: str | None = None,
    source_name: str | None = None,
    source_identifier: str | None = None,
    source_uri: str | None = None,
    file_format: str | None = None,
) -> ScientificInputT:
    """Register a scientific input and its RAW artifact under project_dir/inputs/<kind>.

    Validation happens before any filesystem write, so a rejected registration
    (invalid acquisition data, duplicate id) leaves no partial state on disk. The
    checksum actually persisted is computed from the copy stored inside the
    experiment, not from the original raw_file, so it represents the RAW artifact
    as preserved by the project.
    """

    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw artifact file not found: {raw_file}")

    kind_dir = project_dir / "inputs" / kind
    metadata_file = kind_dir / f"{input_id}.yaml"

    if metadata_file.exists():
        raise FileExistsError(f"'{input_id}' is already registered under {kind_dir}.")

    relative_stored_path = Path("inputs") / kind / input_id / raw_file.name
    file_format_value = (file_format or raw_file.suffix.lstrip(".") or "unknown").lower()
    source_checksum = _compute_checksum(raw_file)
    acquired_at = datetime.now(timezone.utc)

    # Validate everything (acquisition rules, id/name, etc.) before touching disk.
    _build_scientific_input(
        model_cls,
        input_id,
        name,
        external_id,
        method,
        source_name,
        source_identifier,
        source_uri,
        relative_stored_path,
        file_format_value,
        source_checksum,
        acquired_at,
    )

    stored_path = project_dir / relative_stored_path
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_file, stored_path)

    stored_checksum = _compute_checksum(stored_path)

    if stored_checksum != source_checksum:
        raise OSError(
            f"Checksum mismatch after storing raw artifact for '{input_id}': "
            f"expected {source_checksum}, got {stored_checksum}"
        )

    scientific_input = _build_scientific_input(
        model_cls,
        input_id,
        name,
        external_id,
        method,
        source_name,
        source_identifier,
        source_uri,
        relative_stored_path,
        file_format_value,
        stored_checksum,
        acquired_at,
    )

    metadata_file.write_text(
        yaml.safe_dump(scientific_input.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    return scientific_input


def _load_input(project_dir: Path, kind: str, model_cls: Type[ScientificInputT], input_id: str) -> ScientificInputT:
    metadata_file = project_dir / "inputs" / kind / f"{input_id}.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return model_cls.model_validate(data)


def register_ligand(project_dir: Path, **kwargs) -> LigandInput:
    return _register_input(project_dir, kind="ligands", model_cls=LigandInput, **kwargs)


def register_protein(project_dir: Path, **kwargs) -> ProteinInput:
    return _register_input(project_dir, kind="proteins", model_cls=ProteinInput, **kwargs)


def load_ligand(project_dir: Path, input_id: str) -> LigandInput:
    return _load_input(project_dir, kind="ligands", model_cls=LigandInput, input_id=input_id)


def load_protein(project_dir: Path, input_id: str) -> ProteinInput:
    return _load_input(project_dir, kind="proteins", model_cls=ProteinInput, input_id=input_id)
