"""Scientific model persistence: entity and molecular-state registration.

Stores the scientific model under project_dir/scientific, kept entirely
separate from project_dir/inputs (RAW acquisition provenance - see
molecular.registry). A ScientificEntity has no artifact of its own, so its
registration is a single atomic metadata write. A MolecularState's
representation artifact follows the same stage -> publish -> commit
protocol already established by molecular.registry for scientific inputs:
metadata existing on disk is what makes a registration real.

register_state never creates or modifies a ScientificEntity - entity_id
must already be registered, checked before any staging I/O occurs.
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List

import yaml

from molecular.atomic import atomic_write_text, compute_sha256, publish_staged_directory
from molecular.inputs import validate_input_id
from molecular.scientific import MolecularRepresentationRef, MolecularState, ScientificEntity

logger = logging.getLogger(__name__)

STAGING_DIRNAME = ".staging"


class EntityIdentityConflictError(FileExistsError):
    """Raised when entity_id collides - exactly or by casefold - with an
    already-registered ScientificEntity. Subclasses FileExistsError for the
    same reason InputIdentityConflictError does in molecular.registry."""


class StateIdentityConflictError(FileExistsError):
    """Raised when state_id collides - exactly or by casefold - with an
    already-registered MolecularState."""


def _entities_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "entities"


def _states_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "states"


def _state_artifacts_root(project_dir: Path) -> Path:
    return project_dir / "scientific" / "artifacts" / "states"


def _check_entity_identity_conflict(entities_dir: Path, entity_id: str) -> None:
    """Raise EntityIdentityConflictError on an exact or casefold collision
    against an already-registered entity's metadata file. Case-insensitive
    for the same cross-filesystem portability reason as scientific input
    registration in molecular.registry - not a general-purpose mutex; the
    same residual TOCTOU window documented there applies here."""

    if not entities_dir.is_dir():
        return

    folded = entity_id.casefold()

    for entry in entities_dir.iterdir():
        if entry.is_file() and entry.suffix == ".yaml" and entry.stem.casefold() == folded:
            raise EntityIdentityConflictError(
                f"'{entity_id}' is already registered under {entities_dir} "
                f"(conflicts with existing entry '{entry.stem}'; "
                "storage identity is case-insensitive)."
            )


def _check_state_identity_conflict(states_dir: Path, state_id: str) -> None:
    """Raise StateIdentityConflictError on an exact or casefold collision
    against an already-registered state's metadata file."""

    if not states_dir.is_dir():
        return

    folded = state_id.casefold()

    for entry in states_dir.iterdir():
        if entry.is_file() and entry.suffix == ".yaml" and entry.stem.casefold() == folded:
            raise StateIdentityConflictError(
                f"'{state_id}' is already registered under {states_dir} "
                f"(conflicts with existing entry '{entry.stem}'; "
                "storage identity is case-insensitive)."
            )


def _cleanup_directory_best_effort(path: Path, *, description: str) -> None:
    """Remove `path` and everything under it, without ever raising - a
    cleanup failure is logged and swallowed, never masking whatever error is
    already propagating from the caller."""

    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        logger.warning("Failed to clean up %s at %s", description, path, exc_info=True)


def register_entity(project_dir: Path, entity: ScientificEntity) -> ScientificEntity:
    """Register a ScientificEntity's metadata under
    project_dir/scientific/entities.

    There is no artifact to stage for an entity - this is a single atomic
    write, gated by an identity conflict check performed immediately before
    it. Never overwrites an existing entity.
    """

    entities_dir = _entities_dir(project_dir)
    _check_entity_identity_conflict(entities_dir, entity.id)

    metadata_file = entities_dir / f"{entity.id}.yaml"
    serialized_yaml = yaml.safe_dump(entity.model_dump(mode="json"), sort_keys=False)

    atomic_write_text(metadata_file, serialized_yaml)

    return entity


def load_entity(project_dir: Path, entity_id: str) -> ScientificEntity:
    """Load a previously registered ScientificEntity by id.

    Raises FileNotFoundError if no such entity is registered - mirroring
    load_ligand/load_protein in molecular.registry.
    """

    metadata_file = _entities_dir(project_dir) / f"{entity_id}.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return ScientificEntity.model_validate(data)


def register_state(
    project_dir: Path,
    *,
    state_id: str,
    entity_id: str,
    source_file: Path,
    format: str,
) -> MolecularState:
    """Register a MolecularState and its scientific representation artifact
    under project_dir/scientific.

    entity_id must already be registered - this function never creates or
    modifies a ScientificEntity. Stage -> publish -> commit, mirroring
    molecular.registry's registration protocol for scientific inputs: the
    representation file is copied into a private per-attempt staging
    directory and verified there; only once that succeeds is it published
    (renamed) to its final location under
    scientific/artifacts/states/<state_id>/; only once that succeeds is the
    state metadata committed atomically. Metadata existing on disk is what
    makes the registration real.

    Divergence from the illustrative public API in the increment spec: a
    pre-built MolecularState is not accepted as input, because its
    representation.checksum can only be known after the source file has
    been staged and hashed. This function instead builds the
    MolecularRepresentationRef and MolecularState internally from the
    already-staged, already-verified copy - mirroring how
    molecular.registry.register_ligand builds its RawArtifact from a
    raw_file argument rather than accepting a pre-built one.
    """

    validate_input_id(state_id)
    validate_input_id(entity_id)

    if not format:
        raise ValueError("format must be a non-empty string")

    if not source_file.is_file():
        raise FileNotFoundError(f"Representation source file not found: {source_file}")

    # entity_id must already be registered; never created or modified here.
    load_entity(project_dir, entity_id)

    states_dir = _states_dir(project_dir)
    _check_state_identity_conflict(states_dir, state_id)

    artifacts_root = _state_artifacts_root(project_dir)
    relative_path = Path("scientific") / "artifacts" / "states" / state_id / source_file.name

    staging_root = artifacts_root / STAGING_DIRNAME
    attempt_dir = staging_root / uuid.uuid4().hex
    final_dir = artifacts_root / state_id
    artifact_published = False

    try:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        staged_file = attempt_dir / source_file.name
        shutil.copyfile(source_file, staged_file)

        checksum = compute_sha256(staged_file)

        state = MolecularState(
            schema_version="1.0",
            id=state_id,
            entity_id=entity_id,
            representation=MolecularRepresentationRef(
                path=relative_path.as_posix(),
                format=format,
                checksum=checksum,
            ),
        )

        serialized_yaml = yaml.safe_dump(state.model_dump(mode="json"), sort_keys=False)

        # Re-check immediately before publish - narrows, but does not
        # eliminate, the TOCTOU window against the initial check above; same
        # residual risk already accepted for scientific input registration
        # in molecular.registry.
        _check_state_identity_conflict(states_dir, state_id)

        publish_staged_directory(attempt_dir, final_dir)
        artifact_published = True

        metadata_file = states_dir / f"{state_id}.yaml"
        atomic_write_text(metadata_file, serialized_yaml)

    except BaseException:
        if artifact_published:
            _cleanup_directory_best_effort(
                final_dir, description="published representation directory"
            )
        else:
            _cleanup_directory_best_effort(attempt_dir, description="staging directory")
        raise

    return state


def load_state(project_dir: Path, state_id: str) -> MolecularState:
    """Load a previously registered MolecularState by id.

    Raises FileNotFoundError if no such state is registered - mirroring
    load_ligand/load_protein in molecular.registry.
    """

    metadata_file = _states_dir(project_dir) / f"{state_id}.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return MolecularState.model_validate(data)


def list_entity_states(project_dir: Path, entity_id: str) -> List[MolecularState]:
    """List every registered MolecularState whose entity_id matches
    `entity_id`, sorted by state id. Returns an empty list if no states
    directory exists yet, or none match - this is a read-only scan, never a
    write."""

    states_dir = _states_dir(project_dir)

    if not states_dir.is_dir():
        return []

    matches = []

    for entry in sorted(states_dir.iterdir()):
        if not (entry.is_file() and entry.suffix == ".yaml"):
            continue

        data = yaml.safe_load(entry.read_text(encoding="utf-8"))
        state = MolecularState.model_validate(data)

        if state.entity_id == entity_id:
            matches.append(state)

    return matches
