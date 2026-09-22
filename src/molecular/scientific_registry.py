"""Scientific model persistence: entity, molecular-state, and conformer
registration.

Stores the scientific model under project_dir/scientific, kept entirely
separate from project_dir/inputs (RAW acquisition provenance - see
molecular.registry). A ScientificEntity has no artifact of its own, so its
registration is a single atomic metadata write. A MolecularState's or a
Conformer's representation artifact follows the same stage -> publish ->
commit protocol already established by molecular.registry for scientific
inputs: metadata existing on disk is what makes a registration real.

register_state never creates or modifies a ScientificEntity - entity_id
must already be registered, checked before any staging I/O occurs.
register_conformer never creates or modifies a MolecularState - state_id
must already be registered, checked the same way. Conformer ids are
globally unique at the project level, in the same flat namespace style as
entities/states/transformations - never scoped per-state.
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List

import yaml

from molecular.atomic import (
    atomic_write_text,
    check_storage_identity_conflict,
    compute_sha256,
    publish_staged_directory,
)
from molecular.inputs import validate_input_id
from molecular.scientific import (
    Conformer,
    MolecularRepresentationRef,
    MolecularState,
    ScientificEntity,
)

logger = logging.getLogger(__name__)

STAGING_DIRNAME = ".staging"


class EntityIdentityConflictError(FileExistsError):
    """Raised when entity_id collides - exactly or by casefold - with an
    already-registered ScientificEntity. Subclasses FileExistsError for the
    same reason InputIdentityConflictError does in molecular.registry."""


class StateIdentityConflictError(FileExistsError):
    """Raised when state_id collides - exactly or by casefold - with an
    already-registered MolecularState."""


class ConformerIdentityConflictError(FileExistsError):
    """Raised when conformer_id collides - exactly or by casefold - with an
    already-registered Conformer. Subclasses FileExistsError for the same
    reason InputIdentityConflictError does in molecular.registry."""


def _entities_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "entities"


def _states_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "states"


def _state_artifacts_root(project_dir: Path) -> Path:
    return project_dir / "scientific" / "artifacts" / "states"


def _conformers_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "conformers"


def _conformer_artifacts_root(project_dir: Path) -> Path:
    return project_dir / "scientific" / "artifacts" / "conformers"


def _check_entity_identity_conflict(entities_dir: Path, entity_id: str) -> None:
    """Raise EntityIdentityConflictError on an exact or casefold collision
    against an already-registered entity's metadata file. Delegates to the
    shared check_storage_identity_conflict primitive in molecular.atomic -
    see its docstring for the fuller discussion of the residual TOCTOU
    window against concurrent writers."""

    check_storage_identity_conflict(
        entities_dir, entity_id, conflict_error=EntityIdentityConflictError
    )


def _check_state_identity_conflict(states_dir: Path, state_id: str) -> None:
    """Raise StateIdentityConflictError on an exact or casefold collision
    against an already-registered state's metadata file. Delegates to the
    shared check_storage_identity_conflict primitive in molecular.atomic."""

    check_storage_identity_conflict(
        states_dir, state_id, conflict_error=StateIdentityConflictError
    )


def _check_conformer_identity_conflict(conformers_dir: Path, conformer_id: str) -> None:
    """Raise ConformerIdentityConflictError on an exact or casefold
    collision against an already-registered conformer's metadata file.
    Delegates to the shared check_storage_identity_conflict primitive in
    molecular.atomic."""

    check_storage_identity_conflict(
        conformers_dir, conformer_id, conflict_error=ConformerIdentityConflictError
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
    modifies a ScientificEntity. Stage -> verify -> publish -> commit,
    mirroring molecular.registry's registration protocol for scientific
    inputs: the representation file is copied into a private per-attempt
    staging directory; its checksum is computed there and compared against
    the source file's own checksum (computed before the copy), so the
    published copy is proven to match the original byte-for-byte, not just
    internally consistent with itself. Only once that succeeds is it
    published (renamed) to its final location under
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

    source_checksum = compute_sha256(source_file)

    try:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        staged_file = attempt_dir / source_file.name
        shutil.copyfile(source_file, staged_file)

        staged_checksum = compute_sha256(staged_file)

        if staged_checksum != source_checksum:
            raise OSError(
                f"Checksum mismatch after staging representation for '{state_id}': "
                f"expected {source_checksum}, got {staged_checksum}"
            )

        state = MolecularState(
            schema_version="1.0",
            id=state_id,
            entity_id=entity_id,
            representation=MolecularRepresentationRef(
                path=relative_path.as_posix(),
                format=format,
                checksum=staged_checksum,
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


def _remove_state_best_effort(project_dir: Path, state_id: str) -> None:
    """Best-effort compensating removal of a MolecularState's metadata and
    representation artifact directory.

    Not a public delete_state() API: this is a narrowly-scoped internal
    helper meant to be called only by an orchestrating Application Service
    to roll back a state it just created within the same call, when a
    later step of that same operation (e.g. registering a
    StateTransformationRecord) subsequently fails. It must never be used to
    remove a preexisting state - callers can only safely reach for it on a
    state_id they themselves just successfully passed to register_state,
    since register_state already rejects a preexisting state_id outright.

    Never raises: failures are logged as warnings via
    _cleanup_directory_best_effort/best-effort unlink, exactly like the
    cleanup register_state already performs internally on its own failure
    paths - a compensation failure must never mask whatever error is
    already propagating from the caller.
    """

    metadata_file = _states_dir(project_dir) / f"{state_id}.yaml"

    try:
        metadata_file.unlink(missing_ok=True)
    except OSError:
        logger.warning(
            "Failed to clean up %s at %s", "state metadata", metadata_file, exc_info=True
        )

    artifact_dir = _state_artifacts_root(project_dir) / state_id
    _cleanup_directory_best_effort(artifact_dir, description="published representation directory")


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


def register_conformer(
    project_dir: Path,
    *,
    conformer_id: str,
    state_id: str,
    source_file: Path,
    format: str,
) -> Conformer:
    """Register a Conformer and its representation artifact under
    project_dir/scientific.

    state_id must already be registered - this function never creates or
    modifies a MolecularState. Stage -> verify -> publish -> commit,
    mirroring register_state's own protocol exactly: the representation
    file is copied into a private per-attempt staging directory; its
    checksum is computed there and compared against the source file's own
    checksum (computed before the copy); only once that succeeds is it
    published (renamed) to its final location under
    scientific/artifacts/conformers/<conformer_id>/; only once that
    succeeds is the conformer metadata committed atomically. Metadata
    existing on disk is what makes the registration real.

    Registering a Conformer does not assert its geometry is chemically
    valid or genuinely three-dimensional, and does not assert it is
    consistent with its parent state's configurational identity - see
    Conformer's own docstring in molecular.scientific for the exact
    boundary this increment does and does not enforce.
    """

    validate_input_id(conformer_id)
    validate_input_id(state_id)

    if not format:
        raise ValueError("format must be a non-empty string")

    if not source_file.is_file():
        raise FileNotFoundError(f"Representation source file not found: {source_file}")

    # state_id must already be registered; never created or modified here.
    load_state(project_dir, state_id)

    conformers_dir = _conformers_dir(project_dir)
    _check_conformer_identity_conflict(conformers_dir, conformer_id)

    artifacts_root = _conformer_artifacts_root(project_dir)
    relative_path = (
        Path("scientific") / "artifacts" / "conformers" / conformer_id / source_file.name
    )

    staging_root = artifacts_root / STAGING_DIRNAME
    attempt_dir = staging_root / uuid.uuid4().hex
    final_dir = artifacts_root / conformer_id
    artifact_published = False

    source_checksum = compute_sha256(source_file)

    try:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        staged_file = attempt_dir / source_file.name
        shutil.copyfile(source_file, staged_file)

        staged_checksum = compute_sha256(staged_file)

        if staged_checksum != source_checksum:
            raise OSError(
                f"Checksum mismatch after staging representation for '{conformer_id}': "
                f"expected {source_checksum}, got {staged_checksum}"
            )

        conformer = Conformer(
            schema_version="1.0",
            id=conformer_id,
            state_id=state_id,
            representation=MolecularRepresentationRef(
                path=relative_path.as_posix(),
                format=format,
                checksum=staged_checksum,
            ),
        )

        serialized_yaml = yaml.safe_dump(conformer.model_dump(mode="json"), sort_keys=False)

        # Re-check immediately before publish - narrows, but does not
        # eliminate, the TOCTOU window against the initial check above;
        # same residual risk already accepted for state registration above.
        _check_conformer_identity_conflict(conformers_dir, conformer_id)

        publish_staged_directory(attempt_dir, final_dir)
        artifact_published = True

        metadata_file = conformers_dir / f"{conformer_id}.yaml"
        atomic_write_text(metadata_file, serialized_yaml)

    except BaseException:
        if artifact_published:
            _cleanup_directory_best_effort(
                final_dir, description="published representation directory"
            )
        else:
            _cleanup_directory_best_effort(attempt_dir, description="staging directory")
        raise

    return conformer


def load_conformer(project_dir: Path, conformer_id: str) -> Conformer:
    """Load a previously registered Conformer by id.

    Raises FileNotFoundError if no such conformer is registered - mirroring
    load_state/load_entity in this module.
    """

    metadata_file = _conformers_dir(project_dir) / f"{conformer_id}.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return Conformer.model_validate(data)


def list_state_conformers(project_dir: Path, state_id: str) -> List[Conformer]:
    """List every registered Conformer whose state_id matches `state_id`,
    sorted by conformer id. Returns an empty list if no conformers
    directory exists yet, or none match - this is a read-only scan, never
    a write."""

    conformers_dir = _conformers_dir(project_dir)

    if not conformers_dir.is_dir():
        return []

    matches = []

    for entry in sorted(conformers_dir.iterdir()):
        if not (entry.is_file() and entry.suffix == ".yaml"):
            continue

        data = yaml.safe_load(entry.read_text(encoding="utf-8"))
        conformer = Conformer.model_validate(data)

        if conformer.state_id == state_id:
            matches.append(conformer)

    return matches
