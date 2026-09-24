"""State transformation provenance persistence.

Stores StateTransformationRecords under project_dir/scientific/transformations,
kept separate from project_dir/scientific/states and
project_dir/scientific/entities. A transformation has no artifact of its
own in this increment (see molecular.transformation's module docstring for
why it does not duplicate the parent's/child's own representation path or
checksum) - registration is therefore a single atomic metadata write, the
same shape as molecular.scientific_registry.register_entity.

register_state_transformation never creates or modifies a
ScientificEntity, a MolecularState, or any other StateTransformationRecord:
parent_state_id and child_state_id must both already be registered, loaded
and cross-checked before any write occurs.
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml

from molecular.atomic import atomic_write_text, check_storage_identity_conflict
from molecular.scientific_registry import load_state
from molecular.transformation import StateTransformationRecord

logger = logging.getLogger(__name__)


class TransformationIdentityConflictError(FileExistsError):
    """Raised when a StateTransformationRecord's id collides - exactly or
    by casefold - with an already-registered transformation. Subclasses
    FileExistsError for the same reason InputIdentityConflictError does in
    molecular.registry."""


def _transformations_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "transformations"


def _check_transformation_identity_conflict(
    transformations_dir: Path, transformation_id: str
) -> None:
    """Raise TransformationIdentityConflictError on an exact or casefold
    collision against an already-registered transformation's metadata
    file. Delegates to the shared check_storage_identity_conflict
    primitive in molecular.atomic."""

    check_storage_identity_conflict(
        transformations_dir,
        transformation_id,
        conflict_error=TransformationIdentityConflictError,
    )


def register_state_transformation(
    project_dir: Path, transformation: StateTransformationRecord
) -> StateTransformationRecord:
    """Register a StateTransformationRecord's metadata under
    project_dir/scientific/transformations.

    `transformation` must already be a fully-validated StateTransformationRecord
    - its own model validators already guarantee id formats, the
    self-edge rejection (parent_state_id != child_state_id), the closed
    operation vocabulary, and the producer/method_description requirement.
    This function adds the checks that require reading other already-
    registered records from disk, which no model validator can do on its
    own:

    1. parent_state_id must resolve to an already-registered MolecularState
       (FileNotFoundError otherwise);
    2. child_state_id must resolve to an already-registered MolecularState
       (FileNotFoundError otherwise);
    3. parent and child must belong to the same ScientificEntity
       (ValueError otherwise) - this is a registry invariant, not a model
       invariant, precisely because it depends on loading both states;
    4. the transformation id must not collide, exactly or by casefold,
       with an already-registered transformation.

    Multiple transformations may legitimately point at the same
    child_state_id (independent provenance claims), and cycles
    (A -> B -> A) are permitted - neither is rejected here.

    Never creates or modifies a ScientificEntity or a MolecularState. Never
    overwrites an existing transformation.
    """

    parent = load_state(project_dir, transformation.parent_state_id)
    child = load_state(project_dir, transformation.child_state_id)

    if parent.entity_id != child.entity_id:
        raise ValueError(
            f"parent_state_id '{transformation.parent_state_id}' and "
            f"child_state_id '{transformation.child_state_id}' belong to "
            f"different entities ('{parent.entity_id}' vs '{child.entity_id}') "
            "- a transformation must connect two states of the same "
            "ScientificEntity"
        )

    transformations_dir = _transformations_dir(project_dir)
    _check_transformation_identity_conflict(transformations_dir, transformation.id)

    metadata_file = transformations_dir / f"{transformation.id}.yaml"
    serialized_yaml = yaml.safe_dump(transformation.model_dump(mode="json"), sort_keys=False)

    atomic_write_text(metadata_file, serialized_yaml)

    return transformation


def _remove_state_transformation_best_effort(project_dir: Path, transformation_id: str) -> None:
    """Best-effort compensating removal of a StateTransformationRecord's
    metadata file - mirrors _remove_state_best_effort in
    molecular.scientific_registry exactly, including its scoping
    contract: this is not a public delete function, and must only ever be
    called by an orchestrating Application Service to roll back a
    transformation it just registered within the same call, when a later
    step of that same operation subsequently fails. A
    StateTransformationRecord has no artifact of its own (see this
    module's own docstring), so there is no artifact directory to clean
    up alongside the metadata file - unlike _remove_state_best_effort,
    which also removes a representation directory.

    Never raises: failures are logged as warnings and swallowed, never
    masking whatever error is already propagating from the caller.
    """

    metadata_file = _transformations_dir(project_dir) / f"{transformation_id}.yaml"

    try:
        metadata_file.unlink(missing_ok=True)
    except OSError:
        logger.warning(
            "Failed to clean up %s at %s",
            "transformation metadata",
            metadata_file,
            exc_info=True,
        )


def load_state_transformation(project_dir: Path, transformation_id: str) -> StateTransformationRecord:
    """Load a previously registered StateTransformationRecord by id.

    Raises FileNotFoundError if no such transformation is registered -
    mirroring load_entity/load_state in molecular.scientific_registry.
    """

    metadata_file = _transformations_dir(project_dir) / f"{transformation_id}.yaml"
    data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
    return StateTransformationRecord.model_validate(data)


def list_state_transformations(
    project_dir: Path,
    *,
    parent_state_id: Optional[str] = None,
    child_state_id: Optional[str] = None,
) -> List[StateTransformationRecord]:
    """List every registered StateTransformationRecord, optionally filtered
    by parent_state_id and/or child_state_id. Returns an empty list if no
    transformations directory exists yet, or none match - this is a
    read-only scan, never a write."""

    transformations_dir = _transformations_dir(project_dir)

    if not transformations_dir.is_dir():
        return []

    matches = []

    for entry in sorted(transformations_dir.iterdir()):
        if not (entry.is_file() and entry.suffix == ".yaml"):
            continue

        data = yaml.safe_load(entry.read_text(encoding="utf-8"))
        transformation = StateTransformationRecord.model_validate(data)

        if parent_state_id is not None and transformation.parent_state_id != parent_state_id:
            continue
        if child_state_id is not None and transformation.child_state_id != child_state_id:
            continue

        matches.append(transformation)

    return matches
