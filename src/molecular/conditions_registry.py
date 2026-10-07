"""Persistence for ScientificConditionsDeclaration records.

Stores declarations under project_dir/scientific/experimental_conditions,
kept separate from project_dir/scientific/{entities,states,conformers,
transformations,execution_attempts} (Section 18.1 of the frozen spec).

A ScientificConditionsDeclaration has no status machine and no legitimate
overwrite path: register_conditions_declaration is a single, create-once
write, guarded by the same shared identity-conflict primitive
(check_storage_identity_conflict) that registry.py, scientific_registry.py,
transformation_registry.py, and execution_registry.py already use -
casefold-normalized, since `id` is scientist-chosen exactly like
input_id/entity_id/state_id (Section 15.1 of the frozen spec). There is no
mark_* transition anywhere in this module, unlike ExecutionAttempt's or
LigandPromotionAttempt's own registries: a declaration is a single
immutable fact, written once or not at all.

This module owns translation of read-time I/O/parse/schema failures into
ConditionsRecordInvalidError, with __cause__ always preserved - this
applies uniformly whether a record is loaded directly, enumerated, or
consulted as a revision's predecessor or as the colliding record during an
identity-conflict resolution (Section 18.4 of the frozen spec). Translation
of construction-time validation failures (an invalid caller-supplied field)
is conditions_service.py's responsibility, not this module's.

No reservation/claim/owner-token protocol is used here (unlike Increment
018's promotion flow): a declaration is a single, small, purely structural
YAML document with no external artifact and no staging step - the entire
write is one atomic operation (Section 18.6 of the frozen spec).
"""

from pathlib import Path
from typing import Tuple

import yaml
from pydantic import ValidationError

from molecular.atomic import atomic_write_text, check_storage_identity_conflict
from molecular.conditions import ScientificConditionsDeclaration
from molecular.inputs import validate_input_id


class ConditionsIdentityConflictError(FileExistsError):
    """Raised when `conditions_id` collides - exactly or by casefold - with
    an already-persisted declaration under project_dir/scientific/
    experimental_conditions. This is a create-once guard only: there is no
    mark_* transition in this module that this error could ever apply to
    besides the original create.
    """


class ConditionsRecordInvalidError(RuntimeError):
    """Raised when a persisted declaration exists on disk but cannot be
    read back: its YAML fails to parse, or its content fails schema
    validation against ScientificConditionsDeclaration. Reachable via any
    read path - a direct load (`show`), an enumeration (`list`), a
    revision's named predecessor, or the colliding record encountered
    while resolving a ConditionsIdentityConflictError - never only one of
    them. The original yaml.YAMLError or pydantic.ValidationError is always
    preserved as `__cause__`.
    """


def _conditions_dir(project_dir: Path) -> Path:
    return project_dir / "scientific" / "experimental_conditions"


def _persist(project_dir: Path, declaration: ScientificConditionsDeclaration) -> None:
    metadata_file = _conditions_dir(project_dir) / f"{declaration.id}.yaml"
    serialized_yaml = yaml.safe_dump(declaration.model_dump(mode="json"), sort_keys=False)
    atomic_write_text(metadata_file, serialized_yaml)


def register_conditions_declaration(
    project_dir: Path, declaration: ScientificConditionsDeclaration
) -> ScientificConditionsDeclaration:
    """Create-once persist of a freshly-constructed declaration.

    Guards the create path with the same shared, casefold-normalized
    identity-conflict primitive every other registry in this codebase
    uses, raising ConditionsIdentityConflictError on a collision -
    exact or casefold - against an already-persisted declaration's `id`.
    Never overwrites an existing file: recognizing a safe retry versus a
    genuine content conflict under a colliding `id` is conditions_service.py's
    responsibility (a later checkpoint), not this registry's.
    """

    check_storage_identity_conflict(
        _conditions_dir(project_dir),
        declaration.id,
        conflict_error=ConditionsIdentityConflictError,
    )
    _persist(project_dir, declaration)
    return declaration


def load_conditions_declaration(
    project_dir: Path, conditions_id: str
) -> ScientificConditionsDeclaration:
    """Load a previously persisted declaration by id.

    Raises FileNotFoundError if no such declaration is registered -
    mirroring load_state/load_execution_attempt. Raises
    ConditionsRecordInvalidError, with __cause__ preserved, if the file
    exists but its YAML fails to parse or its content fails schema
    validation.
    """

    validate_input_id(conditions_id)
    metadata_file = _conditions_dir(project_dir) / f"{conditions_id}.yaml"

    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"no conditions declaration registered under '{conditions_id}'"
        )

    try:
        data = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        return ScientificConditionsDeclaration.model_validate(data)
    except (yaml.YAMLError, ValidationError) as error:
        raise ConditionsRecordInvalidError(
            f"conditions declaration '{conditions_id}' exists but is not readable/valid"
        ) from error


def list_conditions_declarations(
    project_dir: Path,
) -> Tuple[ScientificConditionsDeclaration, ...]:
    """List every declaration registered under project_dir, in a stable,
    deterministic (declared_at, id) order.

    Returns an empty tuple - never raises - when
    project_dir/scientific/experimental_conditions does not yet exist,
    mirroring check_storage_identity_conflict's own early-return on a
    missing directory. This ordering carries no semantic meaning of
    "current" or "latest": no declaration is ever selected by this
    function, only enumerated.
    """

    directory = _conditions_dir(project_dir)
    if not directory.is_dir():
        return ()

    declarations = [
        load_conditions_declaration(project_dir, path.stem)
        for path in sorted(directory.glob("*.yaml"))
    ]
    return tuple(sorted(declarations, key=lambda d: (d.declared_at, d.id)))
