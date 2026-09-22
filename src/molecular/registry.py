"""Registration persistence: stage -> publish -> commit.

A registration becomes real only when its metadata YAML exists on disk -
that file is the commit marker. The RAW artifact is staged into a private
per-attempt directory first and verified there; only once that succeeds is
it published (renamed) into its final location, and only once *that*
succeeds is the metadata committed atomically. This ordering means a crash
between publish and metadata commit can leave an orphaned RAW directory
with no metadata - a known, accepted residual risk (see module-level notes
in the increment history). An orphan is not a registration. It may be
detected as a storage identity conflict on a later registration attempt
for the same id (see _check_storage_identity_conflict), but this module
does not automatically repair, remove, or recover it.

What is registered here is a ScientificInput / acquisition record, not a
MolecularState - this module knows nothing about scientific state.
"""

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Type, TypeVar

import yaml

from molecular.atomic import atomic_write_text, compute_sha256, publish_staged_directory
from molecular.inputs import (
    AcquisitionMethod,
    AcquisitionRecord,
    LigandInput,
    ProteinInput,
    RawArtifact,
    ScientificInput,
    validate_input_id,
)

logger = logging.getLogger(__name__)

ScientificInputT = TypeVar("ScientificInputT", bound=ScientificInput)

STAGING_DIRNAME = ".staging"


class InputIdentityConflictError(FileExistsError):
    """Raised when input_id collides with an existing registration under a
    kind_dir - an exact duplicate, a casefold collision against a
    differently-cased id, or an orphaned RAW directory with no metadata.

    Storage identity is checked case-insensitively (via .casefold()) no
    matter how the underlying filesystem itself treats case, because two
    distinct logical ids that only differ by case ('Curcumin' vs
    'curcumin') cannot both be given a directory or metadata file in the
    same kind_dir portably across case-sensitive and case-insensitive
    filesystems. This is a storage-layer constraint only - logical identity
    elsewhere in the system remains exact/case-sensitive.

    Subclasses FileExistsError so existing callers that catch
    FileExistsError for "already registered" keep working unchanged.
    """


def _compute_checksum(path: Path) -> str:
    return compute_sha256(path)


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


def _check_storage_identity_conflict(kind_dir: Path, input_id: str) -> None:
    """Raise InputIdentityConflictError if input_id collides - exactly or by
    casefold - with an existing entry under kind_dir: a committed metadata
    file (`<id>.yaml`), a published RAW directory (`<id>/`), or an orphaned
    RAW directory with no metadata at all. The reserved staging namespace
    (`kind_dir/.staging`) is never a candidate - it holds only private,
    uncommitted attempts, never a real or orphaned input.

    Detection is pure Python string comparison over a directory listing; it
    never depends on - and is never masked by - whatever case sensitivity
    the underlying filesystem happens to have. For registrations observed
    sequentially - one call completing before the next begins, the only
    access pattern this increment targets - this check deterministically
    rejects 'Curcumin' once 'curcumin' already occupies the kind_dir
    namespace (or vice versa), on every platform, regardless of whether the
    underlying filesystem is itself case-sensitive.

    This is NOT equivalent to multi-process/multi-thread serialization.
    Two concurrent callers can each run this check - and each see no
    conflict - before either one has published its final RAW directory via
    publish_staged_directory() in atomic.py: there is a residual TOCTOU
    window between this check and the actual filesystem publish. When both
    callers' ids are exactly equal, that window happens to be closed as a
    side effect of publish_staged_directory()'s destination-exists guard
    (the second os.rename fails because the destination is already taken) -
    but that guard is not a general-purpose mutex, and it does not help
    when two ids merely casefold-collide (e.g. 'curcumin' and 'Curcumin'):
    on a case-insensitive filesystem the OS happens to treat those as the
    same path and one rename fails by accident, but on a case-sensitive
    filesystem (e.g. Linux) nothing prevents both renames from succeeding
    independently, which could leave two casefold-colliding registrations
    coexisting. This window is knowingly accepted for this increment: the
    CLI is used as a single process today, multi-process locking/mutex is
    explicitly out of scope, and this protocol makes no promise of
    concurrent-writer safety.
    """

    if not kind_dir.is_dir():
        return

    folded = input_id.casefold()

    for entry in kind_dir.iterdir():
        if entry.name == STAGING_DIRNAME:
            continue

        if entry.is_dir():
            existing_id = entry.name
        elif entry.is_file() and entry.suffix == ".yaml":
            existing_id = entry.stem
        else:
            continue

        if existing_id.casefold() == folded:
            raise InputIdentityConflictError(
                f"'{input_id}' is already registered under {kind_dir} "
                f"(conflicts with existing entry '{existing_id}'; "
                "storage identity is case-insensitive)."
            )


def _cleanup_directory_best_effort(path: Path, *, description: str) -> None:
    """Remove `path` and everything under it, without ever raising.

    A cleanup failure is logged as a warning and swallowed - it must never
    mask whatever error is already propagating from the caller.
    """

    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        logger.warning("Failed to clean up %s at %s", description, path, exc_info=True)


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

    Stage -> publish -> commit: the RAW artifact is copied into a private
    per-attempt staging directory and verified there; only then is it
    published (renamed) to its final location; only once that succeeds is
    the metadata YAML committed atomically. Metadata existing on disk is
    what makes the registration real.

    Validation happens before any filesystem write, so a rejected
    registration (invalid acquisition data, duplicate/conflicting id,
    missing raw file) leaves no partial state on disk under normal
    conditions. If the process is interrupted after the RAW artifact is
    published but before the metadata commit completes, an orphaned RAW
    directory can be left behind - this is documented, accepted residual
    risk, not something this function detects or repairs.
    """

    validate_input_id(input_id)

    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw artifact file not found: {raw_file}")

    kind_dir = project_dir / "inputs" / kind
    metadata_file = kind_dir / f"{input_id}.yaml"
    relative_stored_path = Path("inputs") / kind / input_id / raw_file.name
    file_format_value = (file_format or raw_file.suffix.lstrip(".") or "unknown").lower()

    _check_storage_identity_conflict(kind_dir, input_id)

    source_checksum = _compute_checksum(raw_file)
    acquired_at = datetime.now(timezone.utc)

    # Validate everything (acquisition rules, id/name, etc.) before touching
    # any staging or destination path.
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

    staging_root = kind_dir / STAGING_DIRNAME
    attempt_dir = staging_root / uuid.uuid4().hex
    final_dir = project_dir / relative_stored_path.parent
    raw_published = False

    try:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        staged_file = attempt_dir / raw_file.name
        shutil.copyfile(raw_file, staged_file)

        staged_checksum = _compute_checksum(staged_file)

        if staged_checksum != source_checksum:
            raise OSError(
                f"Checksum mismatch after staging raw artifact for '{input_id}': "
                f"expected {source_checksum}, got {staged_checksum}"
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
            staged_checksum,
            acquired_at,
        )

        serialized_yaml = yaml.safe_dump(
            scientific_input.model_dump(mode="json"), sort_keys=False
        )

        # Re-check immediately before publish - narrows, but does not
        # eliminate, the TOCTOU window against the initial check above.
        _check_storage_identity_conflict(kind_dir, input_id)

        publish_staged_directory(attempt_dir, final_dir)
        raw_published = True

        atomic_write_text(metadata_file, serialized_yaml)

    except BaseException:
        if raw_published:
            _cleanup_directory_best_effort(final_dir, description="published RAW directory")
        else:
            _cleanup_directory_best_effort(attempt_dir, description="staging directory")
        raise

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
