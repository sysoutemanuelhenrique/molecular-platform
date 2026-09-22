"""Direct tests for the register_derived_state application service: no
CliRunner (there is no CLI wiring for it in this increment), no typer.
These tests exercise the orchestration itself - attempt_id correlation,
started/success/error activity logging, and compensating cleanup on
failure. Business rules already covered by test_scientific_registry.py
and test_transformation_registry.py are not repeated here."""

import pytest

import molecular.application as application_module
from molecular.activity import record_activity
from molecular.application import DerivedStateActivityDetails, register_derived_state
from molecular.scientific import ScientificEntity
from molecular.scientific_registry import load_state, register_entity, register_state
from molecular.transformation import TransformationProducer
from molecular.transformation_registry import load_state_transformation


def _entity(entity_id="curcumin", name="Curcumin"):
    return ScientificEntity(schema_version="1.0", id=entity_id, name=name)


def _write_source_file(tmp_path, name="structure.sdf", content=b"fake-sdf-bytes"):
    source_file = tmp_path / name
    source_file.write_bytes(content)
    return source_file


def _setup_entity_with_parent(tmp_path, entity_id="curcumin", parent_state_id="state-a"):
    register_entity(tmp_path, _entity(entity_id=entity_id))
    register_state(
        tmp_path,
        state_id=parent_state_id,
        entity_id=entity_id,
        source_file=_write_source_file(tmp_path, name="parent.sdf"),
        format="sdf",
    )


def _activity_lines(project_dir):
    log_file = project_dir / "logs" / "activity.log"
    if not log_file.exists():
        return []
    return log_file.read_text(encoding="utf-8").splitlines()


def _field(line, name):
    for token in line.split(" "):
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    return None


def _record_activity_raising_only_for(target_outcomes, exc_cls=OSError):
    """Build a record_activity replacement that raises `exc_cls` for the
    given outcome(s) and otherwise delegates to the real implementation -
    lets tests fail exactly one of started/success/error independently,
    with whichever exception type the test wants to simulate."""

    real_record_activity = record_activity

    def _fake(project_dir, command, outcome, *, details=None):
        if outcome in target_outcomes:
            raise exc_cls(f"simulated {outcome} log failure")
        return real_record_activity(project_dir, command, outcome, details=details)

    return _fake


# --- 1. happy path ---


def test_register_derived_state_happy_path(tmp_path):
    _setup_entity_with_parent(tmp_path)

    child, transformation = register_derived_state(
        tmp_path,
        parent_state_id="state-a",
        child_state_id="state-b",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
        transformation_id="transformation-1",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool"),
    )

    assert child.id == "state-b"
    assert child.entity_id == "curcumin"
    assert transformation.parent_state_id == "state-a"
    assert transformation.child_state_id == "state-b"

    assert load_state(tmp_path, "state-b").id == "state-b"
    assert load_state_transformation(tmp_path, "transformation-1").id == "transformation-1"


# --- 2-6. started log content and ordering ---


def test_started_is_written_before_child_persistence(tmp_path, monkeypatch):
    _setup_entity_with_parent(tmp_path)

    def _fail_immediately(*args, **kwargs):
        raise AssertionError("register_state must not be reached in this test")

    monkeypatch.setattr(application_module, "register_state", _fail_immediately)

    with pytest.raises(AssertionError):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=_write_source_file(tmp_path, name="b.sdf"),
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )

    lines = _activity_lines(tmp_path)
    assert len(lines) >= 1
    assert "outcome=started" in lines[0]


def test_started_contains_all_correlation_fields(tmp_path):
    _setup_entity_with_parent(tmp_path)

    register_derived_state(
        tmp_path,
        parent_state_id="state-a",
        child_state_id="state-b",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
        transformation_id="transformation-1",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool"),
    )

    started_line = _activity_lines(tmp_path)[0]

    assert "outcome=started" in started_line
    assert _field(started_line, "attempt_id") is not None
    assert _field(started_line, "parent_state_id") == "state-a"
    assert _field(started_line, "child_state_id") == "state-b"
    assert _field(started_line, "transformation_id") == "transformation-1"


# --- 7-9. attempt_id correlation ---


def test_success_reuses_same_attempt_id_as_started(tmp_path):
    _setup_entity_with_parent(tmp_path)

    register_derived_state(
        tmp_path,
        parent_state_id="state-a",
        child_state_id="state-b",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
        transformation_id="transformation-1",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool"),
    )

    lines = _activity_lines(tmp_path)
    started_line = next(line for line in lines if "outcome=started" in line)
    success_line = next(line for line in lines if "outcome=success" in line)

    assert _field(started_line, "attempt_id") == _field(success_line, "attempt_id")


def test_error_reuses_same_attempt_id_as_started(tmp_path):
    _setup_entity_with_parent(tmp_path)

    with pytest.raises(FileNotFoundError):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=tmp_path / "does-not-exist.sdf",
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )

    lines = _activity_lines(tmp_path)
    started_line = next(line for line in lines if "outcome=started" in line)
    error_line = next(line for line in lines if "outcome=error" in line)

    assert _field(started_line, "attempt_id") == _field(error_line, "attempt_id")


def test_two_attempts_for_same_transformation_id_use_different_attempt_ids(tmp_path):
    _setup_entity_with_parent(tmp_path)

    for _ in range(2):
        with pytest.raises(FileExistsError):
            register_derived_state(
                tmp_path,
                parent_state_id="state-a",
                child_state_id="state-a",  # collides with the parent's own id - fails fast
                source_file=_write_source_file(tmp_path, name="dup.sdf"),
                format="sdf",
                transformation_id="transformation-retry",
                operation="tautomerization",
                producer=TransformationProducer(name="ExternalTool"),
            )

    started_lines = [line for line in _activity_lines(tmp_path) if "outcome=started" in line]

    assert len(started_lines) == 2
    assert _field(started_lines[0], "attempt_id") != _field(started_lines[1], "attempt_id")


# --- 10. started logging failure does not block operation ---


def test_started_logging_failure_does_not_block_operation(tmp_path, monkeypatch):
    _setup_entity_with_parent(tmp_path)

    monkeypatch.setattr(
        application_module, "record_activity", _record_activity_raising_only_for({"started"})
    )

    child, transformation = register_derived_state(
        tmp_path,
        parent_state_id="state-a",
        child_state_id="state-b",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
        transformation_id="transformation-1",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool"),
    )

    assert child.id == "state-b"
    assert transformation.id == "transformation-1"


# --- 11. child registration failure ---


def test_child_registration_failure(tmp_path):
    _setup_entity_with_parent(tmp_path)

    with pytest.raises(FileNotFoundError):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=tmp_path / "does-not-exist.sdf",
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )

    assert not (tmp_path / "scientific" / "states" / "state-b.yaml").exists()

    lines = _activity_lines(tmp_path)
    assert any("outcome=error" in line for line in lines)


# --- 12-13. transformation failure after child creation ---


def test_transformation_validation_failure_after_child_creation(tmp_path):
    _setup_entity_with_parent(tmp_path)

    with pytest.raises(Exception):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=_write_source_file(tmp_path, name="b.sdf"),
            format="sdf",
            transformation_id="transformation-1",
            operation="resonance",  # invalid - not enforced at the Python call boundary
            producer=TransformationProducer(name="ExternalTool"),
        )

    # compensated: the child must not remain
    assert not (tmp_path / "scientific" / "states" / "state-b.yaml").exists()
    assert not (tmp_path / "scientific" / "artifacts" / "states" / "state-b").exists()

    lines = _activity_lines(tmp_path)
    assert any("outcome=error" in line for line in lines)


def test_transformation_metadata_failure(tmp_path, monkeypatch):
    _setup_entity_with_parent(tmp_path)

    def _raise(*args, **kwargs):
        raise OSError("simulated transformation metadata write failure")

    monkeypatch.setattr(application_module, "register_state_transformation", _raise)

    with pytest.raises(OSError):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=_write_source_file(tmp_path, name="b.sdf"),
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )

    assert not (tmp_path / "scientific" / "states" / "state-b.yaml").exists()


# --- 14. compensation removes newly created child ---


def test_compensation_removes_newly_created_child(tmp_path):
    _setup_entity_with_parent(tmp_path)

    with pytest.raises(Exception):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=_write_source_file(tmp_path, name="b.sdf"),
            format="sdf",
            transformation_id="transformation-1",
            operation="resonance",
            producer=TransformationProducer(name="ExternalTool"),
        )

    metadata_file = tmp_path / "scientific" / "states" / "state-b.yaml"
    artifact_dir = tmp_path / "scientific" / "artifacts" / "states" / "state-b"

    assert not metadata_file.exists()
    assert not artifact_dir.exists()


# --- 15. cleanup failure preserves original error ---


def test_cleanup_failure_preserves_original_error(tmp_path, monkeypatch, caplog):
    import logging

    import molecular.scientific_registry as scientific_registry_module

    _setup_entity_with_parent(tmp_path)

    def _raise_rmtree(*args, **kwargs):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(scientific_registry_module.shutil, "rmtree", _raise_rmtree)

    with caplog.at_level(logging.WARNING, logger="molecular.scientific_registry"):
        with pytest.raises(Exception, match="resonance") as exc_info:
            register_derived_state(
                tmp_path,
                parent_state_id="state-a",
                child_state_id="state-b",
                source_file=_write_source_file(tmp_path, name="b.sdf"),
                format="sdf",
                transformation_id="transformation-1",
                operation="resonance",
                producer=TransformationProducer(name="ExternalTool"),
            )

    # the ORIGINAL validation error is what propagated, not a cleanup error
    assert "resonance" in str(exc_info.value) or "operation" in str(exc_info.value).lower()
    assert any("Failed to clean up" in record.getMessage() for record in caplog.records)


# --- 16. preexisting child is never deleted ---


def test_preexisting_child_is_never_deleted(tmp_path):
    _setup_entity_with_parent(tmp_path)

    # register "state-b" directly as a root state (preexisting, not
    # created by register_derived_state at all)
    register_state(
        tmp_path,
        state_id="state-b",
        entity_id="curcumin",
        source_file=_write_source_file(tmp_path, name="preexisting.sdf"),
        format="sdf",
    )

    with pytest.raises(FileExistsError):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",  # already taken by the preexisting root state
            source_file=_write_source_file(tmp_path, name="b-attempt.sdf"),
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )

    # the preexisting state must survive completely untouched
    preexisting = load_state(tmp_path, "state-b")
    assert preexisting.representation.path.endswith("preexisting.sdf")


# --- 17. error logging failure does not mask original exception ---


def test_error_logging_failure_does_not_mask_original_exception(tmp_path, monkeypatch):
    _setup_entity_with_parent(tmp_path)

    monkeypatch.setattr(
        application_module, "record_activity", _record_activity_raising_only_for({"error"})
    )

    with pytest.raises(FileNotFoundError):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=tmp_path / "does-not-exist.sdf",
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )


# --- 18. success logging failure does not turn success into failure ---


def test_success_logging_failure_does_not_turn_success_into_failure(tmp_path, monkeypatch):
    _setup_entity_with_parent(tmp_path)

    monkeypatch.setattr(
        application_module, "record_activity", _record_activity_raising_only_for({"success"})
    )

    child, transformation = register_derived_state(
        tmp_path,
        parent_state_id="state-a",
        child_state_id="state-b",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
        transformation_id="transformation-1",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool"),
    )

    assert child.id == "state-b"
    assert transformation.id == "transformation-1"
    assert load_state(tmp_path, "state-b").id == "state-b"
    assert load_state_transformation(tmp_path, "transformation-1").id == "transformation-1"


# --- 19. simulated crash boundary ---


def test_simulated_crash_boundary_leaves_enough_identifiers_in_started_log(tmp_path, monkeypatch):
    """Simulates an abrupt failure between child creation and transformation
    registration: even considering only the `started` line (as if no
    terminal success/error had ever been written, mimicking a hard
    process crash), it already carries enough identifiers -
    child_state_id and transformation_id - to go verify what actually
    happened on the scientific filesystem."""

    _setup_entity_with_parent(tmp_path)

    def _raise(*args, **kwargs):
        raise OSError("simulated crash boundary")

    monkeypatch.setattr(application_module, "register_state_transformation", _raise)

    with pytest.raises(OSError):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=_write_source_file(tmp_path, name="b.sdf"),
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )

    started_line = _activity_lines(tmp_path)[0]

    assert "outcome=started" in started_line
    assert _field(started_line, "child_state_id") == "state-b"
    assert _field(started_line, "transformation_id") == "transformation-1"

    # compensation already ran (this was a normal exception, not a real
    # process crash) - but the recovery-relevant identifiers were already
    # in the started line regardless of what happens afterward.


# --- 20. started without terminal event does not itself trigger deletion ---


def test_started_without_terminal_event_does_not_trigger_deletion(tmp_path):
    """A `started` entry with no matching success/error must never, by
    itself, cause anything to be deleted - this project implements no
    startup/log-driven auto-repair. Regression guard against that ever
    being added implicitly."""

    _setup_entity_with_parent(tmp_path)

    record_activity(
        tmp_path,
        command="register-derived-state",
        outcome="started",
        details=DerivedStateActivityDetails(
            attempt_id="orphaned-attempt",
            parent_state_id="state-a",
            child_state_id="state-a",
            transformation_id="transformation-orphan",
        ),
    )

    assert load_state(tmp_path, "state-a").id == "state-a"
    assert (tmp_path / "scientific" / "states" / "state-a.yaml").exists()


# --- C-01: KeyboardInterrupt / SystemExit boundary ---


@pytest.mark.parametrize("interrupt_cls", [KeyboardInterrupt, SystemExit])
def test_interrupt_after_child_created_triggers_compensation(tmp_path, monkeypatch, interrupt_cls):
    """A KeyboardInterrupt/SystemExit arriving after the child was
    registered but before the transformation is registered must still
    trigger compensation and a best-effort error log, then propagate
    unchanged - proving register_derived_state's compensation boundary is
    `except BaseException`, not `except Exception`."""

    _setup_entity_with_parent(tmp_path)

    child_metadata_file = tmp_path / "scientific" / "states" / "state-b.yaml"
    child_existed_at_interrupt_time = {"value": None}

    def _raise_interrupt(*args, **kwargs):
        # the child must be REAL on disk at this point - not mocked away -
        # proving it was actually created before the interrupt arrived.
        child_existed_at_interrupt_time["value"] = child_metadata_file.exists()
        raise interrupt_cls()

    monkeypatch.setattr(application_module, "register_state_transformation", _raise_interrupt)

    with pytest.raises(interrupt_cls):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=_write_source_file(tmp_path, name="b.sdf"),
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )

    # the child genuinely existed on disk at the moment of interruption
    assert child_existed_at_interrupt_time["value"] is True

    # ... and was compensated (removed) afterward
    assert not child_metadata_file.exists()
    assert not (tmp_path / "scientific" / "artifacts" / "states" / "state-b").exists()

    # no transformation was ever registered
    assert not (
        tmp_path / "scientific" / "transformations" / "transformation-1.yaml"
    ).exists()

    # the parent remains completely untouched
    parent = load_state(tmp_path, "state-a")
    assert parent.id == "state-a"

    # a best-effort error log was attempted with the same attempt_id as started
    lines = _activity_lines(tmp_path)
    started_line = next(line for line in lines if "outcome=started" in line)
    error_line = next(line for line in lines if "outcome=error" in line)
    assert _field(started_line, "attempt_id") == _field(error_line, "attempt_id")


# --- C-02: id validation happens before the started log ---


@pytest.mark.parametrize(
    "bad_id",
    ["with space", "with=equals", "with\nnewline"],
)
@pytest.mark.parametrize(
    "field_name", ["parent_state_id", "child_state_id", "transformation_id"]
)
def test_invalid_ids_fail_before_started_log(tmp_path, field_name, bad_id):
    """A malformed parent_state_id/child_state_id/transformation_id must be
    rejected before any activity-log I/O - no attempt_id is generated, no
    `started` line is written, and no child or transformation is created."""

    _setup_entity_with_parent(tmp_path)

    kwargs = dict(
        parent_state_id="state-a",
        child_state_id="state-b",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
        transformation_id="transformation-1",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool"),
    )
    kwargs[field_name] = bad_id

    with pytest.raises(ValueError):
        register_derived_state(tmp_path, **kwargs)

    assert _activity_lines(tmp_path) == []
    assert not (tmp_path / "scientific" / "states" / "state-b.yaml").exists()
    assert not (tmp_path / "scientific" / "transformations" / "transformation-1.yaml").exists()


# --- C-03: activity logging failure boundary is Exception, not OSError-only ---


def test_started_logging_runtime_error_does_not_block_operation(tmp_path, monkeypatch):
    _setup_entity_with_parent(tmp_path)

    monkeypatch.setattr(
        application_module,
        "record_activity",
        _record_activity_raising_only_for({"started"}, exc_cls=RuntimeError),
    )

    child, transformation = register_derived_state(
        tmp_path,
        parent_state_id="state-a",
        child_state_id="state-b",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
        transformation_id="transformation-1",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool"),
    )

    assert child.id == "state-b"
    assert transformation.id == "transformation-1"


def test_success_logging_runtime_error_does_not_turn_success_into_failure(tmp_path, monkeypatch):
    _setup_entity_with_parent(tmp_path)

    monkeypatch.setattr(
        application_module,
        "record_activity",
        _record_activity_raising_only_for({"success"}, exc_cls=RuntimeError),
    )

    child, transformation = register_derived_state(
        tmp_path,
        parent_state_id="state-a",
        child_state_id="state-b",
        source_file=_write_source_file(tmp_path, name="b.sdf"),
        format="sdf",
        transformation_id="transformation-1",
        operation="tautomerization",
        producer=TransformationProducer(name="ExternalTool"),
    )

    assert child.id == "state-b"
    assert load_state(tmp_path, "state-b").id == "state-b"
    assert load_state_transformation(tmp_path, "transformation-1").id == "transformation-1"


def test_error_logging_runtime_error_does_not_mask_original_exception(tmp_path, monkeypatch):
    _setup_entity_with_parent(tmp_path)

    monkeypatch.setattr(
        application_module,
        "record_activity",
        _record_activity_raising_only_for({"error"}, exc_cls=RuntimeError),
    )

    # the ORIGINAL persistence/scientific exception must still be the one
    # that propagates, not the RuntimeError from the (failing) error log
    with pytest.raises(FileNotFoundError):
        register_derived_state(
            tmp_path,
            parent_state_id="state-a",
            child_state_id="state-b",
            source_file=tmp_path / "does-not-exist.sdf",
            format="sdf",
            transformation_id="transformation-1",
            operation="tautomerization",
            producer=TransformationProducer(name="ExternalTool"),
        )
