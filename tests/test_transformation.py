"""Model-level tests for state transformation provenance:
StateTransformationRecord, TransformationProducer, TransformationParameter.
No filesystem I/O here - see test_transformation_registry.py for
persistence behavior."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from molecular.transformation import (
    StateTransformationRecord,
    TransformationParameter,
    TransformationProducer,
)


def _producer(**overrides):
    defaults = dict(name="ExternalTool", version="1.0")
    defaults.update(overrides)
    return TransformationProducer(**defaults)


def _record(**overrides):
    defaults = dict(
        schema_version="1.0",
        id="ibuprofen-deprotonation-001",
        parent_state_id="ibuprofen-neutral",
        child_state_id="ibuprofen-anion",
        operation="protonation",
        producer=_producer(),
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return StateTransformationRecord(**defaults)


# --- provenance requirement: producer OR method_description ---


def test_record_accepts_producer_provenance():
    record = _record(producer=_producer(), method_description=None)

    assert record.producer.name == "ExternalTool"
    assert record.method_description is None


def test_record_accepts_manual_provenance():
    record = _record(
        producer=None,
        method_description="Prepared manually in ChemDraw and exported as SDF.",
    )

    assert record.producer is None
    assert record.method_description.startswith("Prepared manually")


def test_record_accepts_producer_and_description_together():
    record = _record(
        producer=_producer(),
        method_description="Cross-checked manually against literature.",
    )

    assert record.producer is not None
    assert record.method_description is not None


def test_record_rejects_neither_producer_nor_description():
    with pytest.raises(ValidationError):
        _record(producer=None, method_description=None)


def test_record_rejects_blank_method_description():
    with pytest.raises(ValidationError):
        _record(producer=None, method_description="   ")


def test_record_rejects_empty_method_description_string():
    with pytest.raises(ValidationError):
        _record(producer=None, method_description="")


# --- TransformationProducer ---


def test_producer_rejects_blank_name():
    with pytest.raises(ValidationError):
        _producer(name="   ")


def test_producer_rejects_empty_name():
    with pytest.raises(ValidationError):
        _producer(name="")


def test_producer_rejects_blank_version():
    with pytest.raises(ValidationError):
        _producer(version="   ")


def test_producer_accepts_missing_version():
    producer = _producer(version=None)

    assert producer.version is None


def test_producer_is_immutable():
    producer = _producer()

    with pytest.raises(ValidationError):
        producer.name = "OtherTool"


# --- TransformationParameter ---


def test_parameter_accepts_str_value():
    parameter = TransformationParameter(name="solvent", value="water")
    assert parameter.value == "water"
    assert isinstance(parameter.value, str)


def test_parameter_accepts_int_value():
    parameter = TransformationParameter(name="charge", value=1)
    assert parameter.value == 1
    assert isinstance(parameter.value, int)


def test_parameter_accepts_float_value():
    parameter = TransformationParameter(name="ph", value=7.4)
    assert parameter.value == 7.4
    assert isinstance(parameter.value, float)


def test_parameter_accepts_bool_value():
    parameter = TransformationParameter(name="aqueous", value=True)
    assert parameter.value is True


def test_parameter_rejects_none_value():
    with pytest.raises(ValidationError):
        TransformationParameter(name="ph", value=None)


def test_parameter_rejects_dict_value():
    with pytest.raises(ValidationError):
        TransformationParameter(name="ph", value={"min": 7.0, "max": 7.4})


def test_parameter_rejects_list_value():
    with pytest.raises(ValidationError):
        TransformationParameter(name="ph", value=[7.0, 7.4])


def test_parameter_rejects_blank_name():
    with pytest.raises(ValidationError):
        TransformationParameter(name="   ", value=7.0)


def test_parameter_is_immutable():
    parameter = TransformationParameter(name="ph", value=7.0)

    with pytest.raises(ValidationError):
        parameter.value = 8.0


# --- StateTransformationRecord: parameters ---


def test_record_accepts_distinct_parameter_names():
    record = _record(
        parameters=(
            TransformationParameter(name="ph_min", value=7.0),
            TransformationParameter(name="ph_max", value=7.4),
        )
    )

    assert len(record.parameters) == 2


def test_record_rejects_duplicate_parameter_names():
    with pytest.raises(ValidationError):
        _record(
            parameters=(
                TransformationParameter(name="ph", value=7.0),
                TransformationParameter(name="ph", value=7.4),
            )
        )


# --- StateTransformationRecord: operation vocabulary ---


def test_record_accepts_protonation():
    record = _record(operation="protonation")
    assert record.operation == "protonation"


def test_record_accepts_tautomerization():
    record = _record(operation="tautomerization")
    assert record.operation == "tautomerization"


def test_record_rejects_stereochemistry_operation():
    with pytest.raises(ValidationError):
        _record(operation="stereochemistry")


def test_record_rejects_resonance_operation():
    with pytest.raises(ValidationError):
        _record(operation="resonance")


def test_record_rejects_unknown_operation():
    with pytest.raises(ValidationError):
        _record(operation="something_else")


# --- StateTransformationRecord: identity ---


def test_record_rejects_invalid_id():
    with pytest.raises(ValidationError):
        _record(id="a/b")


def test_record_rejects_invalid_parent_id():
    with pytest.raises(ValidationError):
        _record(parent_state_id="a/b")


def test_record_rejects_invalid_child_id():
    with pytest.raises(ValidationError):
        _record(child_state_id="a/b")


# --- StateTransformationRecord: self-edge ---


def test_record_rejects_self_edge_directly_by_model():
    with pytest.raises(ValidationError):
        StateTransformationRecord(
            schema_version="1.0",
            id="curcumin-noop-001",
            parent_state_id="state-a",
            child_state_id="state-a",
            operation="tautomerization",
            producer=_producer(),
            created_at=datetime.now(timezone.utc),
        )


# --- StateTransformationRecord: general ---


def test_record_accepts_valid_data():
    record = _record()

    assert record.id == "ibuprofen-deprotonation-001"
    assert record.parent_state_id == "ibuprofen-neutral"
    assert record.child_state_id == "ibuprofen-anion"


def test_record_rejects_extra_fields():
    with pytest.raises(ValidationError):
        StateTransformationRecord(
            schema_version="1.0",
            id="ibuprofen-deprotonation-001",
            parent_state_id="ibuprofen-neutral",
            child_state_id="ibuprofen-anion",
            operation="protonation",
            producer=_producer(),
            created_at=datetime.now(timezone.utc),
            confidence=0.9,
        )


def test_record_is_immutable():
    record = _record()

    with pytest.raises(ValidationError):
        record.operation = "tautomerization"


# --- parameter name case sensitivity is intentional (F-012-C-05) ---


def test_record_accepts_parameter_names_differing_only_by_case():
    """"pH" and "ph" are distinct parameter names by design - parameter
    names are not storage identities and are never casefold-normalized,
    unlike id/parent_state_id/child_state_id."""

    record = _record(
        parameters=(
            TransformationParameter(name="pH", value=7.0),
            TransformationParameter(name="ph", value=7.4),
        )
    )

    assert len(record.parameters) == 2
    assert {parameter.name for parameter in record.parameters} == {"pH", "ph"}
