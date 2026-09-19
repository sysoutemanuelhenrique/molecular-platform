import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AcquisitionMethod = Literal["uploaded", "downloaded"]

_INPUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_input_id(value: str) -> str:
    """Validate that `value` is safe to use as a filesystem path component
    for an input's storage location.

    This is the single canonical authority for input_id format: it accepts
    only ASCII letters, digits, hyphens and underscores, starting with an
    alphanumeric character - which rules out empty strings, path separators
    (forward or backward slash), leading dots (so it can never be mistaken
    for the reserved `.staging` namespace or any other dotfile), `.`/`..`,
    whitespace, and absolute paths. It never sanitizes, normalizes, or
    otherwise changes the value - a rejected value is rejected outright, by
    raising ValueError, never silently altered.
    """

    if not isinstance(value, str) or not _INPUT_ID_PATTERN.match(value):
        raise ValueError(
            "input_id must match ^[A-Za-z0-9][A-Za-z0-9_-]*$ "
            f"(got {value!r})"
        )

    return value


class RawArtifact(BaseModel):
    """The original acquired file or representation, treated as immutable once registered."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    format: str = Field(min_length=1)
    checksum: str = Field(min_length=1)


class AcquisitionRecord(BaseModel):
    """How a scientific input entered the project."""

    model_config = ConfigDict(extra="forbid")

    method: AcquisitionMethod
    source_name: str | None = None
    source_identifier: str | None = None
    source_uri: str | None = None
    acquired_at: datetime
    raw_artifact: RawArtifact

    @model_validator(mode="after")
    def _require_source_when_downloaded(self) -> "AcquisitionRecord":
        if self.method == "downloaded":
            if not self.source_name:
                raise ValueError(
                    "source_name is required when acquisition method is 'downloaded'"
                )
            if not self.source_identifier and not self.source_uri:
                raise ValueError(
                    "source_identifier or source_uri is required when acquisition "
                    "method is 'downloaded'"
                )
        return self


class ScientificInput(BaseModel):
    """A scientist's declared intention to introduce an entity into the project."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    external_id: str | None = None
    acquisition: AcquisitionRecord

    @field_validator("id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        return validate_input_id(value)


class LigandInput(ScientificInput):
    pass


class ProteinInput(ScientificInput):
    pass
