from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AcquisitionMethod = Literal["uploaded", "downloaded"]


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


class LigandInput(ScientificInput):
    pass


class ProteinInput(ScientificInput):
    pass
