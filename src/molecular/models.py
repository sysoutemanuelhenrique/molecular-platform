from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"]
    experiment_id: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1)
    responsible: str | None = Field(default=None, min_length=1)
    summary: str | None = Field(default=None, min_length=1)
    created_at: datetime | None = None
