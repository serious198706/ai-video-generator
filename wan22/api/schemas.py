from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image: str
    prompt: str | None = None
    negative_prompt: str | None = Field(default=None, alias="negativePrompt")
    duration: float = 5
    resolution: Literal["540p", "720p", "1080p"] | None = None
    webhook_url: str | None = Field(default=None, alias="webhookUrl")
    steps: int | None = Field(default=None, ge=1, le=50)
    quality: int | None = Field(default=None, ge=1, le=10)
    seed: int | None = None
    last_image: str | None = Field(default=None, alias="lastImage")

    @field_validator("image", "last_image", "webhook_url", "prompt", "negative_prompt")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("duration")
    @classmethod
    def _duration(cls, value: float) -> float:
        if value <= 0 or value > 15:
            raise ValueError("duration 需在 (0, 15] 秒")
        return value
