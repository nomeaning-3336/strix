"""Shared safety-review data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SafetyVerdict(BaseModel):
    """Strict final output returned by the safety model."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["allow", "block"]
    risk: Literal["low", "medium", "high", "critical"]
    categories: list[str] = Field(default_factory=list, max_length=12)
    reason: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0, le=1)


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    source: Literal["off", "deterministic", "reviewer", "review_error"]
    reason: str
    categories: tuple[str, ...] = ()
    case_id: str | None = None


@dataclass(slots=True)
class InspectionContext:
    evidence_dir: str
    runner: object
    used: bool = False
    incomplete: bool = False
