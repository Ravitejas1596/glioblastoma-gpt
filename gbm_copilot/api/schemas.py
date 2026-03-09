"""
Pydantic Schemas for FastAPI
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question")
    literacy_mode: str = Field(
        default="patient",
        description="Response detail level: patient | caregiver | clinician"
    )
    session_id: str | None = Field(default=None, description="Session ID for multi-turn chat")
    conversation_history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    query_type: str
    citations: list[dict[str, Any]]
    confidence_score: float
    safety_flags: list[str]
    is_blocked: bool
    trial_results: list[dict[str, Any]]
    session_id: str
    expanded_query: str


class IngestRequest(BaseModel):
    quick_mode: bool = Field(default=True, description="Quick mode: 500 abstracts vs 50k")


class HealthResponse(BaseModel):
    status: str
    retrieval_mode: str
    index_stats: dict[str, Any]
