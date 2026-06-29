"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research question or topic", min_length=5)
    session_id: str = Field(default="", description="Optional session ID for continuity")


class ResearchResponse(BaseModel):
    session_id: str
    status: str  # "started" | "completed" | "failed"
    message: str


class ResearchStatusResponse(BaseModel):
    session_id: str
    status: str  # "planning" | "researching" | "synthesizing" | "writing" | "reviewing" | "completed" | "failed"
    sub_queries: list[dict] = []
    result_count: int = 0
    report: str = ""
    errors: list[str] = []


class DocumentUploadResponse(BaseModel):
    filename: str
    chunk_count: int
    message: str


class MemoryResponse(BaseModel):
    topics: list[dict] = []
    summaries: list[dict] = []
