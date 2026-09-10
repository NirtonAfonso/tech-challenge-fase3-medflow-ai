"""Estado compartilhado planejado para o fluxo LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict


class MedicalState(TypedDict, total=False):
    trace_id: str
    question: str
    patient_id: str
    patient_context: dict[str, Any]
    retrieved_documents: list[dict[str, Any]]
    sources: list[str]
    answer: str
    safety_status: str
    requires_human_review: bool
    processing_steps: list[str]
