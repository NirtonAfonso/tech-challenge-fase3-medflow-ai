"""Estado compartilhado do fluxo LangGraph.

``MedicalState`` é o contrato único entre os nós: cada nó lê o que precisa e
devolve apenas as chaves que alterou. ``processing_steps`` acumula a trilha de
execução (usada no log de auditoria e nos testes de roteamento).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

__all__ = ["MedicalState", "new_state", "RouteName"]

RouteName = str


class MedicalState(TypedDict, total=False):
    """Estado do assistente clínico."""

    # Entrada
    trace_id: str
    question: str
    patient_id: str
    requested_k: int

    # Roteamento e segurança
    route: RouteName
    safety_status: str
    safety_rules: list[str]
    safety_rationales: list[str]
    requires_human_review: bool
    output_violations: list[str]

    # Contexto recuperado
    patient_context: dict[str, Any]
    patient_context_text: str
    retrieved_documents: list[dict[str, Any]]
    protocol_context_text: str
    sources: list[dict[str, Any]]
    alerts: list[dict[str, Any]]

    # Saída
    draft_answer: str
    answer: str
    errors: list[str]
    latency_ms: float
    metadata: dict[str, Any]

    # Acumulado por todos os nós (reducer de concatenação).
    processing_steps: Annotated[list[str], operator.add]


def new_state(
    question: str,
    *,
    patient_id: str | None = None,
    trace_id: str | None = None,
    requested_k: int | None = None,
) -> MedicalState:
    """Cria o estado inicial de uma execução."""
    import uuid

    state: MedicalState = {
        "trace_id": trace_id or str(uuid.uuid4()),
        "question": question,
        "patient_id": patient_id or "",
        "processing_steps": [],
        "errors": [],
        "sources": [],
        "alerts": [],
        "retrieved_documents": [],
        "metadata": {},
    }
    if requested_k:
        state["requested_k"] = requested_k
    return state
