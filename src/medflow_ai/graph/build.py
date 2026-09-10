"""Construção do ``StateGraph`` do MedFlow AI.

Grafo (arestas condicionais em destaque)::

    START
      → validate_input
      → classify_request
      ├─[BLOCK]────────────────→ blocked_response ─┐
      ├─[com patient_id]──────→ retrieve_patient → check_alerts →┐
      └─[sem patient_id]──────────────────────────────────────────→ retrieve_protocol
                                                                   → generate_answer
                                                                   → validate_answer
                                                     ├─[seguro]───→ finalize_answer
                                                     └─[revisão]──→ human_review → finalize_answer
                                                                   → audit_log → END

Há três decisões condicionais reais: bloqueio na entrada, presença de paciente
e necessidade de revisão humana na saída.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from medflow_ai.database.repository import PatientRepository
from medflow_ai.graph.nodes import (
    ROUTE_BLOCKED,
    ROUTE_PATIENT_CONTEXT,
    MedFlowNodes,
)
from medflow_ai.graph.state import MedicalState, new_state
from medflow_ai.graph.tools import get_retriever
from medflow_ai.llm.providers import get_chat_model
from medflow_ai.logging_utils.audit import AuditLogger, get_audit_logger
from medflow_ai.rag.retriever import ProtocolRetriever

__all__ = ["build_graph", "MedFlowAssistant", "AssistantResponse"]


def _route_after_classification(state: MedicalState) -> Literal["blocked_response", "retrieve_patient", "retrieve_protocol"]:
    """Aresta condicional 1: bloqueio, com paciente ou apenas protocolo."""
    route = state.get("route")
    if route == ROUTE_BLOCKED:
        return "blocked_response"
    if route == ROUTE_PATIENT_CONTEXT:
        return "retrieve_patient"
    return "retrieve_protocol"


def _route_after_validation(state: MedicalState) -> Literal["human_review", "finalize_answer"]:
    """Aresta condicional 2: revisão humana ou entrega direta."""
    return "human_review" if state.get("requires_human_review") else "finalize_answer"


def build_graph(
    *,
    retriever: ProtocolRetriever | None = None,
    repository: PatientRepository | None = None,
    chat_model: BaseChatModel | None = None,
    audit_logger: AuditLogger | None = None,
    checkpointer: Any | None = None,
):
    """Compila o ``StateGraph`` do assistente.

    Todas as dependências são injetáveis, o que permite testar o grafo com
    dublês (retriever vazio, modelo que falha, repositório inexistente).
    """
    nodes = MedFlowNodes(
        retriever=retriever or get_retriever(),
        repository=repository,
        chat_model=chat_model or get_chat_model(),
        audit_logger=audit_logger,
    )

    graph = StateGraph(MedicalState)
    graph.add_node("validate_input", nodes.validate_input)
    graph.add_node("classify_request", nodes.classify_request)
    graph.add_node("retrieve_patient", nodes.retrieve_patient)
    graph.add_node("check_alerts", nodes.check_alerts)
    graph.add_node("retrieve_protocol", nodes.retrieve_protocol)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("validate_answer", nodes.validate_answer)
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("blocked_response", nodes.blocked_response)
    graph.add_node("finalize_answer", nodes.finalize_answer)
    graph.add_node("audit_log", nodes.audit_log)

    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "classify_request")
    graph.add_conditional_edges(
        "classify_request",
        _route_after_classification,
        {
            "blocked_response": "blocked_response",
            "retrieve_patient": "retrieve_patient",
            "retrieve_protocol": "retrieve_protocol",
        },
    )
    graph.add_edge("retrieve_patient", "check_alerts")
    graph.add_edge("check_alerts", "retrieve_protocol")
    graph.add_edge("retrieve_protocol", "generate_answer")
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_conditional_edges(
        "validate_answer",
        _route_after_validation,
        {"human_review": "human_review", "finalize_answer": "finalize_answer"},
    )
    graph.add_edge("human_review", "finalize_answer")
    graph.add_edge("blocked_response", "audit_log")
    graph.add_edge("finalize_answer", "audit_log")
    graph.add_edge("audit_log", END)

    return graph.compile(checkpointer=checkpointer)


class AssistantResponse(dict):
    """Resposta do assistente (dicionário com atalhos de leitura)."""

    @property
    def answer(self) -> str:
        return str(self.get("answer", ""))

    @property
    def sources(self) -> list[dict[str, Any]]:
        return list(self.get("sources", []))

    @property
    def safety_status(self) -> str:
        return str(self.get("safety_status", ""))

    @property
    def requires_human_review(self) -> bool:
        return bool(self.get("requires_human_review", False))

    @property
    def processing_steps(self) -> list[str]:
        return list(self.get("processing_steps", []))

    @property
    def alerts(self) -> list[dict[str, Any]]:
        return list(self.get("alerts", []))


class MedFlowAssistant:
    """Fachada de alto nível: monta as dependências e executa o grafo."""

    def __init__(
        self,
        *,
        retriever: ProtocolRetriever | None = None,
        repository: PatientRepository | None = None,
        chat_model: BaseChatModel | None = None,
        audit_logger: AuditLogger | None = None,
        checkpointer: Any | None = None,
        with_audit: bool = True,
    ) -> None:
        if repository is None:
            try:
                repository = PatientRepository()
            except FileNotFoundError:
                repository = None  # o fluxo continua respondendo só com protocolos
        if audit_logger is None and with_audit:
            audit_logger = get_audit_logger()
        self.graph = build_graph(
            retriever=retriever,
            repository=repository,
            chat_model=chat_model,
            audit_logger=audit_logger,
            checkpointer=checkpointer,
        )

    def ask(
        self,
        question: str,
        *,
        patient_id: str | None = None,
        trace_id: str | None = None,
        k: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> AssistantResponse:
        """Executa o fluxo completo para uma pergunta clínica."""
        state = new_state(question, patient_id=patient_id, trace_id=trace_id, requested_k=k)
        result = self.graph.invoke(state, config=config or {})
        return AssistantResponse(result)

    def mermaid(self) -> str:
        """Diagrama Mermaid gerado a partir do grafo real (evita diagrama defasado)."""
        return self.graph.get_graph().draw_mermaid()
