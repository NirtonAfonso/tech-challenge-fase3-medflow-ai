"""Nós do fluxo LangGraph do MedFlow AI.

Cada nó é uma função pura sobre ``MedicalState``: recebe o estado, executa uma
responsabilidade única e devolve **apenas** as chaves alteradas. Isso mantém os
nós pequenos, testáveis isoladamente e fáceis de demonstrar.

Ordem lógica::

    validate_input → classify_request → retrieve_patient → check_alerts
                   → retrieve_protocol → generate_answer → validate_answer
                   → [finalize_answer | human_review → finalize_answer] → audit_log
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from medflow_ai.database.repository import PatientNotFoundError, PatientRepository
from medflow_ai.graph.state import MedicalState
from medflow_ai.graph.tools import evaluate_alerts
from medflow_ai.llm.prompts import (
    BLOCKED_RESPONSE,
    HUMAN_REVIEW_NOTICE,
    PROMPT_VERSION,
    build_messages,
    format_protocol_block,
)
from medflow_ai.llm.providers import describe_provider
from medflow_ai.logging_utils.audit import AuditLogger, build_audit_event
from medflow_ai.rag.retriever import ProtocolRetriever
from medflow_ai.safety.policy import (
    POLICY_VERSION,
    SafetyCategory,
    classify_request,
    validate_answer,
)

__all__ = [
    "ROUTE_BLOCKED",
    "ROUTE_PROTOCOL_ONLY",
    "ROUTE_PATIENT_CONTEXT",
    "MedFlowNodes",
]

ROUTE_BLOCKED = "blocked"
ROUTE_PROTOCOL_ONLY = "protocol_only"
ROUTE_PATIENT_CONTEXT = "patient_context"


class MedFlowNodes:
    """Fábrica de nós com as dependências injetadas.

    Injetar repositório, retriever, modelo e logger torna cada nó testável com
    dublês e permite trocar o provedor de LLM sem alterar o grafo.
    """

    def __init__(
        self,
        *,
        retriever: ProtocolRetriever,
        repository: PatientRepository | None = None,
        chat_model: BaseChatModel,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.retriever = retriever
        self.repository = repository
        self.chat_model = chat_model
        self.audit_logger = audit_logger

    # ------------------------------------------------------------------ nós
    def validate_input(self, state: MedicalState) -> dict[str, Any]:
        """Guardrail de ENTRADA: valida a pergunta e classifica o risco."""
        question = (state.get("question") or "").strip()
        errors = list(state.get("errors", []))

        if not question:
            errors.append("Pergunta vazia.")
        if len(question) > 4000:
            errors.append("Pergunta excede o limite de 4000 caracteres.")

        assessment = classify_request(question)
        return {
            "question": question,
            "errors": errors,
            "safety_status": str(assessment.category),
            "safety_rules": assessment.matched_rules,
            "safety_rationales": assessment.rationales,
            "requires_human_review": assessment.requires_human_review,
            "processing_steps": ["validate_input"],
            "metadata": {
                **state.get("metadata", {}),
                "policy_version": POLICY_VERSION,
                "prompt_version": PROMPT_VERSION,
                "llm_provider": describe_provider(self.chat_model),
                "retriever_strategy": self.retriever.strategy,
                "retriever_k": self.retriever.k,
                "started_at": time.time(),
            },
        }

    def classify_request(self, state: MedicalState) -> dict[str, Any]:
        """Define a rota do fluxo a partir da segurança e da presença de paciente."""
        if state.get("safety_status") == str(SafetyCategory.BLOCK):
            route = ROUTE_BLOCKED
        elif state.get("patient_id"):
            route = ROUTE_PATIENT_CONTEXT
        else:
            route = ROUTE_PROTOCOL_ONLY
        return {"route": route, "processing_steps": ["classify_request"]}

    def retrieve_patient(self, state: MedicalState) -> dict[str, Any]:
        """Consulta o prontuário estruturado e monta o contexto minimizado."""
        patient_id = state.get("patient_id", "")
        errors = list(state.get("errors", []))
        if not patient_id:
            return {"processing_steps": ["retrieve_patient:skipped"]}
        if self.repository is None:
            errors.append("Repositório de prontuário indisponível nesta execução.")
            return {"errors": errors, "processing_steps": ["retrieve_patient:unavailable"]}

        try:
            context = self.repository.build_context(patient_id)
        except (PatientNotFoundError, FileNotFoundError) as exc:
            errors.append(f"Falha ao recuperar prontuário: {exc}")
            return {"errors": errors, "processing_steps": ["retrieve_patient:error"]}

        return {
            "patient_context": context.to_dict(),
            "patient_context_text": context.to_prompt_block(),
            "processing_steps": ["retrieve_patient"],
            "metadata": {**state.get("metadata", {}), "patient_context_empty": context.is_empty()},
        }

    def check_alerts(self, state: MedicalState) -> dict[str, Any]:
        """Aplica regras institucionais de alerta sobre o contexto do paciente."""
        patient_id = state.get("patient_id", "")
        if not patient_id or self.repository is None or not state.get("patient_context"):
            return {"processing_steps": ["check_alerts:skipped"]}
        try:
            context = self.repository.build_context(patient_id)
        except (PatientNotFoundError, FileNotFoundError):
            return {"processing_steps": ["check_alerts:error"]}
        return {"alerts": evaluate_alerts(context), "processing_steps": ["check_alerts"]}

    def retrieve_protocol(self, state: MedicalState) -> dict[str, Any]:
        """Recuperação RAG nos protocolos institucionais, com fontes rastreáveis."""
        question = state.get("question", "")
        # A pergunta é enriquecida com as condições do paciente: melhora a
        # recuperação quando o médico escreve "este paciente" sem citar a doença.
        enriched = question
        context = state.get("patient_context") or {}
        conditions = [item.get("descricao", "") for item in context.get("conditions", [])][:3]
        if conditions:
            enriched = f"{question} | condições: {', '.join(conditions)}"

        k = state.get("requested_k") or self.retriever.k
        try:
            chunks = self.retriever.retrieve(enriched, k=k)
        except Exception as exc:  # noqa: BLE001 - falha de retrieval não derruba o fluxo
            errors = list(state.get("errors", []))
            errors.append(f"Falha na recuperação de protocolos: {exc}")
            return {
                "errors": errors,
                "retrieved_documents": [],
                "sources": [],
                "protocol_context_text": "Nenhum trecho de protocolo foi recuperado (falha de retrieval).",
                "processing_steps": ["retrieve_protocol:error"],
            }

        return {
            "retrieved_documents": [
                {"texto": chunk.document.page_content, **chunk.to_source()} for chunk in chunks
            ],
            "sources": self.retriever.sources(chunks),
            "protocol_context_text": format_protocol_block(chunks),
            "processing_steps": ["retrieve_protocol"],
            "metadata": {**state.get("metadata", {}), "retrieved_count": len(chunks)},
        }

    def generate_answer(self, state: MedicalState) -> dict[str, Any]:
        """Gera a resposta com a LLM, ancorada no contexto recuperado."""
        messages = build_messages(
            question=state.get("question", ""),
            patient_context=state.get("patient_context_text", ""),
            protocol_context=state.get(
                "protocol_context_text", "Nenhum trecho de protocolo foi recuperado."
            ),
            safety_status=state.get("safety_status", str(SafetyCategory.SAFE)),
        )
        try:
            response = self.chat_model.invoke(messages)
            draft = str(response.content).strip()
        except Exception as exc:  # noqa: BLE001 - falha do provedor é tratada, não propagada
            errors = list(state.get("errors", []))
            errors.append(f"Falha na geração: {exc}")
            return {
                "draft_answer": "",
                "errors": errors,
                "processing_steps": ["generate_answer:error"],
            }
        return {"draft_answer": draft, "processing_steps": ["generate_answer"]}

    def validate_answer(self, state: MedicalState) -> dict[str, Any]:
        """Guardrail de SAÍDA: audita o rascunho antes de entregá-lo."""
        draft = state.get("draft_answer", "")
        input_category = SafetyCategory(state.get("safety_status", str(SafetyCategory.SAFE)))
        verdict = validate_answer(
            draft,
            sources=state.get("sources", []),
            input_category=input_category,
            require_sources=True,
        )
        requires_review = bool(state.get("requires_human_review")) or not verdict.approved
        return {
            "output_violations": verdict.violations,
            "requires_human_review": requires_review,
            "safety_status": str(
                max(
                    (input_category, verdict.category),
                    key=lambda item: ["SAFE", "CAUTION", "HUMAN_REVIEW", "BLOCK"].index(str(item)),
                )
            ),
            "processing_steps": ["validate_answer"],
        }

    def human_review(self, state: MedicalState) -> dict[str, Any]:
        """Marca o conteúdo como rascunho pendente de validação humana."""
        notice = HUMAN_REVIEW_NOTICE.format(
            safety_status=state.get("safety_status", "HUMAN_REVIEW"),
            rules=", ".join(state.get("safety_rules", [])) or "guardrail de saída",
        )
        draft = state.get("draft_answer", "")
        return {
            "draft_answer": f"{notice}\n{draft}".strip(),
            "requires_human_review": True,
            "processing_steps": ["human_review"],
        }

    def blocked_response(self, state: MedicalState) -> dict[str, Any]:
        """Resposta padrão para solicitações bloqueadas pela política."""
        reason = "; ".join(state.get("safety_rationales", [])) or "Solicitação fora dos limites de atuação."
        return {
            "answer": BLOCKED_RESPONSE.format(reason=reason),
            "draft_answer": "",
            "requires_human_review": True,
            "sources": [],
            "retrieved_documents": [],
            "processing_steps": ["blocked_response"],
        }

    def finalize_answer(self, state: MedicalState) -> dict[str, Any]:
        """Anexa fontes e alertas à resposta final entregue ao médico."""
        parts = [state.get("draft_answer", "").strip()]

        alerts = state.get("alerts", [])
        if alerts:
            parts.append(
                "ALERTAS AUTOMÁTICOS:\n"
                + "\n".join(
                    f"  - [{alert['severidade'].upper()}] {alert['mensagem']} "
                    f"(evidência: {alert['evidencia']}; fonte: {alert['fonte']})"
                    for alert in alerts
                )
            )

        sources = state.get("sources", [])
        if sources:
            parts.append(
                "FONTES CONSULTADAS:\n"
                + "\n".join(
                    f"  [{index}] {source['citation']} (vigência {source['vigencia']}, "
                    f"score {source['score']})"
                    for index, source in enumerate(sources, start=1)
                )
            )
        else:
            parts.append("FONTES CONSULTADAS: nenhuma fonte institucional foi recuperada.")

        if state.get("errors"):
            parts.append("OCORRÊNCIAS TÉCNICAS: " + "; ".join(state["errors"]))

        return {"answer": "\n\n".join(part for part in parts if part), "processing_steps": ["finalize_answer"]}

    def audit_log(self, state: MedicalState) -> dict[str, Any]:
        """Registra o evento de auditoria da execução completa."""
        metadata = state.get("metadata", {})
        started_at = metadata.get("started_at")
        latency_ms = round((time.time() - started_at) * 1000, 2) if started_at else 0.0

        event = build_audit_event(
            trace_id=state.get("trace_id", ""),
            route=state.get("route", "unknown"),
            patient_id=state.get("patient_id") or None,
            sources=[source.get("section_id") for source in state.get("sources", [])],
            safety_status=state.get("safety_status"),
            human_review=bool(state.get("requires_human_review")),
            safety_rules=state.get("safety_rules", []),
            output_violations=state.get("output_violations", []),
            retrieved_documents=len(state.get("retrieved_documents", [])),
            alerts=[alert["id"] for alert in state.get("alerts", [])],
            processing_steps=state.get("processing_steps", []),
            errors=state.get("errors", []),
            latency_ms=latency_ms,
            policy_version=metadata.get("policy_version"),
            prompt_version=metadata.get("prompt_version"),
            llm_provider=metadata.get("llm_provider"),
            retriever_strategy=metadata.get("retriever_strategy"),
            answer_chars=len(state.get("answer", "")),
            status="error" if state.get("errors") else "ok",
        )
        if self.audit_logger is not None:
            self.audit_logger.log(event)
        return {
            "latency_ms": latency_ms,
            "processing_steps": ["audit_log"],
            "metadata": {**metadata, "audit_event": event},
        }
