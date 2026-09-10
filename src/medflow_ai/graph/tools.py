"""Ferramentas LangChain do assistente clínico.

São ``StructuredTool`` reais (``@tool``), com esquema de argumentos e docstring
em português — utilizáveis tanto pelos nós do LangGraph quanto por um agente
ReAct. Cada ferramenta encapsula um acesso a dado (prontuário, protocolo,
pendências) e devolve **dados já minimizados**.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.tools import tool

from medflow_ai.config import get_settings
from medflow_ai.database.repository import PatientNotFoundError, PatientRepository
from medflow_ai.rag.chunking import chunk_documents
from medflow_ai.rag.embeddings import get_embeddings
from medflow_ai.rag.loaders import load_protocol_documents
from medflow_ai.rag.retriever import ProtocolRetriever
from medflow_ai.rag.vector_store import MedFlowVectorStore

__all__ = [
    "get_repository",
    "get_retriever",
    "consultar_prontuario",
    "buscar_protocolo",
    "listar_exames_pendentes",
    "verificar_alertas_clinicos",
    "MEDFLOW_TOOLS",
    "evaluate_alerts",
]


@lru_cache(maxsize=1)
def get_repository() -> PatientRepository:
    """Repositório de prontuário (instância única por processo)."""
    return PatientRepository()


@lru_cache(maxsize=1)
def get_retriever() -> ProtocolRetriever:
    """Retriever de protocolos, reaproveitando índice salvo quando existir."""
    settings = get_settings()
    embeddings = get_embeddings()
    try:
        store = MedFlowVectorStore.load(settings.vector_store_dir, embeddings)
    except FileNotFoundError:
        store = MedFlowVectorStore.from_documents(
            chunk_documents(load_protocol_documents()), embeddings
        )
    return ProtocolRetriever(store)


# ------------------------------------------------------------------ regras de alerta
_ALERT_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "ALERTA-TSH-01",
        "exame": "TSH",
        "condicao": lambda value: value > 10.0,
        "severidade": "media",
        "mensagem": "TSH acentuadamente elevado; revisar adesão, dose e interações de absorção.",
        "fonte": "PROT-END-001",
    },
    {
        "id": "ALERTA-POTASSIO-01",
        "exame": "Potássio",
        "condicao": lambda value: value >= 6.5 or value <= 2.8,
        "severidade": "critica",
        "mensagem": "Potássio em valor crítico; comunicação imediata à equipe médica.",
        "fonte": "PROC-INT-001",
    },
    {
        "id": "ALERTA-LACTATO-01",
        "exame": "Lactato",
        "condicao": lambda value: value > 4.0,
        "severidade": "critica",
        "mensagem": "Lactato acima de 4 mmol/L: valor crítico compatível com sepse grave.",
        "fonte": "PROT-INF-001",
    },
    {
        "id": "ALERTA-CREATININA-01",
        "exame": "Creatinina",
        "condicao": lambda value: value >= 2.0,
        "severidade": "alta",
        "mensagem": "Creatinina elevada; revisar nefrotóxicos e ajustar doses por função renal.",
        "fonte": "PROT-NEF-001",
    },
    {
        "id": "ALERTA-SATO2-01",
        "exame": "Saturação de oxigênio",
        "condicao": lambda value: value < 92.0,
        "severidade": "alta",
        "mensagem": "Saturação abaixo de 92% em ar ambiente: critério de avaliação para internação.",
        "fonte": "PROT-INF-002",
    },
    {
        "id": "ALERTA-TROPONINA-01",
        "exame": "Troponina ultrassensível",
        "condicao": lambda value: value > 52.0,
        "severidade": "critica",
        "mensagem": "Troponina elevada; avaliar injúria miocárdica aguda e acionar Cardiologia.",
        "fonte": "PROT-CAR-001",
    },
    {
        "id": "ALERTA-GLICEMIA-01",
        "exame": "Glicemia de jejum",
        "condicao": lambda value: value > 250.0,
        "severidade": "media",
        "mensagem": "Hiperglicemia importante; avaliar sintomas e descompensação aguda.",
        "fonte": "PROT-END-003",
    },
)

_INTERACTION_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "ALERTA-INTERACAO-01",
        "medicamentos": ("levotiroxina", "carbonato de cálcio"),
        "severidade": "media",
        "mensagem": (
            "Levotiroxina e carbonato de cálcio em uso: manter intervalo mínimo de 4 horas "
            "entre as administrações para não reduzir a absorção."
        ),
        "fonte": "PROT-END-001",
    },
    {
        "id": "ALERTA-INTERACAO-02",
        "medicamentos": ("levotiroxina", "sulfato ferroso"),
        "severidade": "media",
        "mensagem": "Levotiroxina e sulfato ferroso reduzem absorção; separar as administrações.",
        "fonte": "PROT-END-001",
    },
)


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def evaluate_alerts(context: Any) -> list[dict[str, Any]]:
    """Aplica as regras determinísticas de alerta sobre o contexto do paciente.

    Os alertas são **informativos**: sinalizam à equipe, nunca executam conduta.
    """
    alerts: list[dict[str, Any]] = []

    for observation in getattr(context, "observations", []):
        value = _as_float(observation.get("valor"))
        if value is None:
            continue
        for rule in _ALERT_RULES:
            if observation.get("exame") != rule["exame"]:
                continue
            if rule["condicao"](value):
                alerts.append(
                    {
                        "id": rule["id"],
                        "severidade": rule["severidade"],
                        "mensagem": rule["mensagem"],
                        "evidencia": f"{observation['exame']} {observation['valor']} {observation['unidade']} em {observation['data']}",
                        "fonte": rule["fonte"],
                    }
                )

    active = [str(item.get("descricao", "")).casefold() for item in getattr(context, "medications", [])]
    for rule in _INTERACTION_RULES:
        if all(any(drug in item for item in active) for drug in rule["medicamentos"]):
            alerts.append(
                {
                    "id": rule["id"],
                    "severidade": rule["severidade"],
                    "mensagem": rule["mensagem"],
                    "evidencia": " + ".join(rule["medicamentos"]),
                    "fonte": rule["fonte"],
                }
            )

    pending = getattr(context, "pending_exams", [])
    urgent = [item for item in pending if item.get("prioridade") == "urgente"]
    if urgent:
        alerts.append(
            {
                "id": "ALERTA-PENDENCIA-01",
                "severidade": "alta",
                "mensagem": f"{len(urgent)} exame(s) urgente(s) pendente(s) sem resultado liberado.",
                "evidencia": "; ".join(item["exame"] for item in urgent),
                "fonte": "PROC-INT-001",
            }
        )
    return alerts


# ------------------------------------------------------------------ ferramentas
@tool
def consultar_prontuario(patient_id: str, blocos: str = "todos") -> dict[str, Any]:
    """Consulta o prontuário estruturado de um paciente na base institucional.

    Retorna dados JÁ pseudonimizados e minimizados: faixa etária, sexo,
    condições ativas, exames recentes, medicamentos em uso, procedimentos,
    atendimentos e exames pendentes. Nunca retorna nome, CPF, CNS ou contato.

    Args:
        patient_id: identificador do paciente (ex.: "P-DEMO-0001").
        blocos: "todos" ou lista separada por vírgula entre
            conditions, observations, medications, procedures, encounters, pending_exams.
    """
    include = None if blocos.strip().lower() == "todos" else [
        item.strip() for item in blocos.split(",") if item.strip()
    ]
    try:
        context = get_repository().build_context(patient_id, include=include)
    except (PatientNotFoundError, FileNotFoundError) as exc:
        return {"erro": str(exc), "patient_id": patient_id}
    return context.to_dict()


@tool
def buscar_protocolo(pergunta: str, k: int = 4) -> list[dict[str, Any]]:
    """Recupera trechos de protocolos institucionais relevantes para a pergunta.

    Cada resultado traz o texto do trecho e os metadados de rastreabilidade
    (doc_id, seção, versão, vigência e score), permitindo citar a fonte.

    Args:
        pergunta: pergunta clínica em linguagem natural.
        k: número de trechos a recuperar.
    """
    chunks = get_retriever().retrieve(pergunta, k=k)
    return [
        {"texto": chunk.document.page_content, **chunk.to_source()} for chunk in chunks
    ]


@tool
def listar_exames_pendentes(patient_id: str) -> list[dict[str, Any]]:
    """Lista exames solicitados e ainda não liberados para um paciente.

    Args:
        patient_id: identificador do paciente.
    """
    try:
        return get_repository().pending_exams(patient_id)
    except (PatientNotFoundError, FileNotFoundError) as exc:
        return [{"erro": str(exc)}]


@tool
def verificar_alertas_clinicos(patient_id: str) -> list[dict[str, Any]]:
    """Avalia regras institucionais de alerta sobre o prontuário do paciente.

    Sinaliza valores críticos, interações medicamentosas relevantes e pendências
    urgentes. Os alertas são informativos e não substituem avaliação médica.

    Args:
        patient_id: identificador do paciente.
    """
    try:
        context = get_repository().build_context(patient_id)
    except (PatientNotFoundError, FileNotFoundError) as exc:
        return [{"erro": str(exc)}]
    return evaluate_alerts(context)


MEDFLOW_TOOLS = [
    consultar_prontuario,
    buscar_protocolo,
    listar_exames_pendentes,
    verificar_alertas_clinicos,
]
