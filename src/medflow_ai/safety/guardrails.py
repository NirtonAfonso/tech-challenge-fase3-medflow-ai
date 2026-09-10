"""Guardrails iniciais.

Esta versão é propositalmente conservadora e serve apenas como contrato inicial
para testes. A política final deverá combinar regras, prompt de segurança e
validação no LangGraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SafetyDecision(StrEnum):
    ALLOW = "allow"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class SafetyResult:
    decision: SafetyDecision
    reason: str


_HIGH_RISK_MARKERS = (
    "prescreva",
    "receite",
    "prescrever",
    "receitar",
    "dose exata",
    "sem validação médica",
    "sem consultar o médico",
)


def assess_request(text: str) -> SafetyResult:
    """Sinaliza pedidos explícitos de decisão/prescrição autônoma.

    Não é um classificador clínico. É apenas a primeira camada determinística
    do sistema e deve ser complementada posteriormente.
    """

    normalized = text.casefold().strip()
    if any(marker in normalized for marker in _HIGH_RISK_MARKERS):
        return SafetyResult(
            decision=SafetyDecision.HUMAN_REVIEW,
            reason="Pedido potencialmente sensível requer validação humana.",
        )
    return SafetyResult(
        decision=SafetyDecision.ALLOW,
        reason="Nenhum marcador determinístico de alto risco foi encontrado.",
    )
