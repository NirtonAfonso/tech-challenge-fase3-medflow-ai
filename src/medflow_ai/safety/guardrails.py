"""Compatibilidade com a API inicial de guardrails.

A política completa vive em :mod:`medflow_ai.safety.policy`. Este módulo mantém
a interface simplificada (``allow`` / ``human_review``) usada pelos primeiros
testes do repositório, agora delegando a decisão à política oficial para que
não existam duas fontes de verdade.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from medflow_ai.safety.policy import SafetyCategory, classify_request

__all__ = ["SafetyDecision", "SafetyResult", "assess_request"]


class SafetyDecision(StrEnum):
    ALLOW = "allow"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class SafetyResult:
    decision: SafetyDecision
    reason: str
    category: SafetyCategory = SafetyCategory.SAFE


def assess_request(text: str) -> SafetyResult:
    """Versão binária da classificação de segurança.

    ``SAFE`` e ``CAUTION`` viram ``ALLOW``; ``HUMAN_REVIEW`` e ``BLOCK`` viram
    ``HUMAN_REVIEW``. Para roteamento no LangGraph use ``classify_request``,
    que preserva as quatro categorias.
    """
    assessment = classify_request(text)
    reason = assessment.rationales[0] if assessment.rationales else "Sem regra de risco acionada."
    decision = (
        SafetyDecision.HUMAN_REVIEW if assessment.requires_human_review else SafetyDecision.ALLOW
    )
    return SafetyResult(decision=decision, reason=reason, category=assessment.category)
