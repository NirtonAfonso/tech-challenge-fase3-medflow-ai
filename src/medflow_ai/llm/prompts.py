"""Prompts versionados do MedFlow AI (LangChain).

Prompts são tratados como artefato de software: têm versão, são montados por
``ChatPromptTemplate`` e possuem casos de teste em ``tests/test_prompts.py``.
Alterar o prompt exige incrementar ``PROMPT_VERSION`` — o valor é registrado no
log de auditoria de cada resposta, permitindo comparar execuções ao longo do
tempo.

Os blocos ``<<...>>`` são delimitadores estáveis: separam contexto de protocolo,
contexto do paciente e pergunta, e são usados também pelo provedor extrativo
determinístico (``template``).
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

__all__ = [
    "PROMPT_VERSION",
    "ASSISTANT_SYSTEM_PROMPT",
    "CLINICAL_USER_TEMPLATE",
    "HUMAN_REVIEW_NOTICE",
    "BLOCKED_RESPONSE",
    "INSUFFICIENT_EVIDENCE",
    "build_clinical_prompt",
    "build_messages",
    "format_protocol_block",
]

PROMPT_VERSION = "1.0.0"

ASSISTANT_SYSTEM_PROMPT = """Você é o MedFlow AI, assistente clínico ASSISTIVO de uso interno e acadêmico do Hospital Sinapse.
Seu público é composto por médicos assistentes. Você apoia a decisão; você não decide.

LIMITES DE ATUAÇÃO (invioláveis):
1. Você NUNCA prescreve. Não informe dose, posologia, via, intervalo ou duração de medicamento.
2. Você NUNCA fecha diagnóstico definitivo nem dispensa a avaliação de um profissional.
3. Você NUNCA substitui protocolo institucional por conhecimento próprio: se a informação não estiver
   no contexto fornecido, declare explicitamente que não há evidência suficiente nas fontes.
4. Toda afirmação clínica deve citar a fonte institucional entre colchetes, no formato [DOC-ID §SEÇÃO].
5. Dados do paciente vêm pseudonimizados. Não tente identificar o paciente nem reproduzir identificadores.
6. Diante de sinal de alarme ou emergência, oriente o acionamento imediato da equipe médica.

FORMATO DA RESPOSTA (obrigatório, nesta ordem):
RESPOSTA: síntese objetiva em até 6 frases, com as citações [DOC-ID §SEÇÃO] no próprio texto.
CONTEXTO DO PACIENTE: o que dos dados do paciente foi efetivamente usado; escreva "não utilizado" se não houve.
PENDÊNCIAS E ALERTAS: exames pendentes, sinais de alarme ou lacunas relevantes; escreva "nenhuma" se não houver.
LIMITAÇÃO: uma frase indicando que a conduta final é do médico assistente.
"""

CLINICAL_USER_TEMPLATE = """<<PERGUNTA>>
{question}
<</PERGUNTA>>

<<CONTEXTO_PACIENTE>>
{patient_context}
<</CONTEXTO_PACIENTE>>

<<PROTOCOLOS>>
{protocol_context}
<</PROTOCOLOS>>

<<INSTRUCOES>>
Responda apenas com base nos blocos acima. Cite a fonte de cada afirmação clínica no formato [DOC-ID §SEÇÃO].
Se os protocolos recuperados não responderem à pergunta, diga isso claramente em vez de completar com conhecimento próprio.
Classificação de segurança desta solicitação: {safety_status}.
<</INSTRUCOES>>"""

HUMAN_REVIEW_NOTICE = (
    "⚠️ RASCUNHO — PENDENTE DE VALIDAÇÃO HUMANA\n"
    "Esta solicitação foi classificada como {safety_status} pela política de segurança "
    "(regras acionadas: {rules}). O conteúdo abaixo é material de apoio e NÃO constitui "
    "prescrição, diagnóstico ou autorização. A conduta depende de revisão e assinatura "
    "do médico assistente responsável.\n"
)

BLOCKED_RESPONSE = (
    "🚫 SOLICITAÇÃO BLOQUEADA PELA POLÍTICA DE SEGURANÇA CLÍNICA\n"
    "Motivo: {reason}\n"
    "O MedFlow AI não emite prescrição autônoma, não dispensa avaliação médica, não altera "
    "registros clínicos e não opera fora das instruções institucionais.\n"
    "Encaminhe a solicitação ao médico assistente responsável ou reformule a pergunta como "
    "consulta a protocolo (por exemplo: \"qual protocolo institucional trata deste tema?\")."
)

INSUFFICIENT_EVIDENCE = (
    "RESPOSTA: Não há evidência suficiente nos protocolos institucionais recuperados para "
    "responder com segurança a esta pergunta.\n"
    "CONTEXTO DO PACIENTE: {patient_used}\n"
    "PENDÊNCIAS E ALERTAS: recomenda-se consultar o Núcleo de Protocolos Clínicos ou "
    "reformular a pergunta com termos usados nos documentos institucionais.\n"
    "LIMITAÇÃO: a conduta final é do médico assistente."
)


def format_protocol_block(chunks: Sequence[Any], *, max_chars: int = 900) -> str:
    """Formata os chunks recuperados com rótulo de citação estável."""
    if not chunks:
        return "Nenhum trecho de protocolo foi recuperado para esta pergunta."
    blocks: list[str] = []
    for chunk in chunks:
        metadata = chunk.document.metadata
        label = f"[{metadata.get('doc_id', '?')} §{metadata.get('section_number', '?')} {metadata.get('section_title', '')}]"
        text = chunk.document.page_content.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " (...)"
        blocks.append(f"{label}\n{text}")
    return "\n\n".join(blocks)


def build_clinical_prompt() -> ChatPromptTemplate:
    """``ChatPromptTemplate`` do fluxo clínico principal."""
    return ChatPromptTemplate.from_messages(
        [("system", ASSISTANT_SYSTEM_PROMPT), ("human", CLINICAL_USER_TEMPLATE)]
    )


def build_messages(
    *,
    question: str,
    patient_context: str,
    protocol_context: str,
    safety_status: str,
) -> list[BaseMessage]:
    """Renderiza o prompt clínico em mensagens prontas para o modelo."""
    return build_clinical_prompt().format_messages(
        question=question,
        patient_context=patient_context or "Nenhum contexto de paciente foi solicitado.",
        protocol_context=protocol_context,
        safety_status=safety_status,
    )
