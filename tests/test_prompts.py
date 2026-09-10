"""Testes dos prompts versionados e do provedor extrativo."""

from __future__ import annotations

from langchain_core.documents import Document

from medflow_ai.llm.prompts import (
    ASSISTANT_SYSTEM_PROMPT,
    BLOCKED_RESPONSE,
    PROMPT_VERSION,
    build_clinical_prompt,
    build_messages,
    format_protocol_block,
)
from medflow_ai.llm.providers import TemplateChatModel, describe_provider, get_chat_model
from medflow_ai.rag.retriever import ScoredChunk


def _chunk(doc_id: str, numero: str, titulo: str, texto: str) -> ScoredChunk:
    return ScoredChunk(
        document=Document(
            page_content=texto,
            metadata={
                "doc_id": doc_id,
                "section_number": numero,
                "section_title": titulo,
                "section_id": f"{doc_id}#{numero}",
                "citation": f"{doc_id} §{numero} {titulo}",
            },
        ),
        score=0.9,
        rank=1,
    )


def test_prompt_de_sistema_declara_os_limites() -> None:
    for exigencia in ("NUNCA prescreve", "DOC-ID", "não decide", "evidência suficiente"):
        assert exigencia.casefold() in ASSISTANT_SYSTEM_PROMPT.casefold()


def test_prompt_tem_versao_semantica() -> None:
    assert len(PROMPT_VERSION.split(".")) == 3


def test_template_declara_as_variaveis_esperadas() -> None:
    variaveis = set(build_clinical_prompt().input_variables)
    assert variaveis == {"question", "patient_context", "protocol_context", "safety_status"}


def test_mensagens_incluem_delimitadores_de_bloco() -> None:
    mensagens = build_messages(
        question="Quando repetir o TSH?",
        patient_context="Faixa etária: 50-59 anos",
        protocol_context="[PROT-END-001 §7 Monitoramento]\nReavaliar em 6 a 8 semanas.",
        safety_status="CAUTION",
    )
    corpo = mensagens[-1].content
    for delimitador in ("<<PERGUNTA>>", "<<CONTEXTO_PACIENTE>>", "<<PROTOCOLOS>>", "<<INSTRUCOES>>"):
        assert delimitador in corpo
    assert "CAUTION" in corpo


def test_bloco_de_protocolo_rotula_a_fonte() -> None:
    bloco = format_protocol_block(
        [_chunk("PROT-END-001", "7", "Monitoramento", "Reavaliar o TSH em 6 a 8 semanas.")]
    )
    assert "[PROT-END-001 §7 Monitoramento]" in bloco
    assert "Reavaliar o TSH" in bloco


def test_bloco_vazio_declara_ausencia_de_fonte() -> None:
    assert "Nenhum trecho" in format_protocol_block([])


def test_resposta_bloqueada_explica_o_motivo() -> None:
    texto = BLOCKED_RESPONSE.format(reason="Tentativa de burlar validação médica.")
    assert "BLOQUEADA" in texto and "validação médica" in texto


# ----------------------------------------------------------------- provedores
def test_provedor_extrativo_cita_a_fonte_selecionada() -> None:
    mensagens = build_messages(
        question="Em quanto tempo repetir o TSH após ajuste de dose?",
        patient_context="",
        protocol_context=format_protocol_block(
            [
                _chunk("PROT-END-001", "7", "Monitoramento",
                       "Reavaliar o TSH 6 a 8 semanas após o início do tratamento ou após "
                       "qualquer ajuste de dose."),
                _chunk("PROT-CAR-001", "2", "Triagem",
                       "O eletrocardiograma deve ser interpretado em até 10 minutos."),
            ]
        ),
        safety_status="SAFE",
    )
    resposta = TemplateChatModel().invoke(mensagens).content
    assert "[PROT-END-001 §7 Monitoramento]" in resposta
    assert "6 a 8 semanas" in resposta
    for secao in ("RESPOSTA:", "CONTEXTO DO PACIENTE:", "PENDÊNCIAS E ALERTAS:", "LIMITAÇÃO:"):
        assert secao in resposta


def test_provedor_extrativo_admite_falta_de_evidencia() -> None:
    mensagens = build_messages(
        question="Qual a política de estacionamento do hospital?",
        patient_context="",
        protocol_context=format_protocol_block([]),
        safety_status="SAFE",
    )
    resposta = TemplateChatModel().invoke(mensagens).content
    assert "Não há evidência suficiente" in resposta


def test_provedor_e_deterministico() -> None:
    mensagens = build_messages(
        question="Quando repetir o TSH?",
        patient_context="",
        protocol_context=format_protocol_block(
            [_chunk("PROT-END-001", "7", "Monitoramento", "Reavaliar o TSH em 6 a 8 semanas.")]
        ),
        safety_status="SAFE",
    )
    modelo = TemplateChatModel()
    assert modelo.invoke(mensagens).content == modelo.invoke(mensagens).content


def test_fabrica_de_provedores() -> None:
    assert describe_provider(get_chat_model("template")) == "template"
    try:
        get_chat_model("inexistente")
    except ValueError as erro:
        assert "Provedor de LLM desconhecido" in str(erro)
    else:  # pragma: no cover
        raise AssertionError("provedor inválido deveria falhar")
