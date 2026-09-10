"""Testes da trilha de auditoria e da redação de dados sensíveis."""

from __future__ import annotations

import json

from medflow_ai.logging_utils.audit import (
    AuditLogger,
    build_audit_event,
    pseudonymize,
    redact,
    serialize_event,
)


def test_pseudonimizacao_e_estavel_e_nao_expoe_identificador() -> None:
    identificador = "patient-001"
    assert pseudonymize(identificador) == pseudonymize(identificador)
    assert identificador not in pseudonymize(identificador)


def test_evento_guarda_apenas_hash_do_paciente() -> None:
    evento = build_audit_event(
        trace_id="trace-1",
        route="clinical_question",
        patient_id="patient-001",
        sources=["protocol-a"],
        safety_status="CAUTION",
    )
    assert evento["patient_id_hash"] != "patient-001"
    assert evento["sources"] == ["protocol-a"]
    assert evento["safety_status"] == "CAUTION"


def test_redacao_remove_chaves_sensiveis_recursivamente() -> None:
    payload = {
        "api_key": "sk-123",
        "paciente": {"nome": "Maria Silva", "cpf": "123.456.789-00", "idade": 57},
        "itens": [{"email": "a@b.com"}, "texto simples"],
    }
    limpo = redact(payload)
    assert limpo["api_key"] == "[REDACTED]"
    assert limpo["paciente"]["nome"] == "[REDACTED]"
    assert limpo["paciente"]["cpf"] == "[REDACTED]"
    assert limpo["paciente"]["idade"] == 57
    assert limpo["itens"][0]["email"] == "[REDACTED]"


def test_texto_livre_e_anonimizado_antes_de_gravar(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    gravado = logger.log(
        build_audit_event(
            trace_id="t1",
            route="clinical_question",
            patient_id="P-DEMO-0001",
            question="Paciente: Maria da Silva, CPF 123.456.789-00, tem TSH alto?",
            openai_api_key="sk-secreta",
        )
    )
    assert "Maria da Silva" not in gravado["question"]
    assert "123.456.789-00" not in gravado["question"]
    assert gravado["openai_api_key"] == "[REDACTED]"

    conteudo = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "sk-secreta" not in conteudo
    assert "123.456.789-00" not in conteudo


def test_arquivo_e_jsonl_valido_e_acumula_eventos(tmp_path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    for indice in range(3):
        logger.log(build_audit_event(trace_id=f"t{indice}", route="rota"))

    linhas = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 3
    assert all(json.loads(linha)["trace_id"].startswith("t") for linha in linhas)
    assert len(logger.read_events(limit=2)) == 2


def test_serializacao_e_estavel() -> None:
    evento = {"b": 1, "a": 2}
    assert serialize_event(evento) == '{"a": 2, "b": 1}'


def test_leitura_sem_arquivo_retorna_vazio(tmp_path) -> None:
    assert AuditLogger(tmp_path / "inexistente.jsonl").read_events() == []
