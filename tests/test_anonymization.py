"""Testes da anonimização e pseudonimização (requisito obrigatório do desafio)."""

from __future__ import annotations

from datetime import date

import pytest

from medflow_ai.data.anonymization import (
    age_band,
    anonymize_record,
    anonymize_text,
    birthdate_to_age,
    contains_pii,
    pseudonymize,
)

TEXTO_COM_PII = (
    "Paciente: Maria da Silva Souza, CPF 123.456.789-00, CNS 700 5049 3417 8563, "
    "e-mail maria.souza@exemplo.com.br, telefone (11) 98765-4321, "
    "residente na Rua das Acácias, nº 120, CEP 01310-100. "
    "Data de nascimento: 12/03/1975. Prontuário nº 4457821. "
    "Atendida pelo Dr. Carlos Andrade, CRM/SP 123456."
)


@pytest.mark.parametrize(
    "fragmento",
    [
        "Maria da Silva Souza",
        "123.456.789-00",
        "700 5049 3417 8563",
        "maria.souza@exemplo.com.br",
        "(11) 98765-4321",
        "01310-100",
        "12/03/1975",
        "Carlos Andrade",
    ],
)
def test_identificadores_diretos_sao_removidos(fragmento: str) -> None:
    anonimizado, _ = anonymize_text(TEXTO_COM_PII)
    assert fragmento not in anonimizado


def test_relatorio_registra_tipos_encontrados() -> None:
    _, relatorio = anonymize_text(TEXTO_COM_PII)
    tipos = relatorio.counts_by_kind
    for esperado in ("cpf", "cns", "email", "telefone", "cep", "nome", "data_nascimento"):
        assert tipos.get(esperado, 0) >= 1


def test_conteudo_clinico_e_preservado() -> None:
    texto = "TSH 8,4 mUI/L com T4 livre 0,72 ng/dL sugere hipotireoidismo primário."
    anonimizado, relatorio = anonymize_text(texto)
    assert anonimizado == texto
    assert relatorio.total == 0


def test_nao_produz_falso_positivo_em_termo_clinico() -> None:
    assert not contains_pii("Nome do exame: Perfil Tireoidiano Completo.")
    assert not contains_pii("O protocolo recomenda dosagem de TSH e T4 livre.")


def test_pseudonimo_e_estavel_e_nao_reverte_o_valor() -> None:
    valor = "P-DEMO-0001"
    assert pseudonymize(valor) == pseudonymize(valor)
    assert valor not in pseudonymize(valor)
    assert pseudonymize(valor, salt="a") != pseudonymize(valor, salt="b")


def test_registro_estruturado_remove_transforma_e_pseudonimiza() -> None:
    registro = {
        "patient_id": "P000123",
        "name": "João Pereira",
        "cpf": "11122233344",
        "birthdate": "1958-07-09",
        "gender": "M",
        "nota": "Contato: joao@exemplo.com",
    }
    anonimizado, relatorio = anonymize_record(registro, reference_date=date(2026, 3, 1))

    assert "name" not in anonimizado and "cpf" not in anonimizado
    assert anonimizado["age"] == 67
    assert anonimizado["age_band"] == "60-69"
    assert anonimizado["patient_id"] != "P000123"
    assert anonimizado["gender"] == "M"
    assert "joao@exemplo.com" not in anonimizado["nota"]
    assert set(relatorio.fields_removed) == {"name", "cpf"}
    assert set(relatorio.fields_transformed) == {"birthdate", "patient_id"}


@pytest.mark.parametrize(
    ("idade", "faixa"), [(0, "0-9"), (34, "30-39"), (59, "50-59"), (91, "90+"), (None, "desconhecida")]
)
def test_faixa_etaria(idade: int | None, faixa: str) -> None:
    assert age_band(idade) == faixa


def test_conversao_de_data_de_nascimento_aceita_formatos() -> None:
    referencia = date(2026, 3, 1)
    assert birthdate_to_age("1990-03-02", reference=referencia) == 35
    assert birthdate_to_age("02/03/1990", reference=referencia) == 35
    assert birthdate_to_age("texto inválido") is None


def test_datas_soltas_sao_removidas_quando_solicitado() -> None:
    texto = "Consulta realizada em 04/02/2026."
    assert "04/02/2026" in anonymize_text(texto)[0]
    assert "04/02/2026" not in anonymize_text(texto, redact_dates=True)[0]
