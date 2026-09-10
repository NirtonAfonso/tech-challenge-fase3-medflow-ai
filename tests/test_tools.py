"""Testes das ferramentas LangChain expostas pelo assistente."""

from __future__ import annotations

from medflow_ai.graph.tools import (
    MEDFLOW_TOOLS,
    buscar_protocolo,
    consultar_prontuario,
    get_repository,
    get_retriever,
    listar_exames_pendentes,
    verificar_alertas_clinicos,
)


def test_todas_as_ferramentas_tem_nome_e_descricao() -> None:
    assert len(MEDFLOW_TOOLS) == 4
    for ferramenta in MEDFLOW_TOOLS:
        assert ferramenta.name
        assert ferramenta.description and len(ferramenta.description) > 40
        assert ferramenta.args_schema is not None


def test_consultar_prontuario_devolve_contexto_minimizado(database, demo_patient) -> None:
    get_repository.cache_clear()
    resultado = consultar_prontuario.invoke({"patient_id": demo_patient})
    assert resultado["age_band"] == "50-59"
    assert "cpf" not in resultado and "name" not in resultado
    assert resultado["patient_pseudonym"] != demo_patient


def test_consultar_prontuario_aceita_subconjunto(database, demo_patient) -> None:
    get_repository.cache_clear()
    resultado = consultar_prontuario.invoke({"patient_id": demo_patient, "blocos": "pending_exams"})
    assert resultado["pending_exams"]
    assert resultado["conditions"] == []


def test_consultar_prontuario_com_paciente_inexistente(database) -> None:
    get_repository.cache_clear()
    assert "erro" in consultar_prontuario.invoke({"patient_id": "NAO-EXISTE"})


def test_buscar_protocolo_retorna_texto_e_fonte() -> None:
    get_retriever.cache_clear()
    resultados = buscar_protocolo.invoke({"pergunta": "pacote da primeira hora na sepse", "k": 3})
    assert 0 < len(resultados) <= 3
    primeiro = resultados[0]
    assert primeiro["texto"] and primeiro["doc_id"] and primeiro["citation"]
    assert "score" in primeiro


def test_listar_exames_pendentes(database, demo_patient) -> None:
    get_repository.cache_clear()
    pendentes = listar_exames_pendentes.invoke({"patient_id": demo_patient})
    assert {item["exame"] for item in pendentes} >= {"TSH de controle"}


def test_verificar_alertas_detecta_interacao(database, demo_patient) -> None:
    get_repository.cache_clear()
    alertas = verificar_alertas_clinicos.invoke({"patient_id": demo_patient})
    assert any(alerta["id"] == "ALERTA-INTERACAO-01" for alerta in alertas)
    for alerta in alertas:
        assert alerta["fonte"] and alerta["severidade"] and alerta["evidencia"]
