"""Testes da base estruturada de prontuários e da minimização de dados."""

from __future__ import annotations

import pytest

from medflow_ai.data.anonymization import contains_pii
from medflow_ai.database.ingest import build_synthetic_database, ingest_synthea_csv, table_counts
from medflow_ai.database.repository import PatientNotFoundError, PatientRepository
from medflow_ai.database.schema import TABLES, connect, create_schema
from medflow_ai.database.synthetic_patients import DEMO_PATIENT_ID, generate_dataset

DEMO_PATIENT = DEMO_PATIENT_ID


def test_esquema_cria_todas_as_tabelas(tmp_path) -> None:
    caminho = tmp_path / "teste.db"
    with connect(caminho) as conexao:
        create_schema(conexao)
        counts = table_counts(conexao)
    assert set(counts) == set(TABLES)


def test_geracao_e_deterministica_com_o_mesmo_seed() -> None:
    assert generate_dataset(n_patients=5, seed=7).as_tables() == generate_dataset(n_patients=5, seed=7).as_tables()


def test_seeds_diferentes_geram_dados_diferentes() -> None:
    assert generate_dataset(n_patients=5, seed=7).patients != generate_dataset(n_patients=5, seed=8).patients


def test_banco_sintetico_e_populado(database) -> None:
    with connect(database) as conexao:
        counts = table_counts(conexao)
    assert counts["patients"] >= 13
    assert counts["observations"] > 0
    assert counts["lab_orders"] > 0


# ------------------------------------------------------- recuperação exata
def test_paciente_demo_tem_valores_conhecidos(repository: PatientRepository) -> None:
    observacoes = {item["exame"]: item for item in repository.latest_observations(DEMO_PATIENT)}
    assert observacoes["TSH"]["valor"] == "8.40"
    assert observacoes["TSH"]["data"] == "2026-02-04"
    assert observacoes["T4 livre"]["valor"] == "0.72"


def test_historico_de_exame_vem_ordenado_do_mais_recente(repository: PatientRepository) -> None:
    historico = repository.observation_history(DEMO_PATIENT, "TSH")
    assert [item["valor"] for item in historico] == ["8.40", "6.10"]


def test_condicoes_e_medicamentos_ativos(repository: PatientRepository) -> None:
    condicoes = {item["descricao"] for item in repository.conditions(DEMO_PATIENT)}
    assert "Hipotireoidismo primário" in condicoes
    medicamentos = {item["descricao"] for item in repository.medications(DEMO_PATIENT)}
    assert any("Levotiroxina" in item for item in medicamentos)


def test_exames_pendentes_excluem_liberados(repository: PatientRepository) -> None:
    pendentes = {item["exame"] for item in repository.pending_exams(DEMO_PATIENT)}
    assert "TSH de controle" in pendentes
    assert "Perfil lipídico" not in pendentes  # já liberado


def test_paciente_inexistente_gera_erro(repository: PatientRepository) -> None:
    with pytest.raises(PatientNotFoundError):
        repository.demographics("NAO-EXISTE")


# ------------------------------------------------------- minimização de dados
def test_contexto_nao_expoe_identificadores_diretos(repository: PatientRepository) -> None:
    bruto = repository.raw_record(DEMO_PATIENT)
    contexto = repository.build_context(DEMO_PATIENT)
    texto = contexto.to_prompt_block()

    for campo in ("first", "last", "cpf", "cns", "email", "phone", "address"):
        valor = str(bruto[campo])
        assert valor not in texto, f"{campo} vazou para o contexto"
    assert not contains_pii(texto)
    assert bruto["birthdate"] not in texto
    assert contexto.age_band == "50-59"
    assert contexto.patient_pseudonym != DEMO_PATIENT


def test_contexto_respeita_o_subconjunto_solicitado(repository: PatientRepository) -> None:
    contexto = repository.build_context(DEMO_PATIENT, include=["pending_exams"])
    assert contexto.pending_exams
    assert contexto.conditions == []
    assert contexto.medications == []


def test_colunas_seguras_nao_incluem_quase_identificadores(repository: PatientRepository) -> None:
    seguras = set(repository.safe_columns())
    assert "birthdate" not in seguras and "cpf" not in seguras
    assert "gender" in seguras


def test_bloco_de_prompt_lista_pendencias(repository: PatientRepository) -> None:
    bloco = repository.build_context(DEMO_PATIENT).to_prompt_block()
    assert "Exames pendentes:" in bloco
    assert "TSH de controle" in bloco


# ------------------------------------------------------------- ingestão CSV
def test_ingestao_de_csv_no_formato_synthea(tmp_path) -> None:
    (tmp_path / "patients.csv").write_text(
        "Id,BIRTHDATE,GENDER,FIRST,LAST\nP900001,1980-01-01,F,Teste,Sobrenome\n", encoding="utf-8"
    )
    (tmp_path / "conditions.csv").write_text(
        "PATIENT,ENCOUNTER,START,STOP,CODE,DESCRIPTION\nP900001,,2024-01-01,,X,Asma\n", encoding="utf-8"
    )
    destino = tmp_path / "synthea.db"
    counts = ingest_synthea_csv(tmp_path, destino)
    assert counts["patients"] == 1 and counts["conditions"] == 1

    repositorio = PatientRepository(destino)
    assert repositorio.conditions("P900001")[0]["descricao"] == "Asma"


def test_reconstrucao_do_banco_e_idempotente(tmp_path) -> None:
    destino = tmp_path / "rebuild.db"
    primeiro = build_synthetic_database(destino, n_patients=6, seed=42)
    segundo = build_synthetic_database(destino, n_patients=6, seed=42)
    assert primeiro == segundo


def test_paciente_demo_sempre_existe(tmp_path) -> None:
    destino = tmp_path / "demo.db"
    build_synthetic_database(destino, n_patients=1, seed=99)
    assert PatientRepository(destino).exists(DEMO_PATIENT_ID)
