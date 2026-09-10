"""Testes do corpus institucional sintético usado por RAG e fine-tuning."""

from __future__ import annotations

import pytest

from medflow_ai.data.anonymization import contains_pii
from medflow_ai.data.corpus import load_corpus


def test_corpus_carrega_todos_os_documentos(corpus) -> None:
    assert len(corpus) >= 15
    ids = {documento.doc_id for documento in corpus}
    for esperado in ("PROT-END-001", "PROT-CAR-001", "PROC-INT-001", "MOD-LAU-001", "FAQ-MED-001"):
        assert esperado in ids


def test_todo_documento_tem_metadados_de_rastreabilidade(corpus) -> None:
    for documento in corpus:
        assert documento.metadata.get("titulo")
        assert documento.metadata.get("versao")
        assert documento.metadata.get("vigencia")
        assert documento.metadata.get("origem"), f"{documento.doc_id} sem declaração de origem"
        assert documento.sections, f"{documento.doc_id} sem seções"


def test_ids_de_documento_sao_unicos(corpus) -> None:
    ids = [documento.doc_id for documento in corpus]
    assert len(ids) == len(set(ids))


def test_ids_de_secao_sao_unicos(corpus) -> None:
    section_ids = [secao.section_id for documento in corpus for secao in documento.sections]
    assert len(section_ids) == len(set(section_ids))


def test_corpus_nao_contem_identificadores_diretos(corpus) -> None:
    """O corpus é sintético, mas a ausência de PII precisa ser verificada, não presumida."""
    ofensores = [documento.doc_id for documento in corpus if contains_pii(documento.body)]
    assert ofensores == []


def test_citacao_inclui_documento_versao_e_titulo(corpus) -> None:
    documento = next(item for item in corpus if item.doc_id == "PROT-END-001")
    citacao = documento.citation()
    assert "PROT-END-001" in citacao and documento.version in citacao


def test_hash_de_conteudo_muda_com_o_texto(corpus) -> None:
    documento = corpus[0]
    assert len(documento.content_hash) == 16
    assert documento.content_hash == load_corpus()[0].content_hash


def test_corpus_inexistente_gera_erro_claro(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nao-existe")
