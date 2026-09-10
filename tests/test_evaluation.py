"""Testes dos módulos de avaliação (RAG, segurança, prontuário, grafo, geração)."""

from __future__ import annotations

import pytest

from medflow_ai.evaluation import database_eval, graph_eval, rag_eval, safety_eval
from medflow_ai.fine_tuning.evaluate import evaluate_generation, load_test_examples


# ------------------------------------------------------------------- RAG
def test_benchmark_de_rag_tem_gabarito_valido(corpus) -> None:
    secoes = {secao.section_id for documento in corpus for secao in documento.sections}
    itens = rag_eval.load_benchmark()
    assert len(itens) >= 30
    for item in itens:
        assert item.gold_section_id in secoes, f"{item.id} aponta para seção inexistente"
        assert item.gold_section_id.startswith(item.gold_doc_id)


def test_avaliacao_de_retrieval_produz_metricas_coerentes(retriever) -> None:
    metricas = rag_eval.evaluate_retriever(retriever, rag_eval.load_benchmark(), k=5)
    assert metricas.n_questions >= 30
    assert 0.0 <= metricas.hit_at_k <= 1.0
    assert metricas.doc_hit_at_k >= metricas.hit_at_k  # documento é critério mais frouxo
    assert 0.0 <= metricas.mrr <= 1.0
    assert metricas.latencia_media_ms >= 0


def test_recuperacao_supera_baseline_aleatorio(retriever) -> None:
    """Sanidade: o retriever precisa ser muito melhor que sorteio entre ~150 chunks."""
    metricas = rag_eval.evaluate_retriever(retriever, rag_eval.load_benchmark(), k=5)
    assert metricas.hit_at_k > 0.5
    assert metricas.doc_hit_at_k > 0.8


def test_grade_de_experimentos_cobre_as_combinacoes() -> None:
    resultados = rag_eval.run_experiment_grid(
        strategies=("dense", "bm25"), ks=(3,), chunk_sizes=(400,)
    )
    assert len(resultados) == 2
    assert {resultado.strategy for resultado in resultados} == {"dense", "bm25"}


def test_resultados_sao_persistidos(tmp_path) -> None:
    resultados = rag_eval.run_experiment_grid(strategies=("dense",), ks=(3,), chunk_sizes=(400,))
    caminhos = rag_eval.save_results(resultados, tmp_path)
    assert caminhos["json"].exists() and caminhos["csv"].exists()
    assert "hit_at_k" in caminhos["csv"].read_text(encoding="utf-8")


# ------------------------------------------------------------- segurança
@pytest.mark.parametrize(
    "conjunto", ["safety_benchmark", "safety_holdout_v1", "safety_holdout_v2"]
)
def test_conjuntos_de_seguranca_existem_e_sao_avaliaveis(conjunto: str) -> None:
    from medflow_ai.config import get_settings

    caminho = get_settings().project_root / "data" / "benchmarks" / f"{conjunto}.jsonl"
    relatorio = safety_eval.evaluate_safety(safety_eval.load_safety_benchmark(caminho))
    assert relatorio.n_prompts >= 30
    assert 0.0 <= relatorio.accuracy <= 1.0


def test_politica_nunca_subestima_risco_nos_conjuntos_rotulados() -> None:
    """Critério de aceitação clínica: subestimar risco é inaceitável."""
    from medflow_ai.config import get_settings

    raiz = get_settings().project_root / "data" / "benchmarks"
    for conjunto in ("safety_benchmark", "safety_holdout_v1", "safety_holdout_v2"):
        relatorio = safety_eval.evaluate_safety(
            safety_eval.load_safety_benchmark(raiz / f"{conjunto}.jsonl")
        )
        assert relatorio.subestimacao == 0, f"{conjunto} subestimou risco"


def test_matriz_de_confusao_soma_o_total() -> None:
    relatorio = safety_eval.evaluate_safety()
    total = sum(
        quantidade
        for linha in relatorio.confusion.values()
        for quantidade in linha.values()
    )
    assert total == relatorio.n_prompts
    assert "esperado" in relatorio.render_confusion()


# ------------------------------------------------------------- prontuário
def test_avaliacao_do_prontuario_recupera_valores_exatos(repository) -> None:
    casos = database_eval.build_cases(limit=20)
    relatorio = database_eval.evaluate_database(repository, casos)
    assert relatorio.n_cases == len(casos)
    assert relatorio.exact_match == 1.0
    assert relatorio.context_leak_free


# ------------------------------------------------------------------ grafo
def test_avaliacao_do_grafo_roteia_corretamente(assistant) -> None:
    relatorio = graph_eval.evaluate_graph(assistant)
    assert relatorio.route_accuracy == 1.0
    assert relatorio.node_accuracy == 1.0
    assert relatorio.review_accuracy == 1.0
    assert relatorio.falhas == []


# ---------------------------------------------------------------- geração
def test_split_de_teste_vem_de_documentos_held_out() -> None:
    from medflow_ai.fine_tuning.dataset import HELD_OUT_DOCUMENTS

    exemplos = load_test_examples()
    assert exemplos
    assert {exemplo["doc_id"] for exemplo in exemplos} <= set(HELD_OUT_DOCUMENTS)


def test_metricas_de_geracao_distinguem_com_e_sem_rag(retriever) -> None:
    from medflow_ai.llm.prompts import build_messages, format_protocol_block
    from medflow_ai.llm.providers import get_chat_model

    modelo = get_chat_model("template")
    exemplos = load_test_examples()[:8]

    def com_rag(pergunta: str) -> str:
        mensagens = build_messages(
            question=pergunta,
            patient_context="",
            protocol_context=format_protocol_block(retriever.retrieve(pergunta)),
            safety_status="SAFE",
        )
        return str(modelo.invoke(mensagens).content)

    def sem_rag(pergunta: str) -> str:
        mensagens = build_messages(
            question=pergunta, patient_context="",
            protocol_context="Nenhum trecho de protocolo foi recuperado.", safety_status="SAFE",
        )
        return str(modelo.invoke(mensagens).content)

    metricas_com = evaluate_generation(com_rag, sistema="com_rag", examples=exemplos)
    metricas_sem = evaluate_generation(sem_rag, sistema="sem_rag", examples=exemplos)

    assert metricas_com.citacao_presente > metricas_sem.citacao_presente
    assert metricas_com.groundedness > metricas_sem.groundedness
    assert metricas_com.formato == 1.0 and metricas_sem.formato == 1.0
