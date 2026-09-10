"""Testes do pipeline de RAG: chunking, embeddings, vector store e retrievers."""

from __future__ import annotations

import pytest

from medflow_ai.rag.chunking import chunk_documents
from medflow_ai.rag.embeddings import HashingEmbeddings, get_embeddings, normalize_text
from medflow_ai.rag.loaders import load_pdf_documents, load_protocol_documents
from medflow_ai.rag.retriever import BM25Index, ProtocolRetriever
from medflow_ai.rag.vector_store import MedFlowVectorStore


# ------------------------------------------------------------------ embeddings
def test_embedding_e_deterministico() -> None:
    embeddings = HashingEmbeddings(dim=256)
    texto = "Repetir o TSH 6 a 8 semanas após ajuste de dose."
    assert embeddings.embed_query(texto) == embeddings.embed_query(texto)


def test_embedding_tem_dimensao_configurada_e_norma_unitaria() -> None:
    embeddings = HashingEmbeddings(dim=128)
    vetor = embeddings.embed_query("levotiroxina e carbonato de cálcio")
    assert len(vetor) == 128
    assert pytest.approx(sum(valor**2 for valor in vetor), abs=1e-4) == 1.0


def test_textos_relacionados_ficam_mais_proximos_que_textos_distintos() -> None:
    embeddings = HashingEmbeddings(dim=512)
    consulta = embeddings.embed_query("quando repetir o TSH após iniciar levotiroxina")
    relacionado, distante = embeddings.embed_documents(
        [
            "O controle do TSH deve ser feito 6 a 8 semanas após iniciar levotiroxina.",
            "A radiografia de tórax não é rotina na crise asmática.",
        ]
    )
    similar = sum(a * b for a, b in zip(consulta, relacionado))
    diferente = sum(a * b for a, b in zip(consulta, distante))
    assert similar > diferente


def test_normalizacao_remove_acentos_e_caixa() -> None:
    assert normalize_text("Injúria Renal Aguda") == "injuria renal aguda"


def test_backend_desconhecido_falha_explicitamente() -> None:
    with pytest.raises(ValueError, match="Backend de embedding desconhecido"):
        get_embeddings("inexistente")


# -------------------------------------------------------------------- loaders
def test_loader_gera_um_documento_por_secao_com_citacao() -> None:
    documentos = load_protocol_documents()
    assert len(documentos) > 100
    for documento in documentos[:20]:
        assert documento.metadata["citation"]
        assert documento.metadata["doc_id"]
        assert documento.metadata["section_id"]


def test_loader_de_pdf_e_opcional(tmp_path) -> None:
    assert load_pdf_documents(tmp_path / "inexistente") == []


# ------------------------------------------------------------------- chunking
def test_chunking_preserva_metadados_e_cria_chunk_id() -> None:
    chunks = chunk_documents(load_protocol_documents(), chunk_size=300, chunk_overlap=60)
    assert chunks
    for chunk in chunks[:20]:
        assert chunk.metadata["chunk_id"].startswith(chunk.metadata["section_id"])
        assert chunk.metadata["doc_id"]
        assert chunk.metadata["chunk_index"] >= 0


def test_chunk_menor_gera_mais_chunks() -> None:
    secoes = load_protocol_documents()
    assert len(chunk_documents(secoes, chunk_size=250, chunk_overlap=50)) > len(
        chunk_documents(secoes, chunk_size=1500, chunk_overlap=50)
    )


def test_overlap_maior_que_chunk_e_rejeitado() -> None:
    with pytest.raises(ValueError):
        chunk_documents(load_protocol_documents(), chunk_size=200, chunk_overlap=200)


# --------------------------------------------------------------- vector store
def test_vector_store_recupera_o_protocolo_correto(vector_store) -> None:
    resultados = vector_store.similarity_search_with_score(
        "intervalo entre levotiroxina e carbonato de cálcio", k=5
    )
    assert resultados
    doc_ids = {documento.metadata["doc_id"] for documento, _ in resultados}
    assert {"PROT-END-001", "FAQ-MED-001"} & doc_ids


def test_vector_store_persiste_e_recarrega(vector_store, tmp_path) -> None:
    destino = vector_store.save(tmp_path / "indice")
    recarregado = MedFlowVectorStore.load(destino, vector_store.embeddings)
    assert len(recarregado) == len(vector_store)
    consulta = "quais exames pedir na avaliação inicial do hipertenso"
    originais = [documento.metadata["chunk_id"] for documento in vector_store.similarity_search(consulta, k=3)]
    recarregados = [documento.metadata["chunk_id"] for documento in recarregado.similarity_search(consulta, k=3)]
    assert originais == recarregados


def test_carregar_indice_inexistente_gera_erro_claro(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build-index"):
        MedFlowVectorStore.load(tmp_path / "vazio")


def test_mmr_nao_repete_o_mesmo_chunk(vector_store) -> None:
    resultados = vector_store.max_marginal_relevance_search("sepse primeira hora", k=4, fetch_k=15)
    chunk_ids = [documento.metadata["chunk_id"] for documento in resultados]
    assert len(chunk_ids) == len(set(chunk_ids))


# ------------------------------------------------------------------ retriever
@pytest.mark.parametrize("estrategia", ["dense", "mmr", "bm25", "hybrid"])
def test_todas_as_estrategias_retornam_resultados(vector_store, estrategia: str) -> None:
    recuperador = ProtocolRetriever(vector_store, strategy=estrategia, k=4)
    chunks = recuperador.retrieve("qual o pacote da primeira hora na sepse?")
    assert 0 < len(chunks) <= 4
    assert all(chunk.citation for chunk in chunks)


def test_estrategia_invalida_falha(vector_store) -> None:
    recuperador = ProtocolRetriever(vector_store, strategy="inexistente", k=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Estratégia de recuperação desconhecida"):
        recuperador.retrieve("teste")


def test_fontes_sao_desduplicadas_por_secao(retriever) -> None:
    chunks = retriever.retrieve("valores críticos comunicados pelo laboratório")
    fontes = retriever.sources(chunks)
    section_ids = [fonte["section_id"] for fonte in fontes]
    assert len(section_ids) == len(set(section_ids))
    for fonte in fontes:
        assert fonte["doc_id"] and fonte["citation"] and "score" in fonte


def test_bm25_prioriza_termos_raros(vector_store) -> None:
    indice = BM25Index(vector_store.documents)
    resultados = indice.search("coma mixedematoso", k=3)
    assert resultados
    assert any("PROT-END-001" == documento.metadata["doc_id"] for documento, _ in resultados)


def test_consulta_sem_correspondencia_nao_quebra(vector_store) -> None:
    recuperador = ProtocolRetriever(vector_store, strategy="bm25", k=3)
    assert recuperador.retrieve("zzzzqqqq termo inexistente xyzw") == []
