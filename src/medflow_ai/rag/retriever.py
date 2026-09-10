"""Estratégias de recuperação e rastreabilidade de fontes.

Três estratégias comparáveis, todas devolvendo o mesmo contrato
(``ScoredChunk``) para que a avaliação seja justa:

* ``dense``  — similaridade de cosseno no vector store;
* ``mmr``    — Maximum Marginal Relevance sobre o mesmo espaço vetorial;
* ``bm25``   — recuperação lexical clássica (Okapi BM25);
* ``hybrid`` — fusão de rankings (Reciprocal Rank Fusion) entre denso e BM25.

O objetivo de manter mais de uma estratégia é experimental: o relatório compara
as quatro no mesmo conjunto de perguntas com documento-ouro conhecido.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from langchain_core.documents import Document

from medflow_ai.config import get_settings
from medflow_ai.rag.embeddings import normalize_text
from medflow_ai.rag.vector_store import MedFlowVectorStore

__all__ = ["ScoredChunk", "BM25Index", "ProtocolRetriever", "RetrievalStrategy"]

RetrievalStrategy = Literal["dense", "mmr", "bm25", "hybrid"]

_TOKEN_SPLIT = str.maketrans({ch: " " for ch in ".,;:!?()[]{}\"'`/\\|<>—–-\n\t"})


def _tokenize(text: str) -> list[str]:
    return [tok for tok in normalize_text(text).translate(_TOKEN_SPLIT).split() if len(tok) > 1]


@dataclass(frozen=True)
class ScoredChunk:
    """Chunk recuperado com score e metadados de citação."""

    document: Document
    score: float
    rank: int

    @property
    def doc_id(self) -> str:
        return str(self.document.metadata.get("doc_id", "?"))

    @property
    def section_id(self) -> str:
        return str(self.document.metadata.get("section_id", self.doc_id))

    @property
    def citation(self) -> str:
        return str(self.document.metadata.get("citation", self.doc_id))

    def to_source(self) -> dict[str, Any]:
        """Representação enxuta usada em logs e na resposta ao médico."""
        metadata = self.document.metadata
        return {
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "titulo": metadata.get("titulo", ""),
            "secao": metadata.get("section_title", ""),
            "versao": metadata.get("versao", ""),
            "vigencia": metadata.get("vigencia", ""),
            "arquivo": metadata.get("source_path", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "score": round(self.score, 4),
            "rank": self.rank,
            "citation": self.citation,
        }


class BM25Index:
    """Índice Okapi BM25 sobre os mesmos chunks do vector store."""

    def __init__(self, documents: Sequence[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._tokenized = [_tokenize(doc.page_content) for doc in self.documents]
        self._lengths = [len(tokens) for tokens in self._tokenized]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokenized:
            document_frequency.update(set(tokens))
        total = len(self.documents) or 1
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self.documents:
            return []
        scores = []
        for index, frequencies in enumerate(self._term_frequencies):
            length = self._lengths[index] or 1
            score = 0.0
            for term in query_tokens:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_length or 1)
                )
                score += idf * (term_frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: -item[1])
        return [(self.documents[index], score) for index, score in scores[:k]]


class ProtocolRetriever:
    """Fachada de recuperação com estratégia configurável."""

    def __init__(
        self,
        store: MedFlowVectorStore,
        *,
        strategy: RetrievalStrategy | None = None,
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float | None = None,
        min_score: float = 0.0,
    ) -> None:
        settings = get_settings()
        self.store = store
        self.strategy: RetrievalStrategy = (strategy or settings.retriever_strategy)  # type: ignore[assignment]
        self.k = k or settings.retriever_k
        self.fetch_k = fetch_k or max(self.k * 4, 12)
        self.lambda_mult = settings.mmr_lambda if lambda_mult is None else lambda_mult
        self.min_score = min_score
        self._bm25: BM25Index | None = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.store.documents)
        return self._bm25

    def retrieve(self, query: str, *, k: int | None = None) -> list[ScoredChunk]:
        """Recupera chunks relevantes para a pergunta."""
        top_k = k or self.k
        if self.strategy == "dense":
            pairs = self.store.similarity_search_with_score(query, k=top_k)
        elif self.strategy == "mmr":
            pairs = self.store.max_marginal_relevance_search_with_score(
                query, k=top_k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult
            )
        elif self.strategy == "bm25":
            pairs = self.bm25.search(query, k=top_k)
        elif self.strategy == "hybrid":
            pairs = self._hybrid(query, top_k)
        else:
            raise ValueError(f"Estratégia de recuperação desconhecida: {self.strategy!r}")

        chunks = [
            ScoredChunk(document=document, score=score, rank=rank)
            for rank, (document, score) in enumerate(pairs, start=1)
            if score >= self.min_score
        ]
        return chunks

    def _hybrid(self, query: str, top_k: int, *, rrf_k: int = 60) -> list[tuple[Document, float]]:
        """Reciprocal Rank Fusion entre a busca densa e o BM25."""
        pool = max(self.fetch_k, top_k)
        dense = self.store.similarity_search_with_score(query, k=pool)
        lexical = self.bm25.search(query, k=pool)

        fused: dict[str, float] = defaultdict(float)
        registry: dict[str, Document] = {}
        for ranking in (dense, lexical):
            for rank, (document, _) in enumerate(ranking, start=1):
                key = str(document.metadata.get("chunk_id") or id(document))
                registry[key] = document
                fused[key] += 1.0 / (rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
        return [(registry[key], score) for key, score in ordered]

    def sources(self, chunks: Sequence[ScoredChunk]) -> list[dict[str, Any]]:
        """Lista de fontes desduplicada por seção, preservando a ordem de rank."""
        seen: set[str] = set()
        sources: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk.section_id in seen:
                continue
            seen.add(chunk.section_id)
            sources.append(chunk.to_source())
        return sources
