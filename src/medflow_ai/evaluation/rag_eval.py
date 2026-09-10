"""Avaliação quantitativa do pipeline de RAG.

Pergunta de pesquisa central: *dado um conjunto de perguntas clínicas com
documento e seção de referência conhecidos, qual configuração de recuperação
entrega o trecho certo com mais frequência e em posição mais alta?*

Métricas
--------
``hit@k``     proporção de perguntas em que a seção-ouro aparece entre os k primeiros.
``doc_hit@k`` idem, exigindo apenas o documento correto (critério mais frouxo).
``mrr``       Mean Reciprocal Rank da seção-ouro (penaliza acerto em posição baixa).
``recall@k``  igual a ``hit@k`` neste benchmark, pois há exatamente uma seção-ouro
              por pergunta — mantido explícito para leitura direta do relatório.
``latencia_media_ms`` custo médio por consulta.

**Limitação declarada:** o benchmark é construído sobre o mesmo corpus indexado
pelo RAG. Portanto ele mede *qualidade de recuperação*, não generalização para
documentos inéditos. As perguntas foram escritas com vocabulário diferente do
texto-fonte para reduzir casamento lexical trivial, mas isso não elimina a
limitação.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from medflow_ai.config import get_settings
from medflow_ai.rag.chunking import chunk_documents
from medflow_ai.rag.embeddings import get_embeddings
from medflow_ai.rag.loaders import load_protocol_documents
from medflow_ai.rag.retriever import ProtocolRetriever, RetrievalStrategy
from medflow_ai.rag.vector_store import MedFlowVectorStore

__all__ = [
    "BenchmarkItem",
    "RetrievalMetrics",
    "load_benchmark",
    "evaluate_retriever",
    "run_experiment_grid",
    "DEFAULT_BENCHMARK_PATH",
]

DEFAULT_BENCHMARK_PATH = Path("data/benchmarks/rag_benchmark.jsonl")


@dataclass(frozen=True)
class BenchmarkItem:
    """Uma pergunta com referência conhecida."""

    id: str
    question: str
    gold_doc_id: str
    gold_section_id: str
    categoria: str = "geral"


@dataclass
class RetrievalMetrics:
    """Métricas agregadas de uma configuração de recuperação."""

    strategy: str
    k: int
    chunk_size: int
    chunk_overlap: int
    embedding: str
    n_questions: int
    hit_at_k: float
    doc_hit_at_k: float
    mrr: float
    latencia_media_ms: float
    falhas: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_benchmark(path: Path | str | None = None) -> list[BenchmarkItem]:
    """Carrega o benchmark JSONL de perguntas com gabarito."""
    settings = get_settings()
    benchmark_path = Path(path) if path else (settings.project_root / DEFAULT_BENCHMARK_PATH)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark de RAG não encontrado em {benchmark_path}")
    items: list[BenchmarkItem] = []
    with benchmark_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            items.append(
                BenchmarkItem(
                    id=payload["id"],
                    question=payload["question"],
                    gold_doc_id=payload["gold_doc_id"],
                    gold_section_id=payload["gold_section_id"],
                    categoria=payload.get("categoria", "geral"),
                )
            )
    return items


def evaluate_retriever(
    retriever: ProtocolRetriever,
    benchmark: Sequence[BenchmarkItem],
    *,
    k: int | None = None,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
    embedding_name: str = "",
) -> RetrievalMetrics:
    """Executa o benchmark em uma configuração e devolve as métricas."""
    top_k = k or retriever.k
    hits = 0
    doc_hits = 0
    reciprocal_ranks = 0.0
    latencies: list[float] = []
    failures: list[str] = []

    for item in benchmark:
        started = time.perf_counter()
        chunks = retriever.retrieve(item.question, k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        section_ids = [chunk.section_id for chunk in chunks]
        doc_ids = [chunk.doc_id for chunk in chunks]

        if item.gold_section_id in section_ids:
            hits += 1
            reciprocal_ranks += 1.0 / (section_ids.index(item.gold_section_id) + 1)
        else:
            failures.append(item.id)
        if item.gold_doc_id in doc_ids:
            doc_hits += 1

    total = len(benchmark) or 1
    return RetrievalMetrics(
        strategy=str(retriever.strategy),
        k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding=embedding_name,
        n_questions=len(benchmark),
        hit_at_k=round(hits / total, 4),
        doc_hit_at_k=round(doc_hits / total, 4),
        mrr=round(reciprocal_ranks / total, 4),
        latencia_media_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        falhas=failures,
    )


def run_experiment_grid(
    *,
    strategies: Iterable[RetrievalStrategy] = ("dense", "mmr", "bm25", "hybrid"),
    ks: Iterable[int] = (1, 3, 5),
    chunk_sizes: Iterable[int] = (400, 800, 1200),
    chunk_overlap: int = 120,
    benchmark: Sequence[BenchmarkItem] | None = None,
    embedding_backend: str | None = None,
) -> list[RetrievalMetrics]:
    """Roda a grade de experimentos (estratégia × k × tamanho de chunk).

    Reconstrói o índice uma vez por tamanho de chunk (a parte cara) e reaproveita
    o mesmo índice para todas as estratégias e valores de ``k``.
    """
    items = list(benchmark or load_benchmark())
    embeddings = get_embeddings(embedding_backend)
    embedding_name = str(getattr(embeddings, "name", type(embeddings).__name__))
    sections = load_protocol_documents()

    results: list[RetrievalMetrics] = []
    for chunk_size in chunk_sizes:
        overlap = min(chunk_overlap, max(chunk_size // 4, 0))
        chunks = chunk_documents(sections, chunk_size=chunk_size, chunk_overlap=overlap)
        store = MedFlowVectorStore.from_documents(chunks, embeddings)
        for strategy in strategies:
            for k in ks:
                retriever = ProtocolRetriever(
                    store, strategy=strategy, k=k, fetch_k=max(k * 4, 12)
                )
                results.append(
                    evaluate_retriever(
                        retriever,
                        items,
                        k=k,
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                        embedding_name=embedding_name,
                    )
                )
    return results


def save_results(results: Sequence[RetrievalMetrics], output_dir: Path | None = None) -> dict[str, Path]:
    """Persiste os resultados em JSON e CSV dentro de ``artifacts/``."""
    import csv

    target = Path(output_dir or (get_settings().artifacts_dir / "rag"))
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "rag_experiments.json"
    json_path.write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = target / "rag_experiments.csv"
    fields = ["strategy", "k", "chunk_size", "chunk_overlap", "embedding", "n_questions",
              "hit_at_k", "doc_hit_at_k", "mrr", "latencia_media_ms"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            writer.writerow({field: row[field] for field in fields})

    return {"json": json_path, "csv": csv_path}
