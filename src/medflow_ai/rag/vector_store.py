"""Vector store local, persistente e compatível com a interface do LangChain.

Optou-se por uma implementação própria em NumPy em vez de FAISS/Chroma por três
motivos, registrados em ``docs/DECISIONS.md``:

1. o corpus do projeto tem centenas (não milhões) de chunks — busca exata é
   instantânea e evita o índice aproximado;
2. elimina dependência binária pesada, permitindo que a avaliação de RAG rode
   em CI sem GPU e sem download;
3. a persistência é transparente (``.npy`` + ``.jsonl``), o que facilita
   auditoria e reprodutibilidade.

A classe implementa ``VectorStore`` do ``langchain_core``, portanto pode ser
usada com ``.as_retriever()`` e composta em chains normalmente.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from medflow_ai.config import get_settings
from medflow_ai.rag.embeddings import get_embeddings

__all__ = ["MedFlowVectorStore"]

_VECTORS_FILE = "vectors.npy"
_DOCS_FILE = "documents.jsonl"
_MANIFEST_FILE = "manifest.json"


class MedFlowVectorStore(VectorStore):
    """Vector store denso, exato e persistente em disco."""

    def __init__(self, embedding: Embeddings, *, documents: Sequence[Document] | None = None) -> None:
        self._embedding = embedding
        self._documents: list[Document] = list(documents or [])
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        if self._documents:
            self._vectors = self._encode([doc.page_content for doc in self._documents])

    # ------------------------------------------------------------------ infra
    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def documents(self) -> list[Document]:
        return list(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self._embedding.embed_documents(list(texts)), dtype=np.float32)
        return _l2_normalize(vectors)

    # ------------------------------------------------------- VectorStore API
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        text_list = list(texts)
        metadata_list = list(metadatas or [{} for _ in text_list])
        new_documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(text_list, metadata_list)
        ]
        return self.add_documents(new_documents)

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        if not documents:
            return []
        new_vectors = self._encode([doc.page_content for doc in documents])
        if self._vectors.size == 0:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])
        self._documents.extend(documents)
        return [doc.metadata.get("chunk_id", str(index)) for index, doc in enumerate(documents)]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "MedFlowVectorStore":
        metadata_list = metadatas or [{} for _ in texts]
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadata_list)
        ]
        return cls(embedding, documents=documents)

    @classmethod
    def from_documents(  # type: ignore[override]
        cls, documents: list[Document], embedding: Embeddings, **kwargs: Any
    ) -> "MedFlowVectorStore":
        return cls(embedding, documents=documents)

    # ------------------------------------------------------------- pesquisas
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Busca por similaridade de cosseno, retornando o score de cada chunk."""
        if not self._documents:
            return []
        query_vector = _l2_normalize(
            np.asarray([self._embedding.embed_query(query)], dtype=np.float32)
        )[0]
        scores = self._vectors @ query_vector
        top_k = min(k, len(self._documents))
        order = np.argsort(-scores)[:top_k]
        return [(self._documents[int(i)], float(scores[int(i)])) for i in order]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.6,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """MMR: equilibra relevância e diversidade entre os chunks recuperados.

        Evita que os ``k`` resultados sejam trechos quase idênticos do mesmo
        documento, problema comum quando protocolos repetem enunciados.
        """
        if not self._documents:
            return []
        query_vector = _l2_normalize(
            np.asarray([self._embedding.embed_query(query)], dtype=np.float32)
        )[0]
        scores = self._vectors @ query_vector
        candidate_count = min(max(fetch_k, k), len(self._documents))
        candidates = list(np.argsort(-scores)[:candidate_count])

        selected: list[int] = []
        while candidates and len(selected) < min(k, candidate_count):
            if not selected:
                best = candidates.pop(0)
                selected.append(int(best))
                continue
            selected_matrix = self._vectors[selected]
            best_index, best_value = None, -np.inf
            for candidate in candidates:
                redundancy = float(np.max(self._vectors[candidate] @ selected_matrix.T))
                value = lambda_mult * float(scores[candidate]) - (1 - lambda_mult) * redundancy
                if value > best_value:
                    best_index, best_value = candidate, value
            candidates.remove(best_index)  # type: ignore[arg-type]
            selected.append(int(best_index))  # type: ignore[arg-type]

        return [(self._documents[i], float(scores[i])) for i in selected]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.6, **kwargs: Any
    ) -> list[Document]:
        return [
            doc
            for doc, _ in self.max_marginal_relevance_search_with_score(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
            )
        ]

    # ----------------------------------------------------------- persistência
    def save(self, directory: Path | None = None, *, extra: dict[str, Any] | None = None) -> Path:
        """Persiste vetores, documentos e manifesto reprodutível."""
        target = Path(directory or get_settings().vector_store_dir)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / _VECTORS_FILE, self._vectors)
        with (target / _DOCS_FILE).open("w", encoding="utf-8") as handle:
            for document in self._documents:
                handle.write(
                    json.dumps(
                        {"page_content": document.page_content, "metadata": document.metadata},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        manifest = {
            "documents": len(self._documents),
            "dimension": int(self._vectors.shape[1]) if self._vectors.size else 0,
            "embedding": getattr(self._embedding, "name", type(self._embedding).__name__),
            "distinct_doc_ids": sorted(
                {doc.metadata.get("doc_id", "?") for doc in self._documents}
            ),
        }
        manifest.update(extra or {})
        (target / _MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: Path | None = None, embedding: Embeddings | None = None) -> "MedFlowVectorStore":
        """Recarrega um índice salvo sem recalcular embeddings."""
        source = Path(directory or get_settings().vector_store_dir)
        vectors_path = source / _VECTORS_FILE
        docs_path = source / _DOCS_FILE
        if not vectors_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"Índice não encontrado em {source}. Rode 'python -m medflow_ai.cli build-index'."
            )
        store = cls(embedding or get_embeddings())
        documents: list[Document] = []
        with docs_path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                documents.append(
                    Document(page_content=payload["page_content"], metadata=payload["metadata"])
                )
        store._documents = documents
        store._vectors = np.load(vectors_path)
        return store


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)
