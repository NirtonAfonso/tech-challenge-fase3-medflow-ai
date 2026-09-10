"""Backends de embedding do MedFlow AI.

Duas implementações compatíveis com a interface ``Embeddings`` do LangChain:

``HashingEmbeddings``
    Determinística, sem download de modelo e sem GPU. É o backend padrão porque
    torna a avaliação de RAG **reprodutível em CI e offline** — as métricas do
    relatório podem ser regeradas por qualquer pessoa com um único comando.
    Usa *hashing trick* sobre unigramas, bigramas e n-gramas de caractere, com
    TF sublinear e normalização L2 (aproximação de um espaço vetorial esparso
    projetado em dimensão fixa).

``SentenceTransformerEmbeddings``
    Embeddings densos multilíngues de qualidade superior, usados quando a
    dependência opcional ``sentence-transformers`` está instalada (Colab).

A escolha é feita por ``MEDFLOW_EMBEDDING_BACKEND`` (``hashing`` | ``sentence_transformers``).
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
from langchain_core.embeddings import Embeddings

from medflow_ai.config import get_settings

__all__ = [
    "HashingEmbeddings",
    "SentenceTransformerEmbeddings",
    "get_embeddings",
    "normalize_text",
    "cosine_similarity_matrix",
]

_TOKEN_PATTERN = re.compile(r"[0-9a-zà-ÿ]+", re.IGNORECASE)

# Palavras muito frequentes em português que pouco discriminam documentos.
_STOPWORDS = frozenset(
    """a ao aos as à às da das de do dos e em na nas no nos o os ou para por com sem
    que se como quando qual quais um uma uns umas the of and is are to in for on
    ser é são foi era está estão pode podem deve devem não sim mais menos entre
    sobre até após antes cada todo toda todos todas seu sua seus suas este esta
    isso aquele aquela pelo pela pelos pelas""".split()
)


def normalize_text(text: str) -> str:
    """Minúsculas + remoção de acentos, para casar variações de escrita."""
    lowered = text.casefold()
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [tok for tok in _TOKEN_PATTERN.findall(normalized) if tok not in _STOPWORDS]


def _features(text: str, *, char_ngram: int = 4) -> Counter[str]:
    """Extrai unigramas, bigramas e n-gramas de caractere ponderados."""
    tokens = _tokens(text)
    features: Counter[str] = Counter()
    features.update(f"w:{tok}" for tok in tokens)
    features.update(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    for token in tokens:
        if len(token) > char_ngram:
            padded = f"^{token}$"
            for i in range(len(padded) - char_ngram + 1):
                features[f"c:{padded[i:i + char_ngram]}"] += 1
    return features


def _bucket_and_sign(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbeddings(Embeddings):
    """Embedding determinístico baseado em *hashing trick*.

    Não requer rede, GPU nem download de modelo, e produz sempre o mesmo vetor
    para o mesmo texto — o que torna as métricas de RAG reproduzíveis.
    """

    def __init__(self, dim: int | None = None, *, char_ngram: int = 4) -> None:
        self.dim = int(dim or get_settings().embedding_dim)
        if self.dim <= 0:
            raise ValueError("A dimensão do embedding deve ser positiva.")
        self.char_ngram = char_ngram

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}d-char{self.char_ngram}"

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, count in _features(text, char_ngram=self.char_ngram).items():
            bucket, sign = _bucket_and_sign(feature, self.dim)
            # TF sublinear: reduz o peso de repetições dentro do mesmo chunk.
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper fino sobre ``sentence-transformers`` (dependência opcional)."""

    def __init__(self, model_name: str | None = None, **kwargs: Any) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - caminho de ambiente
            raise ImportError(
                "sentence-transformers não está instalado. Use "
                "MEDFLOW_EMBEDDING_BACKEND=hashing ou instale a dependência opcional."
            ) from exc
        self.model_name = model_name or get_settings().embedding_model
        self._model = SentenceTransformer(self.model_name, **kwargs)

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(backend: str | None = None, **kwargs: Any) -> Embeddings:
    """Fábrica de embeddings dirigida por configuração."""
    settings = get_settings()
    chosen = (backend or settings.embedding_backend).strip().lower()
    if chosen in {"hashing", "hash", "default"}:
        return HashingEmbeddings(**kwargs)
    if chosen in {"sentence_transformers", "sentence-transformers", "st"}:
        return SentenceTransformerEmbeddings(**kwargs)
    raise ValueError(
        f"Backend de embedding desconhecido: {chosen!r}. "
        "Use 'hashing' ou 'sentence_transformers'."
    )


def cosine_similarity_matrix(query: Sequence[float], matrix: Iterable[Sequence[float]]) -> np.ndarray:
    """Similaridade de cosseno entre um vetor e uma matriz de vetores."""
    query_vector = np.asarray(query, dtype=np.float32)
    doc_matrix = np.asarray(list(matrix), dtype=np.float32)
    if doc_matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    query_norm = np.linalg.norm(query_vector) or 1.0
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    doc_norms[doc_norms == 0] = 1.0
    return (doc_matrix @ query_vector) / (doc_norms * query_norm)
