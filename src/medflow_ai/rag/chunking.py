"""Chunking com preservação de rastreabilidade.

O corpus já é dividido por seção; aqui as seções longas são subdivididas com
``RecursiveCharacterTextSplitter`` e cada pedaço recebe um ``chunk_id`` estável.
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from medflow_ai.config import get_settings

__all__ = ["chunk_documents", "build_splitter"]

_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Cria o splitter usado no pipeline (separadores adequados a texto clínico)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        keep_separator=True,
    )


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Divide documentos em chunks mantendo os metadados de origem.

    Args:
        documents: documentos por seção produzidos pelos loaders.
        chunk_size: tamanho alvo em caracteres (padrão vem da configuração).
        chunk_overlap: sobreposição em caracteres.

    Returns:
        Lista de chunks com ``chunk_id``, ``chunk_index`` e ``chunk_total``.
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if overlap >= size:
        raise ValueError("chunk_overlap deve ser menor que chunk_size.")

    splitter = build_splitter(size, overlap)
    chunks: list[Document] = []
    for document in documents:
        pieces = splitter.split_text(document.page_content) or [document.page_content]
        for index, piece in enumerate(pieces):
            metadata = dict(document.metadata)
            section_id = metadata.get("section_id", metadata.get("doc_id", "doc"))
            digest = hashlib.sha1(piece.encode("utf-8")).hexdigest()[:8]
            metadata.update(
                {
                    "chunk_id": f"{section_id}::{index}::{digest}",
                    "chunk_index": index,
                    "chunk_total": len(pieces),
                    "chunk_size_chars": len(piece),
                }
            )
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks
