"""Document Loaders do MedFlow AI (LangChain).

O corpus institucional é Markdown versionado; PDFs externos (por exemplo PCDTs
oficiais baixados pelo usuário) são suportados via ``PyPDFLoader`` quando a
dependência ``pypdf`` estiver instalada.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from medflow_ai.config import get_settings
from medflow_ai.data.corpus import ProtocolDocument, load_corpus

__all__ = ["load_protocol_documents", "load_pdf_documents", "section_documents"]


def section_documents(document: ProtocolDocument) -> list[Document]:
    """Converte as seções de um documento em ``Document`` do LangChain."""
    base_metadata = document.to_metadata()
    documents: list[Document] = []
    for section in document.sections:
        metadata = dict(base_metadata)
        metadata.update(
            {
                "section_id": section.section_id,
                "section_number": section.number or "",
                "section_title": section.title,
                "citation": f"{document.doc_id} v{document.version} — {document.title} · §{section.number or '?'} {section.title}",
            }
        )
        documents.append(
            Document(
                page_content=f"{document.title}\n## {section.title}\n{section.text}",
                metadata=metadata,
            )
        )
    return documents


def load_protocol_documents(corpus_dir: Path | None = None) -> list[Document]:
    """Carrega o corpus sintético de protocolos como documentos LangChain.

    Cada seção ``##`` vira um documento com metadados de rastreabilidade
    (``doc_id``, ``section_id``, ``citation``, ``content_hash``).
    """
    documents: list[Document] = []
    for protocol in load_corpus(corpus_dir):
        documents.extend(section_documents(protocol))
    return documents


def load_pdf_documents(pdf_dir: Path | None = None) -> list[Document]:
    """Carrega PDFs opcionais (ex.: PCDTs oficiais) de ``data/raw/pcdt``.

    Retorna lista vazia quando o diretório não existe — o pipeline continua
    funcionando apenas com o corpus sintético versionado.
    """
    directory = pdf_dir or (get_settings().data_dir / "raw" / "pcdt")
    if not directory.exists():
        return []
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        return []

    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as exc:  # pragma: no cover - dependência opcional
        raise ImportError(
            "Para carregar PDFs instale 'pypdf' e 'langchain-community'."
        ) from exc

    documents: list[Document] = []
    for pdf_path in pdfs:
        for page in PyPDFLoader(str(pdf_path)).load():
            metadata = dict(page.metadata)
            metadata.update(
                {
                    "doc_id": pdf_path.stem,
                    "titulo": pdf_path.stem,
                    "tipo": "pdf_externo",
                    "source_path": pdf_path.name,
                    "section_id": f"{pdf_path.stem}#p{metadata.get('page', 0)}",
                    "citation": f"{pdf_path.name} · página {int(metadata.get('page', 0)) + 1}",
                }
            )
            documents.append(Document(page_content=page.page_content, metadata=metadata))
    return documents
