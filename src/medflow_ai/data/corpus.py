"""Leitura do corpus institucional sintético de protocolos.

Os documentos vivem em ``data/synthetic/protocols`` como Markdown com um bloco
de metadados no topo (front matter simples ``chave: valor``). Manter o corpus
versionado em texto — e não em PDF binário — torna o RAG reproduzível em CI e
permite rastrear qualquer alteração de conteúdo pelo próprio git.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from medflow_ai.config import get_settings

__all__ = ["ProtocolDocument", "DocumentSection", "load_corpus", "load_document", "iter_sections"]

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_NUMBERED_HEADING = re.compile(r"^(\d+)\.\s*(.+)$")


@dataclass(frozen=True)
class DocumentSection:
    """Uma seção ``##`` de um documento, com título e corpo."""

    doc_id: str
    section_id: str
    number: str | None
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}".strip()


@dataclass(frozen=True)
class ProtocolDocument:
    """Documento institucional carregado do corpus."""

    doc_id: str
    path: Path
    metadata: dict[str, str]
    body: str
    sections: tuple[DocumentSection, ...] = field(default_factory=tuple)

    @property
    def title(self) -> str:
        return self.metadata.get("titulo", self.doc_id)

    @property
    def doc_type(self) -> str:
        return self.metadata.get("tipo", "documento")

    @property
    def specialty(self) -> str:
        return self.metadata.get("especialidade", "Geral")

    @property
    def version(self) -> str:
        return self.metadata.get("versao", "n/d")

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()[:16]

    def citation(self) -> str:
        """Rótulo curto de citação usado nas respostas do assistente."""
        return f"{self.doc_id} v{self.version} — {self.title}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "titulo": self.title,
            "tipo": self.doc_type,
            "especialidade": self.specialty,
            "versao": self.version,
            "vigencia": self.metadata.get("vigencia", "n/d"),
            "source_path": self.path.name,
            "content_hash": self.content_hash,
        }


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
    return metadata, raw[match.end():]


def iter_sections(doc_id: str, body: str) -> Iterator[DocumentSection]:
    """Divide o corpo do documento em seções de nível ``##``."""
    headings = list(_SECTION_HEADING.finditer(body))
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        raw_title = heading.group(1).strip()
        numbered = _NUMBERED_HEADING.match(raw_title)
        number = numbered.group(1) if numbered else None
        title = numbered.group(2).strip() if numbered else raw_title
        section_id = f"{doc_id}#{number or index + 1}"
        yield DocumentSection(
            doc_id=doc_id,
            section_id=section_id,
            number=number,
            title=title,
            text=body[start:end].strip(),
        )


def load_document(path: Path) -> ProtocolDocument:
    """Carrega um único documento Markdown do corpus."""
    raw = path.read_text(encoding="utf-8")
    metadata, body = _parse_front_matter(raw)
    doc_id = metadata.get("doc_id") or path.stem
    body = body.strip()
    return ProtocolDocument(
        doc_id=doc_id,
        path=path,
        metadata=metadata,
        body=body,
        sections=tuple(iter_sections(doc_id, body)),
    )


def load_corpus(corpus_dir: Path | None = None) -> list[ProtocolDocument]:
    """Carrega todos os documentos do corpus, ordenados por ``doc_id``."""
    directory = corpus_dir or get_settings().corpus_dir
    if not directory.exists():
        raise FileNotFoundError(
            f"Corpus não encontrado em {directory}. "
            "Verifique MEDFLOW_CORPUS_DIR ou o diretório data/synthetic/protocols."
        )
    documents = [load_document(path) for path in sorted(directory.glob("*.md"))]
    if not documents:
        raise FileNotFoundError(f"Nenhum documento .md encontrado em {directory}.")
    return sorted(documents, key=lambda doc: doc.doc_id)
