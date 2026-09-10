"""Fixtures compartilhadas.

Todos os testes rodam em diretório temporário: banco, logs e índice nunca tocam
os artefatos do repositório. Como a configuração é cacheada, cada mudança de
ambiente exige ``reload_settings()`` e a limpeza dos caches de ``graph.tools``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from medflow_ai.config import reload_settings


@pytest.fixture(scope="session")
def sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Diretório isolado usado como raiz de dados/logs durante os testes."""
    root = tmp_path_factory.mktemp("medflow")
    (root / "logs").mkdir()
    return root


@pytest.fixture(scope="session", autouse=True)
def _isolated_environment(sandbox: Path):
    """Aponta banco, logs e índice para o sandbox durante toda a sessão."""
    previous = dict(os.environ)
    os.environ.update(
        {
            "MEDFLOW_DB_PATH": str(sandbox / "hospital.db"),
            "MEDFLOW_LOG_DIR": str(sandbox / "logs"),
            "MEDFLOW_ARTIFACTS_DIR": str(sandbox / "artifacts"),
            "MEDFLOW_VECTOR_STORE_DIR": str(sandbox / "vector_store"),
            "MEDFLOW_PSEUDONYM_SALT": "salt-de-teste",
            "MEDFLOW_SEED": "42",
        }
    )
    reload_settings()
    yield
    os.environ.clear()
    os.environ.update(previous)
    reload_settings()


@pytest.fixture(scope="session")
def database(_isolated_environment) -> Path:
    """Banco sintético construído uma única vez por sessão de testes."""
    from medflow_ai.config import get_settings
    from medflow_ai.database.ingest import build_synthetic_database

    build_synthetic_database(n_patients=12, seed=42)
    return get_settings().database_path


@pytest.fixture(scope="session")
def repository(database: Path):
    from medflow_ai.database.repository import PatientRepository

    return PatientRepository(database)


@pytest.fixture(scope="session")
def corpus():
    from medflow_ai.data.corpus import load_corpus

    return load_corpus()


@pytest.fixture(scope="session")
def vector_store(corpus):
    from medflow_ai.rag.chunking import chunk_documents
    from medflow_ai.rag.embeddings import get_embeddings
    from medflow_ai.rag.loaders import load_protocol_documents
    from medflow_ai.rag.vector_store import MedFlowVectorStore

    chunks = chunk_documents(load_protocol_documents(), chunk_size=400, chunk_overlap=100)
    return MedFlowVectorStore.from_documents(chunks, get_embeddings("hashing"))


@pytest.fixture(scope="session")
def retriever(vector_store):
    from medflow_ai.rag.retriever import ProtocolRetriever

    return ProtocolRetriever(vector_store, strategy="hybrid", k=5)


@pytest.fixture()
def assistant(retriever, repository, tmp_path):
    """Assistente completo, com logger de auditoria isolado por teste."""
    from medflow_ai.graph.build import MedFlowAssistant
    from medflow_ai.llm.providers import get_chat_model
    from medflow_ai.logging_utils.audit import AuditLogger

    return MedFlowAssistant(
        retriever=retriever,
        repository=repository,
        chat_model=get_chat_model("template"),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )


@pytest.fixture(scope="session")
def demo_patient() -> str:
    """Identificador do paciente-âncora usado nos testes determinísticos."""
    from medflow_ai.database.synthetic_patients import DEMO_PATIENT_ID

    return DEMO_PATIENT_ID
