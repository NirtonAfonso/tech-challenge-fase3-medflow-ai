"""Configuração central do projeto baseada em variáveis de ambiente.

Todas as configurações do MedFlow AI passam por aqui. Nenhum módulo deve ler
``os.environ`` diretamente nem embutir caminhos absolutos, para que o projeto
funcione igualmente em execução local, em CI e no Google Colab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Raiz do repositório: .../src/medflow_ai/config.py -> sobe três níveis.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(env_var: str, default: str) -> Path:
    """Resolve um caminho de ambiente; relativos são ancorados na raiz do repo."""
    raw = os.getenv(env_var, default)
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def _int(env_var: str, default: int) -> int:
    try:
        return int(os.getenv(env_var, str(default)))
    except ValueError:
        return default


def _float(env_var: str, default: float) -> float:
    try:
        return float(os.getenv(env_var, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Configuração imutável derivada do ambiente."""

    environment: str = field(default_factory=lambda: os.getenv("MEDFLOW_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Diretórios
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _path("MEDFLOW_DATA_DIR", "data"))
    log_dir: Path = field(default_factory=lambda: _path("MEDFLOW_LOG_DIR", "logs"))
    artifacts_dir: Path = field(default_factory=lambda: _path("MEDFLOW_ARTIFACTS_DIR", "artifacts"))
    database_path: Path = field(
        default_factory=lambda: _path("MEDFLOW_DB_PATH", "data/processed/hospital.db")
    )
    vector_store_dir: Path = field(
        default_factory=lambda: _path("MEDFLOW_VECTOR_STORE_DIR", "data/processed/vector_store")
    )
    corpus_dir: Path = field(
        default_factory=lambda: _path("MEDFLOW_CORPUS_DIR", "data/synthetic/protocols")
    )

    # Reprodutibilidade
    seed: int = field(default_factory=lambda: _int("MEDFLOW_SEED", 42))

    # Pseudonimização — o salt evita que hashes sejam reversíveis por dicionário.
    pseudonym_salt: str = field(
        default_factory=lambda: os.getenv("MEDFLOW_PSEUDONYM_SALT", "medflow-academic-salt")
    )

    # RAG
    chunk_size: int = field(default_factory=lambda: _int("MEDFLOW_CHUNK_SIZE", 400))
    chunk_overlap: int = field(default_factory=lambda: _int("MEDFLOW_CHUNK_OVERLAP", 100))
    retriever_k: int = field(default_factory=lambda: _int("MEDFLOW_RETRIEVER_K", 5))
    retriever_strategy: str = field(
        default_factory=lambda: os.getenv("MEDFLOW_RETRIEVER_STRATEGY", "hybrid")
    )
    mmr_lambda: float = field(default_factory=lambda: _float("MEDFLOW_MMR_LAMBDA", 0.6))
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("MEDFLOW_EMBEDDING_BACKEND", "hashing")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "MEDFLOW_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    embedding_dim: int = field(default_factory=lambda: _int("MEDFLOW_EMBEDDING_DIM", 512))

    # LLM
    llm_provider: str = field(
        default_factory=lambda: os.getenv("MEDFLOW_LLM_PROVIDER", "template")
    )
    base_model_id: str = field(
        default_factory=lambda: os.getenv("MEDFLOW_BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    )
    adapter_path: str = field(default_factory=lambda: os.getenv("MEDFLOW_ADAPTER_PATH", ""))
    max_new_tokens: int = field(default_factory=lambda: _int("MEDFLOW_MAX_NEW_TOKENS", 512))
    temperature: float = field(default_factory=lambda: _float("MEDFLOW_TEMPERATURE", 0.2))

    def ensure_dirs(self) -> None:
        """Cria os diretórios de saída usados em runtime."""
        for directory in (self.data_dir, self.log_dir, self.artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna as configurações (cacheadas) do processo atual."""
    return Settings()


def reload_settings() -> Settings:
    """Relê o ambiente. Útil em testes que manipulam variáveis de ambiente."""
    get_settings.cache_clear()
    return get_settings()


# Compatibilidade com o código inicial do repositório.
settings = get_settings()
