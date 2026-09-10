"""Configuração central do projeto baseada em variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("MEDFLOW_ENV", "development")
    data_dir: Path = Path(os.getenv("MEDFLOW_DATA_DIR", "data"))
    log_dir: Path = Path(os.getenv("MEDFLOW_LOG_DIR", "logs"))
    database_path: Path = Path(
        os.getenv("MEDFLOW_DB_PATH", "data/processed/medflow.db")
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
