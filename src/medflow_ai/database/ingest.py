"""Ingestão do prontuário estruturado no SQLite.

Duas origens suportadas:

* ``build_synthetic_database`` — gera e carrega o dataset sintético do projeto;
* ``ingest_synthea_csv`` — carrega um export CSV real do Synthea (mesmas
  colunas), permitindo trocar a origem sem alterar o restante do sistema.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from medflow_ai.config import get_settings
from medflow_ai.database.schema import TABLES, connect, create_schema
from medflow_ai.database.synthetic_patients import SyntheticDataset, generate_dataset

__all__ = ["build_synthetic_database", "ingest_synthea_csv", "insert_rows", "table_counts"]

# Colunas aceitas por tabela (o que não existir na origem é ignorado).
_COLUMNS: dict[str, tuple[str, ...]] = {
    "patients": ("id", "birthdate", "deathdate", "gender", "first", "last", "cpf", "cns",
                 "email", "phone", "address", "city", "state", "zip"),
    "encounters": ("id", "patient", "start", "stop", "encounterclass", "code",
                   "description", "reasondescription"),
    "conditions": ("patient", "encounter", "start", "stop", "code", "description"),
    "observations": ("patient", "encounter", "date", "code", "description", "value",
                     "units", "type"),
    "medications": ("patient", "encounter", "start", "stop", "code", "description",
                    "dispenses", "reasondescription"),
    "procedures": ("patient", "encounter", "start", "stop", "code", "description",
                   "reasondescription"),
    "lab_orders": ("id", "patient", "encounter", "ordered_at", "exam", "priority",
                   "status", "resulted_at", "result_value", "result_units", "critical_flag"),
}


def insert_rows(connection: sqlite3.Connection, table: str, rows: Sequence[Mapping[str, Any]]) -> int:
    """Insere linhas em uma tabela conhecida, ignorando colunas desconhecidas."""
    if table not in _COLUMNS:
        raise ValueError(f"Tabela desconhecida: {table!r}")
    if not rows:
        return 0
    columns = _COLUMNS[table]
    placeholders = ", ".join("?" for _ in columns)
    statement = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    payload = [tuple(row.get(column) for column in columns) for row in rows]
    connection.executemany(statement, payload)
    connection.commit()
    return len(payload)


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """Contagem de linhas por tabela — usado em testes e no CLI."""
    counts: dict[str, int] = {}
    for table in TABLES:
        counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def build_synthetic_database(
    database_path: Path | None = None,
    *,
    n_patients: int = 40,
    seed: int | None = None,
    reset: bool = True,
) -> dict[str, int]:
    """Cria (ou recria) o banco de demonstração com pacientes sintéticos."""
    settings = get_settings()
    target = Path(database_path or settings.database_path)
    if reset and target.exists():
        target.unlink()

    dataset: SyntheticDataset = generate_dataset(
        n_patients=n_patients, seed=seed if seed is not None else settings.seed
    )
    with connect(target) as connection:
        create_schema(connection)
        for table, rows in dataset.as_tables().items():
            insert_rows(connection, table, rows)
        return table_counts(connection)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key.strip().lower(): value for key, value in row.items()} for row in csv.DictReader(handle)]


def ingest_synthea_csv(
    csv_dir: Path,
    database_path: Path | None = None,
    *,
    reset: bool = True,
) -> dict[str, int]:
    """Ingere um export CSV do Synthea (``patients.csv``, ``conditions.csv``, ...).

    Colunas ausentes viram ``NULL``; colunas extras do Synthea são ignoradas.
    """
    directory = Path(csv_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Diretório de CSVs não encontrado: {directory}")

    target = Path(database_path or get_settings().database_path)
    if reset and target.exists():
        target.unlink()

    with connect(target) as connection:
        create_schema(connection)
        for table in TABLES:
            csv_path = directory / f"{table}.csv"
            if not csv_path.exists():
                continue
            insert_rows(connection, table, _read_csv(csv_path))
        return table_counts(connection)


def export_tables_to_csv(database_path: Path, output_dir: Path) -> list[Path]:
    """Exporta o banco para CSVs (amostra versionável / inspeção manual)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with connect(database_path) as connection:
        for table in TABLES:
            rows: Iterable[sqlite3.Row] = connection.execute(f"SELECT * FROM {table}").fetchall()
            rows = list(rows)
            if not rows:
                continue
            path = output_dir / f"{table}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows([dict(row) for row in rows])
            written.append(path)
    return written
