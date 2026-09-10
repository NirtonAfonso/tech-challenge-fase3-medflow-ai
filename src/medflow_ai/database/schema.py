"""Esquema SQLite do prontuário estruturado.

Os nomes de tabelas e colunas seguem a exportação CSV do **Synthea**
(patients, conditions, observations, medications, procedures, encounters) para
que um export real do Synthea possa ser ingerido sem alterar o código.

Uma tabela adicional, ``lab_orders``, modela **exames solicitados e pendentes**
— informação exigida explicitamente pelo enunciado ("verificar exames
pendentes") e que não existe no export padrão do Synthea.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

__all__ = ["SCHEMA_SQL", "TABLES", "create_schema", "connect"]

TABLES: tuple[str, ...] = (
    "patients",
    "encounters",
    "conditions",
    "observations",
    "medications",
    "procedures",
    "lab_orders",
)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    id            TEXT PRIMARY KEY,
    birthdate     TEXT NOT NULL,
    deathdate     TEXT,
    gender        TEXT,
    first         TEXT,
    last          TEXT,
    cpf           TEXT,
    cns           TEXT,
    email         TEXT,
    phone         TEXT,
    address       TEXT,
    city          TEXT,
    state         TEXT,
    zip           TEXT
);

CREATE TABLE IF NOT EXISTS encounters (
    id                 TEXT PRIMARY KEY,
    patient            TEXT NOT NULL REFERENCES patients(id),
    start              TEXT NOT NULL,
    stop               TEXT,
    encounterclass     TEXT,
    code               TEXT,
    description        TEXT,
    reasondescription  TEXT
);

CREATE TABLE IF NOT EXISTS conditions (
    patient      TEXT NOT NULL REFERENCES patients(id),
    encounter    TEXT,
    start        TEXT NOT NULL,
    stop         TEXT,
    code         TEXT,
    description  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    patient      TEXT NOT NULL REFERENCES patients(id),
    encounter    TEXT,
    date         TEXT NOT NULL,
    code         TEXT,
    description  TEXT NOT NULL,
    value        TEXT,
    units        TEXT,
    type         TEXT
);

CREATE TABLE IF NOT EXISTS medications (
    patient      TEXT NOT NULL REFERENCES patients(id),
    encounter    TEXT,
    start        TEXT NOT NULL,
    stop         TEXT,
    code         TEXT,
    description  TEXT NOT NULL,
    dispenses    INTEGER,
    reasondescription TEXT
);

CREATE TABLE IF NOT EXISTS procedures (
    patient      TEXT NOT NULL REFERENCES patients(id),
    encounter    TEXT,
    start        TEXT NOT NULL,
    stop         TEXT,
    code         TEXT,
    description  TEXT NOT NULL,
    reasondescription TEXT
);

-- Exames solicitados: permite responder "quais exames estão pendentes?".
CREATE TABLE IF NOT EXISTS lab_orders (
    id              TEXT PRIMARY KEY,
    patient         TEXT NOT NULL REFERENCES patients(id),
    encounter       TEXT,
    ordered_at      TEXT NOT NULL,
    exam            TEXT NOT NULL,
    priority        TEXT NOT NULL,
    status          TEXT NOT NULL,   -- pendente | coletado | liberado | cancelado
    resulted_at     TEXT,
    result_value    TEXT,
    result_units    TEXT,
    critical_flag   INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_conditions_patient   ON conditions(patient);
CREATE INDEX IF NOT EXISTS idx_observations_patient ON observations(patient, date DESC);
CREATE INDEX IF NOT EXISTS idx_medications_patient  ON medications(patient);
CREATE INDEX IF NOT EXISTS idx_procedures_patient   ON procedures(patient);
CREATE INDEX IF NOT EXISTS idx_encounters_patient   ON encounters(patient, start DESC);
CREATE INDEX IF NOT EXISTS idx_lab_orders_patient   ON lab_orders(patient, status);
"""


def connect(database_path: Path | str) -> sqlite3.Connection:
    """Abre conexão SQLite com ``row_factory`` em dicionário."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """Cria todas as tabelas e índices (idempotente)."""
    connection.executescript(SCHEMA_SQL)
    connection.commit()
