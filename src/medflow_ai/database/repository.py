"""Camada de acesso ao prontuário estruturado, com minimização de dados.

Regra central do projeto: **a LLM nunca recebe identificadores diretos**. O
repositório lê o prontuário completo do SQLite e devolve um ``PatientContext``
já pseudonimizado e reduzido ao que é necessário para responder à pergunta.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from medflow_ai.config import get_settings
from medflow_ai.data.anonymization import age_band, birthdate_to_age, pseudonymize
from medflow_ai.database.schema import connect
from medflow_ai.database.synthetic_patients import REFERENCE_DATE

__all__ = ["PatientContext", "PatientRepository", "PatientNotFoundError"]

# Campos do prontuário que NUNCA saem do repositório em forma bruta.
# ``birthdate`` é quase-identificador: sai apenas transformado em idade/faixa.
_BLOCKED_COLUMNS = frozenset(
    {"first", "last", "cpf", "cns", "email", "phone", "address", "city", "state",
     "zip", "birthdate", "deathdate"}
)


class PatientNotFoundError(LookupError):
    """Paciente inexistente na base estruturada."""


@dataclass
class PatientContext:
    """Contexto clínico mínimo enviado à LLM (sem identificadores diretos)."""

    patient_id: str
    patient_pseudonym: str
    age: int | None
    age_band: str
    gender: str | None
    conditions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    medications: list[dict[str, Any]] = field(default_factory=list)
    procedures: list[dict[str, Any]] = field(default_factory=list)
    encounters: list[dict[str, Any]] = field(default_factory=list)
    pending_exams: list[dict[str, Any]] = field(default_factory=list)
    reference_date: str = REFERENCE_DATE.isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_empty(self) -> bool:
        return not (self.conditions or self.observations or self.medications or self.pending_exams)

    def to_prompt_block(self, *, max_items: int = 6) -> str:
        """Renderiza o contexto como texto compacto para o prompt da LLM."""
        lines = [
            f"Paciente (pseudonimizado): {self.patient_pseudonym}",
            f"Faixa etária: {self.age_band} anos | Sexo: {self.gender or 'não informado'}",
            f"Data de referência dos dados: {self.reference_date}",
        ]

        if self.conditions:
            lines.append("Condições ativas:")
            lines += [
                f"  - {item['descricao']} (início {item['inicio']})"
                for item in self.conditions[:max_items]
            ]
        if self.observations:
            lines.append("Exames/observações mais recentes:")
            lines += [
                f"  - {item['exame']}: {item['valor']} {item['unidade']} em {item['data']}"
                for item in self.observations[:max_items]
            ]
        if self.medications:
            lines.append("Medicamentos em uso:")
            lines += [
                f"  - {item['descricao']} (desde {item['inicio']})"
                for item in self.medications[:max_items]
            ]
        if self.procedures:
            lines.append("Procedimentos recentes:")
            lines += [
                f"  - {item['descricao']} em {item['data']}" for item in self.procedures[:max_items]
            ]
        if self.pending_exams:
            lines.append("Exames pendentes:")
            lines += [
                f"  - {item['exame']} (solicitado em {item['solicitado_em']}, prioridade {item['prioridade']})"
                for item in self.pending_exams[:max_items]
            ]
        if self.encounters:
            last = self.encounters[0]
            lines.append(f"Último atendimento: {last['descricao']} em {last['inicio']}")
        return "\n".join(lines)


class PatientRepository:
    """Consultas ao prontuário sintético em SQLite."""

    def __init__(self, database_path: Path | str | None = None) -> None:
        self.database_path = Path(database_path or get_settings().database_path)
        if not self.database_path.exists():
            raise FileNotFoundError(
                f"Banco não encontrado em {self.database_path}. "
                "Rode 'python -m medflow_ai.cli build-db' para criá-lo."
            )

    # ------------------------------------------------------------- utilidades
    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with connect(self.database_path) as connection:
            return connection.execute(sql, params).fetchall()

    def list_patient_ids(self, limit: int = 50) -> list[str]:
        rows = self._query("SELECT id FROM patients ORDER BY id LIMIT ?", (limit,))
        return [row["id"] for row in rows]

    def exists(self, patient_id: str) -> bool:
        return bool(self._query("SELECT 1 FROM patients WHERE id = ?", (patient_id,)))

    # ------------------------------------------------------------- consultas
    def demographics(self, patient_id: str) -> dict[str, Any]:
        """Demografia já minimizada: idade/faixa etária e sexo, sem nome ou documentos."""
        rows = self._query(
            "SELECT id, birthdate, gender FROM patients WHERE id = ?", (patient_id,)
        )
        if not rows:
            raise PatientNotFoundError(f"Paciente {patient_id!r} não encontrado.")
        row = rows[0]
        age = birthdate_to_age(row["birthdate"], reference=REFERENCE_DATE)
        return {
            "patient_id": row["id"],
            "patient_pseudonym": pseudonymize(row["id"]),
            "age": age,
            "age_band": age_band(age),
            "gender": row["gender"],
        }

    def conditions(self, patient_id: str, *, active_only: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT description, start, stop FROM conditions WHERE patient = ?"
        if active_only:
            sql += " AND stop IS NULL"
        sql += " ORDER BY start DESC"
        return [
            {"descricao": row["description"], "inicio": row["start"], "fim": row["stop"]}
            for row in self._query(sql, (patient_id,))
        ]

    def latest_observations(self, patient_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Última medida de cada exame, da mais recente para a mais antiga."""
        rows = self._query(
            """
            SELECT description, value, units, date
            FROM observations o
            WHERE patient = ?
              AND date = (SELECT MAX(date) FROM observations
                          WHERE patient = o.patient AND description = o.description)
            GROUP BY description
            ORDER BY date DESC, description
            LIMIT ?
            """,
            (patient_id, limit),
        )
        return [
            {
                "exame": row["description"],
                "valor": row["value"],
                "unidade": row["units"] or "",
                "data": row["date"],
            }
            for row in rows
        ]

    def observation_history(self, patient_id: str, exam: str) -> list[dict[str, Any]]:
        """Série temporal de um exame específico (para avaliar tendência)."""
        rows = self._query(
            """
            SELECT description, value, units, date FROM observations
            WHERE patient = ? AND LOWER(description) = LOWER(?)
            ORDER BY date DESC
            """,
            (patient_id, exam),
        )
        return [
            {
                "exame": row["description"],
                "valor": row["value"],
                "unidade": row["units"] or "",
                "data": row["date"],
            }
            for row in rows
        ]

    def medications(self, patient_id: str, *, active_only: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT description, start, stop FROM medications WHERE patient = ?"
        if active_only:
            sql += " AND stop IS NULL"
        sql += " ORDER BY start DESC"
        return [
            {"descricao": row["description"], "inicio": row["start"], "fim": row["stop"]}
            for row in self._query(sql, (patient_id,))
        ]

    def procedures(self, patient_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT description, start FROM procedures WHERE patient = ? ORDER BY start DESC LIMIT ?",
            (patient_id, limit),
        )
        return [{"descricao": row["description"], "data": row["start"]} for row in rows]

    def encounters(self, patient_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
        rows = self._query(
            """
            SELECT description, start, encounterclass, reasondescription
            FROM encounters WHERE patient = ? ORDER BY start DESC LIMIT ?
            """,
            (patient_id, limit),
        )
        return [
            {
                "descricao": row["description"],
                "inicio": row["start"],
                "classe": row["encounterclass"],
                "motivo": row["reasondescription"],
            }
            for row in rows
        ]

    def pending_exams(self, patient_id: str) -> list[dict[str, Any]]:
        """Exames solicitados e ainda não liberados — requisito explícito do desafio."""
        rows = self._query(
            """
            SELECT exam, ordered_at, priority, status FROM lab_orders
            WHERE patient = ? AND status IN ('pendente', 'coletado')
            ORDER BY ordered_at DESC
            """,
            (patient_id,),
        )
        return [
            {
                "exame": row["exam"],
                "solicitado_em": row["ordered_at"],
                "prioridade": row["priority"],
                "status": row["status"],
            }
            for row in rows
        ]

    # ---------------------------------------------------------------- contexto
    def build_context(
        self,
        patient_id: str,
        *,
        include: Sequence[str] | None = None,
        max_observations: int = 8,
    ) -> PatientContext:
        """Monta o contexto clínico pseudonimizado enviado ao restante do fluxo.

        Args:
            patient_id: identificador do paciente na base.
            include: subconjunto de blocos a incluir (minimização de dados).
                Valores aceitos: ``conditions``, ``observations``, ``medications``,
                ``procedures``, ``encounters``, ``pending_exams``. ``None`` inclui todos.
            max_observations: limite de exames retornados.
        """
        demographics = self.demographics(patient_id)
        wanted = set(
            include
            or ("conditions", "observations", "medications", "procedures", "encounters", "pending_exams")
        )
        return PatientContext(
            **demographics,
            conditions=self.conditions(patient_id) if "conditions" in wanted else [],
            observations=(
                self.latest_observations(patient_id, limit=max_observations)
                if "observations" in wanted
                else []
            ),
            medications=self.medications(patient_id) if "medications" in wanted else [],
            procedures=self.procedures(patient_id) if "procedures" in wanted else [],
            encounters=self.encounters(patient_id) if "encounters" in wanted else [],
            pending_exams=self.pending_exams(patient_id) if "pending_exams" in wanted else [],
        )

    def raw_record(self, patient_id: str) -> dict[str, Any]:
        """Registro bruto do paciente, incluindo PII.

        Usado **apenas** na demonstração de anonimização (notebook/relatório).
        Nunca deve alimentar prompt, log ou resposta.
        """
        rows = self._query("SELECT * FROM patients WHERE id = ?", (patient_id,))
        if not rows:
            raise PatientNotFoundError(f"Paciente {patient_id!r} não encontrado.")
        return dict(rows[0])

    def safe_columns(self) -> list[str]:
        """Colunas de ``patients`` liberadas para uso analítico."""
        rows = self._query("SELECT * FROM patients LIMIT 1")
        if not rows:
            return []
        return [column for column in rows[0].keys() if column not in _BLOCKED_COLUMNS]
