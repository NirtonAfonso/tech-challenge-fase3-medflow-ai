"""Avaliação de recuperação exata no prontuário estruturado.

Pergunta de pesquisa: *quando o valor correto é conhecido, o sistema recupera
exatamente esse valor do banco?* Diferente do RAG, aqui não há ambiguidade: ou
o valor bate, ou não bate. Este é o teste que separa "a LLM disse algo
plausível" de "o dado do paciente foi realmente usado".

O gabarito é gerado a partir do próprio banco sintético (determinístico), o que
mantém o teste válido mesmo se o gerador mudar de seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medflow_ai.config import get_settings
from medflow_ai.database.repository import PatientRepository
from medflow_ai.database.schema import connect

__all__ = ["ExactRetrievalCase", "DatabaseReport", "build_cases", "evaluate_database"]


@dataclass(frozen=True)
class ExactRetrievalCase:
    """Um caso com valor conhecido no banco."""

    patient_id: str
    campo: str
    esperado: str


@dataclass
class DatabaseReport:
    n_cases: int
    exact_match: float
    context_leak_free: bool
    falhas: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_cases(database_path: Path | None = None, *, limit: int = 30) -> list[ExactRetrievalCase]:
    """Deriva casos de teste diretamente do banco (valor conhecido = verdade)."""
    path = Path(database_path or get_settings().database_path)
    cases: list[ExactRetrievalCase] = []
    with connect(path) as connection:
        rows = connection.execute(
            """
            SELECT o.patient, o.description, o.value, o.date
            FROM observations o
            WHERE o.date = (SELECT MAX(date) FROM observations
                            WHERE patient = o.patient AND description = o.description)
            ORDER BY o.patient, o.description
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        for row in rows:
            cases.append(
                ExactRetrievalCase(
                    patient_id=row["patient"],
                    campo=row["description"],
                    esperado=str(row["value"]),
                )
            )
    return cases


def evaluate_database(
    repository: PatientRepository | None = None,
    cases: list[ExactRetrievalCase] | None = None,
) -> DatabaseReport:
    """Mede a taxa de recuperação exata e verifica ausência de PII no contexto."""
    repo = repository or PatientRepository()
    test_cases = cases or build_cases()

    matches = 0
    failures: list[dict[str, Any]] = []
    for case in test_cases:
        observations = repo.latest_observations(case.patient_id, limit=50)
        found = next((item for item in observations if item["exame"] == case.campo), None)
        if found and str(found["valor"]) == case.esperado:
            matches += 1
        else:
            failures.append(
                {
                    "patient_id": case.patient_id,
                    "campo": case.campo,
                    "esperado": case.esperado,
                    "obtido": found["valor"] if found else None,
                }
            )

    # Verificação de minimização: nenhum identificador direto pode aparecer no
    # contexto enviado à LLM, mesmo estando presente na tabela de pacientes.
    leak_free = True
    for patient_id in {case.patient_id for case in test_cases}:
        raw = repo.raw_record(patient_id)
        context_text = repo.build_context(patient_id).to_prompt_block()
        identifiers = [
            str(raw.get(field_name) or "")
            for field_name in ("first", "last", "cpf", "cns", "email", "phone", "address")
        ]
        if any(value and value in context_text for value in identifiers):
            leak_free = False
            break

    total = len(test_cases) or 1
    return DatabaseReport(
        n_cases=len(test_cases),
        exact_match=round(matches / total, 4),
        context_leak_free=leak_free,
        falhas=failures,
    )


def save_report(report: DatabaseReport, output_dir: Path | None = None) -> Path:
    target = Path(output_dir or (get_settings().artifacts_dir / "database"))
    target.mkdir(parents=True, exist_ok=True)
    path = target / "database_report.json"
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
