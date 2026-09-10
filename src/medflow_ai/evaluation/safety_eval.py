"""Avaliação do guardrail de segurança clínica.

Pergunta de pesquisa: *o classificador determinístico roteia corretamente as
solicitações entre SAFE, CAUTION, HUMAN_REVIEW e BLOCK?*

Além da acurácia global, o que interessa clinicamente é a assimetria dos erros:
subestimar o risco (por exemplo, tratar um pedido de prescrição como SAFE) é
muito mais grave do que superestimá-lo. Por isso o relatório separa
``subestimacao`` de ``superestimacao``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from medflow_ai.config import get_settings
from medflow_ai.safety.policy import SafetyCategory, classify_request

__all__ = ["SafetyItem", "SafetyReport", "load_safety_benchmark", "evaluate_safety"]

DEFAULT_SAFETY_BENCHMARK = Path("data/benchmarks/safety_benchmark.jsonl")

_ORDER = ["SAFE", "CAUTION", "HUMAN_REVIEW", "BLOCK"]


@dataclass(frozen=True)
class SafetyItem:
    id: str
    prompt: str
    expected: str


@dataclass
class SafetyReport:
    """Resultado agregado da avaliação de segurança."""

    n_prompts: int
    accuracy: float
    per_category: dict[str, dict[str, Any]]
    confusion: dict[str, dict[str, int]]
    subestimacao: int
    superestimacao: int
    erros: list[dict[str, str]] = field(default_factory=list)
    policy_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render_confusion(self) -> str:
        """Matriz de confusão em texto (linhas = esperado, colunas = previsto)."""
        header = "esperado \\ previsto".ljust(20) + "".join(name.rjust(14) for name in _ORDER)
        lines = [header]
        for expected in _ORDER:
            row = expected.ljust(20)
            row += "".join(str(self.confusion[expected][predicted]).rjust(14) for predicted in _ORDER)
            lines.append(row)
        return "\n".join(lines)


def load_safety_benchmark(path: Path | str | None = None) -> list[SafetyItem]:
    """Carrega o conjunto rotulado de prompts de segurança."""
    benchmark_path = Path(path) if path else (get_settings().project_root / DEFAULT_SAFETY_BENCHMARK)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark de segurança não encontrado em {benchmark_path}")
    items: list[SafetyItem] = []
    with benchmark_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                items.append(SafetyItem(payload["id"], payload["prompt"], payload["expected"]))
    return items


def evaluate_safety(benchmark: Sequence[SafetyItem] | None = None) -> SafetyReport:
    """Executa o benchmark rotulado contra a política vigente."""
    from medflow_ai.safety.policy import POLICY_VERSION

    items = list(benchmark or load_safety_benchmark())
    confusion = {expected: {predicted: 0 for predicted in _ORDER} for expected in _ORDER}
    per_category: dict[str, dict[str, Any]] = {
        name: {"total": 0, "corretos": 0, "acuracia": 0.0} for name in _ORDER
    }
    errors: list[dict[str, str]] = []
    under = over = 0

    for item in items:
        predicted = str(classify_request(item.prompt).category)
        confusion[item.expected][predicted] += 1
        per_category[item.expected]["total"] += 1
        if predicted == item.expected:
            per_category[item.expected]["corretos"] += 1
        else:
            errors.append(
                {"id": item.id, "prompt": item.prompt, "esperado": item.expected, "previsto": predicted}
            )
            if _ORDER.index(predicted) < _ORDER.index(item.expected):
                under += 1
            else:
                over += 1

    for name, stats in per_category.items():
        stats["acuracia"] = round(stats["corretos"] / stats["total"], 4) if stats["total"] else 0.0

    correct = sum(stats["corretos"] for stats in per_category.values())
    return SafetyReport(
        n_prompts=len(items),
        accuracy=round(correct / len(items), 4) if items else 0.0,
        per_category=per_category,
        confusion=confusion,
        subestimacao=under,
        superestimacao=over,
        erros=errors,
        policy_version=POLICY_VERSION,
    )


def save_report(report: SafetyReport, output_dir: Path | None = None) -> Path:
    """Persiste o relatório de segurança em ``artifacts/safety``."""
    target = Path(output_dir or (get_settings().artifacts_dir / "safety"))
    target.mkdir(parents=True, exist_ok=True)
    path = target / "safety_report.json"
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "safety_confusion.txt").write_text(report.render_confusion(), encoding="utf-8")
    return path
