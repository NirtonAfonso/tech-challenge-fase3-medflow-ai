"""Validador dos resultados devolvidos pela execução no Colab.

Depois que o usuário roda o notebook 02 em GPU, é preciso decidir se os
artefatos podem ser citados no README e no relatório. Este módulo faz essa
verificação de forma mecânica, em vez de confiar em inspeção visual:

* os arquivos obrigatórios existem;
* ``training_results.status == "ok"``;
* os campos de rastreabilidade estão preenchidos (commit, modelo, GPU, dtype, seed);
* a loss é numérica e finita;
* a comparação cobre ``base``, ``fine_tuned`` e ``fine_tuned_rag``;
* nenhum segredo aparente e nenhum peso grande contaminou a pasta.

Uso::

    python -m medflow_ai.cli validate-colab-results artifacts/fine_tuning
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medflow_ai.colab import SECRET_PATTERNS, WEIGHT_SUFFIXES

__all__ = ["ValidationResult", "validate_colab_results", "REQUIRED_FILES", "REQUIRED_FIELDS"]

REQUIRED_FILES: tuple[str, ...] = (
    "training_results.json",
    "loss_curve.png",
    "comparacao_sistemas.json",
    "respostas_antes_depois.json",
)

OPTIONAL_FILES: tuple[str, ...] = ("environment.json", "training_config.json")

#: Campos que precisam existir e estar preenchidos em ``training_results.json``.
REQUIRED_FIELDS: tuple[str, ...] = (
    "status", "executado_em", "commit", "seed", "base_model",
    "gpu", "compute_dtype", "metricas_treino", "splits", "parametros",
)

EXPECTED_SYSTEMS: frozenset[str] = frozenset({"base", "fine_tuned", "fine_tuned_rag"})

#: Um artefato de métrica acima disso é sinal de peso disfarçado.
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024

_SECRET_VALUE = re.compile(
    r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"
)


@dataclass
class ValidationResult:
    """Veredito da validação."""

    diretorio: Path
    erros: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    resumo: dict[str, Any] = field(default_factory=dict)

    @property
    def valido(self) -> bool:
        return not self.erros

    def to_dict(self) -> dict[str, Any]:
        return {
            "diretorio": str(self.diretorio),
            "valido": self.valido,
            "erros": list(self.erros),
            "avisos": list(self.avisos),
            "resumo": self.resumo,
        }

    def render(self) -> str:
        linhas = [
            "=" * 78,
            f"VALIDAÇÃO DOS RESULTADOS DO COLAB — {self.diretorio}",
            "=" * 78,
        ]
        if self.resumo:
            for chave, valor in self.resumo.items():
                linhas.append(f"  {chave:24s}: {valor}")
            linhas.append("")
        for erro in self.erros:
            linhas.append(f"  ✖ ERRO   {erro}")
        for aviso in self.avisos:
            linhas.append(f"  ⚠ AVISO  {aviso}")
        linhas.append("")
        linhas.append(
            "RESULTADO: VÁLIDO — os números podem ser citados no README e no relatório."
            if self.valido
            else "RESULTADO: INVÁLIDO — não cite estes números; corrija os erros acima e reexecute."
        )
        return "\n".join(linhas)


def _load_json(caminho: Path, resultado: ValidationResult) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        resultado.erros.append(f"{caminho.name} não é JSON válido: {exc}")
    except OSError as exc:  # pragma: no cover - falha de leitura
        resultado.erros.append(f"{caminho.name} não pôde ser lido: {exc}")
    return None


def _numero_finito(valor: Any) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and math.isfinite(valor)


def _validar_training_results(payload: dict[str, Any], resultado: ValidationResult) -> None:
    status = payload.get("status")
    if status != "ok":
        resultado.erros.append(
            f"training_results.json tem status={status!r}; só 'ok' representa treino concluído."
        )

    for campo in REQUIRED_FIELDS:
        if campo not in payload or payload[campo] in (None, "", [], {}):
            resultado.erros.append(f"training_results.json sem o campo obrigatório: {campo}")

    dtype = str(payload.get("compute_dtype", ""))
    if dtype and dtype not in {"bfloat16", "float16"}:
        resultado.erros.append(
            f"compute_dtype={dtype!r} não é uma precisão de treino válida (bfloat16 ou float16)."
        )

    gpu = payload.get("gpu")
    if isinstance(gpu, dict):
        if not gpu.get("cuda_available"):
            resultado.erros.append("gpu.cuda_available é falso: o treino não rodou em GPU.")
        if gpu.get("bf16_supported") is False and dtype == "bfloat16":
            resultado.erros.append(
                "Incoerência: dtype bfloat16 declarado em GPU sem suporte a BF16."
            )
    elif gpu in (None, "", "n/d"):
        resultado.erros.append("Campo gpu ausente ou vazio; a GPU usada precisa estar registrada.")

    metricas = payload.get("metricas_treino") or {}
    if isinstance(metricas, dict):
        loss = metricas.get("train_loss", metricas.get("loss"))
        if not _numero_finito(loss):
            resultado.erros.append(
                f"metricas_treino sem loss numérica finita (encontrado: {loss!r})."
            )
        else:
            resultado.resumo["train_loss"] = round(float(loss), 4)
        eval_loss = metricas.get("eval_loss")
        if eval_loss is None:
            resultado.avisos.append("Sem eval_loss: o split de validação não foi avaliado.")
        elif _numero_finito(eval_loss):
            resultado.resumo["eval_loss"] = round(float(eval_loss), 4)

    commit = str(payload.get("commit", ""))
    if commit and commit.lower() in {"desconhecido", "unknown"}:
        resultado.erros.append("commit desconhecido: a execução não é rastreável até o código.")

    resultado.resumo.setdefault("status", status)
    resultado.resumo.setdefault("base_model", payload.get("base_model"))
    resultado.resumo.setdefault("commit", commit[:12] or "n/d")
    resultado.resumo.setdefault("compute_dtype", dtype or "n/d")
    if isinstance(gpu, dict):
        resultado.resumo.setdefault("gpu", gpu.get("gpu_name", "n/d"))


def _validar_comparacao(payload: Any, resultado: ValidationResult) -> None:
    sistemas: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("sistemas"), list):
        sistemas = payload["sistemas"]
    elif isinstance(payload, list):
        sistemas = [item for item in payload if isinstance(item, dict)]

    nomes = {str(item.get("sistema", "")) for item in sistemas}
    faltando = EXPECTED_SYSTEMS - nomes
    if faltando:
        resultado.erros.append(
            "comparacao_sistemas.json não cobre "
            f"{', '.join(sorted(faltando))} — a comparação exigida é base × fine_tuned × fine_tuned_rag."
        )
    tamanhos = {item.get("n_exemplos") for item in sistemas if "n_exemplos" in item}
    if len(tamanhos) > 1:
        resultado.erros.append(
            f"Os sistemas foram avaliados em conjuntos de tamanhos diferentes ({tamanhos}); "
            "a comparação precisa usar o mesmo split."
        )
    if nomes:
        resultado.resumo["sistemas_comparados"] = ", ".join(sorted(nomes))


def _varrer_seguranca(diretorio: Path, resultado: ValidationResult) -> None:
    for arquivo in sorted(diretorio.rglob("*")):
        if not arquivo.is_file():
            continue
        nome = arquivo.name.lower()
        if any(padrao in nome for padrao in SECRET_PATTERNS):
            resultado.erros.append(f"Arquivo com aparência de segredo em artifacts/: {arquivo.name}")
        if arquivo.suffix.lower() in WEIGHT_SUFFIXES:
            resultado.erros.append(
                f"Peso de modelo em artifacts/: {arquivo.name}. Pesos ficam no Drive, não no repositório."
            )
        if arquivo.stat().st_size > MAX_ARTIFACT_BYTES:
            resultado.avisos.append(
                f"{arquivo.name} tem {arquivo.stat().st_size / 1024 / 1024:.1f} MB — "
                "grande demais para um artefato de métrica."
            )
        if arquivo.suffix.lower() in {".json", ".txt", ".csv", ".md"}:
            try:
                conteudo = arquivo.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover
                continue
            achado = _SECRET_VALUE.search(conteudo)
            if achado:
                resultado.erros.append(
                    f"{arquivo.name} contém algo com formato de token/credencial "
                    f"({achado.group(0)[:8]}…). Remova antes de versionar."
                )


def validate_colab_results(diretorio: Path | str) -> ValidationResult:
    """Valida a pasta de artefatos do fine-tuning produzida no Colab."""
    pasta = Path(diretorio)
    resultado = ValidationResult(diretorio=pasta)

    if not pasta.exists():
        resultado.erros.append(
            f"Diretório não encontrado: {pasta}. Copie a pasta artifacts/fine_tuning "
            "devolvida pelo Colab (ou extraia o medflow_colab_results.zip) antes de validar."
        )
        return resultado

    for nome in REQUIRED_FILES:
        if not (pasta / nome).exists():
            resultado.erros.append(f"Arquivo obrigatório ausente: {nome}")
    for nome in OPTIONAL_FILES:
        if not (pasta / nome).exists():
            resultado.avisos.append(f"Arquivo opcional ausente: {nome}")

    caminho_treino = pasta / "training_results.json"
    if caminho_treino.exists():
        payload = _load_json(caminho_treino, resultado)
        if isinstance(payload, dict):
            _validar_training_results(payload, resultado)

    caminho_comparacao = pasta / "comparacao_sistemas.json"
    if caminho_comparacao.exists():
        payload = _load_json(caminho_comparacao, resultado)
        if payload is not None:
            _validar_comparacao(payload, resultado)

    caminho_respostas = pasta / "respostas_antes_depois.json"
    if caminho_respostas.exists():
        payload = _load_json(caminho_respostas, resultado)
        if isinstance(payload, dict) and not payload:
            resultado.erros.append("respostas_antes_depois.json está vazio.")

    _varrer_seguranca(pasta, resultado)
    resultado.resumo["arquivos"] = len([p for p in pasta.rglob("*") if p.is_file()])
    return resultado
