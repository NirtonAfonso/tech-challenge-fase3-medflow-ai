"""Bundle leve com os resultados do fine-tuning executado no Colab.

O usuário roda o notebook 02 no Colab e precisa devolver as evidências para
avaliação. Copiar o adapter inteiro é desnecessário e arriscado; este módulo
monta um ZIP pequeno **apenas com métricas e metadados**, e recusa-se a incluir
pesos, checkpoints, `.env`, tokens ou credenciais.

A validação é feita antes de escrever o arquivo: se algo proibido entrar na
lista, a montagem falha em vez de gerar um pacote inseguro.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from medflow_ai.colab import SECRET_PATTERNS, WEIGHT_SUFFIXES, git_info

__all__ = ["BUNDLE_NAME", "BundleReport", "build_results_bundle", "inspect_bundle"]

BUNDLE_NAME = "medflow_colab_results.zip"

#: Extensões permitidas dentro do bundle. Qualquer outra é rejeitada.
ALLOWED_SUFFIXES: frozenset[str] = frozenset({".json", ".csv", ".txt", ".png", ".md", ".jsonl"})

#: Tamanho máximo por arquivo (evita que um peso renomeado entre disfarçado).
MAX_FILE_BYTES = 8 * 1024 * 1024


class BundleSecurityError(RuntimeError):
    """Um arquivo proibido foi oferecido ao bundle."""


@dataclass
class BundleReport:
    """Resultado da montagem do bundle."""

    caminho: Path
    incluidos: list[str] = field(default_factory=list)
    ausentes: list[str] = field(default_factory=list)
    tamanho_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "caminho": str(self.caminho),
            "incluidos": list(self.incluidos),
            "ausentes": list(self.ausentes),
            "tamanho_bytes": self.tamanho_bytes,
            "tamanho_kb": round(self.tamanho_bytes / 1024, 1),
        }

    def render(self) -> str:
        linhas = [f"Bundle: {self.caminho} ({self.tamanho_bytes / 1024:,.1f} KB)"]
        linhas += [f"  ✔ {nome}" for nome in self.incluidos]
        linhas += [f"  · não gerado nesta execução: {nome}" for nome in self.ausentes]
        return "\n".join(linhas)


def _reject_if_unsafe(caminho: Path) -> None:
    nome = caminho.name.lower()
    if any(padrao in nome for padrao in SECRET_PATTERNS):
        raise BundleSecurityError(f"Arquivo com aparência de segredo recusado: {caminho.name}")
    if caminho.suffix.lower() in WEIGHT_SUFFIXES:
        raise BundleSecurityError(f"Peso de modelo recusado no bundle: {caminho.name}")
    if caminho.suffix.lower() not in ALLOWED_SUFFIXES:
        raise BundleSecurityError(
            f"Extensão não permitida no bundle: {caminho.name} "
            f"(permitidas: {', '.join(sorted(ALLOWED_SUFFIXES))})"
        )
    if caminho.stat().st_size > MAX_FILE_BYTES:
        raise BundleSecurityError(
            f"{caminho.name} tem {caminho.stat().st_size / 1024 / 1024:.1f} MB, acima do limite "
            f"de {MAX_FILE_BYTES / 1024 / 1024:.0f} MB para um bundle de métricas."
        )


def build_results_bundle(
    origem: Path | str,
    destino: Path | str,
    *,
    extras: Iterable[Path | str] = (),
    nome: str = BUNDLE_NAME,
) -> BundleReport:
    """Monta o ZIP de resultados a partir do diretório de artefatos do fine-tuning.

    Args:
        origem: diretório com os artefatos (ex.: ``artifacts/fine_tuning``).
        destino: pasta onde o ZIP será escrito (no Colab, a pasta ``bundles`` do Drive).
        extras: arquivos adicionais (por exemplo, o manifesto do dataset).
    """
    origem_dir = Path(origem)
    destino_dir = Path(destino)
    destino_dir.mkdir(parents=True, exist_ok=True)
    caminho_zip = destino_dir / nome

    esperados = [
        "training_results.json",
        "environment.json",
        "loss_curve.png",
        "comparacao_sistemas.json",
        "respostas_antes_depois.json",
        "training_config.json",
        "generation_comparison.json",
    ]

    candidatos: list[Path] = []
    ausentes: list[str] = []
    for nome_arquivo in esperados:
        caminho = origem_dir / nome_arquivo
        (candidatos if caminho.exists() else ausentes).append(
            caminho if caminho.exists() else nome_arquivo  # type: ignore[arg-type]
        )
    for extra in extras:
        caminho = Path(extra)
        (candidatos if caminho.exists() else ausentes).append(
            caminho if caminho.exists() else str(extra)  # type: ignore[arg-type]
        )

    arquivos = [c for c in candidatos if isinstance(c, Path)]
    for arquivo in arquivos:
        _reject_if_unsafe(arquivo)

    procedencia = {
        "gerado_em": datetime.now(UTC).isoformat(),
        "git": git_info(),
        "arquivos": [arquivo.name for arquivo in arquivos],
        "ausentes": [str(nome_arquivo) for nome_arquivo in ausentes],
        "aviso": (
            "Bundle de MÉTRICAS. Não contém adapter, checkpoints, pesos, tokens nem .env. "
            "O adapter treinado permanece no Google Drive."
        ),
    }

    with zipfile.ZipFile(caminho_zip, "w", compression=zipfile.ZIP_DEFLATED) as pacote:
        for arquivo in arquivos:
            pacote.write(arquivo, arcname=arquivo.name)
        pacote.writestr("BUNDLE_INFO.json", json.dumps(procedencia, ensure_ascii=False, indent=2))

    return BundleReport(
        caminho=caminho_zip,
        incluidos=[arquivo.name for arquivo in arquivos] + ["BUNDLE_INFO.json"],
        ausentes=[str(nome_arquivo) for nome_arquivo in ausentes],
        tamanho_bytes=caminho_zip.stat().st_size,
    )


def inspect_bundle(caminho: Path | str) -> dict[str, Any]:
    """Reinspeciona um ZIP já montado: confirma que nada proibido entrou."""
    zip_path = Path(caminho)
    with zipfile.ZipFile(zip_path) as pacote:
        nomes = pacote.namelist()

    problemas: list[str] = []
    for nome in nomes:
        base = Path(nome)
        if any(padrao in base.name.lower() for padrao in SECRET_PATTERNS):
            problemas.append(f"possível segredo: {nome}")
        if base.suffix.lower() in WEIGHT_SUFFIXES:
            problemas.append(f"peso de modelo: {nome}")
        if base.suffix.lower() not in ALLOWED_SUFFIXES and base.name != "BUNDLE_INFO.json":
            problemas.append(f"extensão não permitida: {nome}")

    return {
        "caminho": str(zip_path),
        "arquivos": nomes,
        "tamanho_kb": round(zip_path.stat().st_size / 1024, 1),
        "seguro": not problemas,
        "problemas": problemas,
    }
