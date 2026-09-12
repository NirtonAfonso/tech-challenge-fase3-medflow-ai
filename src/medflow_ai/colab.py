"""Suporte a execução no Google Colab e persistência no Google Drive.

Os notebooks precisam rodar em um runtime Colab novo, com "Executar tudo", sem
depender da máquina local. E como o `/content` do Colab é apagado quando a
sessão encerra, **uma execução só é considerada concluída quando os resultados
estão no Drive**.

Este módulo concentra essa lógica para que os notebooks fiquem curtos e o
comportamento seja testável fora do Colab. Nada aqui importa `google.colab` no
nível do módulo: fora do Colab tudo degrada para o modo local.

Estrutura criada automaticamente em ``/content/drive/MyDrive/MedFlowAI_Fase3``::

    01_preprocessing/{artifacts,datasets,logs}
    02_fine_tuning/{artifacts,adapter,checkpoints,logs,bundles}
    03_rag/{artifacts,indexes,logs}
    04_structured_data/{artifacts,database,logs}
    05_demo/{artifacts,logs,exports}
    shared/{manifests,configs,run_metadata}

Segredos (``.env``, tokens, credenciais) **nunca** são copiados para o Drive.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "DRIVE_ROOT",
    "NOTEBOOK_LAYOUT",
    "REPO_URL",
    "REPO_BRANCH",
    "in_colab",
    "mount_drive",
    "resolve_output_root",
    "ensure_structure",
    "persist",
    "run_metadata",
    "write_run_metadata",
    "PersistenceReport",
    "summarize",
    "SECRET_PATTERNS",
]

REPO_URL = "https://github.com/NirtonAfonso/tech-challenge-fase3-medflow-ai.git"
REPO_BRANCH = "develop"

DRIVE_ROOT = "/content/drive/MyDrive/MedFlowAI_Fase3"

#: Subpastas de cada notebook. A chave é o identificador usado nos notebooks.
NOTEBOOK_LAYOUT: dict[str, tuple[str, ...]] = {
    "01_preprocessing": ("artifacts", "datasets", "logs"),
    "02_fine_tuning": ("artifacts", "adapter", "checkpoints", "logs", "bundles"),
    "03_rag": ("artifacts", "indexes", "logs"),
    "04_structured_data": ("artifacts", "database", "logs"),
    "05_demo": ("artifacts", "logs", "exports"),
    "shared": ("manifests", "configs", "run_metadata"),
}

#: Nomes que nunca podem ser copiados para o Drive nem entrar em um bundle.
SECRET_PATTERNS: tuple[str, ...] = (
    ".env", ".env.local", "id_rsa", "id_ed25519", "credentials.json", "token.json",
    "client_secret", ".netrc", ".git-credentials", "kaggle.json",
)

#: Extensões de peso de modelo — jamais vão para bundles de resultado.
WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".msgpack",
)


# --------------------------------------------------------------------- ambiente
def in_colab() -> bool:
    """Detecta o Google Colab sem quebrar a execução local."""
    if os.environ.get("MEDFLOW_FORCE_COLAB") == "1":
        return True
    if os.environ.get("MEDFLOW_FORCE_COLAB") == "0":
        return False
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def mount_drive(mount_point: str = "/content/drive", *, force_remount: bool = False) -> tuple[bool, str]:
    """Monta o Google Drive quando estiver no Colab.

    Returns:
        ``(montado, mensagem)``. Fora do Colab devolve ``(False, motivo)`` sem
        levantar exceção — os notebooks continuam funcionando localmente.
    """
    if not in_colab():
        return False, "Fora do Google Colab: o Drive não é montado; saídas ficam no repositório local."

    if Path(mount_point, "MyDrive").exists() and not force_remount:
        return True, f"Google Drive já montado em {mount_point}."

    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(mount_point, force_remount=force_remount)
    except Exception as exc:  # noqa: BLE001 - o notebook precisa seguir e avisar
        return False, (
            f"ATENÇÃO: falha ao montar o Google Drive ({exc}). "
            "Os resultados ficarão apenas em /content e serão PERDIDOS quando a sessão encerrar."
        )
    return True, f"Google Drive montado em {mount_point}."


def resolve_output_root(notebook: str, *, drive_root: str | None = None) -> Path:
    """Diretório-raiz de saída deste notebook.

    No Colab com Drive montado, ``<DRIVE_ROOT>/<notebook>``. Caso contrário,
    ``<repo>/artifacts/colab/<notebook>``, para que a execução local produza a
    mesma árvore e os notebooks não precisem de dois caminhos de código.
    """
    if notebook not in NOTEBOOK_LAYOUT:
        raise KeyError(
            f"Notebook desconhecido: {notebook!r}. Use um de: {sorted(NOTEBOOK_LAYOUT)}"
        )

    root = Path(drive_root or DRIVE_ROOT)
    if in_colab() and root.parent.exists():
        return root / notebook

    from medflow_ai.config import get_settings

    return get_settings().artifacts_dir / "colab" / notebook


def ensure_structure(
    notebook: str | None = None, *, drive_root: str | None = None
) -> dict[str, Path]:
    """Cria a estrutura de pastas do notebook (e de ``shared``) com ``exist_ok``.

    Args:
        notebook: identificador do notebook; ``None`` cria a árvore completa.

    Returns:
        Mapa ``nome_da_subpasta -> caminho`` (mais ``_root`` e as pastas de
        ``shared``), pronto para uso direto no notebook.
    """
    alvos = list(NOTEBOOK_LAYOUT) if notebook is None else [notebook, "shared"]
    criados: dict[str, Path] = {}

    for nome in alvos:
        raiz = resolve_output_root(nome, drive_root=drive_root)
        raiz.mkdir(parents=True, exist_ok=True)
        if nome == notebook or notebook is None:
            criados["_root"] = raiz if notebook else criados.get("_root", raiz)
        for sub in NOTEBOOK_LAYOUT[nome]:
            caminho = raiz / sub
            caminho.mkdir(parents=True, exist_ok=True)
            chave = sub if nome == notebook else f"{nome}/{sub}"
            criados[chave] = caminho
    return criados


# ------------------------------------------------------------------ metadados
def git_info(repo_dir: Path | None = None) -> dict[str, str]:
    """Branch e commit SHA da cópia local, para rastreabilidade da execução."""
    cwd = str(repo_dir) if repo_dir else None

    def _run(*args: str) -> str:
        try:
            return subprocess.run(
                args, cwd=cwd, capture_output=True, text=True, timeout=15, check=False
            ).stdout.strip() or "desconhecido"
        except Exception:  # noqa: BLE001 - ambiente sem git
            return "desconhecido"

    return {
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _run("git", "rev-parse", "HEAD"),
        "commit_short": _run("git", "rev-parse", "--short", "HEAD"),
    }


def package_versions(pacotes: Sequence[str] = ()) -> dict[str, str]:
    """Versões instaladas dos pacotes informados (``ausente`` quando não houver)."""
    from importlib import import_module

    padrao = pacotes or (
        "numpy", "langchain", "langchain_core", "langgraph", "torch", "transformers",
        "peft", "trl", "datasets", "accelerate", "bitsandbytes", "pandas", "matplotlib",
    )
    versoes: dict[str, str] = {}
    for nome in padrao:
        try:
            versoes[nome] = getattr(import_module(nome), "__version__", "desconhecida")
        except Exception:  # noqa: BLE001 - pacote opcional ausente
            versoes[nome] = "ausente"
    return versoes


def run_metadata(notebook: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Metadados de rastreabilidade de uma execução (vão para ``shared/run_metadata``)."""
    from medflow_ai.fine_tuning.precision import describe_gpu

    payload: dict[str, Any] = {
        "notebook": notebook,
        "executado_em": datetime.now(UTC).isoformat(),
        "ambiente": "google_colab" if in_colab() else "local",
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "git": git_info(),
        "gpu": describe_gpu(),
        "versoes": package_versions(),
        "drive_root": DRIVE_ROOT if in_colab() else None,
    }
    payload.update(extra or {})
    return payload


def write_run_metadata(
    notebook: str, *, extra: dict[str, Any] | None = None, drive_root: str | None = None
) -> Path:
    """Grava os metadados da execução em ``shared/run_metadata``."""
    pastas = ensure_structure(notebook, drive_root=drive_root)
    destino = pastas["shared/run_metadata"]
    dados = run_metadata(notebook, extra=extra)
    carimbo = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    caminho = destino / f"{carimbo}_{notebook}_{dados['git']['commit_short']}.json"
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    (destino / f"latest_{notebook}.json").write_text(
        json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return caminho


# ----------------------------------------------------------------- persistência
def _is_secret(path: Path) -> bool:
    nome = path.name.lower()
    return any(padrao in nome for padrao in SECRET_PATTERNS)


@dataclass
class PersistenceReport:
    """Resumo do que foi (ou não) persistido."""

    destino: Path
    copiados: list[Path] = field(default_factory=list)
    ignorados: list[str] = field(default_factory=list)
    ausentes: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.copiados if p.exists())

    def render(self) -> str:
        linhas = [f"Persistido em: {self.destino}"]
        for caminho in self.copiados:
            tamanho = caminho.stat().st_size / 1024 if caminho.exists() else 0
            linhas.append(f"  ✔ {caminho.name}  ({tamanho:,.1f} KB)")
        for nome in self.ausentes:
            linhas.append(f"  · não gerado nesta execução: {nome}")
        for nome in self.ignorados:
            linhas.append(f"  ⛔ ignorado por política de segredo: {nome}")
        if not self.copiados:
            linhas.append("  (nenhum arquivo copiado)")
        else:
            linhas.append(f"  total: {len(self.copiados)} arquivo(s), {self.total_bytes/1024:,.1f} KB")
        return "\n".join(linhas)


def persist(
    origens: Iterable[Path | str],
    destino: Path | str,
    *,
    allow_weights: bool = False,
) -> PersistenceReport:
    """Copia arquivos e diretórios para o Drive (ou para a saída local equivalente).

    Args:
        origens: caminhos de arquivo ou diretório. Caminhos inexistentes são
            registrados como "não gerado nesta execução", sem falhar — nem todo
            notebook produz todos os artefatos em toda execução.
        destino: pasta de destino; criada se não existir.
        allow_weights: por padrão, pesos de modelo (``.safetensors``, ``.bin``…)
            NÃO são copiados. O notebook 02 usa ``True`` apenas para o adapter.
    """
    alvo = Path(destino)
    alvo.mkdir(parents=True, exist_ok=True)
    relatorio = PersistenceReport(destino=alvo)

    for origem in origens:
        caminho = Path(origem)
        if not caminho.exists():
            relatorio.ausentes.append(str(caminho))
            continue
        if _is_secret(caminho):
            relatorio.ignorados.append(caminho.name)
            continue

        if caminho.is_dir():
            sub_destino = alvo / caminho.name
            for arquivo in sorted(caminho.rglob("*")):
                if not arquivo.is_file() or _is_secret(arquivo):
                    if arquivo.is_file():
                        relatorio.ignorados.append(arquivo.name)
                    continue
                if not allow_weights and arquivo.suffix.lower() in WEIGHT_SUFFIXES:
                    relatorio.ignorados.append(arquivo.name)
                    continue
                relativo = arquivo.relative_to(caminho)
                final = sub_destino / relativo
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(arquivo, final)
                relatorio.copiados.append(final)
        else:
            if not allow_weights and caminho.suffix.lower() in WEIGHT_SUFFIXES:
                relatorio.ignorados.append(caminho.name)
                continue
            final = alvo / caminho.name
            shutil.copy2(caminho, final)
            relatorio.copiados.append(final)

    return relatorio


def summarize(relatorios: Sequence[PersistenceReport], *, titulo: str = "RESUMO DA PERSISTÊNCIA") -> str:
    """Resumo final impresso ao fim de cada notebook."""
    linhas = ["=" * 78, titulo, "=" * 78]
    total = 0
    for relatorio in relatorios:
        linhas.append(relatorio.render())
        linhas.append("")
        total += len(relatorio.copiados)
    linhas.append(f"TOTAL: {total} arquivo(s) persistido(s).")
    if not in_colab():
        linhas.append(
            "Execução LOCAL: os caminhos acima estão no repositório, não no Google Drive."
        )
    return "\n".join(linhas)
