"""O dataset de fine-tuning não pode depender do estado do banco.

Defeito corrigido: a família ``laudo_preenchido`` era gerada lendo o
``hospital.db`` existente. Como esse banco é mutável (o usuário pode reconstruí-lo
com 8 ou 40 pacientes, e o notebook 04 o recria), o **mesmo comando** produzia
datasets diferentes conforme o que tivesse rodado antes — quebrando o manifesto,
os fingerprints e o split.

Estes testes travam a invariância em quatro estados do ambiente.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from medflow_ai.config import get_settings, reload_settings
from medflow_ai.database.ingest import build_synthetic_database
from medflow_ai.fine_tuning.dataset import (
    HELD_OUT_DOCUMENTS,
    build_sft_dataset,
    generate_examples,
)


def _assinatura(tmp_dir: Path) -> dict:
    """Descreve o dataset de forma comparável (sem o timestamp do manifesto)."""
    splits, stats, caminhos = build_sft_dataset(output_dir=tmp_dir, seed=42)
    manifesto = json.loads(caminhos["manifest"].read_text(encoding="utf-8"))
    return {
        "total": stats.finais,
        "familias": stats.por_familia,
        "documentos": stats.por_documento,
        "splits": {nome: len(itens) for nome, itens in splits.items()},
        "fingerprints": manifesto["fingerprints"],
        "held_out": manifesto["held_out_documents"],
        "documentos_por_split": manifesto["documentos_por_split"],
        "ids": {nome: [item.id for item in itens] for nome, itens in splits.items()},
        "anonimizacao": manifesto["anonimizacao"]["counts_by_kind"],
    }


@pytest.fixture()
def banco_isolado(tmp_path, monkeypatch):
    """Aponta MEDFLOW_DB_PATH para um caminho temporário exclusivo do teste."""
    destino = tmp_path / "estado" / "hospital.db"
    monkeypatch.setenv("MEDFLOW_DB_PATH", str(destino))
    reload_settings()
    yield destino
    monkeypatch.delenv("MEDFLOW_DB_PATH", raising=False)
    reload_settings()


def test_dataset_identico_sem_db_e_com_8_e_com_40_pacientes(tmp_path, banco_isolado) -> None:
    assert not banco_isolado.exists(), "o cenário 1 exige ausência de banco"
    sem_db = _assinatura(tmp_path / "sem_db")

    build_synthetic_database(banco_isolado, n_patients=8, seed=7)
    com_8 = _assinatura(tmp_path / "com_8")

    build_synthetic_database(banco_isolado, n_patients=40, seed=99)
    com_40 = _assinatura(tmp_path / "com_40")

    assert sem_db == com_8 == com_40


def test_geracao_e_estavel_entre_chamadas() -> None:
    primeira = [(item.id, item.fingerprint) for item in generate_examples()]
    segunda = [(item.id, item.fingerprint) for item in generate_examples()]
    assert primeira == segunda


def test_familia_de_laudo_tem_tamanho_fixo(banco_isolado) -> None:
    from medflow_ai.fine_tuning.dataset import SFT_FIXTURE_PATIENTS

    laudos = [item for item in generate_examples() if item.familia == "laudo_preenchido"]
    assert len(laudos) == SFT_FIXTURE_PATIENTS

    build_synthetic_database(banco_isolado, n_patients=3, seed=1)
    apos_db_pequeno = [item for item in generate_examples() if item.familia == "laudo_preenchido"]
    assert [item.id for item in apos_db_pequeno] == [item.id for item in laudos]


def test_pseudonimo_do_exemplo_nao_depende_do_salt_do_ambiente(monkeypatch) -> None:
    """O salt de runtime não pode vazar para dentro do dataset versionado."""
    def laudo_output() -> str:
        return next(
            item.output for item in generate_examples() if item.familia == "laudo_preenchido"
        )

    monkeypatch.setenv("MEDFLOW_PSEUDONYM_SALT", "salt-A")
    reload_settings()
    com_salt_a = laudo_output()

    monkeypatch.setenv("MEDFLOW_PSEUDONYM_SALT", "salt-B-completamente-diferente")
    reload_settings()
    com_salt_b = laudo_output()

    monkeypatch.delenv("MEDFLOW_PSEUDONYM_SALT", raising=False)
    reload_settings()

    assert com_salt_a == com_salt_b


def test_jsonl_versionado_bate_com_o_reproduzido(tmp_path) -> None:
    """O que está no repositório precisa ser exatamente o que o pipeline gera."""
    versionado_dir = get_settings().project_root / "data" / "processed" / "sft"
    if not (versionado_dir / "manifest.json").exists():
        pytest.skip("dataset versionado ausente; rode 'python -m medflow_ai.cli build-dataset'")

    reproduzido = _assinatura(tmp_path / "reproduzido")
    versionado = json.loads((versionado_dir / "manifest.json").read_text(encoding="utf-8"))

    assert reproduzido["fingerprints"] == versionado["fingerprints"]
    assert reproduzido["splits"] == versionado["splits"]
    assert reproduzido["held_out"] == versionado["held_out_documents"]

    for nome in ("train", "validation", "test"):
        linhas = (versionado_dir / f"{nome}.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(linhas) == reproduzido["splits"][nome]


def test_split_mantem_documentos_reservados_fora_do_treino(tmp_path) -> None:
    splits, _, _ = build_sft_dataset(output_dir=tmp_path, seed=42, write=True)
    treino = {item.doc_id for item in splits["train"]} | {item.doc_id for item in splits["validation"]}
    teste = {item.doc_id for item in splits["test"]}
    assert teste == set(HELD_OUT_DOCUMENTS)
    assert not (treino & teste)


def test_fingerprints_nao_se_repetem_entre_splits(tmp_path) -> None:
    splits, _, _ = build_sft_dataset(output_dir=tmp_path, seed=42, write=True)
    conjuntos = {nome: {item.fingerprint for item in itens} for nome, itens in splits.items()}
    assert not (conjuntos["train"] & conjuntos["test"])
    assert not (conjuntos["validation"] & conjuntos["test"])
    assert not (conjuntos["train"] & conjuntos["validation"])


def test_familias_especiais_ficam_no_treino_e_estao_documentadas(tmp_path) -> None:
    """Segurança, contexto de paciente e laudo não pertencem a documento held-out.

    Consequência declarada: essas famílias entram em treino/validação e NÃO são
    avaliadas pelo split de teste, que contém apenas seções de três protocolos
    reservados. A avaliação do comportamento de recusa é feita à parte, pelas
    sondas de segurança de ``fine_tuning.evaluate.SAFETY_PROBES``.
    """
    splits, _, _ = build_sft_dataset(output_dir=tmp_path, seed=42, write=True)
    familias_teste = {item.familia for item in splits["test"]}
    assert familias_teste <= {"protocolo_qa", "procedimento"}

    familias_treino = {item.familia for item in splits["train"]} | {
        item.familia for item in splits["validation"]
    }
    for especial in ("seguranca", "contexto_paciente", "laudo_preenchido"):
        assert especial in familias_treino


def test_construcao_respeita_medflow_data_dir(tmp_path, monkeypatch) -> None:
    """Regressão: `cli build-dataset` escrevia na raiz do repo, ignorando o sandbox.

    O conftest promete que nenhum teste toca os artefatos versionados, mas o
    caminho do dataset era derivado de `project_root` em vez de `MEDFLOW_DATA_DIR`.
    Consequência real: rodar `pytest tests/test_cli.py` sujava
    `data/processed/sft/manifest.json` no repositório.
    """
    from medflow_ai.cli import main
    from medflow_ai.fine_tuning.dataset import default_sft_dir

    monkeypatch.setenv("MEDFLOW_DATA_DIR", str(tmp_path / "dados"))
    reload_settings()

    esperado = tmp_path / "dados" / "processed" / "sft"
    assert default_sft_dir() == esperado

    assert main(["build-dataset", "--seed", "42"]) == 0
    assert (esperado / "manifest.json").exists()
    assert (esperado / "train.jsonl").exists()

    monkeypatch.delenv("MEDFLOW_DATA_DIR", raising=False)
    reload_settings()
    assert default_sft_dir() != esperado
