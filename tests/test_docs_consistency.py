"""Impede que a documentação e o manifesto do dataset divirjam.

Números de dataset aparecem em sete arquivos. Sem esta trava, uma mudança no
pipeline deixa o README afirmando algo que o manifesto contradiz — exatamente o
tipo de inconsistência que desqualifica a evidência de uma entrega.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from medflow_ai.config import get_settings

DOCS = ("README.md", "docs/REPORT.md", "docs/FINE_TUNING.md", "docs/DATASETS.md",
        "docs/REQUIREMENTS_CHECKLIST.md", "docs/DECISIONS.md", "docs/ANONYMIZATION.md",
        "docs/EVALUATION_PLAN.md", "docs/COLAB_RUNBOOK.md")


@pytest.fixture(scope="module")
def manifesto() -> dict:
    caminho = get_settings().project_root / "data" / "processed" / "sft" / "manifest.json"
    if not caminho.exists():
        pytest.skip("dataset ainda não construído")
    return json.loads(caminho.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def textos() -> dict[str, str]:
    raiz = get_settings().project_root
    return {
        nome: (raiz / nome).read_text(encoding="utf-8")
        for nome in DOCS
        if (raiz / nome).exists()
    }


def test_total_de_exemplos_declarado_bate_com_o_manifesto(manifesto, textos) -> None:
    """Só valida menções ao TOTAL do dataset, não a subconjuntos.

    Frases como "22 exemplos de documentos held-out" descrevem o split de teste
    e são legítimas; o que não pode divergir é o total declarado.
    """
    total = manifesto["estatisticas_curadoria"]["finais"]
    padroes = (
        re.compile(r"(?:dataset|conjunto)\s+(?:SFT\s+)?de\s+\*{0,2}(\d+)\*{0,2}\s+exemplos", re.I),
        re.compile(r"\|\s*\*{0,2}Total\*{0,2}\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|"),
        re.compile(r"\*{0,2}(\d+)\*{0,2}\s+exemplos\s+em\s+\d+\s+famílias", re.I),
        re.compile(r"\*{0,2}(\d+)\s+exemplos\*{0,2}\)?\s*\.?\s*(?:É|Suficiente|é pouco)", re.I),
    )
    encontrou = False
    for nome, texto in textos.items():
        for padrao in padroes:
            for valor in padrao.findall(texto):
                encontrou = True
                assert int(valor) == total, (
                    f"{nome} declara total de {valor} exemplos; manifesto diz {total}"
                )
    assert encontrou, "nenhum documento declara o total do dataset"


def test_splits_declarados_batem_com_o_manifesto(manifesto, textos) -> None:
    splits = manifesto["splits"]
    esperado = f"{splits['train']} treino / {splits['validation']} validação / {splits['test']} teste"
    padrao = re.compile(r"(\d+)\s+treino\s*/\s*(\d+)\s+valida\w*\s*/\s*(\d+)\s+teste")
    encontrou = False
    for nome, texto in textos.items():
        for treino, validacao, teste in padrao.findall(texto):
            encontrou = True
            assert (int(treino), int(validacao), int(teste)) == (
                splits["train"], splits["validation"], splits["test"]
            ), f"{nome} declara split diferente do manifesto ({esperado})"
    assert encontrou, "nenhum documento declara os splits — a evidência precisa estar visível"


def test_contagem_de_pii_removida_bate_com_o_manifesto(manifesto, textos) -> None:
    total = sum(manifesto["anonimizacao"]["counts_by_kind"].values())
    classes = len(manifesto["anonimizacao"]["counts_by_kind"])
    padrao = re.compile(r"\*{0,2}(\d+)\s+identificadores\*{0,2}\s+em\s+\*{0,2}(\d+)\s+classes")
    for nome, texto in textos.items():
        for quantidade, tipos in padrao.findall(texto):
            assert (int(quantidade), int(tipos)) == (total, classes), (
                f"{nome} declara {quantidade} identificadores em {tipos} classes; "
                f"manifesto diz {total} em {classes}"
            )


def test_familias_declaradas_existem_no_manifesto(manifesto, textos) -> None:
    familias = set(manifesto["estatisticas_curadoria"]["por_familia"])
    padrao = re.compile(r"`(protocolo_qa|faq_medico|modelo_laudo|modelo_receita|procedimento|"
                        r"seguranca|contexto_paciente|laudo_preenchido)`")
    for nome, texto in textos.items():
        for familia in set(padrao.findall(texto)):
            assert familia in familias, f"{nome} cita família inexistente: {familia}"


def test_documentos_held_out_declarados_batem(manifesto, textos) -> None:
    held_out = set(manifesto["held_out_documents"])
    for nome, texto in textos.items():
        for doc_id in re.findall(r"`(PRO[TC]-[A-Z]{3}-\d{3})`", texto):
            if "held-out" in texto or "reservad" in texto:
                continue  # menção genérica; a checagem forte é a de baixo
    # Todo documento reservado precisa ser citado em pelo menos um documento.
    juntos = "\n".join(textos.values())
    for doc_id in held_out:
        assert doc_id in juntos, f"documento held-out {doc_id} não aparece na documentação"


def test_material_bruto_do_curso_esta_fora_da_arvore() -> None:
    raiz = get_settings().project_root
    assert not (raiz / "docs" / "Fase 3 - Generative AI").exists()
    gitignore = (raiz / ".gitignore").read_text(encoding="utf-8")
    assert "Fase 3 - Generative AI" in gitignore
