"""Verificações de prontidão dos notebooks para um Colab novo.

Estes testes **não** provam execução no Colab — só o usuário pode fazer isso.
O que se verifica é o que é verificável estaticamente e que, quando quebra,
quebra o "Executar tudo" do usuário: JSON válido, sintaxe das células, bootstrap
presente, clone com branch explícita, dependências corretas por notebook, badge
do Colab e persistência no Drive.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from medflow_ai.colab import NOTEBOOK_LAYOUT, REPO_BRANCH
from medflow_ai.config import get_settings

NOTEBOOKS_CPU = (
    "01_data_preprocessing.ipynb",
    "03_rag_pipeline.ipynb",
    "04_structured_patient_data.ipynb",
    "05_medflow_full_demo.ipynb",
)
NOTEBOOK_GPU = "02_fine_tuning_qlora.ipynb"
TODOS = (*NOTEBOOKS_CPU[:1], NOTEBOOK_GPU, *NOTEBOOKS_CPU[1:])

NOTEBOOK_IDS = {
    "01_data_preprocessing.ipynb": "01_preprocessing",
    "02_fine_tuning_qlora.ipynb": "02_fine_tuning",
    "03_rag_pipeline.ipynb": "03_rag",
    "04_structured_patient_data.ipynb": "04_structured_data",
    "05_medflow_full_demo.ipynb": "05_demo",
}


def _dir() -> Path:
    return get_settings().project_root / "notebooks"


def _load(nome: str) -> dict:
    return json.loads((_dir() / nome).read_text(encoding="utf-8"))


def _codigo(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]


# ------------------------------------------------------------------ estrutura
def test_existem_exatamente_cinco_notebooks() -> None:
    encontrados = sorted(p.name for p in _dir().glob("*.ipynb"))
    assert encontrados == sorted(TODOS)


@pytest.mark.parametrize("nome", TODOS)
def test_json_valido_e_formato_nbformat4(nome: str) -> None:
    nb = _load(nome)
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook vazio"
    for celula in nb["cells"]:
        assert celula["cell_type"] in {"code", "markdown"}
        assert isinstance(celula["source"], list)


@pytest.mark.parametrize("nome", TODOS)
def test_todas_as_celulas_de_codigo_tem_sintaxe_valida(nome: str) -> None:
    for indice, fonte in enumerate(_codigo(_load(nome))):
        try:
            ast.parse(fonte)
        except SyntaxError as erro:  # pragma: no cover - falha é o próprio relatório
            pytest.fail(f"{nome} célula de código {indice}: {erro}")


@pytest.mark.parametrize("nome", TODOS)
def test_notebooks_nao_vem_com_saida_salva(nome: str) -> None:
    """Saída salva incha o diff e pode conter dados de execuções antigas."""
    for celula in _load(nome)["cells"]:
        if celula["cell_type"] == "code":
            assert celula.get("outputs", []) == []
            assert celula.get("execution_count") is None


# ------------------------------------------------------------------ bootstrap
@pytest.mark.parametrize("nome", TODOS)
def test_primeira_celula_de_codigo_e_o_bootstrap(nome: str) -> None:
    primeira = _codigo(_load(nome))[0]
    assert "Bootstrap" in primeira
    assert "IN_COLAB" in primeira
    assert "ensure_structure" in primeira
    assert "os.chdir" in primeira


@pytest.mark.parametrize("nome", TODOS)
def test_clone_sempre_declara_a_branch(nome: str) -> None:
    """Clonar a default branch implicitamente traria código diferente do avaliado."""
    for fonte in _codigo(_load(nome)):
        if "git" in fonte and "clone" in fonte:
            assert "--branch" in fonte, f"{nome}: clone sem --branch explícita"
            assert f'REPO_BRANCH = "{REPO_BRANCH}"' in fonte or "REPO_BRANCH" in fonte
            assert "--depth" in fonte


@pytest.mark.parametrize("nome", TODOS)
def test_bootstrap_monta_o_drive_no_colab(nome: str) -> None:
    primeira = _codigo(_load(nome))[0]
    assert "google.colab" in primeira and "drive.mount" in primeira
    assert "/content/drive" in primeira


@pytest.mark.parametrize("nome", TODOS)
def test_bootstrap_imprime_branch_e_commit(nome: str) -> None:
    primeira = _codigo(_load(nome))[0]
    assert "git_info" in primeira
    assert "Commit" in primeira and "Branch" in primeira


@pytest.mark.parametrize("nome", TODOS)
def test_nao_exige_descomentar_nada(nome: str) -> None:
    """'Descomente esta linha' quebra o Executar tudo."""
    for fonte in _codigo(_load(nome)):
        assert "descomente" not in fonte.lower()
        assert "# !pip" not in fonte and "# !git" not in fonte


@pytest.mark.parametrize("nome", TODOS)
def test_nao_usa_magics_de_shell_frageis(nome: str) -> None:
    """`!comando` e `%cd` falham fora do IPython; o bootstrap usa subprocess."""
    for fonte in _codigo(_load(nome)):
        for linha in fonte.splitlines():
            despido = linha.strip()
            assert not despido.startswith("!"), f"{nome}: shell magic '{despido[:40]}'"
            assert not despido.startswith("%cd"), f"{nome}: %cd em vez de os.chdir"


# --------------------------------------------------------------- dependências
@pytest.mark.parametrize("nome", NOTEBOOKS_CPU)
def test_notebooks_de_cpu_instalam_dependencias_de_cpu(nome: str) -> None:
    primeira = _codigo(_load(nome))[0]
    assert "requirements-colab.txt" in primeira
    assert "bitsandbytes" not in primeira, f"{nome} não deve instalar dependência de GPU"


def test_notebook_de_gpu_instala_dependencias_de_treinamento() -> None:
    primeira = _codigo(_load(NOTEBOOK_GPU))[0]
    for pacote in ("transformers", "peft", "trl", "bitsandbytes", "datasets", "accelerate"):
        assert pacote in primeira, f"faltou {pacote} no bootstrap do notebook de GPU"
    assert "_tem_torch_cuda" in primeira, "deve evitar reinstalar torch quando o Colab já tem CUDA"


def test_requirements_colab_existe_e_referencia_o_base() -> None:
    caminho = get_settings().project_root / "requirements-colab.txt"
    conteudo = caminho.read_text(encoding="utf-8")
    assert "-r requirements.txt" in conteudo
    assert "pandas" in conteudo and "matplotlib" in conteudo
    assert "torch" not in conteudo, "dependência de GPU não pertence ao requirements de CPU"


# ---------------------------------------------------------------- Drive/saída
@pytest.mark.parametrize("nome", TODOS)
def test_cada_notebook_usa_a_sua_pasta_do_drive(nome: str) -> None:
    identificador = NOTEBOOK_IDS[nome]
    primeira = _codigo(_load(nome))[0]
    assert f'NOTEBOOK_ID = "{identificador}"' in primeira
    assert identificador in NOTEBOOK_LAYOUT


@pytest.mark.parametrize("nome", TODOS)
def test_cada_notebook_persiste_e_resume_no_final(nome: str) -> None:
    fontes = _codigo(_load(nome))
    juntos = "\n".join(fontes)
    assert "persist(" in juntos, f"{nome} não persiste nada"
    assert "summarize(" in juntos, f"{nome} não imprime resumo dos caminhos persistidos"


@pytest.mark.parametrize("nome", TODOS)
def test_badge_do_colab_aponta_para_a_branch_de_trabalho(nome: str) -> None:
    cabecalho = _markdown(_load(nome))[0]
    assert "colab.research.google.com/github" in cabecalho
    assert f"/blob/{REPO_BRANCH}/notebooks/{nome}" in cabecalho


@pytest.mark.parametrize("nome", TODOS)
def test_cabecalho_declara_o_runtime(nome: str) -> None:
    cabecalho = _markdown(_load(nome))[0]
    assert "Runtime" in cabecalho
    if nome == NOTEBOOK_GPU:
        assert "GPU" in cabecalho and "obrigatória" in cabecalho
    else:
        assert "CPU" in cabecalho


# ------------------------------------------------------- independência mútua
@pytest.mark.parametrize("nome", ("01_data_preprocessing.ipynb", "04_structured_patient_data.ipynb",
                                  "05_medflow_full_demo.ipynb"))
def test_notebooks_que_usam_o_banco_o_constroem(nome: str) -> None:
    """Nenhum notebook pode depender de outro ter rodado antes."""
    juntos = "\n".join(_codigo(_load(nome)))
    assert "build_synthetic_database" in juntos


def test_notebook_de_rag_constroi_o_proprio_indice() -> None:
    juntos = "\n".join(_codigo(_load("03_rag_pipeline.ipynb")))
    assert "MedFlowVectorStore.from_documents" in juntos


# ----------------------------------------------- notebook 05: modo fine-tuned
def test_demo_tem_modo_submissao_com_adapter_real() -> None:
    juntos = "\n".join(_codigo(_load("05_medflow_full_demo.ipynb")))
    assert 'MODO = "auto"' in juntos
    assert "submission" in juntos and "offline" in juntos
    assert 'get_chat_model("hf_local"' in juntos
    assert "adapter_path=" in juntos
    assert "MedFlowAssistant(chat_model=modelo)" in juntos


def test_demo_falha_cedo_sem_adapter_no_modo_submissao() -> None:
    """Regra inegociável: nunca gravar o vídeo com o template achando ser o fine-tuned."""
    juntos = "\n".join(_codigo(_load("05_medflow_full_demo.ipynb")))
    assert "raise FileNotFoundError" in juntos
    assert "adapter_valido" in juntos
    assert 'assert descricao["adapter_carregado"]' in juntos


def test_demo_usa_por_padrao_o_adapter_do_notebook_02() -> None:
    juntos = "\n".join(_codigo(_load("05_medflow_full_demo.ipynb")))
    assert "/content/drive/MyDrive/MedFlowAI_Fase3/02_fine_tuning/adapter" in juntos


def test_demo_rotula_o_baseline_offline() -> None:
    nb = _load("05_medflow_full_demo.ipynb")
    juntos = "\n".join(_markdown(nb) + _codigo(nb))
    assert "NÃO é" in juntos and "baseline" in juntos.lower()


# ---------------------------------------------- notebook 02: artefatos e ZIP
def test_notebook_de_gpu_gera_os_artefatos_obrigatorios() -> None:
    from medflow_ai.fine_tuning.validation import REQUIRED_FILES

    juntos = "\n".join(_codigo(_load(NOTEBOOK_GPU)))
    for arquivo in REQUIRED_FILES:
        assert arquivo in juntos, f"notebook 02 não produz {arquivo}"


def test_notebook_de_gpu_monta_e_audita_o_bundle() -> None:
    juntos = "\n".join(_codigo(_load(NOTEBOOK_GPU)))
    assert "build_results_bundle" in juntos
    assert "inspect_bundle" in juntos
    assert "FT_BUNDLES_DIR" in juntos
    assert "files.download" in juntos


def test_notebook_de_gpu_salva_adapter_no_drive_e_prova_recarga() -> None:
    juntos = "\n".join(_codigo(_load(NOTEBOOK_GPU)))
    assert "allow_weights=True" in juntos
    assert "PeftModel.from_pretrained" in juntos
    assert "FT_ADAPTER_DIR" in juntos


def test_notebook_de_gpu_nao_fixa_precisao() -> None:
    """Regressão: torch.bfloat16 fixo quebraria o treino na T4."""
    juntos = "\n".join(_codigo(_load(NOTEBOOK_GPU)))
    assert "torch.bfloat16" not in juntos
    assert "politica.torch_dtype()" in juntos


def test_notebook_de_gpu_valida_os_proprios_resultados() -> None:
    juntos = "\n".join(_codigo(_load(NOTEBOOK_GPU)))
    assert "validate_colab_results" in juntos


def test_notebook_de_gpu_trata_oom_registrando_a_configuracao_usada() -> None:
    juntos = "\n".join(_codigo(_load(NOTEBOOK_GPU)))
    assert "OutOfMemoryError" in juntos
    assert "USAR_RECUPERACAO_OOM" in juntos
    assert "config_recuperacao" in juntos


def test_token_hf_e_opcional_e_nao_versionado() -> None:
    juntos = "\n".join(_codigo(_load(NOTEBOOK_GPU)))
    assert 'HF_TOKEN = ""' in juntos
    assert "opcional" in juntos.lower() or "OPCIONAL" in juntos


# ------------------------------------------------------------------ segredos
@pytest.mark.parametrize("nome", TODOS)
def test_nenhum_notebook_contem_segredo_aparente(nome: str) -> None:
    import re

    conteudo = (_dir() / nome).read_text(encoding="utf-8")
    for padrao in (r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9]{30,}", r"ghp_[A-Za-z0-9]{30,}"):
        assert not re.search(padrao, conteudo), f"{nome} parece conter um token"
