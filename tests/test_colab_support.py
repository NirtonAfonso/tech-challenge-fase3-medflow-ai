"""Testes do suporte a Colab/Drive, do bundle e do validador de resultados.

Nada aqui pretende provar execução no Colab — isso só o usuário pode fazer. O
que se verifica é a lógica que roda *dentro* do Colab: detecção de ambiente,
criação da árvore no Drive, política de segredos e o veredito do validador.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from medflow_ai.colab import (
    DRIVE_ROOT,
    NOTEBOOK_LAYOUT,
    REPO_BRANCH,
    REPO_URL,
    PersistenceReport,
    ensure_structure,
    git_info,
    in_colab,
    mount_drive,
    persist,
    run_metadata,
    summarize,
    write_run_metadata,
)
from medflow_ai.fine_tuning.bundle import (
    BUNDLE_NAME,
    BundleSecurityError,
    build_results_bundle,
    inspect_bundle,
)
from medflow_ai.fine_tuning.validation import validate_colab_results


# ------------------------------------------------------------------ ambiente
def test_deteccao_de_colab_e_falsa_localmente() -> None:
    assert in_colab() is False


def test_deteccao_pode_ser_forcada(monkeypatch) -> None:
    monkeypatch.setenv("MEDFLOW_FORCE_COLAB", "1")
    assert in_colab() is True
    monkeypatch.setenv("MEDFLOW_FORCE_COLAB", "0")
    assert in_colab() is False


def test_montagem_fora_do_colab_nao_quebra() -> None:
    montado, mensagem = mount_drive()
    assert montado is False
    assert "Colab" in mensagem


def test_constantes_apontam_para_a_branch_de_trabalho() -> None:
    assert REPO_BRANCH == "develop"
    assert REPO_URL.endswith("tech-challenge-fase3-medflow-ai.git")
    assert DRIVE_ROOT == "/content/drive/MyDrive/MedFlowAI_Fase3"


def test_layout_cobre_os_cinco_notebooks_e_shared() -> None:
    assert set(NOTEBOOK_LAYOUT) == {
        "01_preprocessing", "02_fine_tuning", "03_rag", "04_structured_data", "05_demo", "shared",
    }
    assert set(NOTEBOOK_LAYOUT["02_fine_tuning"]) == {
        "artifacts", "adapter", "checkpoints", "logs", "bundles"
    }


# ------------------------------------------------------------------ estrutura
@pytest.mark.parametrize("notebook", ["01_preprocessing", "02_fine_tuning", "03_rag",
                                      "04_structured_data", "05_demo"])
def test_estrutura_e_criada_com_exist_ok(notebook: str) -> None:
    pastas = ensure_structure(notebook)
    for sub in NOTEBOOK_LAYOUT[notebook]:
        assert pastas[sub].is_dir()
    for sub in NOTEBOOK_LAYOUT["shared"]:
        assert pastas[f"shared/{sub}"].is_dir()
    ensure_structure(notebook)  # idempotente


def test_notebook_desconhecido_falha_cedo() -> None:
    with pytest.raises(KeyError, match="Notebook desconhecido"):
        ensure_structure("99_inexistente")


def test_metadados_de_execucao_sao_rastreaveis() -> None:
    dados = run_metadata("03_rag")
    assert dados["notebook"] == "03_rag"
    assert dados["ambiente"] in {"local", "google_colab"}
    assert dados["git"]["commit"]
    assert "gpu" in dados and "versoes" in dados


def test_metadados_sao_gravados_em_shared() -> None:
    caminho = write_run_metadata("05_demo", extra={"cenarios": 5})
    assert caminho.exists()
    payload = json.loads(caminho.read_text(encoding="utf-8"))
    assert payload["cenarios"] == 5
    assert (caminho.parent / "latest_05_demo.json").exists()


def test_git_info_devolve_branch_e_commit() -> None:
    info = git_info()
    assert set(info) == {"branch", "commit", "commit_short"}


# ---------------------------------------------------------------- persistência
def test_persistencia_copia_arquivos_e_relata(tmp_path) -> None:
    origem = tmp_path / "origem"; origem.mkdir()
    (origem / "metricas.json").write_text('{"a": 1}', encoding="utf-8")
    (origem / "grafico.png").write_bytes(b"png")

    relatorio = persist([origem / "metricas.json", origem / "grafico.png"], tmp_path / "drive")
    assert len(relatorio.copiados) == 2
    assert (tmp_path / "drive" / "metricas.json").exists()
    assert "metricas.json" in relatorio.render()


def test_persistencia_registra_arquivo_ausente_sem_falhar(tmp_path) -> None:
    relatorio = persist([tmp_path / "nao_existe.json"], tmp_path / "drive")
    assert relatorio.copiados == []
    assert relatorio.ausentes
    assert "não gerado" in relatorio.render()


def test_persistencia_recusa_segredos(tmp_path) -> None:
    origem = tmp_path / "origem"; origem.mkdir()
    (origem / ".env").write_text("OPENAI_API_KEY=sk-secreta", encoding="utf-8")
    relatorio = persist([origem / ".env"], tmp_path / "drive")
    assert relatorio.copiados == []
    assert ".env" in relatorio.ignorados
    assert not (tmp_path / "drive" / ".env").exists()


def test_persistencia_recusa_pesos_por_padrao(tmp_path) -> None:
    origem = tmp_path / "origem"; origem.mkdir()
    (origem / "adapter_model.safetensors").write_bytes(b"pesos")
    padrao = persist([origem / "adapter_model.safetensors"], tmp_path / "d1")
    assert padrao.copiados == []

    permitido = persist([origem / "adapter_model.safetensors"], tmp_path / "d2", allow_weights=True)
    assert len(permitido.copiados) == 1


def test_persistencia_de_diretorio_preserva_subarvore(tmp_path) -> None:
    origem = tmp_path / "adapter"; (origem / "sub").mkdir(parents=True)
    (origem / "adapter_config.json").write_text("{}", encoding="utf-8")
    (origem / "sub" / "tokenizer.json").write_text("{}", encoding="utf-8")

    relatorio = persist([origem], tmp_path / "drive", allow_weights=True)
    assert (tmp_path / "drive" / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "drive" / "adapter" / "sub" / "tokenizer.json").exists()
    assert len(relatorio.copiados) == 2


def test_resumo_final_lista_tudo(tmp_path) -> None:
    origem = tmp_path / "o"; origem.mkdir()
    (origem / "a.json").write_text("{}", encoding="utf-8")
    texto = summarize([persist([origem / "a.json"], tmp_path / "drive")])
    assert "RESUMO DA PERSISTÊNCIA" in texto and "a.json" in texto
    assert "Execução LOCAL" in texto


# --------------------------------------------------------------------- bundle
@pytest.fixture()
def artefatos_completos(tmp_path):
    pasta = tmp_path / "fine_tuning"; pasta.mkdir()
    (pasta / "training_results.json").write_text(json.dumps({
        "status": "ok", "executado_em": "2026-03-01T10:00:00Z", "commit": "a" * 40,
        "seed": 42, "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "gpu": {"cuda_available": True, "gpu_name": "Tesla T4", "gpu_memory_gb": 14.7,
                "compute_capability": "7.5", "bf16_supported": False},
        "compute_dtype": "float16",
        "metricas_treino": {"train_loss": 0.8123, "eval_loss": 0.9011},
        "splits": {"train": 102, "validation": 13, "test": 22},
        "parametros": {"parametros_treinaveis": 29_884_416, "percentual_treinavel": 0.93},
    }), encoding="utf-8")
    (pasta / "loss_curve.png").write_bytes(b"\x89PNG fake")
    (pasta / "comparacao_sistemas.json").write_text(json.dumps({"sistemas": [
        {"sistema": "base", "n_exemplos": 22}, {"sistema": "fine_tuned", "n_exemplos": 22},
        {"sistema": "fine_tuned_rag", "n_exemplos": 22},
    ]}), encoding="utf-8")
    (pasta / "respostas_antes_depois.json").write_text(json.dumps({"p1": {"base": "a", "fine_tuned": "b"}}), encoding="utf-8")
    (pasta / "environment.json").write_text("{}", encoding="utf-8")
    (pasta / "training_config.json").write_text("{}", encoding="utf-8")
    return pasta


def test_bundle_inclui_metricas_e_procedencia(artefatos_completos, tmp_path) -> None:
    relatorio = build_results_bundle(artefatos_completos, tmp_path / "bundles")
    assert relatorio.caminho.name == BUNDLE_NAME
    with zipfile.ZipFile(relatorio.caminho) as pacote:
        nomes = set(pacote.namelist())
    assert "training_results.json" in nomes and "BUNDLE_INFO.json" in nomes
    assert relatorio.tamanho_bytes > 0


def test_bundle_nunca_carrega_pesos_nem_segredos(artefatos_completos, tmp_path) -> None:
    (artefatos_completos / "adapter_model.safetensors").write_bytes(b"pesos" * 100)
    (artefatos_completos / ".env").write_text("HUGGINGFACE_TOKEN=hf_xxx", encoding="utf-8")

    relatorio = build_results_bundle(artefatos_completos, tmp_path / "bundles")
    with zipfile.ZipFile(relatorio.caminho) as pacote:
        nomes = pacote.namelist()
    assert not any(nome.endswith(".safetensors") for nome in nomes)
    assert ".env" not in nomes
    assert inspect_bundle(relatorio.caminho)["seguro"] is True


def test_bundle_recusa_extra_proibido(artefatos_completos, tmp_path) -> None:
    peso = tmp_path / "adapter_model.bin"; peso.write_bytes(b"x")
    with pytest.raises(BundleSecurityError, match="Peso de modelo"):
        build_results_bundle(artefatos_completos, tmp_path / "bundles", extras=[peso])


def test_bundle_registra_o_que_nao_foi_gerado(tmp_path) -> None:
    vazio = tmp_path / "vazio"; vazio.mkdir()
    relatorio = build_results_bundle(vazio, tmp_path / "bundles")
    assert "loss_curve.png" in relatorio.ausentes


# ------------------------------------------------------------------ validador
def test_validador_aprova_execucao_completa(artefatos_completos) -> None:
    resultado = validate_colab_results(artefatos_completos)
    assert resultado.valido, resultado.erros
    assert resultado.resumo["train_loss"] == 0.8123
    assert resultado.resumo["compute_dtype"] == "float16"
    assert "VÁLIDO" in resultado.render()


def test_validador_reprova_diretorio_inexistente(tmp_path) -> None:
    resultado = validate_colab_results(tmp_path / "nao_existe")
    assert not resultado.valido
    assert "não encontrado" in resultado.erros[0]


def test_validador_reprova_status_diferente_de_ok(artefatos_completos) -> None:
    caminho = artefatos_completos / "training_results.json"
    payload = json.loads(caminho.read_text(encoding="utf-8"))
    payload["status"] = "skipped"
    caminho.write_text(json.dumps(payload), encoding="utf-8")

    resultado = validate_colab_results(artefatos_completos)
    assert not resultado.valido
    assert any("status" in erro for erro in resultado.erros)


def test_validador_reprova_loss_ausente_ou_nao_numerica(artefatos_completos) -> None:
    caminho = artefatos_completos / "training_results.json"
    payload = json.loads(caminho.read_text(encoding="utf-8"))
    payload["metricas_treino"] = {"train_loss": "n/d"}
    caminho.write_text(json.dumps(payload), encoding="utf-8")

    resultado = validate_colab_results(artefatos_completos)
    assert not resultado.valido
    assert any("loss" in erro for erro in resultado.erros)


def test_validador_reprova_comparacao_incompleta(artefatos_completos) -> None:
    (artefatos_completos / "comparacao_sistemas.json").write_text(
        json.dumps({"sistemas": [{"sistema": "base", "n_exemplos": 22}]}), encoding="utf-8"
    )
    resultado = validate_colab_results(artefatos_completos)
    assert not resultado.valido
    assert any("fine_tuned" in erro for erro in resultado.erros)


def test_validador_reprova_treino_sem_gpu(artefatos_completos) -> None:
    caminho = artefatos_completos / "training_results.json"
    payload = json.loads(caminho.read_text(encoding="utf-8"))
    payload["gpu"]["cuda_available"] = False
    caminho.write_text(json.dumps(payload), encoding="utf-8")

    resultado = validate_colab_results(artefatos_completos)
    assert not resultado.valido
    assert any("GPU" in erro or "gpu" in erro for erro in resultado.erros)


def test_validador_detecta_incoerencia_de_precisao(artefatos_completos) -> None:
    caminho = artefatos_completos / "training_results.json"
    payload = json.loads(caminho.read_text(encoding="utf-8"))
    payload["compute_dtype"] = "bfloat16"  # GPU declarada é T4, sem BF16
    caminho.write_text(json.dumps(payload), encoding="utf-8")

    resultado = validate_colab_results(artefatos_completos)
    assert not resultado.valido
    assert any("bfloat16" in erro for erro in resultado.erros)


def test_validador_detecta_token_vazado(artefatos_completos) -> None:
    (artefatos_completos / "environment.json").write_text(
        json.dumps({"token": "hf_" + "a" * 32}), encoding="utf-8"
    )
    resultado = validate_colab_results(artefatos_completos)
    assert not resultado.valido
    assert any("token" in erro or "credencial" in erro for erro in resultado.erros)


def test_validador_detecta_peso_em_artifacts(artefatos_completos) -> None:
    (artefatos_completos / "adapter_model.safetensors").write_bytes(b"pesos")
    resultado = validate_colab_results(artefatos_completos)
    assert not resultado.valido
    assert any("Peso de modelo" in erro for erro in resultado.erros)


def test_validador_reprova_comparacao_com_splits_diferentes(artefatos_completos) -> None:
    (artefatos_completos / "comparacao_sistemas.json").write_text(json.dumps({"sistemas": [
        {"sistema": "base", "n_exemplos": 22}, {"sistema": "fine_tuned", "n_exemplos": 10},
        {"sistema": "fine_tuned_rag", "n_exemplos": 22},
    ]}), encoding="utf-8")
    resultado = validate_colab_results(artefatos_completos)
    assert not resultado.valido
    assert any("tamanhos diferentes" in erro for erro in resultado.erros)


# ------------------------------------------------------------------------ CLI
def test_cli_valida_e_retorna_codigo_de_erro(artefatos_completos, capsys) -> None:
    from medflow_ai.cli import main

    assert main(["validate-colab-results", str(artefatos_completos)]) == 0
    assert "VÁLIDO" in capsys.readouterr().out

    (artefatos_completos / "loss_curve.png").unlink()
    assert main(["validate-colab-results", str(artefatos_completos)]) == 1


def test_cli_inspeciona_bundle(artefatos_completos, tmp_path, capsys) -> None:
    from medflow_ai.cli import main

    relatorio = build_results_bundle(artefatos_completos, tmp_path / "bundles")
    assert main(["inspect-bundle", str(relatorio.caminho)]) == 0
    assert "Seguro para compartilhar" in capsys.readouterr().out
