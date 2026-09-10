"""Testes do pipeline de dataset de fine-tuning (anonimização, curadoria, splits)."""

from __future__ import annotations

import json

from medflow_ai.data.anonymization import contains_pii
from medflow_ai.fine_tuning.config import QLoRAConfig
from medflow_ai.fine_tuning.dataset import (
    HELD_OUT_DOCUMENTS,
    DatasetStats,
    anonymize_examples,
    build_sft_dataset,
    curate,
    generate_examples,
    split_by_document,
)


def test_geracao_cobre_as_familias_exigidas_pelo_enunciado(database) -> None:
    familias = {exemplo.familia for exemplo in generate_examples()}
    for esperada in ("protocolo_qa", "faq_medico", "modelo_laudo", "modelo_receita",
                     "procedimento", "seguranca"):
        assert esperada in familias


def test_exemplos_seguem_o_formato_de_saida_obrigatorio() -> None:
    for exemplo in generate_examples()[:40]:
        for secao in ("RESPOSTA:", "CONTEXTO DO PACIENTE:", "PENDÊNCIAS E ALERTAS:", "LIMITAÇÃO:"):
            assert secao in exemplo.output, exemplo.id


def test_anonimizacao_remove_pii_dos_exemplos(database) -> None:
    brutos = generate_examples()
    com_pii = [exemplo for exemplo in brutos if contains_pii(exemplo.instruction)]
    assert com_pii, "o gerador deve produzir exemplos com PII para exercitar a anonimização"

    limpos, relatorio = anonymize_examples(brutos)
    assert relatorio.total > 0
    assert not any(contains_pii(exemplo.instruction) for exemplo in limpos)
    assert not any(contains_pii(exemplo.output) for exemplo in limpos)


def test_curadoria_remove_duplicados_e_registra_estatisticas() -> None:
    exemplos, _ = anonymize_examples(generate_examples())
    estatisticas = DatasetStats(gerados=len(exemplos))
    curados = curate(list(exemplos) + list(exemplos), estatisticas)

    assert estatisticas.removidos_duplicados == len(curados)
    assert len(curados) == len({exemplo.fingerprint for exemplo in curados})
    assert estatisticas.por_familia and estatisticas.por_documento


def test_curadoria_descarta_exemplos_com_pii_residual(database) -> None:
    """Exemplos ainda não anonimizados são barrados pela curadoria."""
    exemplos = generate_examples()
    estatisticas = DatasetStats(gerados=len(exemplos))
    curate(exemplos, estatisticas)
    assert estatisticas.removidos_pii > 0


def test_split_por_documento_evita_leakage() -> None:
    exemplos = curate(anonymize_examples(generate_examples())[0], DatasetStats())
    splits = split_by_document(exemplos)

    docs_treino = {exemplo.doc_id for exemplo in splits["train"]}
    docs_validacao = {exemplo.doc_id for exemplo in splits["validation"]}
    docs_teste = {exemplo.doc_id for exemplo in splits["test"]}

    assert docs_teste == set(HELD_OUT_DOCUMENTS)
    assert not (docs_teste & docs_treino)
    assert not (docs_teste & docs_validacao)


def test_split_e_reprodutivel_com_o_mesmo_seed() -> None:
    exemplos = curate(anonymize_examples(generate_examples())[0], DatasetStats())
    primeiro = split_by_document(exemplos, seed=7)
    segundo = split_by_document(exemplos, seed=7)
    assert [item.id for item in primeiro["train"]] == [item.id for item in segundo["train"]]


def test_pipeline_completo_escreve_jsonl_e_manifesto(tmp_path, database) -> None:
    splits, estatisticas, caminhos = build_sft_dataset(output_dir=tmp_path, seed=42)

    assert estatisticas.finais > 0
    assert splits["train"] and splits["test"]
    for nome in ("train", "validation", "test", "manifest"):
        assert caminhos[nome].exists()

    linhas = caminhos["train"].read_text(encoding="utf-8").strip().splitlines()
    primeiro = json.loads(linhas[0])
    papeis = [mensagem["role"] for mensagem in primeiro["messages"]]
    assert papeis == ["system", "user", "assistant"]

    manifesto = json.loads(caminhos["manifest"].read_text(encoding="utf-8"))
    assert manifesto["seed"] == 42
    assert manifesto["held_out_documents"] == sorted(HELD_OUT_DOCUMENTS)
    assert manifesto["estatisticas_curadoria"]["finais"] == estatisticas.finais
    assert manifesto["anonimizacao"]["counts_by_kind"]


def test_configuracao_qlora_salva_e_recarrega(tmp_path) -> None:
    config = QLoRAConfig(num_train_epochs=2.0)
    caminho = config.save(tmp_path / "training_config.json")
    recarregada = QLoRAConfig.load(caminho)
    assert recarregada.num_train_epochs == 2.0
    assert recarregada.lora_target_modules == config.lora_target_modules
    assert config.effective_batch_size == 16
