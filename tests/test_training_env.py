"""Testes do módulo de treinamento que NÃO exigem GPU.

O treinamento em si só roda com GPU. O que é testável em CI é o comportamento
defensivo: diagnosticar o ambiente corretamente e recusar-se a produzir
resultados quando não há como treinar.
"""

from __future__ import annotations

from medflow_ai.fine_tuning.train import check_environment, set_seed, train


def test_diagnostico_de_ambiente_e_informativo() -> None:
    relatorio = check_environment()
    assert isinstance(relatorio.ready, bool)
    assert relatorio.python_version
    explicacao = relatorio.explain()
    assert explicacao
    if not relatorio.ready:
        assert "GPU" in explicacao or "Dependências" in explicacao


def test_treino_sem_gpu_nao_inventa_metricas(tmp_path) -> None:
    """Requisito de honestidade: sem ambiente adequado, nada de métricas."""
    relatorio = check_environment()
    resultado = train(output_dir=tmp_path)
    if relatorio.ready:  # pragma: no cover - só executa em máquina com GPU
        assert resultado["status"] in {"ok", "dry_run"}
    else:
        assert resultado["status"] == "skipped"
        assert "metricas_treino" not in resultado
        assert resultado["motivo"]
        assert not list(tmp_path.glob("training_results.json"))


def test_set_seed_nao_falha_sem_torch() -> None:
    set_seed(123)
