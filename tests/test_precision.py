"""Testes da política de precisão FP16/BF16.

O hardware não está disponível em CI, então a detecção é injetada. O que
importa verificar é a **regra**: T4 (compute capability 7.5) precisa treinar em
float16, porque bfloat16 quebraria exatamente no hardware que o projeto
recomenda.
"""

from __future__ import annotations

import pytest

from medflow_ai.fine_tuning.config import QLoRAConfig
from medflow_ai.fine_tuning.precision import PrecisionPolicy, describe_gpu, resolve_precision

T4 = {
    "cuda_available": True, "gpu_name": "Tesla T4", "gpu_memory_gb": 14.7,
    "compute_capability": "7.5", "bf16_supported": False,
}
V100 = {
    "cuda_available": True, "gpu_name": "Tesla V100-SXM2-16GB", "gpu_memory_gb": 16.0,
    "compute_capability": "7.0", "bf16_supported": False,
}
A100 = {
    "cuda_available": True, "gpu_name": "NVIDIA A100-SXM4-40GB", "gpu_memory_gb": 39.6,
    "compute_capability": "8.0", "bf16_supported": True,
}
L4 = {
    "cuda_available": True, "gpu_name": "NVIDIA L4", "gpu_memory_gb": 22.5,
    "compute_capability": "8.9", "bf16_supported": True,
}
CPU = {"cuda_available": False, "gpu_name": "n/d", "gpu_memory_gb": 0.0,
       "compute_capability": "n/d", "bf16_supported": False}


@pytest.mark.parametrize("info", [T4, V100], ids=["T4", "V100"])
def test_gpu_sem_bf16_usa_float16(info: dict) -> None:
    politica = resolve_precision(info)
    assert politica.compute_dtype_name == "float16"
    assert politica.fp16 is True and politica.bf16 is False
    assert politica.trainable is True


@pytest.mark.parametrize("info", [A100, L4], ids=["A100", "L4"])
def test_gpu_com_bf16_usa_bfloat16(info: dict) -> None:
    politica = resolve_precision(info)
    assert politica.compute_dtype_name == "bfloat16"
    assert politica.bf16 is True and politica.fp16 is False


def test_sem_gpu_nao_e_treinavel() -> None:
    politica = resolve_precision(CPU)
    assert politica.compute_dtype_name == "float32"
    assert politica.bf16 is False and politica.fp16 is False
    assert politica.trainable is False
    assert "GPU" in politica.explain()


def test_bf16_declarado_sem_cuda_e_ignorado() -> None:
    """Coerência: sem CUDA não há bf16, mesmo que a detecção diga o contrário."""
    politica = resolve_precision({**CPU, "bf16_supported": True})
    assert politica.bf16_supported is False
    assert politica.compute_dtype_name == "float32"


def test_explicacao_menciona_hardware_e_dtype() -> None:
    texto = resolve_precision(T4).explain()
    assert "Tesla T4" in texto and "7.5" in texto and "float16" in texto


def test_politica_e_serializavel() -> None:
    payload = resolve_precision(A100).to_dict()
    for chave in ("cuda_available", "bf16_supported", "compute_dtype_name", "bf16", "fp16",
                  "gpu_name", "gpu_memory_gb", "compute_capability", "motivo"):
        assert chave in payload


def test_describe_gpu_nao_quebra_sem_torch() -> None:
    """Em CI não há torch: a detecção precisa degradar, não explodir."""
    info = describe_gpu()
    assert set(info) >= {"cuda_available", "gpu_name", "bf16_supported"}
    assert isinstance(info["cuda_available"], bool)


# ------------------------------------------------------- integração com config
def test_config_auto_segue_o_hardware() -> None:
    config = QLoRAConfig()
    assert config.precision == "auto"
    assert config.resolve_precision(T4).fp16 is True
    assert config.resolve_precision(A100).bf16 is True


def test_config_permite_forcar_fp16_em_gpu_com_bf16() -> None:
    politica = QLoRAConfig(precision="fp16").resolve_precision(A100)
    assert politica.fp16 is True and politica.bf16 is False
    assert "forçada" in politica.motivo


def test_forcar_bf16_em_gpu_sem_suporte_falha_cedo() -> None:
    with pytest.raises(ValueError, match="não suporta bfloat16"):
        QLoRAConfig(precision="bf16").resolve_precision(T4)


def test_precision_invalida_e_rejeitada() -> None:
    with pytest.raises(ValueError, match="precision inválida"):
        QLoRAConfig(precision="int4").resolve_precision(T4)


def test_config_nao_fixa_dtype_de_quantizacao() -> None:
    """Regressão: bnb_4bit_compute_dtype fixo em bfloat16 quebrava a T4."""
    assert QLoRAConfig().bnb_4bit_compute_dtype == "auto"
    assert not hasattr(QLoRAConfig(), "bf16"), "o flag fixo bf16 foi substituído por 'precision'"


def test_config_serializa_e_recarrega_precision(tmp_path) -> None:
    caminho = QLoRAConfig(precision="fp16").save(tmp_path / "training_config.json")
    assert QLoRAConfig.load(caminho).precision == "fp16"


# --------------------------------------------------------------- diagnóstico
def test_check_environment_reporta_precisao() -> None:
    from medflow_ai.fine_tuning.train import check_environment

    relatorio = check_environment()
    for campo in ("compute_capability", "bf16_supported", "compute_dtype", "bf16", "fp16"):
        assert hasattr(relatorio, campo)
    assert relatorio.to_dict()["compute_dtype"] in {"float32", "float16", "bfloat16"}


def test_policy_e_imutavel() -> None:
    politica = resolve_precision(T4)
    assert isinstance(politica, PrecisionPolicy)
    with pytest.raises(Exception):
        politica.bf16 = True  # type: ignore[misc]
