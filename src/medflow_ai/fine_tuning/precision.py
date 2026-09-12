"""Política única de precisão numérica (FP16 vs BF16).

Motivação: o projeto declara que uma GPU T4 do Colab gratuito é suficiente, mas
a **T4 (Turing, compute capability 7.5) não suporta bfloat16**. Um valor fixo de
``bf16=True`` faz o treinamento falhar exatamente no hardware que o projeto
recomenda.

A regra é decidida em um único lugar e reutilizada pelo trainer, pelo provedor
de inferência e pelo notebook:

* GPU com BF16 (Ampere/Hopper, cc ≥ 8.0) → **bfloat16**;
* GPU sem BF16 (T4, V100, cc < 8.0)      → **float16**;
* sem GPU                                 → float32, e o treino é recusado.

O módulo não importa ``torch`` no topo: precisa ser importável em CI, onde torch
não está instalado.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["PrecisionPolicy", "resolve_precision", "describe_gpu"]


@dataclass(frozen=True)
class PrecisionPolicy:
    """Decisão de precisão para uma execução concreta."""

    cuda_available: bool
    bf16_supported: bool
    compute_dtype_name: str          # "bfloat16" | "float16" | "float32"
    bf16: bool                       # flag de treino do TrainingArguments
    fp16: bool                       # flag de treino do TrainingArguments
    gpu_name: str = "n/d"
    gpu_memory_gb: float = 0.0
    compute_capability: str = "n/d"
    motivo: str = ""

    @property
    def trainable(self) -> bool:
        """Só há treino QLoRA com GPU CUDA disponível."""
        return self.cuda_available

    def torch_dtype(self) -> Any:
        """Resolve o ``torch.dtype`` correspondente (requer torch instalado)."""
        import torch

        return getattr(torch, self.compute_dtype_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def explain(self) -> str:
        if not self.cuda_available:
            return (
                "Sem GPU CUDA: precisão float32 e treinamento QLoRA indisponível. "
                "No Colab: Ambiente de execução > Alterar o tipo de ambiente > GPU."
            )
        return (
            f"GPU {self.gpu_name} ({self.gpu_memory_gb:.1f} GB, compute capability "
            f"{self.compute_capability}) — BF16 {'suportado' if self.bf16_supported else 'NÃO suportado'}; "
            f"usando {self.compute_dtype_name} (bf16={self.bf16}, fp16={self.fp16}). {self.motivo}"
        )


def describe_gpu() -> dict[str, Any]:
    """Coleta nome, memória e compute capability da GPU, sem falhar se não houver."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "gpu_name": "n/d", "gpu_memory_gb": 0.0,
                "compute_capability": "n/d", "bf16_supported": False}

    if not torch.cuda.is_available():
        return {"cuda_available": False, "gpu_name": "n/d", "gpu_memory_gb": 0.0,
                "compute_capability": "n/d", "bf16_supported": False}

    properties = torch.cuda.get_device_properties(0)
    try:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001 - build de torch sem a checagem
        bf16_supported = properties.major >= 8

    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(properties.total_memory / (1024**3), 2),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "bf16_supported": bf16_supported,
    }


def resolve_precision(info: dict[str, Any] | None = None) -> PrecisionPolicy:
    """Decide a precisão a partir do hardware detectado (ou de ``info`` injetado).

    Args:
        info: dicionário no formato de :func:`describe_gpu`. Injetável em testes
            para verificar a política sem GPU real.
    """
    detected = describe_gpu() if info is None else dict(info)

    cuda = bool(detected.get("cuda_available", False))
    bf16_supported = bool(detected.get("bf16_supported", False)) and cuda

    if not cuda:
        dtype, bf16, fp16 = "float32", False, False
        motivo = "Nenhuma GPU CUDA detectada."
    elif bf16_supported:
        dtype, bf16, fp16 = "bfloat16", True, False
        motivo = "GPU suporta bfloat16 (maior faixa dinâmica, menos risco de overflow)."
    else:
        dtype, bf16, fp16 = "float16", False, True
        motivo = (
            "GPU sem suporte a bfloat16 (ex.: T4/V100, compute capability < 8.0); "
            "usando float16 com escalonamento de gradiente."
        )

    return PrecisionPolicy(
        cuda_available=cuda,
        bf16_supported=bf16_supported,
        compute_dtype_name=dtype,
        bf16=bf16,
        fp16=fp16,
        gpu_name=str(detected.get("gpu_name", "n/d")),
        gpu_memory_gb=float(detected.get("gpu_memory_gb", 0.0)),
        compute_capability=str(detected.get("compute_capability", "n/d")),
        motivo=motivo,
    )
