"""Configuração congelada do fine-tuning QLoRA.

Manter a configuração em código (e não apenas no notebook) garante que o
treinamento executado no Colab seja o mesmo descrito no relatório, e que o
``training_config.json`` salvo junto do adapter possa ser comparado.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from medflow_ai.fine_tuning.precision import PrecisionPolicy

__all__ = ["QLoRAConfig", "DEFAULT_CONFIG"]


@dataclass
class QLoRAConfig:
    """Hiperparâmetros do SFT com QLoRA.

    Escolhas e justificativas estão em ``docs/FINE_TUNING.md``. Os valores
    padrão foram dimensionados para caber em uma GPU T4 de 16 GB (nível
    gratuito do Google Colab).
    """

    # Modelo base
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    fallback_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    trust_remote_code: bool = False

    # Quantização (QLoRA).
    # ``bnb_4bit_compute_dtype = "auto"`` delega a escolha a
    # ``fine_tuning.precision.resolve_precision()``: bfloat16 quando a GPU
    # suporta, float16 em T4/V100. Fixar "bfloat16" aqui quebraria o treino na
    # T4 do Colab gratuito, que é o hardware recomendado pelo projeto.
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "auto"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    lora_bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Treinamento
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_8bit"
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    # ``precision = "auto"`` resolve bf16/fp16 pelo hardware. Use "bf16" ou
    # "fp16" apenas para forçar explicitamente em um experimento controlado.
    precision: str = "auto"
    logging_steps: int = 5
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    seed: int = 42

    # Caminhos
    output_dir: str = "artifacts/fine_tuning"
    adapter_name: str = "medflow-qlora-adapter"
    dataset_dir: str = "data/processed/sft"

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lora_target_modules"] = list(self.lora_target_modules)
        return payload

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: Path | str) -> "QLoRAConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["lora_target_modules"] = tuple(payload.get("lora_target_modules", ()))
        return cls(**payload)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def resolve_precision(self, info: dict[str, Any] | None = None) -> "PrecisionPolicy":
        """Resolve a precisão desta execução, respeitando um override explícito."""
        from medflow_ai.fine_tuning.precision import resolve_precision as _resolve

        policy = _resolve(info)
        choice = (self.precision or "auto").strip().lower()
        if choice == "auto":
            return policy
        if choice not in {"bf16", "fp16"}:
            raise ValueError(
                f"precision inválida: {self.precision!r}. Use 'auto', 'bf16' ou 'fp16'."
            )
        if choice == "bf16" and not policy.bf16_supported:
            raise ValueError(
                "precision='bf16' foi forçado, mas a GPU detectada não suporta bfloat16 "
                f"({policy.gpu_name}, compute capability {policy.compute_capability}). "
                "Use 'auto' ou 'fp16'."
            )
        return replace(
            policy,
            compute_dtype_name="bfloat16" if choice == "bf16" else "float16",
            bf16=choice == "bf16",
            fp16=choice == "fp16",
            motivo=f"Precisão forçada por configuração (precision={choice!r}).",
        )


DEFAULT_CONFIG = QLoRAConfig()
