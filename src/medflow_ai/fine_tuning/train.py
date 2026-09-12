"""Treinamento SFT com QLoRA (execução em GPU).

Este módulo contém o treinamento **real**. Ele não roda em CPU: exige GPU com
suporte a 4-bit (``bitsandbytes``) e as dependências de
``requirements-training.txt``. O caminho de execução previsto é o Google Colab,
via ``notebooks/02_fine_tuning_qlora.ipynb``, que apenas chama estas funções.

Uso::

    python -m medflow_ai.fine_tuning.train --epochs 3 --output artifacts/fine_tuning

Nada aqui gera métricas sintéticas: se o treino não rodar, nenhum arquivo de
resultado é escrito.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medflow_ai.config import get_settings
from medflow_ai.fine_tuning.config import QLoRAConfig
from medflow_ai.fine_tuning.precision import PrecisionPolicy, describe_gpu, resolve_precision

__all__ = ["check_environment", "load_splits", "train", "EnvironmentReport", "resolve_precision"]


@dataclass
class EnvironmentReport:
    """Diagnóstico do ambiente antes de tentar treinar."""

    gpu_available: bool
    gpu_name: str
    gpu_memory_gb: float
    compute_capability: str
    bf16_supported: bool
    compute_dtype: str
    bf16: bool
    fp16: bool
    torch_version: str
    transformers_version: str
    peft_version: str
    trl_version: str
    bitsandbytes_available: bool
    python_version: str
    missing: list[str]

    @property
    def ready(self) -> bool:
        return self.gpu_available and not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "ready": self.ready}

    def explain(self) -> str:
        if self.ready:
            return (
                f"Ambiente pronto: GPU {self.gpu_name} ({self.gpu_memory_gb:.1f} GB, "
                f"compute capability {self.compute_capability}), BF16 "
                f"{'suportado' if self.bf16_supported else 'NÃO suportado'} → precisão "
                f"{self.compute_dtype} (bf16={self.bf16}, fp16={self.fp16}); "
                f"torch {self.torch_version}, transformers {self.transformers_version}."
            )
        problems: list[str] = []
        if not self.gpu_available:
            problems.append(
                "Nenhuma GPU CUDA detectada. O fine-tuning QLoRA exige GPU. "
                "No Colab: Ambiente de execução > Alterar tipo de ambiente > GPU (T4)."
            )
        if self.missing:
            problems.append(
                "Dependências ausentes: " + ", ".join(self.missing)
                + ". Instale com: pip install -r requirements-training.txt"
            )
        return " | ".join(problems)


def _version(module_name: str) -> tuple[str, bool]:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "desconhecida"), True
    except ImportError:
        return "ausente", False


def check_environment(config: QLoRAConfig | None = None) -> EnvironmentReport:
    """Verifica GPU, precisão e dependências sem falhar quando algo está ausente."""
    missing: list[str] = []

    torch_version, has_torch = _version("torch")
    if not has_torch:
        missing.append("torch")

    # A precisão é decidida em um único lugar (fine_tuning.precision) e apenas
    # reportada aqui — trainer, provider e notebook consomem a mesma política.
    policy = (config or QLoRAConfig()).resolve_precision() if has_torch else resolve_precision(
        {"cuda_available": False}
    )

    transformers_version, has_transformers = _version("transformers")
    if not has_transformers:
        missing.append("transformers")
    peft_version, has_peft = _version("peft")
    if not has_peft:
        missing.append("peft")
    trl_version, has_trl = _version("trl")
    if not has_trl:
        missing.append("trl")
    _, has_bnb = _version("bitsandbytes")
    if not has_bnb:
        missing.append("bitsandbytes")
    _, has_datasets = _version("datasets")
    if not has_datasets:
        missing.append("datasets")

    return EnvironmentReport(
        gpu_available=policy.cuda_available,
        gpu_name=policy.gpu_name,
        gpu_memory_gb=policy.gpu_memory_gb,
        compute_capability=policy.compute_capability,
        bf16_supported=policy.bf16_supported,
        compute_dtype=policy.compute_dtype_name,
        bf16=policy.bf16,
        fp16=policy.fp16,
        torch_version=torch_version,
        transformers_version=transformers_version,
        peft_version=peft_version,
        trl_version=trl_version,
        bitsandbytes_available=has_bnb,
        python_version=platform.python_version(),
        missing=missing,
    )


def _manifest_hash(caminho: Path) -> str:
    """Hash do manifesto do dataset: liga o treino ao dado exato que o alimentou."""
    if not caminho.exists():
        return "ausente"
    return hashlib.sha256(caminho.read_bytes()).hexdigest()[:16]


def set_seed(seed: int) -> None:
    """Fixa as sementes de todas as bibliotecas envolvidas."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def load_splits(dataset_dir: Path | str | None = None) -> dict[str, Any]:
    """Carrega os splits JSONL como ``datasets.Dataset``.

    Se os arquivos não existirem, o dataset é construído na hora — o pipeline é
    reprodutível de ponta a ponta a partir do repositório.
    """
    from datasets import Dataset

    settings = get_settings()
    directory = Path(dataset_dir or (settings.project_root / "data" / "processed" / "sft"))
    if not (directory / "train.jsonl").exists():
        from medflow_ai.fine_tuning.dataset import build_sft_dataset

        build_sft_dataset(output_dir=directory)

    splits: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        splits[name] = Dataset.from_list(rows)
    return splits


def _build_model_and_tokenizer(config: QLoRAConfig, policy: "PrecisionPolicy | None" = None):
    """Carrega o modelo base quantizado em 4 bits e prepara para PEFT."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    effective = policy or config.resolve_precision()
    compute_dtype = effective.torch_dtype()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias=config.lora_bias,
        task_type=config.task_type,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, lora_config


def _trainable_parameter_summary(model: Any) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "parametros_treinaveis": trainable,
        "parametros_totais": total,
        "percentual_treinavel": round(100 * trainable / total, 4) if total else 0.0,
    }


def train(
    config: QLoRAConfig | None = None,
    *,
    dataset_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Executa o SFT com QLoRA e salva adapter, configuração e métricas.

    Args:
        config: hiperparâmetros; usa ``QLoRAConfig()`` quando omitido.
        dataset_dir: diretório dos splits JSONL.
        output_dir: destino do adapter e das métricas.
        dry_run: apenas valida ambiente e dataset, sem treinar.

    Returns:
        Dicionário com o resultado. Em caso de ambiente inadequado, retorna
        ``{"status": "skipped", ...}`` — **nenhuma métrica é inventada**.
    """
    settings = get_settings()
    cfg = config or QLoRAConfig()
    target = Path(output_dir or (settings.project_root / cfg.output_dir))
    target.mkdir(parents=True, exist_ok=True)

    environment = check_environment(cfg)
    if not environment.ready:
        return {
            "status": "skipped",
            "motivo": environment.explain(),
            "ambiente": environment.to_dict(),
        }

    set_seed(cfg.seed)
    splits = load_splits(dataset_dir)
    if "train" not in splits:
        return {"status": "error", "motivo": "Split de treino não encontrado."}

    if dry_run:
        return {
            "status": "dry_run",
            "ambiente": environment.to_dict(),
            "splits": {name: len(dataset) for name, dataset in splits.items()},
        }

    from transformers import TrainingArguments
    from trl import SFTConfig, SFTTrainer

    policy = cfg.resolve_precision()
    model, tokenizer, _ = _build_model_and_tokenizer(cfg, policy)
    parameter_summary = _trainable_parameter_summary(model)

    sft_config = SFTConfig(
        output_dir=str(target / "checkpoints"),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        max_length=cfg.max_seq_length,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=policy.bf16,
        fp16=policy.fp16,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy if "validation" in splits else "no",
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=splits["train"],
        eval_dataset=splits.get("validation"),
        processing_class=tokenizer,
    )

    iniciado_em = time.perf_counter()
    train_result = trainer.train()
    metrics: dict[str, Any] = dict(train_result.metrics)
    if "validation" in splits:
        metrics.update({f"eval_{k}": v for k, v in trainer.evaluate().items()})

    adapter_dir = target / cfg.adapter_name
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    cfg.save(target / "training_config.json")

    from medflow_ai.colab import git_info, package_versions

    dataset_manifest = Path(dataset_dir or (settings.project_root / "data" / "processed" / "sft"))
    manifest_hash = _manifest_hash(dataset_manifest / "manifest.json")
    repositorio = git_info(settings.project_root)

    # Estrutura consumida por `medflow_ai.cli validate-colab-results`: qualquer
    # campo ausente ou vazio invalida o resultado, de propósito.
    payload = {
        "status": "ok",
        "executado_em": datetime.now(UTC).isoformat(),
        "commit": repositorio["commit"],
        "branch": repositorio["branch"],
        "seed": cfg.seed,
        "base_model": cfg.base_model_id,
        "gpu": describe_gpu(),
        "compute_dtype": policy.compute_dtype_name,
        "precisao": policy.to_dict(),
        "ambiente": environment.to_dict(),
        "versoes": package_versions(
            ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes")
        ),
        "config": cfg.to_dict(),
        "dataset_manifest_hash": manifest_hash,
        "splits": {name: len(dataset) for name, dataset in splits.items()},
        "parametros": parameter_summary,
        "metricas_treino": metrics,
        "tempo_total_s": round(time.perf_counter() - iniciado_em, 1),
        "log_history": trainer.state.log_history,
        "adapter_path": str(adapter_dir),
    }
    (target / "training_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (target / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "plataforma": platform.platform(),
                "git": repositorio,
                "gpu": payload["gpu"],
                "precisao": policy.to_dict(),
                "versoes": payload["versoes"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA do MedFlow AI")
    parser.add_argument("--model", default=None, help="ID do modelo base no Hugging Face")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--precision", default=None, choices=["auto", "bf16", "fp16"],
                        help="força a precisão; padrão 'auto' decide pela GPU")
    parser.add_argument("--dry-run", action="store_true", help="valida ambiente e dataset e sai")
    parser.add_argument("--check-env", action="store_true", help="apenas diagnostica o ambiente")
    args = parser.parse_args(argv)

    if args.check_env:
        report = check_environment(QLoRAConfig(precision=args.precision or "auto"))
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        print(report.explain())
        return 0 if report.ready else 1

    config = QLoRAConfig()
    if args.model:
        config.base_model_id = args.model
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.precision:
        config.precision = args.precision

    result = train(
        config, dataset_dir=args.dataset_dir, output_dir=args.output, dry_run=args.dry_run
    )
    print(json.dumps({k: v for k, v in result.items() if k != "log_history"},
                     ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {"ok", "dry_run"} else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
