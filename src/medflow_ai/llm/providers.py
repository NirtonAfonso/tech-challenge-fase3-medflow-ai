"""Provedores de LLM intercambiáveis.

Três backends, selecionados por ``MEDFLOW_LLM_PROVIDER``:

``template`` (padrão)
    **Baseline extrativo determinístico**, sem rede e sem GPU. Não é um modelo
    de linguagem: seleciona as sentenças mais similares à pergunta dentro dos
    trechos de protocolo recuperados e as organiza no formato de saída exigido.
    Existe para que o fluxo LangGraph completo — incluindo safety, fontes e
    auditoria — seja executável e testável em CI, e para servir de **referência
    de comparação** contra o modelo base e o modelo fine-tuned.

``hf_local``
    ``transformers`` + (opcionalmente) adapter LoRA/QLoRA treinado no Colab.
    É o caminho usado para demonstrar a LLM customizada.

``openai``
    Provedor externo opcional, desacoplado; exige ``OPENAI_API_KEY``.

Todos implementam ``BaseChatModel`` do LangChain e são intercambiáveis dentro
do mesmo grafo.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from medflow_ai.config import get_settings

__all__ = ["TemplateChatModel", "HuggingFaceChatModel", "get_chat_model", "describe_provider"]

_BLOCK_PATTERN = re.compile(r"<<(?P<name>[A-Z_]+)>>(?P<body>.*?)<</(?P=name)>>", re.DOTALL)
_CITATION_PATTERN = re.compile(r"^\[(?P<label>[^\]]+)\]\s*$", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ0-9])")
# Placeholders de modelos de laudo/receita: [CAMPO_EM_MAIUSCULAS]
_PLACEHOLDER_PATTERN = re.compile(r"\[[A-ZÁÉÍÓÚÂÊÔÃÕÇ_0-9/ ]{3,}\]")


def _parse_blocks(text: str) -> dict[str, str]:
    return {match.group("name"): match.group("body").strip() for match in _BLOCK_PATTERN.finditer(text)}


class TemplateChatModel(BaseChatModel):
    """Baseline extrativo determinístico (sem modelo de linguagem).

    Recebe o prompt clínico já montado, extrai os blocos delimitados e monta a
    resposta selecionando as sentenças de protocolo mais próximas da pergunta.
    Sempre cita a fonte da sentença escolhida e nunca inventa conteúdo, o que o
    torna um piso de *groundedness* muito útil como comparação.
    """

    max_sentences: int = 4
    min_similarity: float = 0.06

    @property
    def _llm_type(self) -> str:
        return "medflow-template-extractive"

    @property
    def provider_name(self) -> str:
        return "template"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(message.content) for message in messages)
        answer = self._compose(_parse_blocks(prompt))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    # ------------------------------------------------------------------ lógica
    def _compose(self, blocks: dict[str, str]) -> str:
        from medflow_ai.llm.prompts import INSUFFICIENT_EVIDENCE

        question = blocks.get("PERGUNTA", "").strip()
        patient_block = blocks.get("CONTEXTO_PACIENTE", "").strip()
        protocol_block = blocks.get("PROTOCOLOS", "").strip()

        patient_used = self._patient_summary(patient_block)
        candidates = self._candidate_sentences(protocol_block)
        selected = self._select(question, candidates)

        if not selected:
            return INSUFFICIENT_EVIDENCE.format(patient_used=patient_used)

        body = " ".join(f"{sentence} [{label}]" for sentence, label in selected)
        alerts = self._alerts(patient_block)
        return (
            f"RESPOSTA: {body}\n"
            f"CONTEXTO DO PACIENTE: {patient_used}\n"
            f"PENDÊNCIAS E ALERTAS: {alerts}\n"
            "LIMITAÇÃO: conteúdo assistivo baseado em protocolos institucionais; "
            "a conduta final é do médico assistente."
        )

    @staticmethod
    def _candidate_sentences(protocol_block: str) -> list[tuple[str, str]]:
        """Sentenças dos protocolos recuperados, cada uma com seu rótulo de citação."""
        sentences: list[tuple[str, str]] = []
        current_label = "fonte não identificada"
        for line in protocol_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            citation = _CITATION_PATTERN.match(stripped)
            if citation:
                current_label = citation.group("label").strip()
                continue
            if stripped.startswith("#") or stripped.startswith("```"):
                continue
            # Linhas de modelo de documento ("IDENTIFICAÇÃO: [PSEUDONIMO_PACIENTE]")
            # não são afirmações clínicas e não devem virar resposta.
            if _PLACEHOLDER_PATTERN.search(stripped):
                continue
            for sentence in _SENTENCE_SPLIT.split(stripped):
                cleaned = sentence.strip()
                if len(cleaned) >= 40:
                    sentences.append((cleaned, current_label))
        return sentences

    def _select(self, question: str, candidates: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        if not candidates or not question:
            return []
        from medflow_ai.rag.embeddings import get_embeddings

        embeddings = get_embeddings("hashing")
        query_vector = np.asarray(embeddings.embed_query(question), dtype=np.float32)
        matrix = np.asarray(
            embeddings.embed_documents([sentence for sentence, _ in candidates]), dtype=np.float32
        )
        scores = matrix @ query_vector

        order = np.argsort(-scores)
        selected: list[tuple[str, str]] = []
        seen: set[str] = set()
        for index in order[: self.max_sentences * 3]:
            if float(scores[index]) < self.min_similarity:
                break
            sentence, label = candidates[int(index)]
            key = sentence[:60].casefold()
            if key in seen:
                continue
            seen.add(key)
            selected.append((sentence, label))
            if len(selected) >= self.max_sentences:
                break
        return selected

    @staticmethod
    def _patient_summary(patient_block: str) -> str:
        if not patient_block or "Nenhum contexto" in patient_block:
            return "não utilizado"
        keep = [
            line.strip()
            for line in patient_block.splitlines()
            if line.strip().startswith(("Faixa etária", "Condições", "Exames", "Medicamentos", "-"))
        ]
        return "; ".join(keep[:8]) if keep else "não utilizado"

    @staticmethod
    def _alerts(patient_block: str) -> str:
        alerts: list[str] = []
        lines = patient_block.splitlines()
        for index, line in enumerate(lines):
            if line.strip().startswith("Exames pendentes"):
                alerts += [
                    item.strip(" -")
                    for item in lines[index + 1 : index + 5]
                    if item.strip().startswith("-")
                ]
        return "; ".join(alerts) if alerts else "nenhuma"


class HuggingFaceChatModel(BaseChatModel):
    """Modelo local via ``transformers``, com adapter PEFT opcional.

    Carrega o modelo base e, quando ``MEDFLOW_ADAPTER_PATH`` aponta para um
    adapter LoRA/QLoRA treinado, aplica-o por cima. Usa o *chat template* do
    tokenizer, o mesmo formato usado no fine-tuning.
    """

    model_id: str = ""
    adapter_path: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.2
    load_in_4bit: bool = True
    _pipeline: Any = None

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    @property
    def _llm_type(self) -> str:
        return "medflow-huggingface"

    @property
    def provider_name(self) -> str:
        return f"hf_local:{self.model_id}" + (f"+adapter" if self.adapter_path else "")

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependência pesada
            raise ImportError(
                "Provedor 'hf_local' requer 'transformers' e 'torch'. "
                "Instale com: pip install -r requirements-training.txt"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model_kwargs: dict[str, Any] = {"dtype": "auto", "device_map": "auto"}
        if self.load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:  # pragma: no cover - bitsandbytes ausente
                pass

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._pipeline = (tokenizer, model)
        return self._pipeline

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        tokenizer, model = self._ensure_pipeline()
        chat = [
            {"role": _role_of(message), "content": str(message.content)} for message in messages
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-4),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=generated.strip()))])


def _role_of(message: BaseMessage) -> str:
    mapping = {"system": "system", "human": "user", "ai": "assistant"}
    return mapping.get(message.type, "user")


def get_chat_model(provider: str | None = None, **kwargs: Any) -> BaseChatModel:
    """Fábrica de modelos de chat dirigida por configuração."""
    settings = get_settings()
    chosen = (provider or settings.llm_provider).strip().lower()

    if chosen in {"template", "extractive", "offline"}:
        return TemplateChatModel(**kwargs)

    if chosen in {"hf_local", "huggingface", "hf"}:
        return HuggingFaceChatModel(
            model_id=kwargs.pop("model_id", settings.base_model_id),
            adapter_path=kwargs.pop("adapter_path", settings.adapter_path),
            max_new_tokens=kwargs.pop("max_new_tokens", settings.max_new_tokens),
            temperature=kwargs.pop("temperature", settings.temperature),
            **kwargs,
        )

    if chosen == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - dependência opcional
            raise ImportError(
                "Provedor 'openai' requer 'langchain-openai'. Instale a dependência opcional."
            ) from exc
        return ChatOpenAI(
            model=kwargs.pop("model", "gpt-4o-mini"),
            temperature=kwargs.pop("temperature", settings.temperature),
            **kwargs,
        )

    raise ValueError(
        f"Provedor de LLM desconhecido: {chosen!r}. Use 'template', 'hf_local' ou 'openai'."
    )


def describe_provider(model: BaseChatModel) -> str:
    """Rótulo do provedor usado no log de auditoria e nos artefatos de avaliação."""
    return str(getattr(model, "provider_name", getattr(model, "_llm_type", type(model).__name__)))
