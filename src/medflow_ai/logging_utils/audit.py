"""Logs estruturados e trilha de auditoria do MedFlow AI.

Requisito do enunciado: "implementar logging detalhado para rastreamento e
auditoria". O que é registrado precisa ser suficiente para reconstruir a
decisão do sistema **sem** guardar dado identificável do paciente.

Cada execução do fluxo gera um evento JSON com ``trace_id``, rota, passos
executados, fontes citadas, status de segurança, latência e resultado. O evento
passa por ``redact`` antes de ser gravado: chaves sensíveis são removidas e
texto livre é anonimizado.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from medflow_ai.config import get_settings

__all__ = [
    "pseudonymize",
    "redact",
    "build_audit_event",
    "serialize_event",
    "AuditLogger",
    "get_audit_logger",
    "SENSITIVE_KEYS",
]

# Chaves que jamais podem ser persistidas em log.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "name", "nome", "first", "last", "cpf", "cns", "rg", "email", "phone",
        "telefone", "address", "endereco", "zip", "cep", "birthdate", "password",
        "senha", "token", "api_key", "apikey", "authorization", "secret",
        "openai_api_key", "huggingface_token", "langchain_api_key",
    }
)

_REDACTED = "[REDACTED]"


def pseudonymize(value: str, *, salt: str | None = None, length: int = 16) -> str:
    """Hash estável e salgado de um identificador (não expõe o valor original)."""
    effective_salt = get_settings().pseudonym_salt if salt is None else salt
    return hashlib.sha256(f"{effective_salt}|{value}".encode("utf-8")).hexdigest()[:length]


def redact(payload: Any, *, anonymize_strings: bool = True) -> Any:
    """Remove chaves sensíveis e anonimiza texto livre recursivamente."""
    # Import local evita ciclo de importação com o pacote de dados.
    from medflow_ai.data.anonymization import anonymize_text

    if isinstance(payload, Mapping):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if str(key).strip().lower() in SENSITIVE_KEYS:
                cleaned[str(key)] = _REDACTED
            else:
                cleaned[str(key)] = redact(value, anonymize_strings=anonymize_strings)
        return cleaned
    if isinstance(payload, (list, tuple)):
        return [redact(item, anonymize_strings=anonymize_strings) for item in payload]
    if isinstance(payload, str) and anonymize_strings:
        return anonymize_text(payload)[0]
    return payload


def build_audit_event(
    *,
    trace_id: str,
    route: str,
    patient_id: str | None = None,
    sources: list[Any] | None = None,
    safety_status: str | None = None,
    human_review: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Monta o evento de auditoria padrão do projeto."""
    event: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "trace_id": trace_id,
        "route": route,
        "patient_id_hash": pseudonymize(patient_id) if patient_id else None,
        "sources": sources or [],
        "safety_status": safety_status,
        "human_review": human_review,
    }
    event.update(extra)
    return event


def serialize_event(event: Mapping[str, Any]) -> str:
    """Serializa o evento em JSON estável (chaves ordenadas)."""
    return json.dumps(event, ensure_ascii=False, sort_keys=True, default=str)


class AuditLogger:
    """Escreve eventos de auditoria em JSONL e no logger padrão do Python.

    O arquivo JSONL é a trilha auditável (uma linha por execução do fluxo);
    o logger padrão facilita a demonstração em terminal e no vídeo.
    """

    def __init__(self, log_path: Path | str | None = None, *, logger_name: str = "medflow.audit") -> None:
        settings = get_settings()
        self.log_path = Path(log_path or (settings.log_dir / "audit.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(logger_name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(settings.log_level.upper())

    def log(self, event: Mapping[str, Any], *, level: int = logging.INFO) -> dict[str, Any]:
        """Redige, persiste e emite um evento. Retorna o evento efetivamente gravado."""
        safe_event = redact(dict(event))
        line = serialize_event(safe_event)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        self._logger.log(level, line)
        return safe_event

    def read_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Lê a trilha de auditoria (usado em testes e na demonstração)."""
        if not self.log_path.exists():
            return []
        with self.log_path.open(encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        return events[-limit:] if limit else events


_default_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Instância compartilhada do logger de auditoria."""
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger()
    return _default_logger
