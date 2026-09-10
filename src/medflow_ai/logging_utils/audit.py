"""Utilidades iniciais para logs de auditoria estruturados."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def pseudonymize(value: str) -> str:
    """Gera identificador estável sem registrar o valor original no log."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def build_audit_event(
    *,
    trace_id: str,
    route: str,
    patient_id: str | None = None,
    sources: list[str] | None = None,
    safety_status: str | None = None,
    human_review: bool = False,
    **extra: Any,
) -> dict[str, Any]:
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


def serialize_event(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, sort_keys=True)
