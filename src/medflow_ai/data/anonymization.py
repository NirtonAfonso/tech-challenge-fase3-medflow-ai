"""Anonimização e pseudonimização de texto clínico (pt-BR).

O Tech Challenge exige demonstrar anonimização mesmo quando os dados já são
sintéticos. Este módulo é determinístico e testável: recebe texto livre ou
registros estruturados e devolve o material com identificadores diretos
removidos, além de um relatório do que foi encontrado.

Escopo intencionalmente explícito (não é um sistema de de-identificação
certificado): cobre os identificadores diretos mais comuns em prontuários
brasileiros. Limitações estão documentadas em ``docs/ANONYMIZATION.md``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping

from medflow_ai.config import get_settings

__all__ = [
    "PIIMatch",
    "AnonymizationReport",
    "anonymize_text",
    "anonymize_record",
    "pseudonymize",
    "age_band",
    "birthdate_to_age",
    "contains_pii",
    "PLACEHOLDERS",
]

PLACEHOLDERS: dict[str, str] = {
    "cpf": "[CPF_REMOVIDO]",
    "cns": "[CNS_REMOVIDO]",
    "rg": "[RG_REMOVIDO]",
    "email": "[EMAIL_REMOVIDO]",
    "telefone": "[TELEFONE_REMOVIDO]",
    "cep": "[CEP_REMOVIDO]",
    "endereco": "[ENDERECO_REMOVIDO]",
    "nome": "[NOME_REMOVIDO]",
    "data_nascimento": "[DATA_NASCIMENTO_REMOVIDA]",
    "data": "[DATA_REMOVIDA]",
    "prontuario": "[PRONTUARIO_REMOVIDO]",
    "crm": "[CRM_REMOVIDO]",
}

# A ordem importa: padrões mais específicos são aplicados primeiro para que um
# CPF não seja capturado por um padrão genérico de número.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # CNS (Cartão Nacional de Saúde): 15 dígitos, com ou sem separadores.
    ("cns", re.compile(r"\b\d{3}[.\s]?\d{4}[.\s]?\d{4}[.\s]?\d{4}\b")),
    # CPF: 000.000.000-00 ou 11 dígitos contíguos.
    ("cpf", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Telefone brasileiro com DDD, fixo ou celular.
    (
        "telefone",
        re.compile(r"(?:\+55[\s-]?)?\(?\d{2}\)?[\s-]?9?\d{4}[\s-]?\d{4}\b"),
    ),
    ("cep", re.compile(r"\b\d{5}-?\d{3}\b")),
    ("rg", re.compile(r"\bRG[:\s]*[\d.\-xX]{7,14}\b", re.IGNORECASE)),
    ("crm", re.compile(r"\bCRM[/\s-]?[A-Z]{0,2}[:\s]*\d{4,7}\b", re.IGNORECASE)),
    (
        "prontuario",
        re.compile(r"\b(?:prontu[áa]rio|matr[íi]cula|registro)\s*(?:n[ºo°.]?\s*)?[:\-]?\s*\d{4,12}\b", re.IGNORECASE),
    ),
    (
        "endereco",
        re.compile(
            r"\b(?:rua|av\.?|avenida|travessa|alameda|pra[çc]a|rodovia)\s+[^\n,;.]{3,60}"
            r"(?:,\s*n[ºo°.]?\s*\d{1,6})?",
            re.IGNORECASE,
        ),
    ),
    (
        "data_nascimento",
        re.compile(
            r"(?P<keep>\b(?:data\s+de\s+nascimento|nascido\s+em|nascida\s+em|dn|d\.n\.)"
            r"[^\S\n]*[:\-]?[^\S\n]*)"
            r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
            re.IGNORECASE,
        ),
    ),
    # Nome próprio precedido de rótulo explícito ("Paciente: Maria da Silva").
    # O rótulo é preservado (grupo ``keep``) para não destruir a estrutura do texto.
    (
        "nome",
        re.compile(
            # ``[^\S\n]`` = espaço que NÃO cruza quebra de linha: impede que o
            # padrão engula o rótulo do campo seguinte ("CPF:", "Contato:").
            r"(?P<keep>\b(?i:paciente|nome\s+completo|nome|sr\.|sra\.|dr\.|dra\.)"
            r"[^\S\n]*[:\-]?[^\S\n]+)"
            r"(?:[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\wÀ-ÿ']{1,20}"
            r"(?:[^\S\n]+(?:d[aeo]s?|e)[^\S\n]+|[^\S\n]+)?){2,5}",
        ),
    ),
)

_DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")

# Campos estruturados considerados identificadores diretos.
DIRECT_IDENTIFIER_FIELDS: frozenset[str] = frozenset(
    {
        "name", "nome", "first", "last", "first_name", "last_name", "given", "family",
        "prefix", "suffix", "maiden", "cpf", "cns", "rg", "ssn", "email", "mail",
        "phone", "telefone", "telecom", "address", "endereco", "street", "line",
        "zip", "cep", "postal_code", "city_of_birth", "birthplace", "passport",
        "drivers", "drivers_license", "responsavel", "mother_name", "nome_mae",
    }
)

# Campos que devem ser transformados (e não apagados).
QUASI_IDENTIFIER_FIELDS: frozenset[str] = frozenset(
    {"birthdate", "birth_date", "data_nascimento", "dob", "nascimento"}
)


@dataclass(frozen=True)
class PIIMatch:
    """Uma ocorrência de identificador direto localizada no texto."""

    kind: str
    start: int
    end: int
    excerpt_hash: str


@dataclass
class AnonymizationReport:
    """Resumo auditável de uma operação de anonimização."""

    matches: list[PIIMatch] = field(default_factory=list)
    fields_removed: list[str] = field(default_factory=list)
    fields_transformed: list[str] = field(default_factory=list)

    @property
    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for match in self.matches:
            counts[match.kind] = counts.get(match.kind, 0) + 1
        return counts

    @property
    def total(self) -> int:
        return len(self.matches) + len(self.fields_removed) + len(self.fields_transformed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts_by_kind": self.counts_by_kind,
            "fields_removed": sorted(self.fields_removed),
            "fields_transformed": sorted(self.fields_transformed),
            "total": self.total,
        }


def pseudonymize(value: str, *, salt: str | None = None, length: int = 16) -> str:
    """Gera um pseudônimo estável (HMAC-like com salt) para um identificador.

    O valor original nunca aparece na saída; com salt configurado por ambiente,
    o hash não é reversível por dicionário de CPFs/IDs conhecidos.
    """
    effective_salt = get_settings().pseudonym_salt if salt is None else salt
    digest = hashlib.sha256(f"{effective_salt}|{value}".encode("utf-8")).hexdigest()
    return digest[:length]


def anonymize_text(text: str, *, redact_dates: bool = False) -> tuple[str, AnonymizationReport]:
    """Remove identificadores diretos de um texto livre.

    Args:
        text: texto clínico bruto.
        redact_dates: quando ``True``, também substitui datas soltas, úteis para
            re-identificação por cruzamento (datas de consulta, por exemplo).

    Returns:
        Tupla ``(texto_anonimizado, relatorio)``.
    """
    report = AnonymizationReport()
    if not text:
        return text, report

    result = text
    for kind, pattern in _PATTERNS:
        placeholder = PLACEHOLDERS[kind]

        def _substitute(match: re.Match[str], kind: str = kind, placeholder: str = placeholder) -> str:
            # Alguns padrões preservam um rótulo ("Paciente:") para manter a
            # legibilidade do texto anonimizado.
            keep = ""
            if "keep" in match.re.groupindex:
                keep = match.group("keep") or ""
            report.matches.append(
                PIIMatch(
                    kind=kind,
                    start=match.start(),
                    end=match.end(),
                    excerpt_hash=pseudonymize(match.group(0)[len(keep):], length=12),
                )
            )
            return f"{keep}{placeholder}"

        result = pattern.sub(_substitute, result)

    if redact_dates:
        def _substitute_date(match: re.Match[str]) -> str:
            report.matches.append(
                PIIMatch(
                    kind="data",
                    start=match.start(),
                    end=match.end(),
                    excerpt_hash=pseudonymize(match.group(0), length=12),
                )
            )
            return PLACEHOLDERS["data"]

        result = _DATE_PATTERN.sub(_substitute_date, result)

    return result, report


def contains_pii(text: str) -> bool:
    """Retorna ``True`` se algum identificador direto for detectado."""
    _, report = anonymize_text(text)
    return bool(report.matches)


def birthdate_to_age(birthdate: str | date | datetime, *, reference: date | None = None) -> int | None:
    """Converte data de nascimento em idade em anos completos."""
    if isinstance(birthdate, datetime):
        parsed: date | None = birthdate.date()
    elif isinstance(birthdate, date):
        parsed = birthdate
    else:
        parsed = None
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(str(birthdate).strip(), fmt).date()
                break
            except ValueError:
                continue
    if parsed is None:
        return None

    today = reference or date.today()
    age = today.year - parsed.year - ((today.month, today.day) < (parsed.month, parsed.day))
    return max(age, 0)


def age_band(age: int | None, *, width: int = 10, cap: int = 90) -> str:
    """Agrupa a idade em faixa etária, com topo agregado para reduzir unicidade."""
    if age is None:
        return "desconhecida"
    if age >= cap:
        return f"{cap}+"
    lower = (age // width) * width
    return f"{lower}-{lower + width - 1}"


def anonymize_record(
    record: Mapping[str, Any],
    *,
    id_fields: Iterable[str] = ("patient_id", "id", "patient"),
    reference_date: date | None = None,
    redact_dates: bool = False,
) -> tuple[dict[str, Any], AnonymizationReport]:
    """Anonimiza um registro estruturado (dict) de prontuário.

    Regras aplicadas:

    * identificadores diretos (nome, CPF, CNS, e-mail, telefone, endereço) são removidos;
    * data de nascimento vira ``age`` e ``age_band``;
    * campos de identificação (``patient_id``) viram pseudônimos estáveis;
    * campos textuais restantes passam pela anonimização de texto livre.
    """
    report = AnonymizationReport()
    anonymized: dict[str, Any] = {}

    for key, value in record.items():
        lowered = key.strip().lower()

        if lowered in DIRECT_IDENTIFIER_FIELDS:
            report.fields_removed.append(key)
            continue

        if lowered in QUASI_IDENTIFIER_FIELDS:
            age = birthdate_to_age(value, reference=reference_date) if value else None
            anonymized["age"] = age
            anonymized["age_band"] = age_band(age)
            report.fields_transformed.append(key)
            continue

        if lowered in {f.lower() for f in id_fields}:
            anonymized[key] = pseudonymize(str(value))
            report.fields_transformed.append(key)
            continue

        if isinstance(value, str):
            cleaned, sub_report = anonymize_text(value, redact_dates=redact_dates)
            report.matches.extend(sub_report.matches)
            anonymized[key] = cleaned
        else:
            anonymized[key] = value

    return anonymized, report
