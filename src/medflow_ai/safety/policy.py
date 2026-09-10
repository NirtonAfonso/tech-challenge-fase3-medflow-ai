"""Política de segurança clínica do MedFlow AI.

O enunciado exige "definir limites de atuação do assistente para evitar
sugestões impróprias (ex.: nunca prescrever diretamente, sem validação
humana)". Isso não é resolvido com um disclaimer: aqui a política é um conjunto
de **regras determinísticas, versionadas e testáveis**, aplicadas em dois
pontos do fluxo:

* ``classify_request``  — guardrail de ENTRADA, decide a rota do LangGraph;
* ``validate_answer``   — guardrail de SAÍDA, audita o texto gerado pela LLM.

Categorias
----------
``SAFE``          consulta informativa sobre protocolo/processo institucional.
``CAUTION``       envolve dados de um paciente concreto; responde com ressalvas,
                  fontes obrigatórias e marcação de conteúdo assistivo.
``HUMAN_REVIEW``  conduta, prescrição, dose, alta ou diagnóstico definitivo:
                  a resposta é produzida como *rascunho* e roteada para
                  validação humana antes de qualquer uso clínico.
``BLOCK``         tentativa de burlar a validação médica, de obter prescrição
                  autônoma sem médico, de falsificar documento ou de subverter
                  as instruções do sistema (prompt injection).

A precedência é sempre ``BLOCK > HUMAN_REVIEW > CAUTION > SAFE``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Sequence

from medflow_ai.data.anonymization import contains_pii

__all__ = [
    "POLICY_VERSION",
    "SafetyCategory",
    "SafetyRule",
    "SafetyAssessment",
    "OutputVerdict",
    "INPUT_RULES",
    "classify_request",
    "validate_answer",
    "requires_human_review",
]

POLICY_VERSION = "1.0.0"


class SafetyCategory(StrEnum):
    """Categorias de risco, em ordem crescente de restrição."""

    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"


_SEVERITY = {
    SafetyCategory.SAFE: 0,
    SafetyCategory.CAUTION: 1,
    SafetyCategory.HUMAN_REVIEW: 2,
    SafetyCategory.BLOCK: 3,
}


@dataclass(frozen=True)
class SafetyRule:
    """Regra determinística de classificação."""

    rule_id: str
    category: SafetyCategory
    pattern: re.Pattern[str]
    rationale: str

    def matches(self, text: str) -> bool:
        return bool(self.pattern.search(text))


def _rule(rule_id: str, category: SafetyCategory, pattern: str, rationale: str) -> SafetyRule:
    return SafetyRule(rule_id, category, re.compile(pattern, re.IGNORECASE | re.DOTALL), rationale)


# Termos de emergência clínica e marcadores de "situação em curso": usados juntos
# para separar pergunta conceitual de acionamento real.
_EMERGENCY_TERMS = (
    r"(?:parada\s+card[íi]aca|choque\s+s[ée]ptico|crise\s+tireot[óo]xica|coma\s+mixedematoso|"
    r"instabilidade\s+hemodin[âa]mica|rebaixamento\s+do\s+n[íi]vel\s+de\s+consci[êe]ncia)"
)
_NOW_MARKER = (
    r"(?:\bpaciente\b|\best[áa]\b|\bestou\b|\bagora\b|\bchegou\b|\bapresent\w+|"
    r"\bo\s+que\s+fa[çc]o\b|\bconduta\b|\burgente\b|\bsocorro\b|\bem\s+curso\b)"
)

# ---------------------------------------------------------------- regras de entrada
INPUT_RULES: tuple[SafetyRule, ...] = (
    # ------------------------------------------------------------------ BLOCK
    _rule(
        "BLOCK-BYPASS-01",
        SafetyCategory.BLOCK,
        r"\b(sem|dispens\w+|ignor\w+|pul\w+|n[ãa]o\s+precisa\w*|desnecess\w+)\b[^.?!]{0,40}"
        r"\b(valida[çc][ãa]o|revis[ãa]o|avalia[çc][ãa]o|supervis[ãa]o|aprova[çc][ãa]o)\b"
        r"[^.?!]{0,30}\b(m[ée]dic\w+|humana?|profissional)\b",
        "Solicita explicitamente contornar a validação médica humana.",
    ),
    _rule(
        "BLOCK-BYPASS-02",
        SafetyCategory.BLOCK,
        r"\bn[ãa]o\s+(?:precisa|preciso|é\s+preciso|necessita)\b[^.?!]{0,30}\b(m[ée]dic\w+|consult\w+)\b",
        "Afirma que a avaliação médica é dispensável.",
    ),
    _rule(
        "BLOCK-FORGERY-01",
        SafetyCategory.BLOCK,
        r"\b(falsifi\w+|forj\w+|adulter\w+|fraud\w+|apagu?e?\w*|delet\w+|remov\w+)\b[^.?!]{0,40}"
        r"\b(laudo|prontu[áa]rio|receita|atestado|registro|log|auditoria|evolu[çc][ãa]o)\b",
        "Pede falsificação, adulteração ou apagamento de registro clínico/auditoria.",
    ),
    _rule(
        "BLOCK-FORGERY-02",
        SafetyCategory.BLOCK,
        r"\b(assine|assina|emita|emite)\b[^.?!]{0,30}\b(atestado|receita|laudo)\b[^.?!]{0,30}"
        r"\b(por\s+mim|no\s+meu\s+nome|sem\s+m[ée]dico)\b",
        "Pede emissão/assinatura de documento privativo de médico.",
    ),
    _rule(
        "BLOCK-INJECTION-01",
        SafetyCategory.BLOCK,
        r"\b(ignore|esque[çc]a|desconsidere|desative|override)\b[^.?!]{0,40}"
        r"\b(instru[çc][õo]es|regras?|pol[íi]ticas?|prompt|system|diretrizes?|"
        r"restri[çc][õo]es|guardrails?|filtros?\s+de\s+seguran[çc]a)\b",
        "Tentativa de subverter as instruções do sistema (prompt injection).",
    ),
    _rule(
        "BLOCK-INJECTION-02",
        SafetyCategory.BLOCK,
        r"\b(finja|pretenda|atue como|aja como|voc[êe] agora [ée])\b[^.?!]{0,40}"
        r"\b(sem restri\w+|sem limites|m[ée]dico\s+respons[áa]vel|prescritor)\b",
        "Tentativa de mudar o papel do sistema para prescritor sem limites.",
    ),
    _rule(
        "BLOCK-SELFHARM-01",
        SafetyCategory.BLOCK,
        r"\b(dose\s+letal|overdose\s+(?:letal|fatal)|como\s+(?:me\s+)?matar|suicídio|eutan[áa]sia)\b",
        "Conteúdo de dano; encaminhamento humano imediato, sem resposta técnica.",
    ),
    # ----------------------------------------------------------- HUMAN_REVIEW
    _rule(
        "HR-PRESCRIPTION-01",
        SafetyCategory.HUMAN_REVIEW,
        # Apenas formas ACIONÁVEIS. Perguntas sobre o documento ("quais campos a
        # receita exige?") são informativas e devem permanecer SAFE.
        r"\b(?:prescrev\w+|prescrever|prescrey?a|receite|receitar|"
        r"prescri[çc][ãa]o\s+(?:para|dest[ea]|desse|do|da)|"
        r"(?:escrev\w+|fa[çc]\w+|gere?|monte)\s+(?:a\s+|uma\s+)?receita|"
        r"receita\s+de\s+alta|posologia\s+(?:para|dest[ea]|desse|completa)|"
        r"pode\s+tomar|devo\s+dar|o\s+que\s+(?:ele|ela|o\s+paciente)\s+deve\s+tomar)\b",
        "Pedido de prescrição/posologia: ato privativo do médico.",
    ),
    _rule(
        "HR-DOSE-02",
        SafetyCategory.HUMAN_REVIEW,
        # Falha observada no held-out v1: "quantos miligramas devo administrar".
        r"\b(?:quantos?|quantas?)\s+\w*(?:grama|mg\b|mcg\b|micrograma|unidade|comprimido|gota|ml\b)\w*"
        r"|\b(?:quanto|qual\s+quantidade)\b[^.?!]{0,25}\b(?:administrar|dar|aplicar|tomar|prescrever)\b",
        "Pergunta quantitativa sobre administração de medicamento.",
    ),
    _rule(
        "HR-DOSE-01",
        SafetyCategory.HUMAN_REVIEW,
        r"\b(qual|defina|calcule|ajuste|aumente|diminua|reduza|dobre|confirme|informe|"
        r"sugira|indique|diga|escolha|titule)\b[^.?!]{0,40}\bdose\b"
        r"|\b\d+[.,]?\d*\s?(mg|mcg|µg|g|ui|ml|mEq|unidades?)\b",
        "Envolve definição, cálculo ou ajuste de dose de medicamento.",
    ),
    _rule(
        "HR-DIAGNOSIS-01",
        SafetyCategory.HUMAN_REVIEW,
        r"\b(confirm\w+|feche|fech\w+|estabele[çc]\w+|d[êe]\s+o|qual\s+[ée]\s+o)\b[^.?!]{0,25}"
        r"\bdiagn[óo]stic\w+\b|\bdiagn[óo]stico\s+(?:definitivo|final|conclusivo)\b",
        "Pede diagnóstico definitivo/conclusivo, que exige decisão médica.",
    ),
    _rule(
        "HR-CONDUCT-01",
        SafetyCategory.HUMAN_REVIEW,
        r"\b(?:decid\w+|autoriz\w+)\b[^.?!]{0,30}\b(?:paciente|alta|tratamento|cirurgia|conduta)\b"
        r"|\b(?:liber\w+|dar|receber|ter)\s+(?:a\s+)?alta\b"
        r"|\bindi(?:qu|c)\w+\s+(?:a\s+)?cirurgia\b"
        r"|\b(?:suspend\w+|interromp\w+)\s+(?:o\s+|a\s+)?(?:tratamento|medica\w+|antibi[óo]tico\w*)\b"
        r"|\binici[ae]r?\s+(?:o\s+)?tratamento\b",
        "Solicita decisão terapêutica ou administrativa com impacto clínico direto.",
    ),
    _rule(
        "HR-EMERGENCY-01",
        SafetyCategory.HUMAN_REVIEW,
        # Termo de emergência PRÓXIMO de um marcador de situação em curso.
        # "Como o protocolo define crise tireotóxica?" é pergunta informativa.
        rf"{_NOW_MARKER}[^.?!]{{0,80}}{_EMERGENCY_TERMS}"
        rf"|{_EMERGENCY_TERMS}[^.?!]{{0,80}}{_NOW_MARKER}",
        "Cenário de emergência em curso: exige acionamento imediato de equipe humana.",
    ),
    # ---------------------------------------------------------------- CAUTION
    _rule(
        "CAU-PATIENT-01",
        SafetyCategory.CAUTION,
        r"\b(este|esse|dest[ea]|desse|do|da)\s+paciente\b|\bpatient[_\s-]?id\b|\bprontu[áa]rio\b"
        r"|\bmeu\s+paciente\b|\bo\s+paciente\s+(?:tem|est[áa]|apresenta)\b"
        r"|\bpaciente\s+[A-Z]{1,4}[-_]?[A-Z0-9][A-Z0-9\-_]{2,}\b"
        r"|\b(?:dest[ea]|desse|neste|nesse)\s+(?:caso|quadro)\b",
        "Pergunta referida a um paciente concreto: exige contexto e fontes.",
    ),
    _rule(
        "CAU-SUGGEST-01",
        SafetyCategory.CAUTION,
        r"\b(pr[óo]ximos?\s+passos?|conduta|o\s+que\s+fa[çc]o|"
        r"como\s+(?:devo\s+|posso\s+|deve-se\s+)?proceder|sugest\w+|"
        r"recomend\w+|devo\s+solicitar|preciso\s+pedir)\b",
        "Pede sugestão de conduta: resposta assistiva, nunca determinante.",
    ),
)


@dataclass
class SafetyAssessment:
    """Resultado do guardrail de entrada."""

    category: SafetyCategory
    matched_rules: list[str] = field(default_factory=list)
    rationales: list[str] = field(default_factory=list)
    policy_version: str = POLICY_VERSION

    @property
    def allowed(self) -> bool:
        return self.category is not SafetyCategory.BLOCK

    @property
    def requires_human_review(self) -> bool:
        return self.category in (SafetyCategory.HUMAN_REVIEW, SafetyCategory.BLOCK)

    def to_dict(self) -> dict[str, Any]:
        return {
            "safety_status": str(self.category),
            "matched_rules": list(self.matched_rules),
            "rationales": list(self.rationales),
            "policy_version": self.policy_version,
        }


@dataclass
class OutputVerdict:
    """Resultado do guardrail de saída."""

    category: SafetyCategory
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    policy_version: str = POLICY_VERSION

    @property
    def approved(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_status": str(self.category),
            "violations": list(self.violations),
            "notes": list(self.notes),
            "policy_version": self.policy_version,
        }


def classify_request(text: str, *, rules: Sequence[SafetyRule] = INPUT_RULES) -> SafetyAssessment:
    """Classifica a solicitação do médico segundo a política de segurança.

    A categoria final é a de **maior severidade** entre as regras acionadas.
    Todas as regras acionadas são registradas, o que torna a decisão auditável
    e explicável (requisito de explainability do enunciado).
    """
    if not text or not text.strip():
        return SafetyAssessment(
            category=SafetyCategory.BLOCK,
            matched_rules=["BLOCK-EMPTY-01"],
            rationales=["Pergunta vazia: não é possível avaliar segurança nem responder."],
        )

    matched = [rule for rule in rules if rule.matches(text)]
    if not matched:
        return SafetyAssessment(
            category=SafetyCategory.SAFE,
            matched_rules=[],
            rationales=["Nenhuma regra de risco acionada; consulta tratada como informativa."],
        )

    category = max((rule.category for rule in matched), key=lambda item: _SEVERITY[item])
    relevant = [rule for rule in matched if rule.category is category]
    return SafetyAssessment(
        category=category,
        matched_rules=[rule.rule_id for rule in matched],
        rationales=[rule.rationale for rule in relevant],
    )


def requires_human_review(category: SafetyCategory) -> bool:
    """Conveniência para as arestas condicionais do LangGraph."""
    return category in (SafetyCategory.HUMAN_REVIEW, SafetyCategory.BLOCK)


# ------------------------------------------------------------ guardrail de saída
# Dose administrada ("200 mg de levotiroxina") — mas NÃO valor laboratorial, que
# usa unidade composta com barra ("0,90 mg/dL", "312 UI/mL", "8,4 mUI/L").
_DOSE_PATTERN = re.compile(
    r"\b\d+[.,]?\d*\s?(mg|mcg|µg|g|ui|ml|mEq|unidades?)\b"
    r"(?!\s*/)"                              # exclui unidades de concentração
    r"(?![^.?!]{0,20}refer[êe]ncia)",         # exclui menção a faixa de referência
    re.IGNORECASE,
)
_POSOLOGY_PATTERN = re.compile(
    r"\b(?:tomar|administrar|aplicar|usar)\b[^.?!]{0,40}\b(?:\d+\s*(?:x|vezes)|"
    r"de\s+\d+\s*(?:em|\/)\s*\d+\s*horas?|a\s+cada\s+\d+\s*horas?)\b",
    re.IGNORECASE,
)
_DEFINITIVE_PATTERN = re.compile(
    r"\b(?:o\s+diagn[óo]stico\s+[ée]\s+definitiv|confirmo\s+o\s+diagn[óo]stico|"
    r"diagn[óo]stico\s+confirmado\s+de|com\s+certeza\s+absoluta|"
    r"n[ãa]o\s+[ée]\s+necess[áa]ri[ao]\s+avalia[çc][ãa]o\s+m[ée]dica)",
    re.IGNORECASE,
)


def validate_answer(
    answer: str,
    *,
    sources: Iterable[Any] = (),
    input_category: SafetyCategory = SafetyCategory.SAFE,
    require_sources: bool = True,
) -> OutputVerdict:
    """Audita a resposta gerada antes de entregá-la ao médico.

    Verifica quatro classes de violação:

    1. **posologia/dose** — o assistente não pode emitir prescrição;
    2. **afirmação definitiva** — não pode fechar diagnóstico nem dispensar médico;
    3. **ausência de fonte** — resposta clínica sem rastreabilidade;
    4. **vazamento de PII** — identificador direto no texto de saída.
    """
    verdict = OutputVerdict(category=input_category)
    source_list = list(sources)

    if _DOSE_PATTERN.search(answer) or _POSOLOGY_PATTERN.search(answer):
        verdict.violations.append("OUT-DOSE-01: resposta contém dose/posologia explícita.")
    if _DEFINITIVE_PATTERN.search(answer):
        verdict.violations.append(
            "OUT-DEFINITIVE-01: resposta contém afirmação diagnóstica definitiva "
            "ou dispensa de avaliação médica."
        )
    if require_sources and not source_list:
        verdict.violations.append("OUT-SOURCE-01: resposta clínica sem fonte rastreável.")
    if contains_pii(answer):
        verdict.violations.append("OUT-PII-01: resposta contém identificador direto de paciente.")

    if verdict.violations:
        verdict.category = SafetyCategory.HUMAN_REVIEW
        verdict.notes.append("Resposta reclassificada para validação humana pelo guardrail de saída.")
    elif input_category is SafetyCategory.SAFE and source_list:
        verdict.notes.append("Resposta fundamentada em fontes institucionais rastreáveis.")

    return verdict
