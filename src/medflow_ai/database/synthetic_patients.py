"""Gerador determinístico de prontuários sintéticos.

Por que gerar em vez de baixar o Synthea: o Synthea exige Java e produz saída
de centenas de MB, inviável em CI. Este gerador produz o **mesmo esquema** do
export CSV do Synthea, com volume pequeno e conteúdo clinicamente coerente com
os protocolos do corpus institucional — e é totalmente determinístico dado um
seed, o que torna os testes de recuperação exata reproduzíveis.

Os registros incluem propositalmente identificadores diretos falsos (nome, CPF,
CNS, e-mail, telefone, endereço) para que o pipeline de anonimização tenha algo
concreto para remover e para que a demonstração antes/depois seja verificável.

**Nenhum dado real de paciente é usado ou versionado.**
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

__all__ = ["SyntheticDataset", "generate_dataset", "REFERENCE_DATE", "DEMO_PATIENT_ID"]

# Data de referência fixa: garante idades e datas estáveis entre execuções.
REFERENCE_DATE = date(2026, 3, 1)
DEMO_PATIENT_ID = "P-DEMO-0001"

_FIRST_NAMES = [
    "Ana", "Bruno", "Carla", "Diego", "Elisa", "Fábio", "Gabriela", "Heitor",
    "Isabela", "João", "Karina", "Lucas", "Mariana", "Nelson", "Olívia",
    "Paulo", "Renata", "Sérgio", "Tatiana", "Vitor",
]
_LAST_NAMES = [
    "Almeida", "Barbosa", "Cardoso", "Duarte", "Esteves", "Ferreira", "Gomes",
    "Henriques", "Ibrahim", "Jorge", "Klein", "Lima", "Moreira", "Nogueira",
    "Oliveira", "Pereira", "Queiroz", "Ramos", "Santos", "Teixeira",
]
_STREETS = [
    "Rua das Acácias", "Avenida Bandeirantes", "Rua Doutor Freire", "Travessa São Jorge",
    "Alameda dos Ipês", "Avenida Portugal", "Rua Marechal Deodoro",
]
_CITIES = [("São Paulo", "SP"), ("Belo Horizonte", "MG"), ("Curitiba", "PR"), ("Recife", "PE")]


@dataclass(frozen=True)
class ObservationSpec:
    """Especificação de um exame com faixa de valores plausível."""

    description: str
    code: str
    units: str
    low: float
    high: float
    decimals: int = 2


@dataclass(frozen=True)
class ClinicalProfile:
    """Perfil clínico sintético coerente com um protocolo institucional."""

    key: str
    label: str
    protocol_doc_id: str
    conditions: tuple[str, ...]
    observations: tuple[ObservationSpec, ...]
    medications: tuple[str, ...]
    procedures: tuple[str, ...] = ()
    encounter_class: str = "ambulatorial"
    pending_exams: tuple[str, ...] = ()
    age_range: tuple[int, int] = (30, 80)


PROFILES: tuple[ClinicalProfile, ...] = (
    ClinicalProfile(
        key="hipotireoidismo_primario",
        label="Hipotireoidismo primário em acompanhamento",
        protocol_doc_id="PROT-END-001",
        conditions=("Hipotireoidismo primário", "Dislipidemia"),
        observations=(
            ObservationSpec("TSH", "3016-3", "mUI/L", 5.5, 14.0),
            ObservationSpec("T4 livre", "3024-7", "ng/dL", 0.45, 0.85),
            ObservationSpec("Anti-TPO", "8099-6", "UI/mL", 90.0, 600.0, 0),
            ObservationSpec("Colesterol LDL", "13457-7", "mg/dL", 110.0, 190.0, 0),
        ),
        medications=("Levotiroxina sódica comprimido",),
        pending_exams=("TSH de controle", "T4 livre de controle"),
        age_range=(35, 78),
    ),
    ClinicalProfile(
        key="hipotireoidismo_subclinico",
        label="Hipotireoidismo subclínico em investigação",
        protocol_doc_id="PROT-END-001",
        conditions=("Hipotireoidismo subclínico",),
        observations=(
            ObservationSpec("TSH", "3016-3", "mUI/L", 4.6, 8.9),
            ObservationSpec("T4 livre", "3024-7", "ng/dL", 0.95, 1.5),
        ),
        medications=(),
        pending_exams=("TSH de confirmação (segunda dosagem)",),
        age_range=(28, 70),
    ),
    ClinicalProfile(
        key="hipertireoidismo",
        label="Tireotoxicose em investigação",
        protocol_doc_id="PROT-END-002",
        conditions=("Tireotoxicose", "Fibrilação atrial"),
        observations=(
            ObservationSpec("TSH", "3016-3", "mUI/L", 0.001, 0.09, 3),
            ObservationSpec("T4 livre", "3024-7", "ng/dL", 2.1, 4.8),
            ObservationSpec("Frequência cardíaca", "8867-4", "bpm", 96.0, 132.0, 0),
        ),
        medications=("Metimazol comprimido", "Propranolol comprimido"),
        pending_exams=("TRAb", "Cintilografia de tireoide"),
        age_range=(25, 72),
    ),
    ClinicalProfile(
        key="diabetes_tipo_2",
        label="Diabetes mellitus tipo 2 em acompanhamento",
        protocol_doc_id="PROT-END-003",
        conditions=("Diabetes mellitus tipo 2", "Obesidade"),
        observations=(
            ObservationSpec("Hemoglobina glicada", "4548-4", "%", 6.8, 11.2, 1),
            ObservationSpec("Glicemia de jejum", "1558-6", "mg/dL", 118.0, 265.0, 0),
            ObservationSpec("Creatinina", "2160-0", "mg/dL", 0.7, 1.9),
            ObservationSpec("Relação albumina/creatinina urinária", "9318-7", "mg/g", 12.0, 320.0, 0),
        ),
        medications=("Metformina comprimido",),
        pending_exams=("Retinografia anual", "Relação albumina/creatinina urinária"),
        age_range=(40, 82),
    ),
    ClinicalProfile(
        key="hipertensao",
        label="Hipertensão arterial sistêmica",
        protocol_doc_id="PROT-CAR-002",
        conditions=("Hipertensão arterial sistêmica",),
        observations=(
            ObservationSpec("Pressão arterial sistólica", "8480-6", "mmHg", 132.0, 186.0, 0),
            ObservationSpec("Pressão arterial diastólica", "8462-4", "mmHg", 84.0, 112.0, 0),
            ObservationSpec("Potássio", "2823-3", "mEq/L", 3.2, 5.4, 1),
            ObservationSpec("Creatinina", "2160-0", "mg/dL", 0.7, 1.6),
        ),
        medications=("Losartana comprimido", "Hidroclorotiazida comprimido"),
        pending_exams=("MAPA 24 horas",),
        age_range=(38, 85),
    ),
    ClinicalProfile(
        key="dor_toracica",
        label="Dor torácica em investigação no pronto-socorro",
        protocol_doc_id="PROT-CAR-001",
        conditions=("Dor torácica a esclarecer", "Doença arterial coronariana"),
        observations=(
            ObservationSpec("Troponina ultrassensível", "67151-1", "ng/L", 4.0, 180.0, 0),
            ObservationSpec("Frequência cardíaca", "8867-4", "bpm", 62.0, 118.0, 0),
            ObservationSpec("Pressão arterial sistólica", "8480-6", "mmHg", 96.0, 168.0, 0),
        ),
        medications=("Ácido acetilsalicílico comprimido",),
        procedures=("Eletrocardiograma de 12 derivações",),
        encounter_class="emergencia",
        pending_exams=("Troponina ultrassensível (segunda amostra)", "Ecocardiograma transtorácico"),
        age_range=(45, 84),
    ),
    ClinicalProfile(
        key="sepse",
        label="Sepse de foco urinário em internação",
        protocol_doc_id="PROT-INF-001",
        conditions=("Sepse", "Infecção do trato urinário"),
        observations=(
            ObservationSpec("Lactato", "2524-7", "mmol/L", 2.1, 6.4, 1),
            ObservationSpec("Leucócitos", "6690-2", "/mm3", 14000.0, 28000.0, 0),
            ObservationSpec("Creatinina", "2160-0", "mg/dL", 1.1, 3.2),
            ObservationSpec("Pressão arterial média", "8478-0", "mmHg", 54.0, 78.0, 0),
        ),
        medications=("Antimicrobiano de amplo espectro conforme padronização",),
        procedures=("Coleta de hemocultura em dois sítios",),
        encounter_class="internacao",
        pending_exams=("Hemocultura (resultado final)", "Lactato de controle"),
        age_range=(52, 88),
    ),
    ClinicalProfile(
        key="pneumonia",
        label="Pneumonia adquirida na comunidade",
        protocol_doc_id="PROT-INF-002",
        conditions=("Pneumonia adquirida na comunidade",),
        observations=(
            ObservationSpec("Saturação de oxigênio", "2708-6", "%", 86.0, 95.0, 0),
            ObservationSpec("Proteína C reativa", "1988-5", "mg/L", 48.0, 260.0, 0),
            ObservationSpec("Ureia", "3094-0", "mg/dL", 28.0, 92.0, 0),
        ),
        medications=("Antimicrobiano conforme padronização institucional",),
        procedures=("Radiografia de tórax",),
        encounter_class="internacao",
        pending_exams=("Hemocultura (resultado final)",),
        age_range=(30, 86),
    ),
    ClinicalProfile(
        key="asma",
        label="Asma com exacerbação recente",
        protocol_doc_id="PROT-PNE-001",
        conditions=("Asma persistente moderada",),
        observations=(
            ObservationSpec("Saturação de oxigênio", "2708-6", "%", 89.0, 97.0, 0),
            ObservationSpec("Pico de fluxo expiratório", "33452-4", "L/min", 180.0, 420.0, 0),
        ),
        medications=("Corticoide inalatório com formoterol", "Salbutamol inalatório"),
        encounter_class="emergencia",
        pending_exams=("Espirometria ambulatorial",),
        age_range=(18, 65),
    ),
    ClinicalProfile(
        key="injuria_renal",
        label="Injúria renal aguda em investigação",
        protocol_doc_id="PROT-NEF-001",
        conditions=("Injúria renal aguda", "Hipertensão arterial sistêmica"),
        observations=(
            ObservationSpec("Creatinina", "2160-0", "mg/dL", 1.8, 4.6),
            ObservationSpec("Potássio", "2823-3", "mEq/L", 4.8, 6.8, 1),
            ObservationSpec("Débito urinário 24h", "9187-6", "mL", 220.0, 900.0, 0),
        ),
        medications=("Furosemida comprimido",),
        encounter_class="internacao",
        pending_exams=("Ultrassonografia de rins e vias urinárias", "Creatinina de controle"),
        age_range=(48, 88),
    ),
    ClinicalProfile(
        key="avc",
        label="AVC isquêmico agudo",
        protocol_doc_id="PROT-NEU-001",
        conditions=("Acidente vascular cerebral isquêmico",),
        observations=(
            ObservationSpec("Glicemia capilar", "2339-0", "mg/dL", 78.0, 240.0, 0),
            ObservationSpec("Pressão arterial sistólica", "8480-6", "mmHg", 138.0, 208.0, 0),
        ),
        medications=(),
        procedures=("Tomografia de crânio sem contraste",),
        encounter_class="emergencia",
        pending_exams=("Ecocardiograma transtorácico", "Doppler de carótidas"),
        age_range=(55, 90),
    ),
)

_PROFILE_BY_KEY = {profile.key: profile for profile in PROFILES}


@dataclass
class SyntheticDataset:
    """Conjunto de tabelas sintéticas prontas para carga no SQLite."""

    patients: list[dict[str, Any]] = field(default_factory=list)
    encounters: list[dict[str, Any]] = field(default_factory=list)
    conditions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    medications: list[dict[str, Any]] = field(default_factory=list)
    procedures: list[dict[str, Any]] = field(default_factory=list)
    lab_orders: list[dict[str, Any]] = field(default_factory=list)

    def as_tables(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "patients": self.patients,
            "encounters": self.encounters,
            "conditions": self.conditions,
            "observations": self.observations,
            "medications": self.medications,
            "procedures": self.procedures,
            "lab_orders": self.lab_orders,
        }

    def summary(self) -> dict[str, int]:
        return {name: len(rows) for name, rows in self.as_tables().items()}


def _digits(rng: random.Random, count: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(count))


def _fake_identity(rng: random.Random) -> dict[str, str]:
    """Identificadores diretos FALSOS, presentes para exercitar a anonimização."""
    first = rng.choice(_FIRST_NAMES)
    last = f"{rng.choice(_LAST_NAMES)} {rng.choice(_LAST_NAMES)}"
    city, state = rng.choice(_CITIES)
    slug = f"{first}.{last.split()[0]}".lower()
    return {
        "first": first,
        "last": last,
        "cpf": f"{_digits(rng, 3)}.{_digits(rng, 3)}.{_digits(rng, 3)}-{_digits(rng, 2)}",
        "cns": f"{_digits(rng, 3)} {_digits(rng, 4)} {_digits(rng, 4)} {_digits(rng, 4)}",
        "email": f"{slug}@exemplo-sintetico.test",
        "phone": f"({_digits(rng, 2)}) 9{_digits(rng, 4)}-{_digits(rng, 4)}",
        "address": f"{rng.choice(_STREETS)}, nº {rng.randint(10, 1999)}",
        "city": city,
        "state": state,
        "zip": f"{_digits(rng, 5)}-{_digits(rng, 3)}",
    }


def _round(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}" if decimals else str(int(round(value)))


def _generate_patient(
    rng: random.Random,
    patient_id: str,
    profile: ClinicalProfile,
    dataset: SyntheticDataset,
) -> None:
    identity = _fake_identity(rng)
    age = rng.randint(*profile.age_range)
    birthdate = REFERENCE_DATE - timedelta(days=age * 365 + rng.randint(0, 364))

    dataset.patients.append(
        {
            "id": patient_id,
            "birthdate": birthdate.isoformat(),
            "deathdate": None,
            "gender": rng.choice(["F", "M"]),
            **identity,
        }
    )

    encounter_count = rng.randint(2, 4)
    encounter_ids: list[str] = []
    for index in range(encounter_count):
        days_ago = 25 * (encounter_count - index) + rng.randint(0, 12)
        start = REFERENCE_DATE - timedelta(days=days_ago)
        encounter_id = f"{patient_id}-ENC{index + 1:02d}"
        encounter_ids.append(encounter_id)
        dataset.encounters.append(
            {
                "id": encounter_id,
                "patient": patient_id,
                "start": start.isoformat(),
                "stop": (start + timedelta(days=rng.randint(0, 4))).isoformat(),
                "encounterclass": profile.encounter_class,
                "code": profile.key,
                "description": f"Atendimento {profile.encounter_class} — {profile.label}",
                "reasondescription": profile.conditions[0],
            }
        )

    for condition in profile.conditions:
        onset = REFERENCE_DATE - timedelta(days=rng.randint(120, 2200))
        dataset.conditions.append(
            {
                "patient": patient_id,
                "encounter": encounter_ids[0],
                "start": onset.isoformat(),
                "stop": None,
                "code": profile.key,
                "description": condition,
            }
        )

    for spec in profile.observations:
        # Duas medidas por exame: permite mostrar tendência no contexto clínico.
        for offset, encounter_id in zip((1, 0), (encounter_ids[-2], encounter_ids[-1])):
            observed_at = REFERENCE_DATE - timedelta(days=25 * (offset + 1) + rng.randint(0, 6))
            raw_value = rng.uniform(spec.low, spec.high)
            dataset.observations.append(
                {
                    "patient": patient_id,
                    "encounter": encounter_id,
                    "date": observed_at.isoformat(),
                    "code": spec.code,
                    "description": spec.description,
                    "value": _round(raw_value, spec.decimals),
                    "units": spec.units,
                    "type": "numeric",
                }
            )

    for medication in profile.medications:
        started = REFERENCE_DATE - timedelta(days=rng.randint(30, 900))
        dataset.medications.append(
            {
                "patient": patient_id,
                "encounter": encounter_ids[0],
                "start": started.isoformat(),
                "stop": None,
                "code": profile.key,
                "description": medication,
                "dispenses": rng.randint(1, 12),
                "reasondescription": profile.conditions[0],
            }
        )

    for procedure in profile.procedures:
        performed = REFERENCE_DATE - timedelta(days=rng.randint(1, 60))
        dataset.procedures.append(
            {
                "patient": patient_id,
                "encounter": encounter_ids[-1],
                "start": performed.isoformat(),
                "stop": performed.isoformat(),
                "code": profile.key,
                "description": procedure,
                "reasondescription": profile.conditions[0],
            }
        )

    for index, exam in enumerate(profile.pending_exams):
        ordered = REFERENCE_DATE - timedelta(days=rng.randint(1, 20))
        # Um dos exames pode já ter sido liberado, para que o filtro de
        # "pendentes" seja de fato exercitado nos testes.
        status = "pendente" if index == 0 or rng.random() < 0.6 else "liberado"
        dataset.lab_orders.append(
            {
                "id": f"{patient_id}-LAB{index + 1:02d}",
                "patient": patient_id,
                "encounter": encounter_ids[-1],
                "ordered_at": ordered.isoformat(),
                "exam": exam,
                "priority": "urgente" if profile.encounter_class != "ambulatorial" else "rotina",
                "status": status,
                "resulted_at": None if status == "pendente" else ordered.isoformat(),
                "result_value": None,
                "result_units": None,
                "critical_flag": 0,
            }
        )


def _append_demo_patient(dataset: SyntheticDataset) -> None:
    """Paciente-âncora com valores fixos, usado em testes e na demonstração.

    Ter um paciente com valores conhecidos permite testar *recuperação exata*
    do prontuário sem depender do gerador aleatório.
    """
    dataset.patients.append(
        {
            "id": DEMO_PATIENT_ID,
            "birthdate": "1968-05-14",
            "deathdate": None,
            "gender": "F",
            "first": "Mariana",
            "last": "Costa Ribeiro",
            "cpf": "529.982.247-25",
            "cns": "700 5049 3417 8563",
            "email": "mariana.costa@exemplo-sintetico.test",
            "phone": "(11) 98765-4321",
            "address": "Rua das Acácias, nº 120",
            "city": "São Paulo",
            "state": "SP",
            "zip": "01310-100",
        }
    )
    dataset.encounters.extend(
        [
            {
                "id": f"{DEMO_PATIENT_ID}-ENC01",
                "patient": DEMO_PATIENT_ID,
                "start": "2025-12-02",
                "stop": "2025-12-02",
                "encounterclass": "ambulatorial",
                "code": "hipotireoidismo_primario",
                "description": "Consulta ambulatorial de Endocrinologia",
                "reasondescription": "Hipotireoidismo primário",
            },
            {
                "id": f"{DEMO_PATIENT_ID}-ENC02",
                "patient": DEMO_PATIENT_ID,
                "start": "2026-02-04",
                "stop": "2026-02-04",
                "encounterclass": "ambulatorial",
                "code": "hipotireoidismo_primario",
                "description": "Retorno ambulatorial de Endocrinologia",
                "reasondescription": "Hipotireoidismo primário",
            },
        ]
    )
    dataset.conditions.extend(
        [
            {
                "patient": DEMO_PATIENT_ID,
                "encounter": f"{DEMO_PATIENT_ID}-ENC01",
                "start": "2019-08-11",
                "stop": None,
                "code": "hipotireoidismo_primario",
                "description": "Hipotireoidismo primário",
            },
            {
                "patient": DEMO_PATIENT_ID,
                "encounter": f"{DEMO_PATIENT_ID}-ENC01",
                "start": "2021-03-02",
                "stop": None,
                "code": "dislipidemia",
                "description": "Dislipidemia",
            },
        ]
    )
    dataset.observations.extend(
        [
            {"patient": DEMO_PATIENT_ID, "encounter": f"{DEMO_PATIENT_ID}-ENC01", "date": "2025-12-02",
             "code": "3016-3", "description": "TSH", "value": "6.10", "units": "mUI/L", "type": "numeric"},
            {"patient": DEMO_PATIENT_ID, "encounter": f"{DEMO_PATIENT_ID}-ENC02", "date": "2026-02-04",
             "code": "3016-3", "description": "TSH", "value": "8.40", "units": "mUI/L", "type": "numeric"},
            {"patient": DEMO_PATIENT_ID, "encounter": f"{DEMO_PATIENT_ID}-ENC02", "date": "2026-02-04",
             "code": "3024-7", "description": "T4 livre", "value": "0.72", "units": "ng/dL", "type": "numeric"},
            {"patient": DEMO_PATIENT_ID, "encounter": f"{DEMO_PATIENT_ID}-ENC02", "date": "2026-02-04",
             "code": "8099-6", "description": "Anti-TPO", "value": "312", "units": "UI/mL", "type": "numeric"},
            {"patient": DEMO_PATIENT_ID, "encounter": f"{DEMO_PATIENT_ID}-ENC02", "date": "2026-02-04",
             "code": "2160-0", "description": "Creatinina", "value": "0.90", "units": "mg/dL", "type": "numeric"},
        ]
    )
    dataset.medications.extend(
        [
            {"patient": DEMO_PATIENT_ID, "encounter": f"{DEMO_PATIENT_ID}-ENC01", "start": "2019-08-20",
             "stop": None, "code": "levotiroxina", "description": "Levotiroxina sódica comprimido",
             "dispenses": 12, "reasondescription": "Hipotireoidismo primário"},
            {"patient": DEMO_PATIENT_ID, "encounter": f"{DEMO_PATIENT_ID}-ENC01", "start": "2021-03-10",
             "stop": None, "code": "carbonato_calcio", "description": "Carbonato de cálcio comprimido",
             "dispenses": 6, "reasondescription": "Suplementação"},
        ]
    )
    dataset.procedures.append(
        {"patient": DEMO_PATIENT_ID, "encounter": f"{DEMO_PATIENT_ID}-ENC02", "start": "2026-02-04",
         "stop": "2026-02-04", "code": "usg_tireoide", "description": "Ultrassonografia de tireoide",
         "reasondescription": "Bócio difuso"}
    )
    dataset.lab_orders.extend(
        [
            {"id": f"{DEMO_PATIENT_ID}-LAB01", "patient": DEMO_PATIENT_ID,
             "encounter": f"{DEMO_PATIENT_ID}-ENC02", "ordered_at": "2026-02-04",
             "exam": "TSH de controle", "priority": "rotina", "status": "pendente",
             "resulted_at": None, "result_value": None, "result_units": None, "critical_flag": 0},
            {"id": f"{DEMO_PATIENT_ID}-LAB02", "patient": DEMO_PATIENT_ID,
             "encounter": f"{DEMO_PATIENT_ID}-ENC02", "ordered_at": "2026-02-04",
             "exam": "T4 livre de controle", "priority": "rotina", "status": "pendente",
             "resulted_at": None, "result_value": None, "result_units": None, "critical_flag": 0},
            {"id": f"{DEMO_PATIENT_ID}-LAB03", "patient": DEMO_PATIENT_ID,
             "encounter": f"{DEMO_PATIENT_ID}-ENC02", "ordered_at": "2026-02-04",
             "exam": "Perfil lipídico", "priority": "rotina", "status": "liberado",
             "resulted_at": "2026-02-05", "result_value": "LDL 148", "result_units": "mg/dL",
             "critical_flag": 0},
        ]
    )


def generate_dataset(*, n_patients: int = 40, seed: int = 42) -> SyntheticDataset:
    """Gera o conjunto sintético completo de forma determinística.

    Args:
        n_patients: número de pacientes gerados aleatoriamente (além do demo).
        seed: semente do gerador — a mesma semente sempre produz o mesmo banco.
    """
    rng = random.Random(seed)
    dataset = SyntheticDataset()
    _append_demo_patient(dataset)

    for index in range(n_patients):
        profile = PROFILES[index % len(PROFILES)]
        patient_id = f"P{index + 1:06d}"
        _generate_patient(rng, patient_id, profile, dataset)

    return dataset
