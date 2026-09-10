---
doc_id: MOD-LAU-001
titulo: Modelos Institucionais de Laudo
versao: 1.7
vigencia: 2025-07-14
especialidade: Apoio Diagnóstico
tipo: modelo_documento
origem: Documento sintético acadêmico do "Hospital Sinapse" (fictício). Os exemplos usam pacientes fictícios. Não é um documento oficial.
---

# Modelos Institucionais de Laudo

## 1. Regras gerais de redação de laudo
Todo laudo institucional deve conter: identificação pseudonimizada do paciente no corpo do texto, data e hora do exame, indicação clínica, técnica utilizada, achados descritivos, impressão diagnóstica e identificação do profissional responsável. A impressão diagnóstica não deve repetir integralmente a descrição: deve sintetizar e responder à pergunta clínica.

## 2. Modelo — Laudo de perfil tireoidiano
```
IDENTIFICAÇÃO: [PSEUDONIMO_PACIENTE] — sexo [SEXO] — faixa etária [FAIXA_ETARIA]
DATA DO EXAME: [DATA]
INDICAÇÃO CLÍNICA: [INDICACAO]
RESULTADOS:
  TSH: [VALOR] mUI/L (referência do método: [REF])
  T4 livre: [VALOR] ng/dL (referência do método: [REF])
  Anti-TPO: [VALOR] UI/mL (referência do método: [REF])
IMPRESSÃO: padrão laboratorial compatível com [INTERPRETACAO], a ser correlacionado ao quadro clínico.
OBSERVAÇÃO: resultado isolado não estabelece diagnóstico. Correlacionar com avaliação médica.
RESPONSÁVEL TÉCNICO: [PROFISSIONAL]
```

## 3. Modelo — Laudo de eletrocardiograma
```
IDENTIFICAÇÃO: [PSEUDONIMO_PACIENTE] — faixa etária [FAIXA_ETARIA]
DATA E HORA: [DATA_HORA]
INDICAÇÃO CLÍNICA: [INDICACAO]
TÉCNICA: ECG de 12 derivações, velocidade 25 mm/s, amplitude 10 mm/mV.
DESCRIÇÃO: ritmo [RITMO]; frequência [FC] bpm; eixo [EIXO]; intervalo PR [PR] ms;
duração do QRS [QRS] ms; intervalo QT corrigido [QTC] ms; alterações de repolarização: [REPOL].
IMPRESSÃO: [IMPRESSAO].
OBSERVAÇÃO: comparar com traçado prévio quando disponível.
RESPONSÁVEL: [PROFISSIONAL]
```

## 4. Modelo — Laudo de radiografia de tórax
```
IDENTIFICAÇÃO: [PSEUDONIMO_PACIENTE] — faixa etária [FAIXA_ETARIA]
DATA: [DATA]
INDICAÇÃO CLÍNICA: [INDICACAO]
TÉCNICA: radiografia de tórax em incidências [INCIDENCIAS].
DESCRIÇÃO: campos pulmonares [DESCRICAO_PULMOES]; seios costofrênicos [SEIOS];
área cardíaca [AREA_CARDIACA]; mediastino [MEDIASTINO]; estruturas ósseas [OSSEO].
IMPRESSÃO: [IMPRESSAO].
RESPONSÁVEL: [PROFISSIONAL]
```

## 5. Exemplo preenchido (paciente fictício)
```
IDENTIFICAÇÃO: PSEUDO-7f3a91c2 — sexo feminino — faixa etária 50-59
DATA DO EXAME: 2026-02-04
INDICAÇÃO CLÍNICA: fadiga e ganho ponderal; investigação de disfunção tireoidiana.
RESULTADOS: TSH 8,4 mUI/L; T4 livre 0,72 ng/dL; Anti-TPO 312 UI/mL.
IMPRESSÃO: padrão laboratorial compatível com hipotireoidismo primário de provável
etiologia autoimune, a ser correlacionado ao quadro clínico.
```
