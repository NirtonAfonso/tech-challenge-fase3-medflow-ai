# Anonimização e pseudonimização

**Implementação:** `src/medflow_ai/data/anonymization.py` · **Testes:** `tests/test_anonymization.py`
**Demonstração antes/depois:** `notebooks/01_data_preprocessing.ipynb` (seção 2)

---

## 1. Por que isso existe mesmo com dados sintéticos

O enunciado exige "preparar os dados com técnicas de preprocessing, **anonimização** e curadoria". Com
dados sintéticos já limpos, a anonimização não teria o que remover e a demonstração seria vazia.

Por isso (ADR-012) os pacientes sintéticos carregam identificadores diretos **falsos** — nome, CPF,
CNS, telefone, e-mail, endereço — e o pipeline os remove de verdade. O resultado é verificável: a
construção do dataset remove **80 identificadores em 8 classes**, registrados em
`data/processed/sft/manifest.json` → `anonimizacao`.

---

## 2. O que é removido, transformado e preservado

| Identificador | Tratamento | Placeholder |
|---|---|---|
| Nome (com rótulo: `Paciente:`, `Dr.`, `Nome:`) | removido, rótulo preservado | `[NOME_REMOVIDO]` |
| CPF | removido | `[CPF_REMOVIDO]` |
| CNS (Cartão Nacional de Saúde) | removido | `[CNS_REMOVIDO]` |
| RG | removido | `[RG_REMOVIDO]` |
| CRM | removido | `[CRM_REMOVIDO]` |
| E-mail | removido | `[EMAIL_REMOVIDO]` |
| Telefone (fixo/celular, com DDD) | removido | `[TELEFONE_REMOVIDO]` |
| CEP | removido | `[CEP_REMOVIDO]` |
| Endereço (rua, avenida, travessa…) | removido | `[ENDERECO_REMOVIDO]` |
| Número de prontuário/matrícula | removido | `[PRONTUARIO_REMOVIDO]` |
| **Data de nascimento** | **transformada** em idade e faixa etária | `[DATA_NASCIMENTO_REMOVIDA]` |
| **`patient_id`** | **pseudonimizado** (SHA-256 com salt) | hash de 16 caracteres |
| Datas de atendimento | removidas **opcionalmente** (`redact_dates=True`) | `[DATA_REMOVIDA]` |
| **Conteúdo clínico** | **preservado integralmente** | — |

A distinção entre *remover* e *transformar* é deliberada: a idade tem valor clínico (o protocolo de
hipotireoidismo tem conduta distinta acima de 70 anos), a data de nascimento não. A faixa etária
agrega o topo em `90+` para reduzir unicidade.

### Exemplo verificável

```text
ANTES:
Paciente: Maria da Silva Souza, CPF 123.456.789-00, CNS 700 5049 3417 8563,
e-mail maria.souza@exemplo.com.br, telefone (11) 98765-4321, residente na
Rua das Acácias, nº 120, CEP 01310-100. Data de nascimento: 12/03/1975.
Prontuário nº 4457821. Atendida pelo Dr. Carlos Andrade, CRM/SP 123456.
Diagnóstico: hipotireoidismo primário, TSH 8,4 mUI/L em 04/02/2026.

DEPOIS (redact_dates=True):
Paciente: [NOME_REMOVIDO], CPF [CPF_REMOVIDO], CNS [CNS_REMOVIDO],
e-mail [EMAIL_REMOVIDO], telefone [TELEFONE_REMOVIDO], residente na
[ENDERECO_REMOVIDO], CEP [CEP_REMOVIDO]. [DATA_NASCIMENTO_REMOVIDA].
[PRONTUARIO_REMOVIDO]. Atendida pelo Dr. [NOME_REMOVIDO], [CRM_REMOVIDO].
Diagnóstico: hipotireoidismo primário, TSH 8,4 mUI/L em [DATA_REMOVIDA].
```

O conteúdo clínico — `hipotireoidismo primário`, `TSH 8,4 mUI/L` — atravessa intacto. Isso é testado
(`test_conteudo_clinico_e_preservado`).

---

## 3. Pseudonimização

```python
pseudonymize(valor) = sha256(f"{salt}|{valor}")[:16]
```

O salt vem de `MEDFLOW_PSEUDONYM_SALT`. Sem salt, um hash puro de CPF seria trivialmente reversível
por dicionário — o espaço de CPFs válidos é pequeno o bastante para força bruta.

Propriedades garantidas por teste: estabilidade (mesmo valor → mesmo pseudônimo), não reversibilidade
por inspeção (o valor original não aparece na saída) e sensibilidade ao salt.

> ⚠️ O salt padrão (`medflow-academic-salt`) é público. **Troque-o em qualquer uso não acadêmico.**

---

## 4. Três camadas de aplicação

| Camada | Onde | O que garante |
|---|---|---|
| **Dataset de treino** | `fine_tuning/dataset.py` | nenhum exemplo com PII entra no fine-tuning; exemplos residuais são descartados pela curadoria |
| **Contexto do paciente** | `database/repository.py` | *minimização*: os campos identificadores nem chegam a sair do banco |
| **Trilha de auditoria** | `logging_utils/audit.py::redact` | chaves sensíveis → `[REDACTED]`; texto livre anonimizado antes de gravar |

A camada do meio é a mais forte: não é filtragem de texto, é **arquitetura**. `PatientContext` só
possui campos não identificadores; `build_context()` nunca lê nome ou CPF. O teste
`test_contexto_nao_expoe_identificadores_diretos` compara o registro bruto com o texto que sobe ao
prompt, campo a campo.

---

## 5. Falsos positivos e falsos negativos

### Falsos positivos evitados (testados)

| Texto | Comportamento correto |
|---|---|
| `Nome do exame: Perfil Tireoidiano Completo` | não redigido (a palavra seguinte é minúscula) |
| `TSH 8,4 mUI/L com T4 livre 0,72 ng/dL` | intacto |
| `trace d8d80869-1234-97fc-0bc888eca5c2` | intacto |
| `chunk PROT-END-001#7::0::5c006de6` | intacto |

O caso do UUID foi um **defeito real encontrado ao executar o notebook 05**: o padrão de telefone
casava dígitos dentro do `trace_id` e corrompia a trilha de auditoria. Os padrões numéricos passaram a
exigir ausência de caractere de palavra ou hífen adjacente, e há teste de regressão.

### Falsos negativos conhecidos

1. **Nome sem rótulo.** "Conversei com Maria Souza ontem" não é capturado. Exigir rótulo é a escolha
   que evita redigir termos clínicos capitalizados.
2. **Cidade e UF.** Permanecem no texto. São quase-identificadores; em produção seriam generalizados
   para região.
3. **Datas de atendimento.** Só removidas com `redact_dates=True`, porque a data do exame é
   clinicamente relevante na maioria dos usos.
4. **PII em formato não previsto.** Passaporte, número de convênio, identificadores estrangeiros.

---

## 6. O que este módulo **não** é

Não é um sistema de de-identificação certificado (HIPAA *Safe Harbor*, LGPD Art. 12). É uma
implementação acadêmica, determinística e auditável, construída para demonstrar a técnica e proteger
dados **sintéticos**. Uso com dados reais exigiria validação clínica, NER treinado em português
médico, política de retenção, controle de acesso e avaliação formal de risco de reidentificação.

---

## 7. Como verificar

```bash
pytest tests/test_anonymization.py -v
python -m medflow_ai.cli build-dataset      # imprime os identificadores removidos por classe
```

E no notebook `01_data_preprocessing.ipynb`, seção 2, com o registro bruto e o anonimizado lado a lado.
