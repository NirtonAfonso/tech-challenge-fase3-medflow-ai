# Datasets — decisões, origem e licenças

> **Status: congelado.** As escolhas abaixo são as efetivamente usadas pelo sistema. A justificativa
> completa de cada recusa está em [`DECISIONS.md`](DECISIONS.md) (ADR-002 e ADR-011).

---

## 1. Quadro geral

| Finalidade | Fonte adotada | Volume | Licença / origem |
|---|---|---|---|
| RAG (conhecimento) | **corpus institucional sintético** — `data/synthetic/protocols/` | 15 documentos, 119 seções, ~6.000 palavras | autoral, criado para este trabalho |
| Fine-tuning | derivado do mesmo corpus + exemplos de segurança e de prontuário | 137 exemplos | autoral |
| Prontuário estruturado | gerador determinístico com **esquema do Synthea** | 41 pacientes, 257 observações, 69 solicitações de exame | autoral; esquema compatível com [Synthea](https://github.com/synthetichealth/synthea) (Apache-2.0) |
| Benchmark de RAG | perguntas com seção-ouro conhecida | 40 | autoral |
| Benchmark de segurança | prompts rotulados em 4 categorias, 3 conjuntos | 112 | autoral |

**Nenhum dado real de paciente é usado, baixado ou versionado.**

---

## 2. Corpus institucional sintético

`data/synthetic/protocols/*.md` — 15 documentos Markdown com bloco de metadados
(`doc_id`, `titulo`, `versao`, `vigencia`, `especialidade`, `tipo`, `origem`).

| Tipo | Documentos | Cobre a exigência do enunciado |
|---|---|---|
| Protocolo clínico | `PROT-END-001/002/003`, `PROT-CAR-001/002`, `PROT-INF-001/002`, `PROT-NEU-001`, `PROT-PNE-001`, `PROT-NEF-001` | "protocolos médicos do hospital" |
| Procedimento interno | `PROC-INT-001` (exames, prazos, valores críticos), `PROC-INT-002` (alta segura) | "procedimentos internos" |
| Modelo de documento | `MOD-LAU-001` (laudos), `MOD-REC-001` (receitas) | "modelos de laudos, receitas" |
| FAQ | `FAQ-MED-001` | "perguntas frequentes feitas por médicos" |

Especialidades cobertas: Endocrinologia, Cardiologia, Infectologia, Pneumologia, Neurologia,
Nefrologia, Apoio Diagnóstico, Farmácia Clínica, Qualidade Assistencial.

**Cada arquivo declara sua própria origem**, por exemplo:

```yaml
origem: Documento sintético acadêmico do "Hospital Sinapse" (fictício), estruturado a
        partir de convenções públicas de diretrizes clínicas. Não é um documento oficial.
```

Isso é verificado por teste (`test_todo_documento_tem_metadados_de_rastreabilidade`), junto com a
ausência de PII no corpus e a unicidade de `doc_id` e `section_id`.

### Por que autoral, e não PCDT/MedQuAD/PubMedQA

Resumo (detalhes em ADR-002):

| Fonte sugerida/considerada | Decisão | Motivo principal |
|---|---|---|
| **PCDT — Ministério da Saúde** | não usar como corpus principal | seria desonesto apresentar política pública oficial como "protocolo interno" de um hospital fictício; PDFs mudam de versão sem aviso, quebrando a reprodutibilidade do benchmark |
| **MedQuAD** | não usar como base principal | inglês; QA de saúde ao consumidor; não representa o fluxo "prontuário + protocolo interno"; atribuição varia por subconjunto |
| **PubMedQA** | não usar como base principal | formato yes/no/maybe sobre literatura; é benchmark biomédico, não corpus de protocolo institucional |
| **MIMIC-IV** | não usar | exige credenciamento e treinamento formal; desproporcional ao escopo |
| **MedMCQA** | não usar | múltipla escolha, pouco representativo do fluxo clínico proposto |

**Porta de entrada mantida aberta.** O loader aceita PDFs externos em `data/raw/pcdt/` (diretório
ignorado pelo Git). Quem quiser rodar o sistema sobre PCDTs oficiais só precisa colocar os arquivos e
reindexar:

```bash
mkdir -p data/raw/pcdt && cp ~/Downloads/pcdt_*.pdf data/raw/pcdt/
pip install pypdf
python -m medflow_ai.cli build-index    # PDFs entram com citação por página
```

Antes de usar PCDTs reais em um trabalho publicado, registre aqui: URL exata, data de acesso, versão
do documento e a licença/termos de uso vigentes no portal do Ministério da Saúde.

---

## 3. Prontuário estruturado

Esquema **idêntico ao export CSV do Synthea**, mais uma tabela:

| Tabela | Origem do esquema | Papel |
|---|---|---|
| `patients` | Synthea | demografia (com PII **falsa**, para exercitar a anonimização) |
| `encounters` | Synthea | atendimentos |
| `conditions` | Synthea | condições ativas |
| `observations` | Synthea | exames com valor e unidade |
| `medications` | Synthea | medicamentos em uso |
| `procedures` | Synthea | procedimentos realizados |
| `lab_orders` | **acrescentada** | exames solicitados e **pendentes** — exigência do enunciado ausente do Synthea |

Os 41 pacientes seguem 11 perfis clínicos coerentes com os protocolos do corpus (hipotireoidismo,
tireotoxicose, diabetes tipo 2, hipertensão, dor torácica, sepse, pneumonia, asma, injúria renal,
AVC), mais o paciente-âncora `P-DEMO-0001` com valores fixos usado em testes e na demonstração.

Para ingerir um export real do Synthea:

```bash
python -m medflow_ai.cli build-db --synthea-csv /caminho/para/csv
```

---

## 4. Benchmarks

### `data/benchmarks/rag_benchmark.jsonl` — 40 perguntas

Cada linha: `id`, `question`, `gold_doc_id`, `gold_section_id`, `categoria`. As perguntas foram
escritas com **vocabulário diferente** do texto-fonte, para reduzir casamento lexical trivial. Um
teste valida que todo `gold_section_id` existe de fato no corpus.

> **Limitação declarada.** O benchmark é construído sobre o mesmo corpus indexado pelo RAG. Ele mede
> **qualidade de recuperação**, não generalização para documentos inéditos.

### `data/benchmarks/safety_*.jsonl` — 112 prompts

| Arquivo | Prompts | Papel |
|---|---|---|
| `safety_benchmark.jsonl` | 48 | desenvolvimento — as regras foram ajustadas nele |
| `safety_holdout_v1.jsonl` | 32 | held-out; aposentado após motivar correções |
| `safety_holdout_v2.jsonl` | 32 | held-out congelado; **é este o número de generalização** |

A metodologia está em [`SAFETY_POLICY.md`](SAFETY_POLICY.md) §5 e ADR-008.

---

## 5. Controle de leakage

| Risco | Controle |
|---|---|
| QA de treino e de teste da mesma seção | **split por documento**: `PROT-NEF-001`, `PROT-PNE-001`, `PROC-INT-002` inteiros no teste; interseção verificada por teste |
| Regras de segurança ajustadas no conjunto de avaliação | três conjuntos com papéis distintos; medições congeladas em `artifacts/safety/` |
| Benchmark de RAG sobre o próprio corpus | **não é eliminável** neste desenho; declarado no módulo, no notebook 03 e no README |
| Split mudar entre execuções | manifesto com seed, contagens, documentos por split e *fingerprint* de cada exemplo |

---

## 6. Política de versionamento de dados

| Caminho | Versionado? | Por quê |
|---|---|---|
| `data/synthetic/protocols/` | ✅ sim | corpus é fonte da verdade; precisa ser auditável no `git` |
| `data/benchmarks/` | ✅ sim | gabaritos pequenos e essenciais à reprodutibilidade |
| `data/processed/sft/` | ✅ sim | dataset anonimizado, pequeno, com manifesto |
| `data/processed/hospital.db` | ❌ não | regenerável (`build-db`), e `.gitignore` bloqueia `*.db` |
| `data/processed/vector_store/` | ❌ não | regenerável (`build-index`) |
| `data/raw/` | ❌ não | downloads locais; nunca dados restritos |
| adapters / checkpoints | ❌ não | pesos grandes ficam no Drive ou Hugging Face |

Regra permanente: **nunca versionar dado real de paciente neste repositório.**
