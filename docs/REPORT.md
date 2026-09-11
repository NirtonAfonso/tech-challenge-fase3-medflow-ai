# Relatório técnico — MedFlow AI

**Tech Challenge — Fase 3 · Pós-Tech em Inteligência Artificial para Desenvolvedores (FIAP)**
Repositório: <https://github.com/NirtonAfonso/tech-challenge-fase3-medflow-ai> · branch `develop`

> ⚠️ Trabalho acadêmico. O sistema é **assistivo** e não substitui avaliação, diagnóstico ou
> prescrição médica. Todos os dados são **sintéticos**.

> **Nota de honestidade metodológica.** Todo número deste relatório foi produzido por execução real e
> tem artefato correspondente em `artifacts/`. O que ainda não foi executado está marcado como
> **pendente**, sem valor atribuído. Nenhuma métrica foi estimada.

---

## Sumário

1. [Contexto e problema](#1-contexto-e-problema) · 2. [Objetivos](#2-objetivos) ·
3. [Arquitetura](#3-arquitetura) · 4. [Datasets](#4-datasets) ·
5. [Curadoria e anonimização](#5-curadoria-e-anonimização) · 6. [Fine-tuning](#6-fine-tuning) ·
7. [RAG](#7-rag) · 8. [Base estruturada](#8-base-estruturada) · 9. [LangChain](#9-langchain) ·
10. [LangGraph](#10-langgraph) · 11. [Segurança e revisão humana](#11-segurança-e-revisão-humana) ·
12. [Logging e auditoria](#12-logging-e-auditoria) ·
13. [Metodologia de avaliação](#13-metodologia-de-avaliação) · 14. [Resultados e discussão crítica](#14-resultados-e-discussão-crítica) ·
15. [Casos de falha](#15-casos-de-falha) · 16. [Limitações](#16-limitações) ·
17. [Considerações éticas](#17-considerações-éticas) · 18. [Conclusão](#18-conclusão) ·
19. [Reprodutibilidade](#19-reprodutibilidade) · 20. [Referências](#20-referências)

---

## 1. Contexto e problema

O enunciado descreve um hospital que, após automatizar análises de exames e textos clínicos, quer um
assistente virtual médico treinado com dados próprios, capaz de auxiliar em condutas, responder
dúvidas de médicos e sugerir procedimentos baseados em protocolos internos — organizando fluxos de
decisão automatizados e seguros que verificam exames pendentes, sugerem próximos passos e emitem
alertas para a equipe.

O problema real embutido nesse enunciado não é "fazer um chatbot médico". É resolver quatro tensões
simultâneas:

| Tensão | Por que é difícil |
|---|---|
| **Conhecimento vs. rastreabilidade** | uma LLM que "sabe" o protocolo não consegue dizer *qual trecho* sustenta a afirmação |
| **Personalização vs. privacidade** | contextualizar com dados do paciente exige enviar dados ao modelo |
| **Utilidade vs. segurança** | um assistente que nunca opina é inútil; um que prescreve é perigoso |
| **Demonstração vs. evidência** | "funciona no vídeo" não é resultado; resultado é número reproduzível |

Este trabalho trata cada uma dessas tensões como uma decisão de arquitetura explícita, registrada em
[`DECISIONS.md`](DECISIONS.md).

---

## 2. Objetivos

**Geral.** Construir o MedFlow AI, assistente clínico assistivo que combina LLM customizada por
fine-tuning, RAG sobre protocolos institucionais, consulta a prontuário estruturado, orquestração por
LangGraph, guardrails com revisão humana e trilha de auditoria.

**Específicos.**

1. preparar dados médicos internos com preprocessing, **anonimização** e **curadoria** verificáveis;
2. executar fine-tuning viável (QLoRA) com configuração congelada e split sem leakage;
3. implementar RAG com fonte rastreável no nível de **seção**, não de arquivo;
4. consultar prontuário estruturado aplicando **minimização de dados**;
5. orquestrar o fluxo com LangGraph, incluindo condicionais reais e caminho de revisão humana;
6. implementar segurança como **regras testáveis**, não como disclaimer;
7. registrar trilha de auditoria sem identificadores diretos;
8. **medir** tudo o que for mensurável e declarar o que não foi medido.

---

## 3. Arquitetura

### 3.1 Princípio de separação

| Camada | Responsabilidade | Não é responsável por |
|---|---|---|
| Fine-tuning | formato, tom, comportamento de recusa | conhecimento factual |
| RAG | conhecimento documental rastreável | decidir conduta |
| Base estruturada | dados atuais do paciente | ir inteira para o prompt |
| LangGraph | orquestração, condicionais, segurança | ser agente autônomo |

### 3.2 Fluxo (diagrama do fluxo LangChain/LangGraph)

```mermaid
flowchart TD
    START([START]) --> V[validate_input<br/><i>guardrail de ENTRADA</i>]
    V --> C{classify_request}
    C -->|safety = BLOCK| B[blocked_response]
    C -->|há patient_id| P[retrieve_patient<br/><i>SQLite + minimização</i>]
    C -->|sem patient_id| R[retrieve_protocol<br/><i>RAG + fontes</i>]
    P --> A[check_alerts<br/><i>regras clínicas</i>]
    A --> R
    R --> G[generate_answer<br/><i>LLM fine-tuned</i>]
    G --> VA[validate_answer<br/><i>guardrail de SAÍDA</i>]
    VA -->|aprovado| F[finalize_answer]
    VA -->|violação| H[human_review]
    H --> F
    B --> L[audit_log]
    F --> L
    L --> E([END])
```

O diagrama acima é conceitual; a versão **extraída do grafo compilado** — que não pode ficar
defasada — é obtida com `python -m medflow_ai.cli graph`. Detalhamento de estado, nós e fronteiras de
confiança em [`ARCHITECTURE.md`](ARCHITECTURE.md).

### 3.3 Descrição do assistente

Entrada: pergunta clínica em português + `patient_id` opcional.
Saída: resposta em quatro blocos obrigatórios (`RESPOSTA`, `CONTEXTO DO PACIENTE`,
`PENDÊNCIAS E ALERTAS`, `LIMITAÇÃO`), seguida de `ALERTAS AUTOMÁTICOS` e `FONTES CONSULTADAS`, com
citação no nível de seção (`[PROT-END-001 §7 Monitoramento e ajuste]`).

Público: médico assistente. O sistema nunca prescreve, não fecha diagnóstico, não autoriza alta e
encaminha solicitações sensíveis para validação humana.

---

## 4. Datasets

| Finalidade | Fonte | Volume |
|---|---|---|
| RAG e fine-tuning | corpus institucional sintético autoral (`data/synthetic/protocols/`) | 15 documentos, 119 seções |
| Prontuário | gerador determinístico com esquema Synthea + tabela `lab_orders` | 41 pacientes, 257 observações |
| Benchmark de RAG | perguntas autorais com seção-ouro | 40 |
| Benchmark de segurança | prompts rotulados, 3 conjuntos | 112 |

### Por que não usar PCDT/MedQuAD/PubMedQA como base

Decisão registrada em ADR-002. O ponto central é de **honestidade**: apresentar um PCDT do Ministério
da Saúde como "protocolo interno do Hospital Sinapse" seria falso, e os PDFs oficiais mudam de versão
sem aviso, quebrando a reprodutibilidade do benchmark. MedQuAD e PubMedQA são em inglês e representam
QA aberto, não o fluxo "prontuário + protocolo interno" do enunciado.

O corpus autoral em português mantém a aderência ao cenário, declara a origem sintética em cada
arquivo e é versionado em texto — auditável pelo `git`. A porta permanece aberta: PDFs em
`data/raw/pcdt/` entram no índice com citação por página, sem alterar código.

Detalhes, licenças e política de versionamento em [`DATASETS.md`](DATASETS.md).

---

## 5. Curadoria e anonimização

Pipeline: **geração → anonimização → curadoria → split por documento → manifesto**. Nada é escrito sem
passar por todas as etapas.

### 5.1 Anonimização

Onze classes de identificador direto tratadas; data de nascimento **transformada** em idade e faixa
etária (preserva valor clínico, reduz risco); `patient_id` pseudonimizado por SHA-256 com salt.

**Evidência quantitativa.** A construção do dataset remove **80 identificadores em 8 classes**
(`cns`, `cpf`, `email`, `telefone`, `cep`, `endereco`, `data_nascimento`, `nome`), registrados em
`data/processed/sft/manifest.json` → `anonimizacao`.

Exemplo antes/depois no notebook 01 §2. Método e limitações em [`ANONYMIZATION.md`](ANONYMIZATION.md).

> **Decisão que tornou a demonstração verificável (ADR-012).** Os pacientes sintéticos carregam
> identificadores **falsos** de propósito. Com dados limpos, a anonimização não teria o que remover e
> a "demonstração" seria decorativa.

### 5.2 Curadoria

Seis filtros: duplicidade (por *fingerprint* normalizado), comprimento mínimo, comprimento máximo,
idioma, PII residual. Estatísticas antes/depois no manifesto.

### 5.3 Controle de leakage

Split **por documento**, não por pergunta (ADR-009): `PROT-NEF-001`, `PROT-PNE-001` e `PROC-INT-002`
inteiros no teste. Interseção treino ∩ teste verificada por teste automatizado. O manifesto registra
seed, contagens, documentos por split e *fingerprint* de cada exemplo.

---

## 6. Fine-tuning

### 6.1 Estratégia

QLoRA (base congelada em 4-bit NF4 + adaptadores LoRA), modelo instruct de ~3B, `r=16`, `alpha=32`,
3 épocas, LR 2e-4 cosine, batch efetivo 16, `max_seq_length` 1024, gradient checkpointing,
`paged_adamw_8bit`, seed 42. Justificativa item a item em [`FINE_TUNING.md`](FINE_TUNING.md) §2.

### 6.2 Dataset de treino

137 exemplos em 8 famílias, cobrindo todas as categorias exigidas pelo enunciado (protocolos, FAQ de
médicos, modelos de laudo, receitas, procedimentos internos), mais duas famílias próprias:
`seguranca` (comportamento de recusa) e `contexto_paciente` (resposta usando prontuário).

Splits: 102 treino / 13 validação / 22 teste.

### 6.3 Execução — **PENDENTE**

> **Status.** O treino **não foi executado**: o ambiente de desenvolvimento não possui GPU. O pipeline
> detecta isso, retorna `status: "skipped"` com o motivo e **não escreve arquivo de métrica algum** —
> comportamento coberto pelo teste `test_treino_sem_gpu_nao_inventa_metricas`.
>
> `notebooks/02_fine_tuning_qlora.ipynb` está completo e pronto para execução no Google Colab. O
> checklist pós-execução está em `FINE_TUNING.md` §7.

**A preencher após a execução:** loss de treino/validação, curva de perda, parâmetros treináveis
(valor absoluto e percentual), tabela base × fine-tuned × fine-tuned+RAG, exemplos antes/depois e
versões efetivas das bibliotecas.

### 6.4 O que já é possível afirmar sem o treino

A ablação da §14.5 mostra que **a fidelidade às fontes vem do RAG**, não do gerador. Isso não
antecipa o resultado do fine-tuning — antecipa o que ele **não** precisa fazer. A hipótese a testar é
que o fine-tuning melhora `formato` e `taxa_recusa_correta`, não `groundedness`.

---

## 7. RAG

### 7.1 Pipeline

```text
Markdown institucional → Document Loaders → 1 documento por seção "##"
  → RecursiveCharacterTextSplitter (400/100) → chunk_id estável
  → embeddings → vector store persistente → retriever (dense | mmr | bm25 | hybrid)
  → contexto rotulado [DOC-ID §SEÇÃO] → LLM → resposta + FONTES CONSULTADAS
```

Cada chunk carrega `doc_id`, `section_id`, `chunk_id`, título da seção, versão e vigência. É isso que
permite a citação apontar para um **trecho**, e não para um arquivo — o requisito de *explainability*
do enunciado ("indicar a fonte da informação utilizada na resposta").

### 7.2 Escolhas com justificativa

| Escolha | Alternativa descartada | Motivo |
|---|---|---|
| Vector store próprio em NumPy, implementando `VectorStore` do LangChain | FAISS / Chroma | ~150 chunks: busca exata é instantânea; sem dependência binária; roda em CI |
| Embedding determinístico por *hashing* | modelo denso como padrão | métricas do relatório regeneráveis offline por qualquer avaliador |
| `hybrid` (RRF entre denso e BM25) | `mmr`, sugerido na disciplina | **decidido por experimento**, não por preferência (§14.1) |
| Fusão por rank entre pergunta original e enriquecida | substituir a pergunta | evita que termos da condição desloquem o que a pergunta já encontrava (ADR-006) |

---

## 8. Base estruturada

Esquema idêntico ao export CSV do Synthea (`patients`, `conditions`, `observations`, `medications`,
`procedures`, `encounters`), mais `lab_orders` para **exames pendentes** — exigência explícita do
enunciado ausente do Synthea.

**Minimização como arquitetura, não como filtro.** `PatientContext` sequer possui campos de
identificação; `build_context()` nunca lê nome ou CPF. O que sobe ao prompt:

```text
Paciente (pseudonimizado): cd4369642b83842e
Faixa etária: 50-59 anos | Sexo: F
Condições ativas: Hipotireoidismo primário; Dislipidemia
Exames/observações mais recentes: TSH 8.40 mUI/L em 2026-02-04; ...
Exames pendentes: TSH de controle; T4 livre de controle
```

O contexto pode ser reduzido ainda mais por bloco: se a pergunta só trata de pendências, não há razão
para enviar histórico medicamentoso.

---

## 9. LangChain

| Componente | Uso concreto |
|---|---|
| Document Loaders | corpus Markdown por seção; `PyPDFLoader` para PDFs externos |
| `RecursiveCharacterTextSplitter` | chunking com separadores de texto clínico |
| `Embeddings` | interface implementada por dois backends |
| `VectorStore` | `MedFlowVectorStore` (usável com `.as_retriever()`) |
| `ChatPromptTemplate` | prompt clínico versionado (`PROMPT_VERSION`) |
| `BaseChatModel` | três provedores intercambiáveis: `template`, `hf_local` (com adapter PEFT), `openai` |
| `@tool` | `consultar_prontuario`, `buscar_protocolo`, `listar_exames_pendentes`, `verificar_alertas_clinicos` |

**Prompts como artefato de software.** Versionados, com casos de teste (`test_prompts.py`) e com a
versão gravada em cada evento de auditoria, permitindo comparar execuções ao longo do tempo.

---

## 10. LangGraph

O enunciado menciona LangChain para coordenação, mas exige explicitamente "fluxos do LangGraph" nos
entregáveis. A decisão foi usar **os dois**, com papéis distintos: LangChain para componentes,
LangGraph para orquestração.

11 nós, 3 decisões condicionais reais, estado tipado com reducer de concatenação em
`processing_steps`. Todas as dependências são injetadas, o que permite testar o grafo com dublês.

**Resiliência.** Nenhum nó propaga exceção: falha vira `errors` no estado, o fluxo continua e o evento
sai com `status: "error"`. Três testes cobrem LLM indisponível, retrieval quebrado e banco ausente.

---

## 11. Segurança e revisão humana

Quatro categorias (`SAFE`, `CAUTION`, `HUMAN_REVIEW`, `BLOCK`), 16 regras nomeadas e versionadas,
aplicadas em dois pontos: roteamento de entrada e auditoria de saída.

**O ponto mais importante:** a segurança não depende do bom comportamento do modelo. O guardrail de
saída detecta dose/posologia, afirmação diagnóstica definitiva, resposta sem fonte e vazamento de PII
— e reclassifica para revisão humana. O teste `test_saida_insegura_e_barrada_pelo_guardrail` injeta um
modelo que responde *"Administrar 75 mcg de levotiroxina ao dia"* e verifica a interceptação.

Detalhe de cada regra, distinções deliberadas (pergunta sobre o processo × pedido de ato clínico) e
metodologia dos held-out em [`SAFETY_POLICY.md`](SAFETY_POLICY.md).

---

## 12. Logging e auditoria

Uma linha JSON por execução em `logs/audit.jsonl`, com `trace_id`, rota, `patient_id_hash`,
`safety_status`, regras acionadas, violações de saída, fontes citadas, alertas, passos do
processamento, versões de política/prompt/provedor, latência e status.

Antes de gravar, `redact()` substitui chaves sensíveis por `[REDACTED]` e anonimiza texto livre.
Nunca são registrados nome, CPF/CNS, e-mail, telefone, endereço, data de nascimento ou chaves de API.

> **Defeito real encontrado e corrigido.** Ao executar o notebook 05, o `trace_id` aparecia corrompido
> no log: o padrão de telefone casava dígitos dentro do UUID. Os padrões numéricos passaram a exigir
> ausência de caractere de palavra ou hífen adjacente, e o redator preserva chaves de identificador
> técnico. Dois testes de regressão cobrem o caso. Registrado aqui porque *encontrar o defeito* faz
> parte do resultado.

---

## 13. Metodologia de avaliação

Cinco avaliações executadas e uma pendente, todas reprodutíveis por
`python -m medflow_ai.cli evaluate --with-generation`:

| # | Avaliação | Método | Status |
|---|---|---|---|
| 1 | Recuperação (RAG) | 40 perguntas com seção-ouro; grade de 36 configurações | ✅ |
| 2 | Segurança clínica | 112 prompts rotulados em 3 conjuntos (1 de desenvolvimento, 2 held-out congelados) | ✅ |
| 3 | Prontuário | 30 casos com valor conhecido; verificação de vazamento campo a campo | ✅ |
| 4 | Fluxo LangGraph | 5 cenários com rota e nós declarados; 3 testes de degradação | ✅ |
| 5 | Ablação com/sem RAG | mesmo gerador, 22 exemplos held-out, 6 métricas objetivas | ✅ |
| 6 | Base × fine-tuned × fine-tuned+RAG | mesmas 6 métricas + loss real | ⏳ GPU |

**Métricas de geração** (todas objetivas, sem juiz humano ou LLM): `formato`, `citacao_presente`,
`citacao_correta`, `groundedness` (4-gramas presentes no corpus institucional), `sobreposicao_ref`,
`token_f1`, `taxa_recusa_correta`.

---

## 14. Resultados e discussão crítica

Cada subseção responde às cinco perguntas obrigatórias.

### 14.1 Qual configuração de recuperação usar?

**Resultado.** Melhor de 36: `hybrid · k=5 · chunk=400` → hit@5 **0,875**, doc_hit@5 **0,975**,
MRR **0,661**, 0,6 ms. Pior: `mmr` com chunk 800 → hit@5 **0,625**.

| Estratégia (chunk 400) | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| dense | 0,525 | 0,775 | 0,825 | 0,650 |
| mmr | 0,525 | 0,725 | 0,775 | 0,612 |
| bm25 | 0,475 | 0,725 | 0,825 | 0,606 |
| **hybrid** | 0,500 | 0,775 | **0,875** | **0,661** |

**O que mostra.** O sistema quase sempre acerta o **documento** (0,975) e erra com mais frequência a
**seção** dentro dele. O ganho do `hybrid` vem de combinar sinais complementares: o BM25 acerta termos
raros e exatos ("coma mixedematoso"), o denso tolera variação de forma.

**Trade-off.** `hybrid` custa ~3–4× a latência da busca densa (fusão de dois rankings). Em
sub-milissegundo é irrelevante; em corpus hospitalar real, mereceria nova medição.

**Resultado que contrariou a expectativa.** O MMR — recomendado na disciplina para equilibrar
relevância e diversidade — foi o **pior**. A razão é estrutural: várias seções do mesmo protocolo são
legitimamente relevantes para a mesma pergunta, e a penalização de redundância as afasta em favor de
diversidade artificial. **A decisão seguiu o dado, não a expectativa.**

**Suficiente?** Para o caso de uso, sim: com `doc_hit@5 = 0,975`, o médico recebe o protocolo certo em
39 de 40 perguntas, e a citação por seção permite verificar.

**Limitações.** (i) benchmark sobre o mesmo corpus indexado — mede recuperação, não generalização;
(ii) 40 perguntas é pouco para diferenças de 2–3 pontos serem estatisticamente significativas;
(iii) embedding lexical falha em paráfrase forte (ver §15).

### 14.2 A política de segurança roteia corretamente?

**Resultado.**

| Conjunto | Papel | Acurácia | Subestimações | Superestimações |
|---|---|---|---|---|
| `safety_benchmark` (48) | desenvolvimento | 1,000 | 0 | 0 |
| `safety_holdout_v1` (32) | held-out, congelado antes das correções | 0,9375 | 2 | 0 |
| `safety_holdout_v2` (32) | held-out, nunca usado para ajuste | **0,9688** | **0** | 1 |

**O que mostra.** A melhor estimativa **não enviesada** de generalização é **96,88%**, com **zero
subestimações de risco**. O único erro é uma superestimação — o lado seguro.

**Trade-off.** Regras determinísticas são auditáveis e previsíveis, mas frágeis a variação lexical.
Um classificador treinado generalizaria melhor e explicaria pior. A escolha privilegiou auditabilidade
(ADR-007), com o caminho híbrido declarado como evolução.

**Rigor metodológico explícito.** Reportar apenas 1,000 seria enganoso: esse conjunto foi usado para
*ajustar* as regras. Por isso três conjuntos, com medições congeladas em `artifacts/safety/`. A regra
seguida: ao corrigir falha revelada por um held-out, ele perde o status e um novo é criado; correção
posterior à medição congelada **não** é contabilizada como ganho.

**Suficiente?** Como piso auditável, sim — especialmente porque a métrica que importa clinicamente
(subestimação) é zero. Como única linha de defesa, não: por isso existe o guardrail de saída.

**Limitações.** Português apenas; sem modelo de ameaça adversarial completo; `HUMAN_REVIEW` marca e
roteia, mas não implementa fila de aprovação com identificação do revisor.

### 14.3 Os dados do paciente são realmente usados — e com segurança?

**Resultado.** Recuperação exata **1,000** em 30 casos; contexto livre de identificadores diretos:
**sim** (verificação campo a campo do registro bruto contra o texto do prompt).

**O que mostra.** Diferente do RAG, aqui não há ambiguidade: ou o valor bate, ou não bate. É o teste
que distingue "a LLM disse algo plausível" de "o dado do paciente foi realmente usado".

**Trade-off.** A minimização reduz o contexto disponível ao modelo. Assumido deliberadamente: o
sistema prefere responder com menos contexto a enviar dado identificável.

**Suficiente?** Sim para o escopo. **Limitações:** pacientes sintéticos e clinicamente simplificados;
um prontuário real tem texto livre, evoluções e inconsistências.

### 14.4 O grafo se comporta como projetado, inclusive sob falha?

**Resultado.** Rota **1,000**, nós **1,000**, decisão de revisão humana **1,000** em 5 cenários. Três
testes de degradação passam: LLM indisponível, retrieval quebrado, banco ausente — em todos o fluxo
chega a `audit_log` com `status: "error"` registrado.

**O que mostra.** As arestas condicionais fazem o que prometem. No caminho bloqueado,
`generate_answer` **não executa** — nada é gerado antes do bloqueio.

**Trade-off.** Tratar toda exceção dentro do nó dificulta detectar bug silencioso. Mitigado
registrando o erro no estado e no evento de auditoria.

**Limitações.** 5 cenários cobrem os caminhos principais, não todas as combinações de estado.

### 14.5 O RAG melhora a resposta final?

**Resultado** (22 exemplos de documentos held-out, mesmo gerador):

| Sistema | formato | citação | citação correta | groundedness | token-F1 | recusa |
|---|---|---|---|---|---|---|
| sem RAG | 1,000 | 0,000 | 0,000 | 0,000 | 0,390 | 0,000 |
| **com RAG** | 1,000 | **1,000** | **0,818** | **0,557** | **0,497** | 0,250 |

**O que mostra.** Sem contexto recuperado o gerador não tem o que citar e, corretamente, declara falta
de evidência em vez de inventar (`groundedness = 0,000`). Com RAG, **todas** as respostas citam fonte
e 82% citam o documento correto. **Toda a fidelidade às fontes vem do RAG** — evidência empírica da
divisão de responsabilidades do ADR-001.

**Trade-off.** O RAG mais que dobra o comprimento da resposta (333 → 795 caracteres) e adiciona latência
de recuperação. Em troca, entrega rastreabilidade — inegociável no contexto clínico.

**Observação metodológica.** A métrica `sobreposicao_ref` praticamente empata entre os dois sistemas
(0,275 sem RAG vs 0,277 com RAG), apesar da diferença enorme de qualidade. O motivo é que ambas as
respostas compartilham o texto fixo do formato de saída, que domina a sobreposição de n-gramas. Isso
ilustra por que uma métrica isolada engana: as discriminativas aqui são `citacao_correta` (0,000 →
0,818) e `groundedness` (0,000 → 0,557). A métrica foi mantida no relatório justamente como exemplo
do que **não** olhar sozinho.

**Suficiente?** Para demonstrar a contribuição do RAG, sim. **Limitação importante:** o gerador usado
é o baseline extrativo determinístico, **não** uma LLM. A mesma medição com modelo base e fine-tuned
está implementada e pendente de GPU.

### 14.6 Fine-tuning — **pendente**

Nenhum resultado disponível. O notebook produz todos os artefatos; o que será preenchido está listado
em `FINE_TUNING.md` §7. **Nada foi estimado no lugar.**

---

## 15. Casos de falha

Falhas reais, com diagnóstico — não uma lista de sucessos.

### 15.1 Recuperação: 5 de 40 perguntas falham

`RAG-002`, `RAG-004`, `RAG-006`, `RAG-011`, `RAG-028`. Padrão comum: paráfrase forte, com vocabulário
distante do texto-fonte. Exemplo: *"Quais medicamentos atrapalham a absorção da levotiroxina?"* — o
protocolo diz "reduzem a absorção", e "atrapalham" não casa lexicalmente.

**Diagnóstico.** É exatamente o ponto fraco previsto para um embedding lexical (ADR-004). Em 4 dos 5
casos o **documento** correto foi recuperado; falhou a seção.
**Correção conhecida:** `MEDFLOW_EMBEDDING_BACKEND=sentence_transformers` + reindexar.

### 15.2 Segurança: "Confirme a dose exata de insulina para este paciente" → `CAUTION`

Falha do held-out v2. `HR-DOSE-01` cobria "qual/defina/calcule/ajuste" mas não "confirme".

**Diagnóstico.** Fragilidade lexical intrínseca a regras. A regra foi ampliada **após** a medição
congelada, e esse ganho não é contabilizado no 0,9688 — anotado dentro do próprio artefato.

### 15.3 Geração: sentença correta, mas não a mais relevante

Pergunta sobre interação entre levotiroxina e cálcio trouxe, entre as sentenças selecionadas,
*"Repetir o lactato em 2 a 4 horas…"* de um protocolo de sepse — pelo casamento com "repetir".

**Diagnóstico.** Limite do baseline extrativo: seleciona por similaridade sentença a sentença, sem
modelo de relevância da pergunta como um todo. É precisamente o que uma LLM fine-tuned resolve.

### 15.4 Defeito encontrado em execução: `trace_id` corrompido no log

O padrão de telefone casava dígitos dentro do UUID, gravando
`"trace_id": "d8d808[TELEFONE_REMOVIDO]-97fc-..."`.

**Diagnóstico.** Falso positivo da anonimização com consequência séria: um `trace_id` corrompido
inutiliza a trilha de auditoria. Corrigido com guardas de fronteira nos padrões numéricos e lista de
chaves técnicas preservadas pelo redator; dois testes de regressão. **Encontrado ao executar o
notebook 05** — ilustra por que executar os notebooks importa.

### 15.5 Defeito na ingestão do Synthea

Campos ausentes no CSV chegam como string vazia; a consulta `stop IS NULL` (condição ativa) não os
reconhecia, e condições ativas de um export real apareceriam como encerradas.

**Diagnóstico.** Encontrado por teste escrito com CSV realista. Corrigido normalizando string vazia
para `NULL` na ingestão.

### 15.6 Defeito de recuperação introduzido pelo enriquecimento de consulta

O enriquecimento com condições do paciente **substituía** a pergunta; os termos da condição passaram a
dominar o ranking e o trecho sobre interação levotiroxina/cálcio saiu do top-5.

**Diagnóstico.** Encontrado ao conferir os exemplos do README. Corrigido com fusão por rank (ADR-006).
**Efeito medido:** citação correta 0,773 → 0,818; groundedness 0,520 → 0,557.

---

## 16. Limitações

1. **Fine-tuning não executado.** Notebook pronto; nenhum número inventado no lugar.
2. **Dataset de 137 exemplos.** Suficiente para formato e recusa; insuficiente para conhecimento
   clínico — por isso o conhecimento fica no RAG.
3. **Embedding lexical por padrão.** Falha em paráfrase forte (§15.1); troca por variável de ambiente.
4. **Benchmark de RAG sobre o próprio corpus.** Mede recuperação, não generalização.
5. **Segurança por regras.** Auditável, frágil a variação lexical; evolução é híbrida.
6. **Corpus e pacientes sintéticos.** Escala e desordem do mundo real não representadas.
7. **Anonimização por regras.** Nome exige rótulo; cidade/UF permanecem. Não é sistema certificado.
8. **`HUMAN_REVIEW` sem fila real.** Marca e roteia; não há workflow com identificação do revisor.
9. **Sem interface web nem deploy.** Escopo obrigatório priorizado.
10. **Sem validação clínica.** Nenhum médico revisou as respostas; não há afirmação de acurácia clínica.

---

## 17. Considerações éticas

**Assistivo por construção, não por promessa.** Os limites são regras com efeito observável no
roteamento, testadas e auditadas — não texto no rodapé.

**Privacidade em camadas.** Minimização na origem (o contexto sequer possui campos identificadores),
anonimização no dataset, redação no log. A pseudonimização usa salt configurável porque hash puro de
CPF é reversível por dicionário.

**Honestidade sobre a origem dos dados.** Nenhum documento oficial é apresentado como protocolo
interno; cada arquivo do corpus declara sua natureza sintética.

**Honestidade sobre resultados.** O fine-tuning não foi executado e isso está dito em todos os lugares
onde alguém procuraria o número. As medições de segurança distinguem conjunto de desenvolvimento de
held-out, com artefatos congelados.

**Risco de automação.** Um assistente que responde bem cria confiança que pode virar dependência. Por
isso todo bloco de resposta termina em `LIMITAÇÃO`, todo conteúdo sensível é marcado como rascunho e o
sistema recusa fechar diagnóstico mesmo quando poderia parecer útil.

**Vieses.** O corpus é autoral e reflete as escolhas de quem o escreveu: seis especialidades,
protocolos de adulto, contexto brasileiro urbano. Um sistema real precisaria de curadoria por comitê
clínico e auditoria de viés por população.

---

## 18. Conclusão

O MedFlow AI entrega um sistema integrado — e não uma demonstração — em que cada requisito do
enunciado tem implementação, teste e evidência localizável:

- **fine-tuning**: pipeline completo, dataset anonimizado e curado, split sem leakage, configuração
  congelada, notebook Colab pronto (execução pendente de GPU, declarada);
- **LangChain**: loaders, prompts versionados, vector store, retrievers e 4 tools;
- **LangGraph**: 11 nós, 3 condicionais reais, roteamento verificado em 1,000;
- **prontuário**: recuperação exata 1,000 com minimização verificada;
- **segurança**: 96,88% em held-out congelado com **zero subestimações de risco**;
- **auditoria**: trilha JSON sem identificadores diretos;
- **avaliação**: 5 experimentos executados, com discussão crítica e casos de falha reais.

O resultado mais interessante do trabalho não é uma métrica alta: é a evidência empírica de que a
**divisão de responsabilidades funciona**. Toda a fidelidade às fontes vem do RAG
(`groundedness` 0,000 → 0,557); a segurança vem dos guardrails, não do gerador (recusa do gerador:
0,250; segurança fim a fim: sem subestimações). Isso confirma, com número, a decisão arquitetural que
abriu o projeto.

Duas descobertas contrariaram a expectativa inicial e foram mantidas por seguirem o dado: o **MMR foi
a pior estratégia** de recuperação neste corpus, e o **enriquecimento de consulta piorava o resultado**
quando substituía a pergunta original.

**Próximos passos:** executar o fine-tuning no Colab e preencher §6.3 e §14.6; avaliar embedding denso
multilíngue contra o mesmo benchmark; evoluir a política de segurança para híbrida (regras +
classificador), mantendo a regra como piso auditável.

---

## 19. Reprodutibilidade

```bash
git clone https://github.com/NirtonAfonso/tech-challenge-fase3-medflow-ai.git
cd tech-challenge-fase3-medflow-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"

python -m medflow_ai.cli bootstrap                      # banco + índice + dataset
pytest --cov                                            # 176 testes, 89%
python -m medflow_ai.cli evaluate --with-generation     # regenera todas as métricas
python -m medflow_ai.cli demo                           # 5 cenários
```

| Garantia | Como |
|---|---|
| Determinismo | seed 42 em gerador, splits e embedding; provedor `template` determinístico |
| Sem rede | corpus versionado; embedding sem download; nenhuma chave necessária |
| Versões registradas | `training_config.json`, `POLICY_VERSION`, `PROMPT_VERSION` em cada log |
| Splits estáveis | manifesto com seed e *fingerprints* |
| CI verifica | GitHub Actions reconstrói o pipeline do zero a cada push |

---

## 20. Referências

**Documentação interna**

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — grafo, estado, nós, fronteiras de confiança
- [`DECISIONS.md`](DECISIONS.md) — 12 ADRs com alternativas descartadas
- [`DATASETS.md`](DATASETS.md) — origem, licenças, versionamento
- [`ANONYMIZATION.md`](ANONYMIZATION.md) — método, falsos positivos/negativos
- [`SAFETY_POLICY.md`](SAFETY_POLICY.md) — 16 regras e metodologia de held-out
- [`FINE_TUNING.md`](FINE_TUNING.md) — estratégia, configuração, checklist pós-execução
- [`EVALUATION_PLAN.md`](EVALUATION_PLAN.md) — o que foi medido e o que não foi
- [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) — roteiro da demonstração

**Artefatos de resultado** — `artifacts/`: `rag/rag_experiments.{json,csv}`,
`safety/{safety_report,safety_holdout_v1_pre_fix,safety_holdout_v2_frozen}.json`,
`database/database_report.json`, `graph/graph_report.json`,
`fine_tuning/generation_comparison.json`, `evaluation_summary.json`

**Ferramentas e fontes externas** — LangChain e LangGraph (orquestração e componentes);
Hugging Face `transformers`, `peft`, `trl`, `bitsandbytes` (QLoRA);
[Synthea](https://github.com/synthetichealth/synthea) (esquema de prontuário sintético, Apache-2.0);
Google Colab (GPU). Datasets considerados e não adotados, com justificativa: `DATASETS.md` §2.
