# Decisões arquiteturais (ADRs)

Registro das escolhas técnicas que não são óbvias, com o contexto que as motivou, as alternativas
descartadas e as consequências assumidas. Uma decisão sem alternativa descartada não é uma decisão.

---

## ADR-001 — Separar conhecimento (RAG) de comportamento (fine-tuning)

**Status:** aceita · **Data:** 2026-02

**Contexto.** O enunciado exige fine-tuning *e* pipeline LangChain com dados atualizados do paciente.
Uma leitura ingênua levaria a treinar o modelo com todos os protocolos, esperando que ele "saiba"
medicina institucional.

**Decisão.** O fine-tuning adapta **formato, tom e comportamento de recusa**. O conhecimento factual
rastreável fica no **RAG**. O dado do paciente vem sempre do **banco**, nunca da memória do modelo.

**Alternativas descartadas.**
- *Memorizar protocolos por fine-tuning*: impossibilita citar a fonte de um trecho específico,
  exige retreinar a cada revisão de protocolo e não permite auditar de onde veio a afirmação.
- *Só RAG, sem fine-tuning*: descumpre requisito explícito do enunciado.

**Consequências.** O ganho de fidelidade é atribuível ao RAG, e isso foi **medido**: sem contexto
recuperado, `groundedness = 0,000` e nenhuma citação; com RAG, `0,557` e 100% de respostas com fonte.
Em troca, o sistema depende da qualidade do retriever — se ele falha, a resposta correta é declarar
falta de evidência, e não completar com conhecimento paramétrico.

---

## ADR-002 — Corpus institucional sintético autoral em vez de MedQuAD/PubMedQA/PCDT

**Status:** aceita · **Data:** 2026-02

**Contexto.** O enunciado sugere PubMedQA e MedQuAD como datasets, e o cenário fala em "protocolos
internos do hospital".

**Decisão.** O corpus principal é um conjunto autoral de 15 documentos em português
(`data/synthetic/protocols/`), cada um declarando explicitamente sua origem sintética.

**Alternativas descartadas e por quê.**

| Alternativa | Motivo da recusa |
|---|---|
| **MedQuAD** como base principal | inglês; QA aberto de saúde ao consumidor; não representa o fluxo "prontuário + protocolo interno"; requer verificação de atribuição por subconjunto |
| **PubMedQA** como base principal | formato yes/no/maybe sobre literatura; útil como benchmark biomédico, não como corpus de protocolo institucional |
| **PCDT do Ministério da Saúde** como "protocolos internos" | seria **desonesto** apresentar documento oficial de política pública como protocolo interno de um hospital fictício; além disso, PDFs oficiais mudam de versão sem aviso, quebrando a reprodutibilidade do benchmark |
| **MIMIC-IV** | exige credenciamento e treinamento formal; complexidade desproporcional ao escopo |

**Consequências.** Ganha-se aderência ao cenário, idioma correto, rastreabilidade total (o corpus é
versionado em texto no próprio repositório) e reprodutibilidade em CI. Perde-se realismo de escala e
a possibilidade de comparar com benchmarks públicos. **Mitigação implementada:** o loader aceita PDFs
externos em `data/raw/pcdt/`, então quem quiser rodar com PCDTs oficiais só precisa colocar os
arquivos e reindexar — sem alterar código.

---

## ADR-003 — Vector store próprio em NumPy em vez de FAISS ou Chroma

**Status:** aceita · **Data:** 2026-02

**Contexto.** O corpus indexado tem ~150 chunks. FAISS e Chroma são o padrão de mercado.

**Decisão.** Implementar `MedFlowVectorStore` sobre NumPy, **implementando a interface `VectorStore`
do `langchain_core`** — portanto plenamente utilizável com `.as_retriever()` e composição de chains.

**Justificativa.** Em 150 vetores, a busca exata é instantânea; um índice aproximado só adicionaria
erro. Elimina dependência binária pesada, permitindo que a avaliação de RAG rode em CI sem GPU e sem
download. A persistência é transparente (`vectors.npy` + `documents.jsonl` + `manifest.json`), o que
facilita auditoria.

**Consequências.** Não escala para milhões de vetores — e isso está declarado. A troca por FAISS é
localizada: basta uma classe que implemente a mesma interface.

---

## ADR-004 — Embedding determinístico por *hashing* como padrão

**Status:** aceita · **Data:** 2026-02

**Contexto.** Métricas de RAG no relatório precisam ser regeneráveis por qualquer avaliador. Um
modelo de embedding denso exige download de centenas de MB e pode mudar de versão.

**Decisão.** Backend padrão `hashing` (n-gramas de palavra e caractere, TF sublinear, normalização
L2). Backend `sentence_transformers` disponível por variável de ambiente.

**Consequências assumidas.** O backend é essencialmente lexical: falha em paráfrases fortes. Isso foi
**medido**, não presumido — 5 das 40 perguntas do benchmark falham, e a análise dessas falhas está no
notebook 03. A troca é de uma variável de ambiente e uma reindexação.

---

## ADR-005 — Estratégia de recuperação escolhida por experimento, não por preferência

**Status:** aceita · **Data:** 2026-02

**Contexto.** Quatro estratégias implementadas: `dense`, `mmr`, `bm25`, `hybrid`.

**Decisão.** Rodar a grade de 36 configurações (4 × 3 valores de `k` × 3 tamanhos de chunk) e adotar
como padrão a melhor: **`hybrid`, `k=5`, `chunk=400`** (hit@5 = 0,875; MRR = 0,661).

**Resultado que contrariou a expectativa inicial.** O MMR — recomendado na disciplina para equilibrar
relevância e diversidade — foi a **pior** estratégia neste corpus (hit@5 = 0,625 com chunk 800). A
razão é que várias seções do mesmo protocolo são legitimamente relevantes para a mesma pergunta, e a
penalização de redundância as afasta em favor de diversidade artificial. A decisão seguiu o dado.

**Consequências.** `hybrid` custa ~3–4× a latência da busca densa (fusão de dois rankings). Em
sub-milissegundo isso é irrelevante; em corpus hospitalar real, mereceria nova medição.

---

## ADR-006 — Fusão por rank entre pergunta original e pergunta enriquecida

**Status:** aceita · **Data:** 2026-02

**Contexto.** O médico escreve "este paciente" sem nomear a doença. Enriquecer a consulta com as
condições ativas do prontuário parece óbvio.

**Problema observado.** A primeira implementação **substituía** a consulta pela versão enriquecida.
Os termos da condição passaram a dominar o ranking e deslocaram trechos que a pergunta original já
encontrava (o trecho sobre interação levotiroxina/cálcio sumiu do top-5).

**Decisão.** Executar as duas buscas e fundi-las por **Reciprocal Rank Fusion**. O enriquecimento
pode acrescentar, nunca deslocar.

**Efeito medido** (22 exemplos held-out): citação correta 0,773 → 0,818; groundedness 0,520 → 0,557.

---

## ADR-007 — Segurança como regras determinísticas, não como prompt

**Status:** aceita · **Data:** 2026-02

**Contexto.** O caminho fácil seria escrever "não prescreva" no prompt do sistema e confiar no modelo.

**Decisão.** A política vive em `safety/policy.py`: regras nomeadas, versionadas, com justificativa
textual, aplicadas **antes** (roteamento) e **depois** (auditoria da saída) da geração.

**Justificativa.** Uma instrução em prompt é um pedido; uma regra é uma garantia. O teste
`test_saida_insegura_e_barrada_pelo_guardrail` prova o ponto: um modelo que devolve
"Administrar 75 mcg de levotiroxina ao dia" tem a resposta interceptada e reclassificada para revisão
humana, sem que o prompt tenha sido obedecido.

**Consequências.** Regras são frágeis a variação lexical — e isso foi medido em conjuntos held-out
(ver ADR-008). O caminho de evolução é híbrido (regras + classificador), mantendo a regra como piso
auditável.

---

## ADR-008 — Conjuntos held-out congelados para medir a política de segurança

**Status:** aceita · **Data:** 2026-02

**Contexto.** As regras de segurança foram ajustadas olhando os erros do conjunto rotulado. Reportar
a acurácia nesse mesmo conjunto seria enganoso: ele deixou de ser teste no momento em que virou
material de ajuste.

**Decisão.** Três conjuntos, com papéis distintos e registro público do que aconteceu com cada um:

| Conjunto | Papel | Resultado |
|---|---|---|
| `safety_benchmark` (48) | desenvolvimento — as regras foram ajustadas nele | 1,000 |
| `safety_holdout_v1` (32) | held-out; medição **congelada antes** das correções em `artifacts/safety/safety_holdout_v1_pre_fix.json` | 0,9375 |
| `safety_holdout_v2` (32) | held-out, nunca usado para ajustar; medição congelada em `artifacts/safety/safety_holdout_v2_frozen.json` | **0,9688** |

**Regra que seguimos.** Ao corrigir uma falha revelada por um held-out, esse conjunto perde o status
de held-out e um novo é criado. Uma correção feita **após** a medição congelada não é contabilizada
como ganho — está anotada dentro do próprio artefato.

**Número a citar no relatório:** **0,9688**, com **zero subestimações de risco**.

---

## ADR-009 — Split por documento, não por pergunta

**Status:** aceita · **Data:** 2026-02

**Decisão.** Três documentos **inteiros** (`PROT-NEF-001`, `PROT-PNE-001`, `PROC-INT-002`) são
reservados ao teste do fine-tuning.

**Justificativa.** Um split aleatório de perguntas colocaria QA da mesma seção nos dois lados,
tornando o benchmark trivial. O teste `test_split_por_documento_evita_leakage` verifica que a
interseção é vazia; o manifesto registra seed, contagens e *fingerprints*.

**Limitação que este ADR NÃO resolve.** O benchmark de **RAG** é construído sobre o mesmo corpus
indexado. Ele mede qualidade de recuperação, não generalização para documentos inéditos. Está
declarado no docstring do módulo, no notebook 03 e no README.

---

## ADR-010 — Provedor `template` como baseline executável, não como substituto de LLM

**Status:** aceita · **Data:** 2026-02

**Contexto.** O fine-tuning exige GPU. Sem um caminho offline, o fluxo LangGraph, os guardrails e a
auditoria não seriam testáveis em CI.

**Decisão.** Um terceiro provedor, `template`, seleciona extrativamente as sentenças dos trechos
recuperados mais próximas da pergunta e as formata na estrutura de saída obrigatória.

**O que ele é e o que não é.** É um **piso de groundedness** (nunca inventa: só reordena texto das
fontes) e um baseline de comparação legítimo. **Não é** um modelo de linguagem, e isso é declarado no
docstring, no README e no relatório. Sua `taxa_recusa_correta` de 0,250 é justamente a evidência
empírica do que o fine-tuning precisa ensinar — e do porquê a segurança não pode depender do gerador.

---

## ADR-011 — Gerador determinístico de pacientes em vez de baixar o Synthea

**Status:** aceita · **Data:** 2026-02

**Decisão.** Gerar 41 pacientes sintéticos com o **mesmo esquema do export CSV do Synthea**, mantendo
`ingest_synthea_csv()` pronto para um export real.

**Justificativa.** O Synthea exige Java e produz centenas de MB — inviável em CI. Um gerador com seed
permite testes de **recuperação exata** (o valor correto é conhecido por construção), que é o teste
que distingue "a LLM disse algo plausível" de "o dado do paciente foi realmente usado".

**Extensão necessária.** O export do Synthea não modela exames **pendentes**, informação exigida pelo
enunciado ("verificar exames pendentes"). Foi acrescentada a tabela `lab_orders`.

---

## ADR-012 — Identificadores diretos falsos **dentro** dos dados sintéticos

**Status:** aceita · **Data:** 2026-02

**Contexto.** Com dados sintéticos limpos, a anonimização não teria o que remover, e a "demonstração
de anonimização" viraria decoração.

**Decisão.** Os pacientes sintéticos carregam nome, CPF, CNS, telefone, e-mail e endereço **falsos**,
e a família de exemplos `laudo_preenchido` os injeta no enunciado do exemplo de treino.

**Consequência verificável.** A construção do dataset remove **80 identificadores em 8 classes**
(registrado em `data/processed/sft/manifest.json` → `anonimizacao`), e o teste
`test_curadoria_descarta_exemplos_com_pii_residual` prova que exemplos não anonimizados são barrados.

**Cuidado assumido.** Os valores são sintéticos e não correspondem a pessoas reais; o CPF do paciente
demo é um número de teste amplamente usado em documentação.
