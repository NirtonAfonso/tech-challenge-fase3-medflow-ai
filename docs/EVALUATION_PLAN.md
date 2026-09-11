# Plano de avaliação — executado

Este documento descreve o que foi **efetivamente medido**, com que método e onde estão os artefatos.
Itens ainda não executados estão marcados como pendentes, sem número atribuído.

Regra de discussão aplicada a todo experimento (originada do feedback da Fase 1):

> 1. Qual pergunta este experimento responde? 2. O que o resultado mostra? 3. Qual trade-off apareceu?
> 4. É suficiente para o caso de uso? 5. Quais limitações permanecem?

Reproduzir tudo: `python -m medflow_ai.cli evaluate --with-generation`

---

## 1. Recuperação (RAG) — **executado**

**Pergunta.** Dadas 40 perguntas clínicas com seção-ouro conhecida, qual configuração recupera o
trecho certo com mais frequência e em posição mais alta?

**Método.** Grade de 36 configurações: 4 estratégias (`dense`, `mmr`, `bm25`, `hybrid`) × `k` ∈ {1,3,5}
× chunk ∈ {400, 800, 1200}. Métricas: `hit@k` (seção correta), `doc_hit@k` (documento correto), `MRR`,
latência.

**Resultado.** Melhor: `hybrid · k=5 · chunk=400` → hit@5 = **0,875**, doc_hit@5 = **0,975**,
MRR = **0,661**, 0,6 ms. Pior estratégia: `mmr` (0,625 com chunk 800).

**Artefatos.** `artifacts/rag/rag_experiments.{json,csv}` · notebook 03 · README §5.1
**Limitação.** Benchmark sobre o mesmo corpus indexado: mede recuperação, não generalização.

---

## 2. Segurança clínica — **executado**

**Pergunta.** A política roteia corretamente entre `SAFE`, `CAUTION`, `HUMAN_REVIEW` e `BLOCK`? E,
mais importante: ela **subestima risco** alguma vez?

**Método.** 112 prompts rotulados em três conjuntos com papéis distintos (desenvolvimento e dois
held-out congelados). Métricas: acurácia por categoria, matriz de confusão, e a separação entre
**subestimação** (grave) e **superestimação** (aceitável).

**Resultado.** Desenvolvimento 1,000 · held-out v1 1,000 · **held-out v2 0,9688** —
**zero subestimações em todos os conjuntos**. O único erro é uma superestimação.

**Artefatos.** `artifacts/safety/*.json`, `safety_confusion.txt` · `docs/SAFETY_POLICY.md` §5
**Limitação.** Regras são frágeis a variação lexical; o caminho é um classificador híbrido.

---

## 3. Prontuário estruturado — **executado**

**Pergunta.** Quando o valor correto é conhecido, o sistema recupera exatamente esse valor? E o
contexto enviado à LLM está livre de identificadores diretos?

**Método.** Gabarito derivado do próprio banco determinístico; comparação exata de valor. Verificação
campo a campo do registro bruto contra o texto do prompt.

**Resultado.** Recuperação exata **1,000** em 30 casos · contexto livre de PII: **sim**.

**Artefatos.** `artifacts/database/database_report.json` · notebook 04 · `test_database.py`

---

## 4. Fluxo LangGraph — **executado**

**Pergunta.** O grafo percorre a rota esperada por tipo de solicitação, executa os nós corretos e
decide corretamente sobre revisão humana?

**Método.** 5 cenários declarando rota, nós obrigatórios e expectativa de revisão humana. Mais três
testes de degradação: LLM indisponível, retrieval quebrado, banco ausente.

**Resultado.** Rota **1,000** · nós **1,000** · revisão humana **1,000** · nenhuma falha.

**Artefatos.** `artifacts/graph/graph_report.json` · `test_graph.py`

---

## 5. Contribuição do RAG na resposta (ablação) — **executado**

**Pergunta.** O contexto recuperado melhora a resposta final, ou o gerador já seria suficiente?

**Método.** Mesmo gerador, mesmos 22 exemplos de documentos held-out, com e sem contexto recuperado.
Métricas: formato, citação presente, citação correta, groundedness contra o corpus, token-F1.

**Resultado.**

| Sistema | formato | citação | citação correta | groundedness | token-F1 |
|---|---|---|---|---|---|
| sem RAG | 1,000 | 0,000 | 0,000 | 0,000 | 0,390 |
| com RAG | 1,000 | 1,000 | 0,818 | 0,557 | 0,497 |

**Interpretação.** Toda a fidelidade às fontes vem do RAG. Sem contexto, o sistema corretamente
declara falta de evidência em vez de inventar.

**Artefatos.** `artifacts/fine_tuning/generation_comparison.json` · notebook 03 §6

---

## 6. Fine-tuning: base × fine-tuned × fine-tuned + RAG — **pendente de GPU**

**Pergunta.** O fine-tuning melhora aderência ao formato e comportamento de recusa? E ele acrescenta
fidelidade além do que o RAG já dá?

**Método (implementado, pronto para rodar).** Mesmas seis métricas da seção 5, aplicadas aos três
sistemas sobre o mesmo split de teste congelado; mais loss de treino/validação do `Trainer` e
exemplos qualitativos antes/depois.

**Status.** Nenhum número disponível. `notebooks/02_fine_tuning_qlora.ipynb` produz todos os
artefatos; `docs/FINE_TUNING.md` §7 tem o checklist pós-execução.

---

## 7. O que deliberadamente **não** foi medido

| Item | Por quê |
|---|---|
| Acurácia clínica das respostas | exigiria validação por médico; fora do escopo acadêmico deste trabalho |
| Comparação com outros assistentes comerciais | sem acesso controlado e sem benchmark comum |
| Latência sob carga | não há deploy; o escopo obrigatório foi priorizado |
| Robustez adversarial completa | cobre prompt injection direta; não cobre injeção via documento |
