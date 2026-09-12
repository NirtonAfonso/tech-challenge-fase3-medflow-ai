# Fine-tuning — estratégia, configuração e execução

**Configuração em código:** `src/medflow_ai/fine_tuning/config.py` (`QLoRAConfig`)
**Treinamento:** `src/medflow_ai/fine_tuning/train.py` · **Notebook:** `notebooks/02_fine_tuning_qlora.ipynb`
**Avaliação comparável:** `src/medflow_ai/fine_tuning/evaluate.py`

> **Status atual:** o pipeline está completo e validado; **o treino ainda não foi executado**, por
> falta de GPU no ambiente de desenvolvimento. Nenhum número de loss, acurácia ou comparação
> base × fine-tuned aparece neste repositório antes de existir de verdade. A seção 7 explica
> exatamente o que produzir e onde colar.

---

## 1. O que o fine-tuning deve ensinar (e o que não deve)

Decisão registrada em ADR-001: o fine-tuning adapta **comportamento**, não **conhecimento**.

| O modelo ajustado deve aprender | O modelo ajustado **não** deve aprender |
|---|---|
| o formato de saída em 4 blocos (`RESPOSTA` / `CONTEXTO DO PACIENTE` / `PENDÊNCIAS E ALERTAS` / `LIMITAÇÃO`) | o conteúdo dos protocolos (isso muda e precisa de fonte rastreável → RAG) |
| citar no padrão `[DOC-ID §SEÇÃO]` | valores de exame de pacientes (isso vem do banco) |
| recusar prescrição, dose e diagnóstico definitivo | decidir conduta |
| declarar falta de evidência em vez de completar | — |
| tom clínico assistivo, em português | — |

**Por que essa divisão importa na prática.** Se o modelo memorizasse os protocolos, não haveria como
apontar *qual trecho* sustentou a afirmação, e uma revisão de protocolo exigiria retreinar. A
avaliação da ablação confirma a divisão empiricamente: a fidelidade às fontes (`groundedness`) vem do
RAG, não do gerador.

---

## 2. Estratégia técnica

| Item | Escolha | Justificativa |
|---|---|---|
| Método | **QLoRA** — base congelada em 4-bit NF4 + adaptadores LoRA treináveis | full fine-tuning de 3B não cabe em T4 (16 GB); QLoRA treina ~0,5% dos parâmetros |
| Quantização | NF4, *double quant*, compute em bfloat16 | padrão do QLoRA; NF4 preserva melhor a distribuição dos pesos que INT4 uniforme |
| Modelo base | instruct de ~3B: **`Qwen/Qwen2.5-3B-Instruct`** | **não gated** (sem aceite de licença nem token), *chat template* nativo, multilíngue com bom português, cabe em T4 |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` | razão α/r = 2, ponto de partida consolidado para SFT |
| Módulos alvo | todas as projeções de atenção e MLP (`q,k,v,o,gate,up,down`) | adaptar só atenção limita a mudança de estilo de geração |
| Épocas | 3 | dataset pequeno; mais épocas tendem a decorar |
| LR / scheduler | `2e-4`, cosine, warmup 3% | LR típico de LoRA (1–2 ordens acima de full FT) |
| Batch | 2 × 8 acumulação = **16 efetivo** | batch pequeno cabe na memória; acumulação estabiliza o gradiente |
| `max_seq_length` | 1024 | cobre o exemplo mais longo do dataset com folga |
| Gradient checkpointing | ativado | troca ~30% de tempo por memória — necessário em T4 |
| Otimizador | `paged_adamw_8bit` | evita pico de memória do otimizador |
| Seed | 42, gravada em `training_config.json` | reprodutibilidade |

### Por que não…

- **…full fine-tuning?** Inviável no orçamento de GPU e desnecessário: o objetivo é comportamento.
- **…um modelo de 7B+?** Não cabe em T4 com folga para o otimizador, mesmo em 4-bit.
- **…um modelo de 1B?** Perde qualidade de instrução em português; o formato de 4 blocos com citação
  exige alguma capacidade de seguir instrução estruturada.
- **…DPO/ORPO?** Exigiria pares de preferência que não temos. SFT é adequado para ensinar formato.

---

## 3. Dataset

Construído pelo pipeline **geração → anonimização → curadoria → split por documento → manifesto**
(`python -m medflow_ai.cli build-dataset`). Detalhes em `notebooks/01_data_preprocessing.ipynb`.

| Família | Exemplos | Cobre a exigência do enunciado |
|---|---|---|
| `protocolo_qa` | 82 | protocolos médicos |
| `procedimento` | 14 | procedimentos internos |
| `faq_medico` | 12 | perguntas frequentes de médicos |
| `laudo_preenchido` | 10 | modelos de laudo (com anonimização exercitada) |
| `modelo_receita` | 6 | receitas |
| `modelo_laudo` | 5 | modelos de laudo |
| `seguranca` | 5 | comportamento de recusa |
| `contexto_paciente` | 3 | resposta usando prontuário |
| **Total** | **137** | — |

**Splits:** 102 treino / 13 validação / 22 teste.
**Teste = 3 documentos inteiros reservados:** `PROT-NEF-001`, `PROT-PNE-001`, `PROC-INT-002`
(ADR-009; interseção com treino verificada por teste automatizado).

**Formato:** `messages` com papéis `system` / `user` / `assistant`, consumido diretamente pelo
`SFTTrainer` do TRL. O `system` é o prompt versionado do assistente — o mesmo usado em produção — para
que o modelo treine exatamente sob a instrução que receberá depois.

### Limitação de volume, declarada

137 exemplos é pouco. É suficiente para adaptar **formato e comportamento de recusa** e insuficiente
para ensinar conhecimento clínico — que é precisamente a divisão desejada. Caminhos de ampliação, em
ordem de custo-benefício:

1. mais documentos no corpus institucional (cada documento rende ~8 exemplos);
2. múltiplas paráfrases de pergunta por seção (com cuidado para não duplicar semanticamente);
3. mais exemplos de `contexto_paciente` gerados a partir do banco sintético;
4. subconjunto curado de MedQuAD traduzido, aceitando o custo de verificação de licença (ADR-002).

---

## 4. Execução

```bash
# 1. Abra notebooks/02_fine_tuning_qlora.ipynb no Google Colab
# 2. Ambiente de execução → Alterar o tipo de ambiente → GPU (T4 basta)
# 3. Execute as células em ordem
```

Ou por linha de comando, em máquina com GPU:

```bash
pip install -r requirements-training.txt
python -m medflow_ai.fine_tuning.train --check-env     # diagnóstico
python -m medflow_ai.fine_tuning.train --dry-run       # valida ambiente + dataset
python -m medflow_ai.fine_tuning.train                 # treina
```

### Recusa explícita de fabricar resultado

Sem GPU ou sem dependências, o pipeline retorna `status: "skipped"` com o motivo e **não escreve
arquivo de métrica algum**. Isso é testado (`test_treino_sem_gpu_nao_inventa_metricas`):

```text
$ python -m medflow_ai.fine_tuning.train --check-env
Nenhuma GPU CUDA detectada. O fine-tuning QLoRA exige GPU. No Colab: Ambiente de
execução > Alterar tipo de ambiente > GPU (T4). | Dependências ausentes: torch,
transformers, peft, trl, bitsandbytes, datasets.
```

---

## 5. Avaliação: como comparar sistemas sem juiz humano

Comparar respostas livres exige métricas objetivas. `fine_tuning/evaluate.py` implementa seis, todas
calculadas sobre o **split de teste congelado** (documentos held-out) e aplicáveis igualmente aos três
sistemas:

| Métrica | O que responde |
|---|---|
| `formato` | a resposta traz os 4 blocos obrigatórios? |
| `citacao_presente` | há citação `[DOC-ID §SEÇÃO]`? |
| `citacao_correta` | a citação aponta para o documento de referência do exemplo? |
| `groundedness` | fração dos 4-gramas da resposta presentes **em algum documento do corpus** — o que não está no corpus foi inventado |
| `sobreposicao_ref` | fração dos 4-gramas presentes na resposta de referência (inflada pelo texto fixo do formato — use com cautela) |
| `token_f1` | F1 de tokens contra a referência |
| `taxa_recusa_correta` | em 4 sondas de prescrição/decisão, o sistema recusa? |

### Baseline já medido (sem LLM)

Esta comparação **já existe** e foi produzida por execução real
(`artifacts/fine_tuning/generation_comparison.json`), usando o provedor extrativo determinístico:

| Sistema | formato | citação | citação correta | groundedness | token-F1 | recusa |
|---|---|---|---|---|---|---|
| `template_sem_rag` | 1,000 | 0,000 | 0,000 | 0,000 | 0,390 | 0,000 |
| `template_com_rag` | 1,000 | **1,000** | **0,818** | **0,557** | **0,497** | 0,250 |

**Leitura.** O RAG é responsável por toda a fidelidade às fontes. A `taxa_recusa_correta` de 0,250 do
baseline é o alvo natural do fine-tuning — e é também a razão pela qual a segurança **não** é
delegada ao gerador: o guardrail de saída intercepta o que o gerador deixa passar.

---

## 6. O que esperar (e o que não afirmar sem medir)

Hipóteses a **testar**, não a assumir:

| Hipótese | Como será verificada |
|---|---|
| fine-tuning ↑ `formato` em relação ao modelo base | tabela da seção 10 do notebook 02 |
| fine-tuning ↑ `taxa_recusa_correta` (famílias `seguranca`) | mesma tabela |
| fine-tuning **não** ↑ `groundedness` sozinho | comparar `fine_tuned` vs `fine_tuned_rag` |
| fine-tuning + RAG é a melhor configuração global | comparação das três colunas |

Se algum resultado contrariar a hipótese, **o resultado prevalece** e vai para o relatório com a
análise do porquê. É exatamente o que já aconteceu com o MMR na avaliação de RAG (ADR-005).

---

## 7. Checklist pós-execução

Depois de rodar o notebook 02 em GPU:

- [ ] `artifacts/fine_tuning/training_results.json` com `status: "ok"` e loss real
- [ ] `artifacts/fine_tuning/loss_curve.png`
- [ ] `artifacts/fine_tuning/comparacao_sistemas.json` (base × fine-tuned × fine-tuned+RAG)
- [ ] `artifacts/fine_tuning/respostas_antes_depois.json` (evidência qualitativa)
- [ ] adapter salvo no Google Drive ou Hugging Face Hub — **nunca** commitado no Git
- [ ] versões efetivas das bibliotecas (célula 2 do notebook) coladas em `docs/REPORT.md`
- [ ] tabela comparativa e discussão crítica (5 perguntas) em `docs/REPORT.md` §6 e §14
- [ ] `README.md` §5.6 atualizado: trocar "pendente de execução" pelos números reais
- [ ] `docs/REQUIREMENTS_CHECKLIST.md`: marcar os itens de fine-tuning

Para usar o adapter no assistente:

```bash
export MEDFLOW_LLM_PROVIDER=hf_local
export MEDFLOW_ADAPTER_PATH=/caminho/para/medflow-qlora-adapter
python -m medflow_ai.cli ask "Quando repetir o TSH deste paciente?" --patient-id P-DEMO-0001
```

---

## 8. Riscos operacionais do Colab

| Risco | Mitigação implementada |
|---|---|
| runtime reiniciado no meio do treino | `save_strategy="epoch"`, `save_total_limit=2` |
| GPU indisponível | célula de diagnóstico interrompe com mensagem acionável |
| modelo *gated* no Hugging Face | o modelo padrão é não gated, então o token é opcional; se você trocar `base_model_id` por um modelo gated, informe `HUGGINGFACE_TOKEN`. **Não existe fallback automático de modelo**: uma falha de download interrompe a execução em vez de trocar o modelo silenciosamente por baixo do relatório. |
| OOM em T4 | `gradient_checkpointing`, batch 2, `paged_adamw_8bit`, `max_seq_length=1024` |
| divergência notebook × repositório | o notebook **chama** `train()` do pacote, não reimplementa |
