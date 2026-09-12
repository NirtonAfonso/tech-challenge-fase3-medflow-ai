# Runbook — executar o MedFlow AI no Google Colab

Guia operacional para você rodar os cinco notebooks, obter os artefatos reais e devolvê-los para que
README e relatório sejam atualizados **somente** com números medidos.

> **Estado atual da entrega.** Tudo que não depende de GPU já foi executado e medido (RAG, prontuário,
> segurança, grafo). O **fine-tuning ainda não rodou** — é exatamente o que este runbook destrava.

---

## Antes de começar

| Item | Valor |
|---|---|
| Repositório | <https://github.com/NirtonAfonso/tech-challenge-fase3-medflow-ai> |
| Branch | **`develop`** (os notebooks clonam essa branch explicitamente) |
| Conta | a mesma no GitHub e no Google Drive |
| Custo | nenhum: o nível gratuito do Colab basta |

### O Google Drive é montado automaticamente

Você **não precisa criar pasta nenhuma**. No primeiro notebook o Colab pedirá autorização de acesso ao
Drive; aceite. A partir daí, esta árvore é criada sozinha (`exist_ok`, então rodar de novo não apaga
nada):

```text
/content/drive/MyDrive/MedFlowAI_Fase3/
├── 01_preprocessing/    artifacts/ · datasets/ · logs/
├── 02_fine_tuning/      artifacts/ · adapter/ · checkpoints/ · logs/ · bundles/
├── 03_rag/              artifacts/ · indexes/ · logs/
├── 04_structured_data/  artifacts/ · database/ · logs/
├── 05_demo/             artifacts/ · logs/ · exports/
└── shared/              manifests/ · configs/ · run_metadata/
```

**Por que o Drive importa:** o `/content` do Colab é apagado quando a sessão encerra. Uma execução só
está concluída quando os resultados estão no Drive. Cada notebook imprime, no final, a lista de
arquivos persistidos e onde ficaram.

> 🔒 Segredos (`.env`, tokens, credenciais) **nunca** são copiados para o Drive nem entram no ZIP de
> resultados. Isso é imposto por código, não por disciplina.

### Links diretos

| # | Notebook | Runtime | Abrir |
|---|---|---|---|
| 01 | Preparação de dados | CPU | [Colab](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/01_data_preprocessing.ipynb) |
| 02 | **Fine-tuning QLoRA** | **GPU** | [Colab](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/02_fine_tuning_qlora.ipynb) |
| 03 | Pipeline de RAG | CPU | [Colab](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/03_rag_pipeline.ipynb) |
| 04 | Prontuário estruturado | CPU | [Colab](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/04_structured_patient_data.ipynb) |
| 05 | Demonstração ponta a ponta | CPU / GPU | [Colab](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/05_medflow_full_demo.ipynb) |

> Os links apontam para `develop`. **Depois do merge final para `main`**, troque `/blob/develop/` por
> `/blob/main/` nestes links e nos badges de `notebooks/README.md` e do topo de cada notebook.

---

## Ordem recomendada

```text
01 → 02 → 03 → 04 → 05
```

Cada notebook constrói os próprios pré-requisitos (banco, índice, dataset), então a ordem é uma
sugestão de narrativa, não uma dependência técnica. A **única** dependência real: o notebook 05 em modo
submissão precisa do adapter produzido pelo 02.

---

## Notebook 01 — Preparação de dados, anonimização e curadoria

| | |
|---|---|
| Runtime | CPU (padrão; não mexa) |
| O que clicar | `Ambiente de execução` → `Executar tudo` |
| Pasta no Drive | `MedFlowAI_Fase3/01_preprocessing/` |
| Duração | ~2 min |

**Produz:** `artifacts/01_preprocessing_report.json`, os três JSONL do dataset e o `manifest.json`
em `datasets/`, e uma cópia do manifesto em `shared/manifests/`.

**Critério de sucesso:**
- imprime "Interseção treino ∩ teste: VAZIA ✅";
- imprime "Invariância confirmada ✅" (o dataset não muda com o tamanho do banco);
- o relatório de anonimização mostra identificadores removidos em 8 classes.

**Baixar manualmente:** nada. Fica tudo no Drive.

---

## Notebook 02 — Fine-tuning QLoRA ⚠️ **exige GPU**

| | |
|---|---|
| Runtime | `Ambiente de execução` → `Alterar o tipo de ambiente de execução` → **GPU** (T4 basta) → `Salvar` |
| O que clicar | depois de escolher a GPU, `Executar tudo` |
| Pasta no Drive | `MedFlowAI_Fase3/02_fine_tuning/` |
| Duração | 20–60 min conforme a fila do Colab |

**Escolha a GPU ANTES de executar.** Se rodar em CPU, o notebook para na célula de diagnóstico com a
mensagem de como corrigir — ele não treina em CPU e não inventa métrica.

**Precisão:** decidida pelo hardware. T4 (compute capability 7.5) → **FP16**; GPU Ampere ou mais nova →
**BF16**. Você não precisa configurar nada.

**Produz, todos persistidos no Drive:**

| Subpasta | Conteúdo |
|---|---|
| `adapter/` | adapter LoRA + tokenizer — **é o que o notebook 05 usa** |
| `checkpoints/` | checkpoints intermediários do Trainer |
| `artifacts/` | `training_results.json`, `environment.json`, `training_config.json`, `loss_curve.png`, `comparacao_sistemas.json`, `respostas_antes_depois.json` |
| `bundles/` | **`medflow_colab_results.zip`** — o pacote que você devolve |
| `logs/` | logs da execução |

**Critério de sucesso:**
1. o diagnóstico imprime o nome da GPU e a precisão escolhida;
2. a curva de perda aparece e é salva;
3. a recarga do adapter funciona e as respostas "ANTES / DEPOIS" aparecem lado a lado;
4. a tabela compara `base`, `fine_tuned` e `fine_tuned_rag`;
5. o validador imprime **`RESULTADO: VÁLIDO`**;
6. a auditoria do ZIP imprime **`Seguro: True`**.

Se o validador reprovar, **não** cite os números: ele diz exatamente o que faltou.

**Se der OOM:** o notebook explica como ligar `USAR_RECUPERACAO_OOM = True` (batch 1, sequência 768,
batch efetivo preservado). A configuração reduzida fica registrada em `training_results.json` — a
metodologia não muda em silêncio.

**Baixar manualmente:** `medflow_colab_results.zip`. O notebook tenta o download automático; se o
navegador bloquear, pegue em `MedFlowAI_Fase3/02_fine_tuning/bundles/`.

> O ZIP tem só métricas (algumas centenas de KB). O adapter **não** vai nele — fica no Drive.

---

## Notebook 03 — Pipeline de RAG

| | |
|---|---|
| Runtime | CPU |
| O que clicar | `Executar tudo` |
| Pasta no Drive | `MedFlowAI_Fase3/03_rag/` |
| Duração | ~3 min (ou ~8 min com a seção densa) |

**Produz:** `rag_experiments.json`/`.csv` (36 configurações), `rag_hit_at_k.png`,
`rag_mrr_por_chunk.png`, `generation_comparison.json` e o índice vetorial em `indexes/`.

**Seção 7 é opcional** (embedding denso, baixa ~100 MB). Ela é **aditiva**: os números oficiais do
relatório continuam sendo os do backend determinístico. Para pular, ponha `EXECUTAR_DENSO = False`.

**Critério de sucesso:** os dois gráficos aparecem e a melhor configuração sai como
`hybrid · k=5 · chunk=400`.

---

## Notebook 04 — Prontuário estruturado

| | |
|---|---|
| Runtime | CPU |
| O que clicar | `Executar tudo` |
| Pasta no Drive | `MedFlowAI_Fase3/04_structured_data/` |
| Duração | ~1 min |

**Produz:** `04_minimizacao_evidencia.json`, `database_report.json`, exportação CSV das tabelas e o
próprio `hospital.db` em `database/`.

**Critério de sucesso:** imprime "Campos identificadores presentes no prompt: NENHUM ✅" e
"recuperação exata: 1.000".

---

## Notebook 05 — Demonstração ponta a ponta

| | |
|---|---|
| Runtime | CPU no modo offline · **GPU** no modo submissão |
| Pasta no Drive | `MedFlowAI_Fase3/05_demo/` |
| Adapter | usa por padrão `MedFlowAI_Fase3/02_fine_tuning/adapter/` |

### Para a demonstração oficial do vídeo

1. selecione **GPU** (o modelo fine-tuned precisa dela);
2. na célula de configuração, deixe `MODO = "auto"` (usa o fine-tuned se o adapter existir) ou
   `MODO = "submission"` para **exigir** o fine-tuned;
3. `Executar tudo`;
4. confirme que a saída diz **"MODO SUBMISSÃO — LLM CUSTOMIZADA POR FINE-TUNING"** e lista
   `adapter_carregado: True`.

> Se o adapter não existir, o modo submissão **falha de propósito**, com a mensagem de rodar o
> notebook 02 antes. Isso existe para impedir que o vídeo seja gravado com o baseline achando que é a
> LLM customizada.

**Os oito itens que o vídeo precisa mostrar** estão nos cenários C1 a C5 mais as seções 4 a 6:
protocolo, contexto do paciente, exame pendente e alerta, revisão humana, bloqueio de bypass, fontes,
trilha de processamento e logs de auditoria.

**Produz:** `05_demo_<modo>.json` com os cenários e as avaliações, a transcrição em `exports/`
(útil como legenda do vídeo), o diagrama Mermaid e a cópia da trilha de auditoria em `logs/`.

---

## Depois de rodar tudo

1. **Devolva o bundle.** Anexe `medflow_colab_results.zip` numa nova sessão do Claude Code e peça a
   atualização de README e relatório **somente com os números reais**. O caminho é
   `MedFlowAI_Fase3/02_fine_tuning/bundles/medflow_colab_results.zip`.

2. **Valide localmente**, se quiser conferir antes:

   ```bash
   unzip medflow_colab_results.zip -d artifacts/fine_tuning
   python -m medflow_ai.cli validate-colab-results artifacts/fine_tuning
   python -m medflow_ai.cli inspect-bundle medflow_colab_results.zip
   ```

3. **Rode a CI**: um push em `develop` já dispara o workflow; confirme que ficou verde.

4. **Preencha integrantes e RMs** no README §17.

5. **Grave o vídeo** seguindo [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md), com o notebook 05 em modo
   submissão. Adicione o link no README §17.

6. **Merge final para `main`** e troque `/blob/develop/` por `/blob/main/` nos badges do Colab.

---

## Nota administrativa — histórico do Git

Os PDFs originais das aulas foram removidos da árvore atual e o `.gitignore` impede que voltem. **O
histórico não foi reescrito**: os blobs antigos continuam acessíveis por `git log`/`git show`.

Antes de tornar o repositório público ou entregá-lo formalmente, considere:

- `git checkout --orphan` + commit único, ou um squash do histórico; ou
- `git filter-repo --path "docs/Fase 3 - Generative AI" --invert-paths` (requer force-push e
  coordenação com quem tiver clones).

Não fiz isso automaticamente porque reescrever histórico é irreversível e afeta qualquer clone
existente — é uma decisão sua.

---

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| "Nenhuma GPU CUDA detectada" no notebook 02 | runtime em CPU | `Alterar o tipo de ambiente de execução` → GPU → `Salvar` → `Executar tudo` |
| Drive não monta | autorização recusada ou expirada | reexecute a primeira célula e aceite o pedido de acesso |
| "Os resultados ficarão apenas em /content" | falha no mount | não continue: resolva o Drive antes, senão perde tudo ao encerrar |
| OOM no treino | VRAM insuficiente | ligue `USAR_RECUPERACAO_OOM = True` conforme a célula indica |
| Notebook 05 falha pedindo adapter | notebook 02 ainda não rodou | rode o 02, ou use `MODO = "offline"` sabendo que **não** é a LLM customizada |
| Seção densa do 03 não roda | sem rede para o Hugging Face | é opcional; os números oficiais não dependem dela |
| Validador reprova | artefato faltando ou incompleto | ele lista exatamente o que falta; reexecute a parte correspondente |
