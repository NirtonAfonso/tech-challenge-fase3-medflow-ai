# Notebooks

Experimentos reprodutíveis, projetados para rodar em um **runtime Colab novo com "Executar tudo"**, sem
depender de máquina local. A lógica reutilizável vive em `src/medflow_ai/` — os notebooks **chamam** o
pacote, não reimplementam, para que notebook e repositório nunca divirjam.

👉 **Guia passo a passo para o usuário:** [`../docs/COLAB_RUNBOOK.md`](../docs/COLAB_RUNBOOK.md)

| # | Notebook | Runtime | Abrir no Colab | Saída no Drive |
|---|---|---|---|---|
| 01 | Preparação de dados, anonimização e curadoria | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/01_data_preprocessing.ipynb) | `01_preprocessing/` |
| 02 | **Fine-tuning QLoRA** | ⚠️ **GPU** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/02_fine_tuning_qlora.ipynb) | `02_fine_tuning/` |
| 03 | Pipeline de RAG: construção e avaliação | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/03_rag_pipeline.ipynb) | `03_rag/` |
| 04 | Prontuário estruturado e minimização | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/04_structured_patient_data.ipynb) | `04_structured_data/` |
| 05 | Demonstração ponta a ponta | CPU / GPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirtonAfonso/tech-challenge-fase3-medflow-ai/blob/develop/notebooks/05_medflow_full_demo.ipynb) | `05_demo/` |

> Os badges apontam para a branch **`develop`**. Depois do merge final para `main`, troque
> `/blob/develop/` por `/blob/main/` aqui, no topo de cada notebook e em `docs/COLAB_RUNBOOK.md`.

## O que o bootstrap faz sozinho

A primeira célula de cada notebook é autossuficiente:

1. detecta Colab (sem quebrar execução local);
2. monta o Google Drive e avisa, em alto e bom som, se o mount falhar;
3. clona a branch `develop` com `git clone --depth 1 --branch develop` — **nunca** a branch default
   implicitamente — ou atualiza o clone existente com `fetch` + `checkout` + `pull --ff-only`;
4. faz `os.chdir()` para a raiz e ajusta o `sys.path`;
5. instala `requirements-colab.txt` (ou as dependências de treinamento, no notebook 02) e o pacote em
   modo editável;
6. imprime versão do Python, ambiente, **branch e commit SHA**;
7. cria a estrutura de saída no Drive e grava os metadados da execução.

Você não precisa descomentar nada, nem criar pasta, nem definir variável de ambiente.

## Independência entre notebooks

Cada notebook constrói os próprios pré-requisitos (banco sintético, índice vetorial, dataset). A
ordem 01 → 02 → 03 → 04 → 05 é uma narrativa, não uma dependência — com **uma exceção**: o notebook 05
em modo submissão precisa do adapter que o notebook 02 grava em
`MedFlowAI_Fase3/02_fine_tuning/adapter/`. Sem ele, o modo submissão **falha de propósito**, em vez de
usar o baseline fingindo ser a LLM customizada.

## Execução local

Funciona também fora do Colab: o bootstrap detecta o repositório, pula clone e Drive, e escreve a mesma
árvore em `artifacts/colab/` (ignorada pelo Git).

```bash
pip install -r requirements-colab.txt && pip install -e .
jupyter lab notebooks/
```

O notebook 02 continua exigindo GPU: sem ela, o pipeline recusa treinar e **não escreve métrica nenhuma**.

## Estrutura de cada seção

Seguindo o feedback recebido na Fase 1, cada experimento responde a cinco perguntas:
**qual pergunta responde · o que o resultado mostra · qual trade-off apareceu · é suficiente para o caso
de uso · quais limitações permanecem**. Gráfico sem pergunta explícita não entra.
