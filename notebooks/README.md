# Notebooks

Experimentos reprodutíveis. A lógica reutilizável vive em `src/medflow_ai/` — os notebooks **chamam**
o pacote, não reimplementam, para que notebook e repositório nunca divirjam.

| Notebook | O que demonstra | Executa sem GPU? |
|---|---|---|
| `01_data_preprocessing.ipynb` | corpus institucional, anonimização **antes/depois**, curadoria com estatísticas, split por documento com interseção verificada | ✅ |
| `02_fine_tuning_qlora.ipynb` | diagnóstico de GPU, baseline antes do treino, SFT QLoRA, salvamento e recarga do adapter, comparação base × fine-tuned × fine-tuned+RAG | ❌ **requer GPU** (Google Colab) |
| `03_rag_pipeline.ipynb` | chunking rastreável, grade de 36 experimentos com gráficos, análise das falhas e ablação com/sem RAG | ✅ |
| `04_structured_patient_data.ipynb` | SQLite com esquema Synthea, consultas e evidência de **minimização de dados** | ✅ |
| `05_medflow_full_demo.ipynb` | os 5 cenários do vídeo, guardrail de saída e trilha de auditoria | ✅ |

## Antes de rodar

```bash
pip install -r requirements.txt && pip install -e .
pip install matplotlib pandas            # usados nos gráficos dos notebooks 02 e 03
python -m medflow_ai.cli bootstrap       # banco, índice e dataset
```

Os notebooks localizam a raiz do projeto sozinhos, então funcionam tanto em execução local quanto no
Colab (o notebook 02 traz a célula de `git clone`).

## Estrutura de cada seção

Seguindo o feedback recebido na Fase 1, cada experimento responde a cinco perguntas:
**qual pergunta responde · o que o resultado mostra · qual trade-off apareceu · é suficiente para o
caso de uso · quais limitações permanecem**. Gráfico sem pergunta explícita não entra.
