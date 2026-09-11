# Artefatos de resultado

Métricas **reais**, produzidas por execução e versionadas como evidência. Regenerar tudo:

```bash
python -m medflow_ai.cli evaluate --with-generation
```

| Arquivo | Conteúdo | Gerado por |
|---|---|---|
| `evaluation_summary.json` | resumo consolidado das quatro avaliações + ablação | `cli evaluate` |
| `rag/rag_experiments.json` · `.csv` | grade de 36 configurações de recuperação (estratégia × k × chunk) | `rag_eval.run_experiment_grid` |
| `safety/safety_report.json` | acurácia por categoria e matriz de confusão (conjunto de desenvolvimento) | `safety_eval.evaluate_safety` |
| `safety/safety_confusion.txt` | matriz de confusão em texto | idem |
| `safety/safety_holdout_v1_pre_fix.json` | **medição congelada** do held-out v1, antes das correções que ele motivou | congelado manualmente |
| `safety/safety_holdout_v2_frozen.json` | **medição congelada** do held-out v2 — é este o número de generalização | congelado manualmente |
| `database/database_report.json` | recuperação exata do prontuário e verificação de vazamento de PII | `database_eval` |
| `graph/graph_report.json` | roteamento, nós executados e decisão de revisão humana por cenário | `graph_eval` |
| `fine_tuning/generation_comparison.json` | ablação com/sem RAG em 22 exemplos held-out | `cli evaluate --with-generation` |

## Ainda não gerados (dependem de GPU)

Produzidos por `notebooks/02_fine_tuning_qlora.ipynb`:

- `fine_tuning/training_results.json` — loss real e configuração efetiva
- `fine_tuning/loss_curve.png`
- `fine_tuning/comparacao_sistemas.json` — base × fine-tuned × fine-tuned + RAG
- `fine_tuning/respostas_antes_depois.json` — evidência qualitativa

## Regras

- **nada aqui é estimado.** Se um arquivo não existe, a medição não foi feita;
- arquivos marcados como *congelados* não devem ser regenerados: eles registram o estado em que a
  medição foi tomada (ver `docs/DECISIONS.md`, ADR-008);
- pesos de modelo, checkpoints e arquivos grandes **não** entram aqui — vão para Google Drive ou
  Hugging Face, com o caminho registrado no `.env`.
