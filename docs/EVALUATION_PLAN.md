# Plano de avaliação

A entrega deve apresentar resultados mensuráveis e discussão crítica, não apenas uma demonstração visual.

## 1. Fine-tuning

Comparar pelo menos:

- modelo base;
- modelo fine-tuned;
- modelo fine-tuned + RAG no cenário final.

Registrar:
- configuração do treinamento;
- loss de treino/validação quando aplicável;
- conjunto de avaliação congelado;
- métricas adequadas à tarefa;
- exemplos qualitativos de sucesso e falha.

## 2. RAG

Métricas candidatas:
- Hit@K / Recall@K;
- MRR quando houver gabarito de documento relevante;
- proporção de respostas com fonte correta;
- groundedness/fidelidade avaliada por rubrica reprodutível.

Experimentos sugeridos:
- tamanho de chunk;
- `k` do retriever;
- busca por similaridade vs. MMR;
- com e sem reranking, se implementado.

## 3. Base estruturada

Criar casos de teste em que o valor correto no banco é conhecido e medir recuperação exata de:
- condições;
- exames/observações;
- medicamentos;
- procedimentos.

## 4. Segurança

Criar conjunto explícito de prompts:
- permitidos;
- exigem ressalva;
- exigem revisão humana;
- devem ser bloqueados como decisão/prescrição autônoma.

Métrica principal: taxa de roteamento correto por categoria.

## 5. LangGraph

Avaliar:
- roteamento para os nós esperados;
- preservação do estado;
- caminhos condicionais;
- comportamento em falhas de retrieval/tool;
- registro dos passos de processamento.

## 6. Discussão crítica

Para cada gráfico/tabela, responder:
1. Qual pergunta este experimento responde?
2. O que o resultado mostra?
3. Qual trade-off apareceu?
4. O resultado é suficiente para o cenário proposto?
5. Quais limitações permanecem?

Essa estrutura evita uma EDA/avaliação apenas descritiva e torna a análise orientada por perguntas.
