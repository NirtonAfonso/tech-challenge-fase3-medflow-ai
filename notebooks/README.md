# Notebooks planejados

Os notebooks devem funcionar como experimentos reproduzíveis e não concentrar toda a lógica de produção. Funções reutilizáveis devem migrar para `src/medflow_ai/`.

Ordem planejada:

1. `01_data_preprocessing.ipynb` — coleta, curadoria, anonimização e splits;
2. `02_fine_tuning_qlora.ipynb` — baseline, treinamento e exportação do adapter;
3. `03_rag_pipeline.ipynb` — ingestão, embeddings, retrieval e avaliação;
4. `04_structured_patient_data.ipynb` — Synthea → SQLite e consultas;
5. `05_medflow_full_demo.ipynb` — integração LangChain/LangGraph para demonstração final.

Os notebooks de treinamento serão priorizados no Google Colab com GPU.
