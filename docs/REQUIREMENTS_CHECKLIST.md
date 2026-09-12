# Checklist oficial — Tech Challenge Fase 3

Controle interno de cobertura dos requisitos.

## Semântica dos estados

Um item **implementado** não é um item **executado**. A distinção existe porque afirmar que o
fine-tuning "funciona" antes de rodá-lo em GPU seria falso.

| Símbolo | Significado |
|---|---|
| ✅ | **executado e verificado** — existe artefato ou teste que prova |
| 🧩 | **implementado e pronto** — o código existe e é testado, mas o resultado final depende de execução externa |
| ⏳ | **depende de execução externa** — GPU no Google Colab |
| 📝 | **ação manual do grupo** — gravar vídeo, preencher dados, publicar |

> **Regra do projeto:** ✅ só é marcado com *implementação + teste/resultado + evidência localizável*.
> Marcar sem evidência é proibido.

---

## 1. Fine-tuning de LLM com dados médicos internos

| Item | Estado | Evidência |
|---|---|---|
| Modelo base selecionado e justificado | ✅ | `fine_tuning/config.py`; `docs/FINE_TUNING.md` §2 |
| Dataset de treinamento definido | ✅ | `data/processed/sft/` (137 exemplos) |
| Exemplos de **protocolos médicos** | ✅ | família `protocolo_qa` (82) |
| Exemplos de **perguntas frequentes de médicos** | ✅ | família `faq_medico` (12) |
| Exemplos de **modelos de laudo** | ✅ | famílias `modelo_laudo` (5) + `laudo_preenchido` (10) |
| Exemplos de **receitas** | ✅ | família `modelo_receita` (6) |
| Exemplos de **procedimentos internos** | ✅ | família `procedimento` (14) |
| Preprocessing implementado | ✅ | `fine_tuning/dataset.py`; notebook 01 |
| **Anonimização** demonstrável | ✅ | 80 identificadores em 8 classes; notebook 01 §2; `test_anonymization.py` |
| **Curadoria** | ✅ | `curate()`; estatísticas em `manifest.json` |
| Split sem leakage | ✅ | split por documento; `test_dataset_invariance.py` |
| Dataset independente do estado do banco | ✅ | verificado com 0, 8 e 40 pacientes |
| Configuração e seeds registradas | ✅ | `QLoRAConfig`; `training_config.json` |
| Seleção de precisão FP16/BF16 pelo hardware | ✅ | `fine_tuning/precision.py`; 17 testes com GPU injetada |
| Estratégia LoRA/QLoRA implementada | 🧩 | `fine_tuning/train.py` — código pronto, treino não executado |
| Notebook Colab **pronto para execução** | 🧩 | `notebooks/02_fine_tuning_qlora.ipynb`; 54 testes de prontidão |
| Notebook Colab **executado** | ⏳ | requer GPU; `docs/COLAB_RUNBOOK.md` §Notebook 02 |
| **Treino executado em GPU** | ⏳ | pipeline recusa fabricar métrica (`test_treino_sem_gpu_nao_inventa_metricas`) |
| **Adapter/checkpoint gerado** | ⏳ | idem |
| **Comparação base × fine-tuned** | ⏳ | código pronto (`fine_tuning/evaluate.py`); ablação sem LLM já medida |
| Validação dos resultados do Colab | 🧩 | `cli validate-colab-results`; 15 testes |

---

## 2. Assistente médico com LangChain

| Item | Estado | Evidência |
|---|---|---|
| Pipeline integrando a LLM customizada | 🧩 | `llm/providers.py` (`hf_local` + adapter) — falta o adapter real |
| Document Loaders | ✅ | `rag/loaders.py` (Markdown + `PyPDFLoader`) |
| Prompts estruturados e versionados | ✅ | `llm/prompts.py`; `test_prompts.py` |
| Vector store e retriever | ✅ | `MedFlowVectorStore` implementa `VectorStore`; 4 estratégias |
| Tools LangChain | ✅ | 4 `@tool` em `graph/tools.py` |
| **Consulta a base estruturada** | ✅ | `database/repository.py`; `test_database.py` |
| **Contextualização com dados atuais do paciente** | ✅ | README §4.2; notebook 05 cenário C2 |
| Verificar exames pendentes | ✅ | tabela `lab_orders`; notebook 05 cenário C3 |
| Emitir alertas para a equipe | ✅ | `evaluate_alerts`; cenário C3 |
| RAG com metadados e fontes | ✅ | citação `[DOC-ID §SEÇÃO]` + bloco `FONTES CONSULTADAS` |

---

## 3. Fluxos LangGraph

| Item | Estado | Evidência |
|---|---|---|
| Estado compartilhado | ✅ | `MedicalState` com reducer em `processing_steps` |
| Nós especializados | ✅ | 11 nós em `graph/nodes.py` |
| **Arestas condicionais reais** | ✅ | 3 decisões; `graph/build.py` |
| Fluxo automatizado demonstrável | ✅ | `cli demo`; notebook 05 |
| Diagrama atualizado | ✅ | gerado do grafo compilado: `cli graph` |
| Teste de roteamento | ✅ | rota 1,000 · nós 1,000 · revisão 1,000 |
| Resiliência a falhas | ✅ | 3 testes: LLM indisponível, retrieval quebrado, sem banco |

---

## 4. Segurança e validação

| Item | Estado | Evidência |
|---|---|---|
| Limites explícitos de atuação | ✅ | `docs/SAFETY_POLICY.md` §1 |
| Bloquear prescrição/decisão autônoma | ✅ | `HR-PRESCRIPTION-01`, `HR-DOSE-01/02`, `OUT-DOSE-01` |
| Nó de validação/safety | ✅ | `validate_input` + `validate_answer` |
| Caminho de human review | ✅ | nó `human_review`; README §4.3 |
| Teste de prompts permitidos e proibidos | ✅ | 112 prompts rotulados |
| **Zero subestimação de risco** | ✅ | 3 conjuntos; teste trava a CI se houver |
| Generalização medida em held-out | ✅ | 0,9688 em `safety_holdout_v2` (congelado) |
| Logging detalhado e auditável | ✅ | `logs/audit.jsonl`; `cli logs` |
| Pseudonimização em logs | ✅ | `patient_id_hash`; `test_audit.py` |
| **Explainability / fonte da resposta** | ✅ | citação por seção em toda resposta clínica |
| Guardrail independe do modelo | ✅ | `test_saida_insegura_e_barrada_pelo_guardrail` |
| Sem fallback silencioso para `template` | ✅ | notebook 05 modo submissão; `HuggingFaceChatModel` falha sem adapter |
| Segredos fora do Drive e do bundle | ✅ | `colab.persist`, `bundle.build_results_bundle`; 8 testes |

---

## 5. Organização do código

| Item | Estado | Evidência |
|---|---|---|
| Estrutura modular em Python | ✅ | 11 subpacotes em `src/medflow_ai/` |
| Configuração centralizada por ambiente | ✅ | `config.py` |
| `.env.example` completo | ✅ | 25 variáveis documentadas |
| `.gitignore` protege segredos, dados, pesos e material do curso | ✅ | `test_docs_consistency.py` |
| Material bruto do curso fora da árvore | ✅ | removido em `f092ba1` |
| **Testes automatizados** | ✅ | 338 testes |
| **CI/CD** | ✅ | GitHub Actions: 3.11/3.12, pipeline do zero, invariância, notebooks |
| CLI reproduzível | ✅ | 11 comandos |
| README completo com exemplos de uso | ✅ | 17 seções, saídas reais verificadas |
| Notebooks prontos para Colab | ✅ | bootstrap, badges, Drive; 54 testes de prontidão |
| Runbook de execução | ✅ | `docs/COLAB_RUNBOOK.md` |

---

## 6. Entregáveis

| Item | Estado | Evidência |
|---|---|---|
| Pipeline de fine-tuning no repositório | ✅ | `src/medflow_ai/fine_tuning/` + notebook 02 |
| Integração LangChain | ✅ | seção 2 |
| Fluxos LangGraph | ✅ | seção 3 |
| Dataset anonimizado / sintético versionado | ✅ | `data/synthetic/`, `data/processed/sft/` |
| Código reproduzível | ✅ | `cli bootstrap`; CI reconstrói do zero a cada push |
| **Relatório técnico** | ✅ | `docs/REPORT.md` |
| — processo de fine-tuning | ✅ | REPORT §6; `docs/FINE_TUNING.md` |
| — descrição do assistente | ✅ | REPORT §3 |
| — diagrama do fluxo | ✅ | REPORT §3; `cli graph` |
| — avaliação do modelo | 🧩 ⏳ | RAG/safety/DB/grafo medidos; fine-tuning pendente de GPU |
| — análise dos resultados | ✅ | REPORT §14 (5 perguntas por experimento) |
| **Roteiro do vídeo** | ✅ | `docs/VIDEO_SCRIPT.md` |
| **Vídeo gravado (≤ 15 min)** | 📝 | ação do grupo |
| — mostra LLM personalizada | ⏳ 📝 | depende do treino; notebook 05 em modo submissão |
| — mostra fluxo automatizado | ✅ | `cli demo`; notebook 05 |
| — mostra pergunta clínica contextualizada | ✅ | cenário C2 |
| — mostra logs e validação | ✅ | notebook 05 §5; cenários C4 e C5 |
| Integrantes e RMs no README | 📝 | ação do grupo |
| Link do vídeo no README | 📝 | ação do grupo |
| Merge para `main` | 📝 | após o vídeo; trocar badges de `develop` para `main` |

---

## 7. O que falta, em ordem

1. **Executar `notebooks/02_fine_tuning_qlora.ipynb` no Colab com GPU** seguindo
   [`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md). Devolver `medflow_colab_results.zip`.
2. **Validar** com `python -m medflow_ai.cli validate-colab-results artifacts/fine_tuning` e atualizar
   README §5.6 e REPORT §6.3/§14.6 **somente** com os números reais.
3. **Rodar o notebook 05 em modo submissão** (com o adapter) e gravar o vídeo.
4. **Preencher integrantes/RMs** e o link do vídeo no README §17.
5. **Merge para `main`** e trocar os badges do Colab para `main`.
