# Checklist oficial — Tech Challenge Fase 3

Controle interno de cobertura dos requisitos. **Regra do projeto:** um item só é marcado quando existe
*implementação + teste/resultado + evidência localizável*. Marcar sem evidência é proibido.

Legenda: ✅ concluído e verificável · ⏳ pendente de recurso externo · 📝 depende de ação do grupo

---

## 1. Fine-tuning de LLM com dados médicos internos

| Item | Status | Evidência |
|---|---|---|
| Modelo base selecionado e justificado | ✅ | `fine_tuning/config.py`; `docs/FINE_TUNING.md` §2 |
| Dataset de treinamento definido | ✅ | `data/processed/sft/` (137 exemplos) |
| Exemplos de **protocolos médicos** | ✅ | família `protocolo_qa` (82) |
| Exemplos de **perguntas frequentes de médicos** | ✅ | família `faq_medico` (12) |
| Exemplos de **modelos de laudo** | ✅ | famílias `modelo_laudo` (5) + `laudo_preenchido` (10) |
| Exemplos de **receitas** | ✅ | família `modelo_receita` (6) |
| Exemplos de **procedimentos internos** | ✅ | família `procedimento` (14) |
| Preprocessing implementado | ✅ | `fine_tuning/dataset.py`; notebook 01 |
| **Anonimização** demonstrável | ✅ | 80 identificadores removidos em 8 classes; notebook 01 §2; `test_anonymization.py` |
| **Curadoria** (duplicados, comprimento, idioma, PII) | ✅ | `curate()`; estatísticas em `manifest.json` |
| Split sem leakage | ✅ | split por documento; `test_split_por_documento_evita_leakage` |
| Configuração e seeds registradas | ✅ | `QLoRAConfig`; `training_config.json` |
| Estratégia LoRA/QLoRA implementada | ✅ | `fine_tuning/train.py` |
| Notebook Colab funcional | ✅ | `notebooks/02_fine_tuning_qlora.ipynb` |
| **Treino executado em GPU** | ⏳ | requer Colab; pipeline recusa fabricar métrica (`test_treino_sem_gpu_nao_inventa_metricas`) |
| **Adapter/checkpoint gerado** | ⏳ | idem |
| **Comparação base × fine-tuned** | ⏳ | código pronto (`fine_tuning/evaluate.py`); baseline sem LLM já medido |

---

## 2. Assistente médico com LangChain

| Item | Status | Evidência |
|---|---|---|
| Pipeline integrando a LLM customizada | ✅ | `llm/providers.py` (`hf_local` com adapter) |
| Document Loaders | ✅ | `rag/loaders.py` (Markdown + `PyPDFLoader`) |
| Prompts estruturados e versionados | ✅ | `llm/prompts.py` (`PROMPT_VERSION`); `test_prompts.py` |
| Vector store e retriever | ✅ | `MedFlowVectorStore` implementa `VectorStore`; 4 estratégias |
| Tools LangChain | ✅ | 4 `@tool` em `graph/tools.py`; `test_tools.py` |
| **Consulta a base estruturada** | ✅ | `database/repository.py`; `test_database.py` |
| **Contextualização com dados atuais do paciente** | ✅ | README §4.2 (alerta só existe por causa do prontuário) |
| RAG com metadados e fontes | ✅ | citação `[DOC-ID §SEÇÃO]` + bloco `FONTES CONSULTADAS` |

---

## 3. Fluxos LangGraph

| Item | Status | Evidência |
|---|---|---|
| Estado compartilhado | ✅ | `MedicalState` com reducer em `processing_steps` |
| Nós especializados | ✅ | 11 nós em `graph/nodes.py` |
| **Arestas condicionais reais** | ✅ | 3 decisões; `graph/build.py` |
| Fluxo automatizado demonstrável | ✅ | `cli demo` (5 cenários); notebook 05 |
| Diagrama atualizado | ✅ | gerado do grafo compilado: `cli graph` |
| Teste de roteamento | ✅ | `graph_eval`: rota 1,000 · nós 1,000 · revisão 1,000 |
| Resiliência a falhas | ✅ | 3 testes: LLM indisponível, retrieval quebrado, sem banco |

---

## 4. Segurança e validação

| Item | Status | Evidência |
|---|---|---|
| Limites explícitos de atuação | ✅ | `docs/SAFETY_POLICY.md` §1; prompt do sistema |
| Bloquear prescrição/decisão autônoma | ✅ | `HR-PRESCRIPTION-01`, `HR-DOSE-01/02`, `OUT-DOSE-01` |
| Nó de validação/safety | ✅ | `validate_input` (entrada) + `validate_answer` (saída) |
| Caminho de human review | ✅ | nó `human_review`; README §4.3 |
| Teste de prompts permitidos e proibidos | ✅ | 112 prompts rotulados; `test_safety.py` |
| **Zero subestimação de risco** | ✅ | 3 conjuntos; teste trava a CI se houver |
| Generalização medida em held-out | ✅ | 0,9688 em `safety_holdout_v2` (congelado) |
| Logging detalhado e auditável | ✅ | `logs/audit.jsonl`; `cli logs` |
| Pseudonimização em logs | ✅ | `patient_id_hash`; `test_audit.py` |
| **Explainability / fonte da resposta** | ✅ | citação por seção em toda resposta clínica |
| Guardrail independe do modelo | ✅ | `test_saida_insegura_e_barrada_pelo_guardrail` |

---

## 5. Organização do código

| Item | Status | Evidência |
|---|---|---|
| Estrutura modular em Python | ✅ | 10 subpacotes em `src/medflow_ai/` |
| Configuração centralizada por ambiente | ✅ | `config.py`; nenhum módulo lê `os.environ` direto |
| `.env.example` completo | ✅ | 25 variáveis documentadas |
| `.gitignore` protege segredos, dados e pesos | ✅ | — |
| **Testes automatizados** | ✅ | 176 testes, 89% de cobertura |
| **CI/CD** | ✅ | GitHub Actions: matriz 3.11/3.12, pipeline do zero, artefatos |
| CLI reproduzível | ✅ | 9 comandos |
| README completo com exemplos de uso | ✅ | 17 seções, saídas reais verificadas |

> Os três itens em negrito atacam diretamente o feedback recebido na Fase 2 ("ausência de testes
> automatizados, ausência de CI/CD, README sem exemplos claros de uso").

---

## 6. Entregáveis

| Item | Status | Evidência |
|---|---|---|
| Pipeline de fine-tuning no repositório | ✅ | `src/medflow_ai/fine_tuning/` + notebook 02 |
| Integração LangChain | ✅ | seção 2 desta tabela |
| Fluxos LangGraph | ✅ | seção 3 desta tabela |
| Dataset anonimizado / sintético versionado | ✅ | `data/synthetic/`, `data/processed/sft/` |
| Código reproduzível | ✅ | `cli bootstrap` reconstrói tudo; CI faz isso a cada push |
| **Relatório técnico** | ✅ | `docs/REPORT.md` |
| — processo de fine-tuning | ✅ | REPORT §6; `docs/FINE_TUNING.md` |
| — descrição do assistente | ✅ | REPORT §3 |
| — diagrama do fluxo | ✅ | REPORT §3; `docs/ARCHITECTURE.md`; `cli graph` |
| — avaliação do modelo | ✅ (parcial ⏳) | REPORT §13–14; fine-tuning pendente de GPU |
| — análise dos resultados | ✅ | REPORT §14 (5 perguntas por experimento) |
| **Roteiro do vídeo** | ✅ | `docs/VIDEO_SCRIPT.md` |
| **Vídeo gravado (≤ 15 min)** | 📝 | ação do grupo |
| — mostra LLM personalizada | ⏳📝 | depende do treino em GPU |
| — mostra fluxo automatizado | ✅ | `cli demo` |
| — mostra pergunta clínica contextualizada | ✅ | `cli ask --patient-id` |
| — mostra logs e validação | ✅ | `cli logs`; cenários 4 e 5 |
| Integrantes e RMs no README | 📝 | ação do grupo |
| Link do vídeo no README | 📝 | ação do grupo |

---

## 7. O que falta, em ordem

1. **Executar `notebooks/02_fine_tuning_qlora.ipynb` no Colab com GPU** e colar os números reais em
   `README.md` §5.6 e `docs/REPORT.md` §6 e §13 (checklist em `docs/FINE_TUNING.md` §7).
2. **Gravar o vídeo** seguindo `docs/VIDEO_SCRIPT.md`.
3. **Preencher integrantes/RMs** e o link do vídeo no README.
