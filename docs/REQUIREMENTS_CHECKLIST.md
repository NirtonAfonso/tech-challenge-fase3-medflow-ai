# Checklist oficial — Tech Challenge Fase 3

Documento de controle interno para garantir que cada requisito do enunciado tenha implementação e evidência na entrega.

## 1. Fine-tuning de LLM com dados médicos internos

- [ ] Selecionar modelo base (decisão inicial: modelo pequeno compatível com QLoRA/Colab).
- [ ] Definir dataset de treinamento.
- [ ] Incluir exemplos que representem protocolos médicos.
- [ ] Incluir perguntas frequentes de médicos.
- [ ] Incluir exemplos sintéticos de laudos, receitas e procedimentos internos.
- [ ] Implementar preprocessing.
- [ ] Implementar anonimização/pseudonimização demonstrável.
- [ ] Implementar curadoria: duplicados, registros inválidos, padronização e qualidade.
- [ ] Separar treino/validação/teste evitando leakage.
- [ ] Registrar configuração de treinamento e seeds.
- [ ] Salvar adapter/checkpoint final sem versionar pesos grandes diretamente no Git.
- [ ] Comparar modelo base e modelo ajustado.

## 2. Assistente médico com LangChain

- [ ] Pipeline com a LLM customizada.
- [ ] Document Loaders para o corpus escolhido.
- [ ] Prompt(s) estruturados e versionados.
- [ ] Consulta a base estruturada de prontuários/registros sintéticos.
- [ ] Contextualização da resposta com informações atualizadas do paciente.
- [ ] RAG para protocolos/documentos com metadados e fontes.

## 3. LangGraph

- [ ] Definir estado compartilhado.
- [ ] Implementar nós especializados.
- [ ] Implementar arestas e pelo menos um fluxo condicional relevante.
- [ ] Fluxo automatizado demonstrável no vídeo.
- [ ] Diagrama do grafo atualizado.

## 4. Segurança e validação

- [ ] Limites explícitos de atuação.
- [ ] Bloquear prescrição/decisão clínica autônoma sem validação humana.
- [ ] Nó de validação/safety.
- [ ] Caminho de human review.
- [ ] Logging detalhado e auditável.
- [ ] Pseudonimização de identificadores em logs.
- [ ] Explainability/rastreabilidade: fontes da resposta.
- [ ] Testes de cenários seguros e inseguros.

## 5. Organização do código

- [x] Estrutura modular em Python criada.
- [x] README inicial criado.
- [x] `.env.example` criado.
- [x] `.gitignore` protege segredos, dados brutos e pesos.
- [x] CI inicial para testes criada.
- [ ] README final com instalação, execução, arquitetura e exemplos.

## 6. Entregáveis

- [ ] Repositório com pipeline de fine-tuning.
- [ ] Integração LangChain.
- [ ] Fluxos LangGraph.
- [ ] Dataset anonimizado ou amostra sintética versionada.
- [ ] Relatório técnico detalhado.
- [ ] Explicação do processo de fine-tuning no relatório.
- [ ] Descrição do assistente no relatório.
- [ ] Diagrama do fluxo LangChain/LangGraph.
- [ ] Avaliação do modelo e análise crítica dos resultados.
- [ ] Vídeo de até 15 minutos.
- [ ] Vídeo mostra treinamento/LLM personalizada.
- [ ] Vídeo mostra fluxo automatizado.
- [ ] Vídeo mostra pergunta clínica contextualizada.
- [ ] Vídeo mostra logs e validação.

## Regra do projeto

Tudo que valer requisito deve ter **implementação + teste/resultado + evidência clara no README/relatório/vídeo**. Não depender de o avaliador descobrir sozinho que a funcionalidade existe.
