# Estratégia de datasets

> Status: candidatos pesquisados; seleção final ainda não congelada.

## Objetivo

Não usar um único dataset para tentar resolver problemas diferentes. O projeto separará dados de **treinamento**, **conhecimento RAG**, **prontuário estruturado** e **avaliação**.

## Candidatos

### MedQuAD

**Uso pretendido:** instruction tuning / perguntas e respostas médicas.

Pontos fortes:
- grande quantidade de pares pergunta-resposta;
- variedade de tipos de pergunta médica;
- estrutura adequada para converter em exemplos supervisionados.

Cuidados:
- conteúdo majoritariamente em inglês;
- verificar e documentar a licença/atribuição na versão efetivamente usada;
- não misturar exemplos idênticos entre treino e teste.

### PubMedQA

**Uso pretendido:** benchmark complementar de conhecimento/raciocínio biomédico.

Pontos fortes:
- perguntas apoiadas por literatura biomédica;
- conjunto rotulado por especialistas disponível.

Cuidados:
- formato yes/no/maybe não representa sozinho o uso conversacional do assistente;
- usar preferencialmente como avaliação complementar, não como único corpus de treinamento.

### Protocolos Clínicos e Diretrizes Terapêuticas (PCDT)

**Uso pretendido:** corpus principal de RAG em português e fonte rastreável.

Pontos fortes:
- protocolos clínicos em português;
- aderência ao cenário de protocolos internos do hospital;
- permite demonstrar recuperação + fonte.

Cuidados:
- registrar versão/data/origem de cada documento;
- separar documentos usados em treino daqueles reservados à avaliação quando houver QA derivado.

### Protocolos Clínicos BR / PCDT-QA / HealthBench-BR

**Uso pretendido:** avaliar adaptação ao contexto brasileiro e, após verificação final de licença e splits, complementar treino/benchmark.

Cuidados:
- congelar versões/commits utilizados;
- respeitar separação por protocolo para reduzir leakage;
- documentar claramente qual parte entrou em treino e qual ficou somente para teste.

### Synthea

**Uso pretendido:** dados estruturados de pacientes totalmente sintéticos.

Possíveis tabelas locais:
- patients;
- conditions;
- observations;
- medications;
- procedures;
- encounters.

Pontos fortes:
- dados sintéticos;
- adequado para demonstrar consulta a prontuários sem expor pacientes reais;
- exportações estruturadas úteis para SQLite/FHIR/CSV.

## Combinação recomendada neste momento

| Finalidade | Fonte inicial |
|---|---|
| Fine-tuning | MedQuAD + exemplos sintéticos hospitalares + subconjunto em português após curadoria |
| RAG | PCDTs/protocolos clínicos em português |
| Prontuário estruturado | Synthea |
| Avaliação biomédica | PubMedQA + conjunto brasileiro separado do treino |
| Avaliação de segurança | conjunto sintético próprio de prompts permitidos/bloqueados |

## Regras de versionamento

- nunca versionar prontuários reais;
- `data/raw/` fica ignorado por padrão;
- versionar apenas amostras pequenas, sintéticas e autorizadas;
- registrar URL/origem, licença, data e versão/commit de cada dataset no relatório;
- gerar manifesto dos splits para reprodutibilidade.
