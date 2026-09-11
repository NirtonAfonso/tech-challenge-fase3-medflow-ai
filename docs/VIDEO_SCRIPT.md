# Roteiro do vídeo — MedFlow AI (≤ 15 minutos)

**Objetivo:** demonstrar, em ordem, tudo que o enunciado exige ver funcionando, com evidência na tela.

O enunciado pede que o vídeo mostre: **treinamento e funcionamento da LLM personalizada**,
**execução de um fluxo automatizado**, **resposta a perguntas clínicas contextualizadas** e
**logs e validação das respostas**. Cada bloco abaixo está amarrado a um desses itens.

---

## Preparação (antes de gravar)

```bash
cd tech-challenge-fase3-medflow-ai
source .venv/bin/activate
python -m medflow_ai.cli bootstrap        # banco + índice + dataset
rm -f logs/audit.jsonl                    # log limpo para a demonstração ficar legível
pytest -q                                 # confirme que está verde antes de gravar
```

**Abas/janelas a deixar prontas:** terminal em fonte grande; editor com
`src/medflow_ai/safety/policy.py` e `src/medflow_ai/graph/build.py`; navegador com o repositório na
aba de Actions (CI verde) e `artifacts/rag/rag_experiments.csv`.

**Regra de ouro:** nada de slides longos. Terminal rodando > slide explicando.

---

## Bloco 1 — Problema e proposta · 0:00 – 1:15

**Tela:** README, seção 1.

**Fala.** "O hospital quer um assistente médico treinado com dados próprios. O desafio real não é
fazer um chatbot: é resolver quatro tensões — conhecimento versus rastreabilidade, personalização
versus privacidade, utilidade versus segurança, e demonstração versus evidência."

"O MedFlow AI resolve isso separando responsabilidades: o **fine-tuning** ensina formato e
comportamento; o **RAG** guarda o conhecimento rastreável; o **banco** tem o dado do paciente; o
**LangGraph** orquestra com segurança. Vou mostrar cada parte funcionando e o número que a sustenta."

---

## Bloco 2 — Arquitetura, do código · 1:15 – 2:30

**Tela:** terminal.

```bash
python -m medflow_ai.cli graph
```

**Fala.** "Este diagrama não foi desenhado: foi extraído do grafo compilado, então não pode ficar
defasado. Onze nós, três decisões condicionais: bloqueio na entrada, presença de paciente, e
necessidade de revisão humana na saída."

Aponte no editor (`graph/build.py`) as duas funções de aresta condicional.

---

## Bloco 3 — Dados, anonimização e curadoria · 2:30 – 4:30

**Tela:** notebook `01_data_preprocessing.ipynb`, seção 2, e depois terminal.

**Fala.** "Os dados internos do hospital são 15 documentos sintéticos em português: protocolos, FAQ
de médicos, modelos de laudo e receita, procedimentos internos. Cada arquivo declara que é sintético
— não apresentamos documento oficial como protocolo interno."

Mostre o **antes/depois** do registro do paciente no notebook.

**Fala.** "Os pacientes sintéticos carregam CPF, CNS, telefone e endereço **falsos de propósito**.
Sem isso, a anonimização não teria o que remover e a demonstração seria decorativa."

```bash
python -m medflow_ai.cli build-dataset
```

**Fala.** "Oitenta identificadores removidos, em oito classes. E o split é **por documento**: três
protocolos inteiros ficam de fora do treino, para o benchmark não ser trivial."

---

## Bloco 4 — Fine-tuning · 4:30 – 6:30

**Tela:** `notebooks/02_fine_tuning_qlora.ipynb` (percorrer rapidamente), depois terminal.

**Fala.** "QLoRA em modelo instruct de 3B: base congelada em 4 bits, adaptadores LoRA treináveis.
O notebook faz diagnóstico de GPU, baseline **antes** do treino, o SFT, salva o adapter, recarrega e
compara base × fine-tuned × fine-tuned + RAG."

```bash
python -m medflow_ai.fine_tuning.train --check-env
```

**Fala.** "E aqui está um ponto de honestidade: sem GPU, o pipeline **se recusa a produzir métrica**.
Nenhum número de treino foi inventado neste trabalho."

> **Se o treino já tiver sido executado**, use este bloco para mostrar a curva de perda
> (`artifacts/fine_tuning/loss_curve.png`), a tabela comparativa e dois exemplos antes/depois.
> **Se não tiver**, diga com todas as letras que está pendente e mostre o notebook pronto.

---

## Bloco 5 — Fluxo automatizado ponta a ponta · 6:30 – 9:30

**Tela:** terminal.

```bash
python -m medflow_ai.cli demo
```

Comente cada cenário enquanto rola:

| Cenário | O que apontar na tela |
|---|---|
| 1 — protocolo, sem paciente | rota `protocol_only`, `safety=SAFE`, bloco `FONTES CONSULTADAS` |
| 2 — pergunta com paciente | rota `patient_context`, os nós `retrieve_patient` e `check_alerts` na trilha |
| 3 — exames pendentes e alertas | bloco `ALERTAS AUTOMÁTICOS` |
| 4 — pedido de prescrição | cabeçalho `RASCUNHO — PENDENTE DE VALIDAÇÃO HUMANA` |
| 5 — tentativa de burla | `SOLICITAÇÃO BLOQUEADA` e a ausência de `generate_answer` na trilha |

**Fala no cenário 5.** "Repare na sequência de passos: `validate_input`, `classify_request`,
`blocked_response`. O nó de geração **não executou**. Nada foi gerado antes do bloqueio."

---

## Bloco 6 — Pergunta clínica contextualizada · 9:30 – 11:00

**Tela:** terminal.

```bash
python -m medflow_ai.cli ask "Este paciente usa levotiroxina; há interação relevante e quando repetir o TSH?" --patient-id P-DEMO-0001
```

**Fala.** "Este é o requisito de contextualização. Repare no alerta: levotiroxina e carbonato de
cálcio, intervalo mínimo de 4 horas. Nenhuma dessas informações estava na pergunta — o alerta veio do
**prontuário**, a regra das 4 horas veio do **protocolo**, e a resposta cita a seção exata."

Mostre em seguida o que sobe ao modelo:

```bash
python -m medflow_ai.cli ask "Quais exames estão pendentes?" --patient-id P-DEMO-0001 --json | head -40
```

**Fala.** "O nome, o CPF e o telefone estão no banco, mas **não** no contexto. O paciente vai
pseudonimizado e a data de nascimento vira faixa etária. Minimização de dados é arquitetura aqui, não
filtro de texto."

---

## Bloco 7 — Segurança e validação · 11:00 – 12:45

**Tela:** editor em `safety/policy.py`, depois terminal.

**Fala.** "A segurança não é um disclaimer: são 16 regras nomeadas, versionadas e testadas, aplicadas
na entrada e na saída."

```bash
python -m medflow_ai.cli evaluate | head -40
```

**Fala.** "Cento e doze prompts rotulados em três conjuntos. O número que importa é o do held-out que
**nunca** foi usado para ajustar regra: 96,88%. E, mais importante que a acurácia: **zero
subestimações de risco** — nenhum pedido perigoso foi tratado como seguro. Errar para o lado seguro é
aceitável; para o lado inseguro, não."

```bash
pytest tests/test_graph.py::test_saida_insegura_e_barrada_pelo_guardrail -v
```

**Fala.** "Este teste injeta uma LLM que responde 'administrar 75 mcg de levotiroxina ao dia' e
verifica que o guardrail de saída intercepta. A segurança não depende do modelo se comportar bem."

---

## Bloco 8 — Logs, auditoria e evidência · 12:45 – 14:00

**Tela:** terminal.

```bash
python -m medflow_ai.cli logs --limit 2
```

**Fala.** "Uma linha por execução: `trace_id`, rota, hash do paciente, regras acionadas, violações,
fontes citadas, passos, versões de política e prompt, latência. Sem nome, sem CPF, sem chave."

```bash
pytest -q
```

**Fala.** "Cento e setenta e seis testes, 89% de cobertura. E a CI reconstrói o pipeline do zero a
cada push." — mostre a aba do GitHub Actions verde e o `artifacts/rag/rag_experiments.csv`.

---

## Bloco 9 — Resultados, limitações e conclusão · 14:00 – 15:00

**Tela:** README, seção 5.

**Fala.** "Resultados medidos: recuperação 0,875 de hit@5 na melhor de 36 configurações; recuperação
exata do prontuário 1,000; roteamento do grafo 1,000; segurança 96,88% em held-out com zero
subestimações."

"O resultado mais interessante não é uma métrica alta, é uma confirmação: sem RAG a fidelidade às
fontes é **zero**; com RAG, 0,557 e todas as respostas citam fonte. A separação entre conhecimento e
comportamento funciona — e isso está medido, não afirmado."

"Duas descobertas contrariaram a expectativa e foram mantidas porque seguimos o dado: o MMR foi a
**pior** estratégia de recuperação neste corpus, e enriquecer a consulta com as condições do paciente
**piorava** o resultado quando substituía a pergunta original."

"Limitações declaradas: o fine-tuning ainda não foi executado em GPU — e nenhum número foi inventado
no lugar; o dataset tem 137 exemplos; o benchmark de RAG roda sobre o próprio corpus; a política de
segurança é por regras, frágil a variação lexical. Tudo está no README, seção 15."

"O MedFlow AI é assistivo, acadêmico, e não substitui avaliação médica. Obrigado."

---

## Checklist de gravação

- [ ] `pytest -q` verde antes de começar
- [ ] `logs/audit.jsonl` apagado
- [ ] terminal em fonte grande, tema de alto contraste
- [ ] CI verde visível em alguma tomada
- [ ] os cinco cenários do `demo` aparecem
- [ ] o bloco `FONTES CONSULTADAS` aparece pelo menos uma vez em close
- [ ] a trilha de passos do cenário bloqueado aparece em close
- [ ] o log de auditoria aparece em close
- [ ] a pendência do fine-tuning é dita em voz alta
- [ ] duração total ≤ 15 minutos
- [ ] link do vídeo adicionado ao README §17 após publicar
