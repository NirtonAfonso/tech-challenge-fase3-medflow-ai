---
doc_id: MOD-REC-001
titulo: Modelos Institucionais de Receita e Regras de Prescrição
versao: 2.3
vigencia: 2025-12-19
especialidade: Farmácia Clínica
tipo: modelo_documento
origem: Documento sintético acadêmico do "Hospital Sinapse" (fictício). Não é um documento oficial e não contém posologias para uso real.
---

# Modelos Institucionais de Receita e Regras de Prescrição

## 1. Princípio institucional
A prescrição é ato privativo do médico. Sistemas de apoio à decisão, incluindo o assistente digital, podem apresentar o modelo de documento, verificar campos obrigatórios e sinalizar interações, mas nunca emitem prescrição, dose ou posologia de forma autônoma. Qualquer conteúdo gerado por sistema é rascunho sujeito a revisão e assinatura médica.

## 2. Campos obrigatórios da receita
Identificação do prescritor com registro profissional; identificação do paciente; data; denominação do medicamento pela denominação comum brasileira; concentração; forma farmacêutica; posologia com dose, via, intervalo e duração; quantidade total; orientações ao paciente; e assinatura.

## 3. Modelo — Receita simples
```
RECEITUÁRIO — [UNIDADE]
Paciente: [PSEUDONIMO_PACIENTE]        Data: [DATA]
1) [MEDICAMENTO] [CONCENTRACAO] — [FORMA]
   Uso [VIA]. [DOSE] a cada [INTERVALO] por [DURACAO].
   Orientações: [ORIENTACOES]
   Quantidade: [QUANTIDADE]
Prescritor: [PROFISSIONAL] — [REGISTRO]
Assinatura: ______________________
```

## 4. Modelo — Orientação de alta com plano terapêutico
```
PLANO TERAPÊUTICO DE ALTA
Paciente: [PSEUDONIMO_PACIENTE]     Data da alta: [DATA]
Diagnósticos: [DIAGNOSTICOS]
Medicamentos mantidos: [LISTA]
Medicamentos iniciados: [LISTA]
Medicamentos suspensos: [LISTA E MOTIVO]
Exames pendentes e responsável: [LISTA]
Sinais de alerta para retorno imediato: [LISTA]
Retorno: [DATA/SERVIÇO]
Prescritor: [PROFISSIONAL] — [REGISTRO]
```

## 5. Verificações automáticas permitidas
O sistema pode checar: ausência de campo obrigatório; duplicidade terapêutica; alergia registrada em prontuário; interação medicamentosa relevante; e necessidade de ajuste por função renal estimada. Todas as verificações produzem alerta para o prescritor, jamais alteração automática da prescrição.

## 6. Prescrições que exigem dupla checagem
Anticoagulantes, insulinas, opioides, eletrólitos concentrados, quimioterápicos e antimicrobianos de uso restrito exigem dupla checagem por outro profissional habilitado antes da administração.
