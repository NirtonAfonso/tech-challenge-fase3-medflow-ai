# Dados

Estrutura prevista:

```text
data/
├── raw/        # não versionado; downloads locais
├── processed/  # artefatos processados; versionar apenas amostras pequenas e seguras
└── synthetic/  # dados sintéticos permitidos para demonstração/testes
```

## Política

**Nunca adicionar dados reais de pacientes a este repositório.**

Cada dataset utilizado deverá possuir no relatório:
- origem;
- licença;
- versão/data de acesso;
- finalidade (treino, RAG, prontuário ou avaliação);
- método de preprocessing/anonimização;
- estratégia de split.
