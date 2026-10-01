# Minicurso FEMa — Código de Apoio (Dia 1 + Dia 2)

Código de apoio para os dois dias do minicurso (4h cada). Versão enxuta — pensada
para dar tempo de **codar ao vivo** durante a aula, não só rodar células prontas.

## Estrutura

```
src/
  distance_models.py   # KNN do zero (funções puras) + versões vetorizadas compatíveis com scikit-learn
  viz.py                # fronteiras de decisão, scatter ponderado, matriz de confusão, resíduos
  metrics_utils.py       # wrappers de métricas de classificação e regressão
  datasets_utils.py      # carregamento e normalização de datasets (breast_cancer, digits, iris, wine, diabetes, sintético)
  fema.py                # FEMaClassifier / FEMaRegressor, com as 5 funções de base num dict (BASIS_FUNCTIONS)

notebooks/
  # Dia 1 — Fundamentos (4h)
  01_knn_do_zero.ipynb                  -> Bloco 4 / "Pausa para código" #1
  02_normalizacao.ipynb                  -> Slide "Normalização de Dados"
  03_tuning_grid_random.ipynb            -> "Pausa para código" #2 (Grid/Random Search)
  04_metricas.ipynb                      -> "Pausa para código" #3 (métricas)
  05_comparacao_modelos_lineares.ipynb    -> Fechamento Dia 1: KNN vs. modelos lineares em vários datasets

  # Dia 2 — FEMa (4h)
  06_fema_do_zero.ipynb                 -> Codar o FEMa ao vivo: distância -> funções de base -> classe -> fronteiras
  07_estudo_de_caso_digits.ipynb          -> Caso real: EDA (Digits, 64 features/10 classes) -> tuning -> comparação -> métricas
```

Todos os notebooks já foram executados (saídas/gráficos salvos nos próprios .ipynb) —
basta abrir e ler, ou rodar de novo do zero (Kernel > Restart & Run All).

## O FEMa, em poucas linhas

`src/fema.py` tem só o essencial: um dicionário `BASIS_FUNCTIONS` com as 5 funções do
pôster (Shepard, Wendland C2, Laplaciana, Inverse Multiquadratic, Radial) e uma classe
`FEMaClassifier`/`FEMaRegressor` que busca os `k` vizinhos mais próximos por força
bruta e tira a média ponderada pelo peso da função de base escolhida — a mesma lógica
do KNN ponderado do Dia 1, só trocando "1/distância" fixo por uma função plugável.

Segue o padrão `fit`/`predict` do scikit-learn, então funciona direto com
`GridSearchCV`, `cross_val_score` e `plot_decision_boundary` já usados no Dia 1.

**Deixado de propósito como próximo passo (não entra no minicurso):**
- **Estruturas de dados espaciais** (KD-Tree/Ball-Tree) para acelerar a busca de vizinhos
  além da força bruta — importa em bases maiores que as usadas aqui.
- **Verificação estatística rigorosa** (Friedman + Wilcoxon + Holm + rank-biserial, como
  no pôster do CIC) para confirmar se a função de base "vencedora" é significativamente
  melhor, ou só parece melhor por acaso.

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Nota sobre datasets

`synthetic_nonlinear` (usado em `05_comparacao_modelos_lineares.ipynb`) é gerado localmente
com `make_regression` + uma componente não-linear — usado no lugar de `california_housing`
porque este exigiria baixar dados da internet. Se seu ambiente tiver acesso à rede, você pode
trocar por `fetch_california_housing()` em `src/datasets_utils.py` sem mudar mais nada.
