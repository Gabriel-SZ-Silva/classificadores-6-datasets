# Arquitetura

O projeto é um pacote Python (`src/classificadores`) com uma linha de comando e um notebook do Colab gerado a partir do mesmo código. Cada módulo tem uma responsabilidade.

```mermaid
flowchart TB
    subgraph fontes["Fontes de dados"]
        K[("API pública do Kaggle<br/>5 datasets")]
        G[("GitHub raw<br/>Eleições")]
    end

    subgraph pacote["src/classificadores"]
        CFG["config.py<br/>constantes e os 6 datasets<br/>(alvo, colunas, pré-tratamento)"]
        DAD["dados.py<br/>download com cache em data/raw,<br/>limpeza e separação X / y"]
        MOD["modelos.py<br/>pré-processamento e<br/>dicionário dos 4 classificadores"]
        AVA["avaliacao.py<br/>treino/teste, CV estratificada,<br/>RandomizedSearchCV e métricas"]
        DIA["diagnostico.py<br/>viés x variância,<br/>curvas de aprendizado, RF x GB"]
        GRA["graficos.py<br/>matrizes, ROC, curvas,<br/>mapas de calor"]
        REL["relatorio.py<br/>CSVs, tabelas.md, figuras,<br/>painel.html"]
        PIP["pipeline.py<br/>laço sobre os datasets"]
        CLI["cli.py<br/>python -m classificadores"]
    end

    subgraph saidas["resultados/"]
        CSV[("metricas.csv<br/>diagnostico.csv<br/>escolha_rf_gb.csv<br/>hiperparametros.csv")]
        MD["tabelas.md"]
        FIG["figuras/*.png"]
        HTML["painel.html"]
    end

    NB["notebooks/<br/>classificadores_6_datasets.ipynb"]

    K & G --> DAD
    CFG --> DAD & MOD & AVA & DIA
    CLI --> PIP --> DAD --> AVA
    MOD --> AVA --> DIA --> REL
    GRA --> REL
    REL --> CSV & MD & FIG & HTML
    pacote -. "scripts/gerar_notebook.py" .-> NB
```

## Módulos

| Módulo | O que faz |
|---|---|
| `config.py` | Constantes do experimento (semente, tamanho do teste, folds, iterações da busca, limiares do diagnóstico) e a descrição de cada dataset: origem, coluna-alvo, colunas usadas ou removidas e a função de pré-tratamento. |
| `dados.py` | Lê o CSV do cache `data/raw`; se faltar, baixa da API pública do Kaggle ou do GitHub e guarda no cache. Remove linhas sem rótulo e classes raras e separa `X` e `y`. |
| `modelos.py` | Monta o `ColumnTransformer` (imputação, escala e one-hot) e devolve **um dicionário** com os 4 classificadores, cada um com o seu `Pipeline` e o seu espaço de hiperparâmetros. |
| `avaliacao.py` | Divide treino e teste, cria a validação cruzada estratificada, roda o `RandomizedSearchCV` com várias métricas ao mesmo tempo e calcula as métricas de teste, a matriz de confusão e a curva ROC. Percorre o dicionário de modelos com um único `for`. |
| `diagnostico.py` | Classifica cada modelo como alta variância, alto viés ou equilibrado, gera as curvas de aprendizado do Random Forest e do Gradient Boosting e escreve a justificativa da escolha entre os dois. |
| `graficos.py` | Todas as figuras, com a mesma cor para cada modelo em todos os gráficos. |
| `relatorio.py` | Tabelas comparativas (CSV e Markdown), figuras em PNG e o painel HTML de arquivo único. |
| `pipeline.py` | Percorre os 6 datasets e chama as etapas acima. |
| `cli.py` | Comandos `baixar` e `executar`. |

## Fluxo de um dataset

```mermaid
sequenceDiagram
    participant P as pipeline
    participant D as dados
    participant A as avaliacao
    participant M as modelos
    participant G as diagnostico
    P->>D: carregar_dataset(nome)
    D-->>P: X, y
    P->>A: avaliar_dataset(X, y, criar_modelos())
    A->>A: dividir (80/20 estratificado) e criar_cv (5 folds)
    loop para cada um dos 4 classificadores
        A->>M: pipeline + espaço de busca
        A->>A: RandomizedSearchCV (CV automática, 5 métricas, treino e validação)
        A->>A: métricas no teste, matriz de confusão, pontuações para a ROC
    end
    A-->>P: ResultadoDataset
    P->>G: curvas_aprendizado (RF e GB)
    P->>G: diagnosticar (viés x variância, escolha RF x GB)
```

## Decisões de projeto

- **Pré-processamento dentro do `Pipeline`.** Imputação, escala e one-hot são ajustados só nos folds de treino de cada validação cruzada, o que evita vazamento de dados.
- **Uma divisão treino/teste por dataset.** O conjunto de teste é separado uma vez e só é usado no fim; a escolha de hiperparâmetros e o diagnóstico usam apenas a validação cruzada.
- **Métricas macro.** Precisão, recall e F1 são médias macro (todas as classes pesam igual), porque Fishmorph e Pokémon são desbalanceados.
- **ROC sem `probability=True` no SVM.** A curva ROC só depende da ordem das pontuações, então o SVM usa `decision_function`. Isso evita a validação cruzada interna que o `probability=True` faria e deixa o SVM várias vezes mais rápido.
- **`HistGradientBoostingClassifier`.** É gradient boosting de árvores, como o `GradientBoostingClassifier`, mas usa histogramas. No Fishmorph a versão clássica leva cerca de 1 minuto por ajuste, contra 3 segundos. A clássica continua disponível com `--gb-classico`.
- **Notebook gerado, não copiado.** `scripts/gerar_notebook.py` transforma cada módulo em uma célula. O notebook do Colab e o pacote não divergem.
