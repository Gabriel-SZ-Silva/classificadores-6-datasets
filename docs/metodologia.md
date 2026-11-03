# Metodologia

## 1. Preparação dos dados

| Dataset | Alvo | Tratamento |
|---|---|---|
| Câncer de Mama | `diagnosis` (M/B) | remove `id` e a coluna vazia `Unnamed: 32` |
| Eleições (Literary Digest) | `vencedor_real` (republicano venceu?) | alvo derivado de `pct_real_rep > pct_real_dem`; removidos o resultado real (vazamento) e também ano e candidatos, que sozinhos indicam o vencedor. Linhas sem resultado real (quase todo 1932) saem. |
| Fishmorph (Ordem) | `Order` | removidas as ordens com menos de 10 espécies (9 ordens, 38 linhas), que inviabilizam divisão e validação estratificadas |
| Cardiopatia | `HeartDisease` | `Cholesterol` e `RestingBP` iguais a 0 viram valor ausente ("não medido") |
| Estresse Estudantil | `stress_level` (0, 1, 2) | nenhum |
| Ultimate Pokémon | `type_1` (18 tipos) | só os 21 atributos numéricos de batalha e biologia; nome, descrição, habilidades e `type_2` ficam de fora porque praticamente entregam o tipo |

Dentro do pipeline: numéricos recebem imputação (média ou mediana, escolhida na busca) e padronização; categóricos recebem imputação pela moda e one-hot.

## 2. Divisão e validação cruzada

1. **Treino e teste:** 80% / 20%, estratificado pela classe, com semente fixa (42).
2. **Validação cruzada automática:** `StratifiedKFold` com 5 folds (ou menos, se a menor classe do treino não tiver 5 exemplos), embaralhado.
3. **Busca de hiperparâmetros:** `RandomizedSearchCV` com 10 combinações por modelo, escolhidas pelo F1 macro médio nos folds de validação. A mesma busca registra as 5 métricas em treino e validação de cada fold.
4. **Teste:** o melhor modelo é reajustado no treino inteiro e avaliado uma única vez no teste.

## 3. Métricas

| Métrica | Como é calculada |
|---|---|
| Acurácia | proporção de acertos |
| Precisão | média macro da precisão de cada classe |
| Recall | média macro do recall de cada classe |
| F1 | média macro do F1 de cada classe |
| ROC AUC | binário: AUC da classe positiva; multiclasse: média macro um-contra-todos. As figuras mostram a curva micro-média. |
| Matriz de confusão | contagens no teste; cor normalizada por linha (proporção de cada classe real) |

## 4. Diagnóstico de viés e variância

Para cada modelo, com os melhores hiperparâmetros, compara-se o F1 macro médio nos folds de **treino** com o dos folds de **validação**:

| Diagnóstico | Regra |
|---|---|
| Alta variância (overfitting) | F1 treino − F1 validação > 0,10 |
| Alto viés (underfitting) | F1 treino < 0,80 (o modelo não ajusta nem os dados de treino) |
| Viés e variância altos | as duas condições |
| Equilibrado | nenhuma das duas |

Os limiares estão em `config.py` (`LIMIAR_GAP_VARIANCIA`, `LIMIAR_TREINO_VIES`).

As **curvas de aprendizado** (F1 de treino e de validação em 5 tamanhos de treino, de 30% a 100%) completam o diagnóstico do Random Forest e do Gradient Boosting. Se a curva de validação ainda sobe no fim, mais dados reduziriam a variância. Se já está estável, o ganho viria de novos atributos.

## 5. Random Forest x Gradient Boosting

Os dois combinam árvores de decisão, mas atacam problemas opostos:

- **Random Forest (bagging)** treina árvores profundas em amostras bootstrap e tira a média. A média reduz a **variância**, e o viés continua baixo, como o de uma árvore profunda.
- **Gradient Boosting (boosting)** soma árvores rasas em sequência, cada uma corrigindo o erro das anteriores. Isso reduz o **viés**, e a variância é controlada pela taxa de aprendizado, pela profundidade e pela regularização.

A escolha segue três passos:

1. **Problema do dataset.** É lido nos dois candidatos. Se os dois superajustam (gap > 0,10), o dataset tem alta variância, e a teoria aponta o Random Forest. Se os dois não ajustam nem o treino (F1 de treino < 0,80), tem alto viés, e a teoria aponta o Gradient Boosting. Se só um superajusta, o diagnóstico é misto. Se nenhum, é equilibrado.
2. **Linha de base.** A árvore de decisão isolada, aprendiz-base dos dois, mostra quanto cada ensemble melhorou a validação e reduziu o gap.
3. **Decisão.** Vence quem teve o maior F1 na validação cruzada. Se a diferença for menor que o desvio-padrão entre os folds, é empate técnico, e vence o de **menor gap**, isto é, o que generaliza de forma mais estável. O texto diz se o resultado confirma ou contraria a teoria.

A curva de aprendizado do escolhido diz ainda se mais dados ajudariam. O teste não entra nessa decisão, para que a escolha não seja feita olhando a resposta.
