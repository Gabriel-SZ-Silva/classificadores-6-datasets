# Resultados

Gerado em 02/11/2026 22:13.

## Acurácia no teste

| Dataset | Decision Tree | Random Forest | Gradient Boosting | SVM |
|---|:---:|:---:|:---:|:---:|
| Câncer de Mama | 0.886 | 0.965 | 0.956 | **0.982** |
| Eleições (Literary Digest) | 0.690 | 0.759 | 0.724 | **0.793** |
| Fishmorph (Ordem) | 0.627 | 0.758 | 0.799 | **0.809** |
| Cardiopatia | 0.821 | **0.886** | 0.870 | 0.864 |
| Estresse Estudantil | 0.873 | **0.886** | 0.864 | **0.886** |
| Ultimate Pokémon | 0.229 | 0.288 | **0.307** | 0.224 |

## Precisão no teste

| Dataset | Decision Tree | Random Forest | Gradient Boosting | SVM |
|---|:---:|:---:|:---:|:---:|
| Câncer de Mama | 0.879 | 0.967 | 0.961 | **0.986** |
| Eleições (Literary Digest) | 0.715 | 0.757 | 0.721 | **0.791** |
| Fishmorph (Ordem) | 0.534 | 0.677 | 0.766 | **0.789** |
| Cardiopatia | 0.820 | **0.885** | 0.868 | 0.863 |
| Estresse Estudantil | 0.885 | 0.889 | 0.864 | **0.916** |
| Ultimate Pokémon | 0.198 | 0.234 | **0.268** | 0.174 |

## Recall no teste

| Dataset | Decision Tree | Random Forest | Gradient Boosting | SVM |
|---|:---:|:---:|:---:|:---:|
| Câncer de Mama | 0.875 | 0.957 | 0.945 | **0.976** |
| Eleições (Literary Digest) | 0.704 | 0.760 | 0.721 | **0.791** |
| Fishmorph (Ordem) | 0.585 | **0.771** | 0.631 | 0.694 |
| Cardiopatia | 0.824 | **0.884** | 0.868 | 0.862 |
| Estresse Estudantil | 0.873 | **0.887** | 0.864 | 0.886 |
| Ultimate Pokémon | 0.195 | 0.229 | **0.252** | 0.192 |

## F1 no teste

| Dataset | Decision Tree | Random Forest | Gradient Boosting | SVM |
|---|:---:|:---:|:---:|:---:|
| Câncer de Mama | 0.877 | 0.962 | 0.952 | **0.981** |
| Eleições (Literary Digest) | 0.688 | 0.757 | 0.721 | **0.791** |
| Fishmorph (Ordem) | 0.548 | 0.703 | 0.680 | **0.727** |
| Cardiopatia | 0.820 | **0.884** | 0.868 | 0.862 |
| Estresse Estudantil | 0.875 | 0.887 | 0.863 | **0.889** |
| Ultimate Pokémon | 0.191 | 0.219 | **0.249** | 0.179 |

## ROC AUC no teste

| Dataset | Decision Tree | Random Forest | Gradient Boosting | SVM |
|---|:---:|:---:|:---:|:---:|
| Câncer de Mama | 0.896 | 0.996 | 0.993 | **0.997** |
| Eleições (Literary Digest) | 0.815 | 0.822 | **0.861** | 0.827 |
| Fishmorph (Ordem) | 0.788 | 0.978 | **0.982** | 0.969 |
| Cardiopatia | 0.889 | 0.927 | 0.913 | **0.931** |
| Estresse Estudantil | 0.964 | **0.985** | 0.982 | 0.984 |
| Ultimate Pokémon | 0.616 | **0.781** | 0.774 | 0.750 |

## F1 na validação cruzada (média dos folds)

| Dataset | Decision Tree | Random Forest | Gradient Boosting | SVM |
|---|:---:|:---:|:---:|:---:|
| Câncer de Mama | 0.937 | 0.960 | 0.967 | **0.974** |
| Eleições (Literary Digest) | **0.842** | 0.795 | 0.806 | 0.805 |
| Fishmorph (Ordem) | 0.495 | **0.641** | 0.621 | 0.638 |
| Cardiopatia | 0.828 | 0.856 | **0.867** | 0.851 |
| Estresse Estudantil | 0.885 | 0.886 | **0.887** | 0.883 |
| Ultimate Pokémon | 0.201 | **0.252** | 0.246 | 0.238 |

## Diagnóstico de viés e variância

| Dataset | Decision Tree | Random Forest | Gradient Boosting | SVM |
|---|:---:|:---:|:---:|:---:|
| Câncer de Mama | Equilibrado (gap 0.047) | Equilibrado (gap 0.036) | Equilibrado (gap 0.033) | Equilibrado (gap 0.008) |
| Eleições (Literary Digest) | Equilibrado (gap 0.069) | Alta variância (overfitting) (gap 0.205) | Alta variância (overfitting) (gap 0.153) | Equilibrado (gap 0.084) |
| Fishmorph (Ordem) | Alta variância (overfitting) (gap 0.472) | Alta variância (overfitting) (gap 0.290) | Alta variância (overfitting) (gap 0.379) | Alta variância (overfitting) (gap 0.294) |
| Cardiopatia | Equilibrado (gap 0.026) | Equilibrado (gap 0.046) | Alta variância (overfitting) (gap 0.133) | Equilibrado (gap 0.004) |
| Estresse Estudantil | Equilibrado (gap 0.012) | Equilibrado (gap 0.099) | Alta variância (overfitting) (gap 0.113) | Alta variância (overfitting) (gap 0.117) |
| Ultimate Pokémon | Viés e variância altos (gap 0.429) | Alta variância (overfitting) (gap 0.727) | Alta variância (overfitting) (gap 0.754) | Alta variância (overfitting) (gap 0.653) |

## Random Forest x Gradient Boosting

**Câncer de Mama**: Gradient Boosting. O dataset está **equilibrado**: os dois ensembles generalizam bem (gap de 0.036 no Random Forest e 0.033 no Gradient Boosting), sem sinal de viés nem de variância altos. Como linha de base, a árvore de decisão isolada teve F1 de validação 0.937 (gap 0.047); o Random Forest chegou a 0.960 e o Gradient Boosting a 0.967. A diferença de F1 entre os dois (0.007) é menor que o desvio-padrão entre os folds (0.016), um empate técnico. O critério de desempate é a estabilidade: o **Gradient Boosting** tem o menor gap (0.033 contra 0.036). A curva de aprendizado do Gradient Boosting já está estável: mais dados do mesmo tipo ajudariam pouco.

**Eleições (Literary Digest)**: Gradient Boosting. O dataset tem **alta variância**: os dois ensembles ajustam o treino muito melhor que a validação (gap de 0.205 no Random Forest e 0.153 no Gradient Boosting). Na teoria, é o caso do Random Forest, cuja média de árvores reduz variância. Como linha de base, a árvore de decisão isolada teve F1 de validação 0.842 (gap 0.069); o Random Forest chegou a 0.795 e o Gradient Boosting a 0.806. A diferença de F1 entre os dois (0.011) é menor que o desvio-padrão entre os folds (0.072), um empate técnico. O critério de desempate é a estabilidade: o **Gradient Boosting** tem o menor gap (0.153 contra 0.205). A evidência contraria a teoria: com os hiperparâmetros encontrados (taxa de aprendizado, profundidade, regularização), o Gradient Boosting controlou o problema melhor que o Random Forest neste dataset. A curva de aprendizado do Gradient Boosting já está estável: mais dados do mesmo tipo ajudariam pouco.

**Fishmorph (Ordem)**: Random Forest. O dataset tem **alta variância**: os dois ensembles ajustam o treino muito melhor que a validação (gap de 0.290 no Random Forest e 0.379 no Gradient Boosting). Na teoria, é o caso do Random Forest, cuja média de árvores reduz variância. Como linha de base, a árvore de decisão isolada teve F1 de validação 0.495 (gap 0.472); o Random Forest chegou a 0.641 e o Gradient Boosting a 0.621. A diferença de F1 entre os dois (0.019) é menor que o desvio-padrão entre os folds (0.034), um empate técnico. O critério de desempate é a estabilidade: o **Random Forest** tem o menor gap (0.290 contra 0.379). A evidência confirma a teoria. A curva de aprendizado do Random Forest ainda sobe no fim (+0.038 de F1 na segunda metade): mais exemplos tendem a reduzir a variância restante.

**Cardiopatia**: Random Forest. O diagnóstico é **misto**: o Gradient Boosting superajusta (gap 0.133), o outro não (gap 0.046). Como linha de base, a árvore de decisão isolada teve F1 de validação 0.828 (gap 0.026); o Random Forest chegou a 0.856 e o Gradient Boosting a 0.867. A diferença de F1 entre os dois (0.011) é menor que o desvio-padrão entre os folds (0.040), um empate técnico. O critério de desempate é a estabilidade: o **Random Forest** tem o menor gap (0.046 contra 0.133). A curva de aprendizado do Random Forest já está estável: mais dados do mesmo tipo ajudariam pouco.

**Estresse Estudantil**: Random Forest. O diagnóstico é **misto**: o Gradient Boosting superajusta (gap 0.113), o outro não (gap 0.099). Como linha de base, a árvore de decisão isolada teve F1 de validação 0.885 (gap 0.012); o Random Forest chegou a 0.886 e o Gradient Boosting a 0.887. A diferença de F1 entre os dois (0.002) é menor que o desvio-padrão entre os folds (0.026), um empate técnico. O critério de desempate é a estabilidade: o **Random Forest** tem o menor gap (0.099 contra 0.113). A curva de aprendizado do Random Forest já está estável: mais dados do mesmo tipo ajudariam pouco.

**Ultimate Pokémon**: Random Forest. O dataset tem **alta variância**: os dois ensembles ajustam o treino muito melhor que a validação (gap de 0.727 no Random Forest e 0.754 no Gradient Boosting). Na teoria, é o caso do Random Forest, cuja média de árvores reduz variância. Como linha de base, a árvore de decisão isolada teve F1 de validação 0.201 (gap 0.429); o Random Forest chegou a 0.252 e o Gradient Boosting a 0.246. A diferença de F1 entre os dois (0.006) é menor que o desvio-padrão entre os folds (0.033), um empate técnico. O critério de desempate é a estabilidade: o **Random Forest** tem o menor gap (0.727 contra 0.754). A evidência confirma a teoria. O F1 de validação baixo, mesmo com o treino quase perfeito, mostra que estes atributos explicam pouco do alvo: trocar de modelo não muda muito o resultado. A curva de aprendizado do Random Forest ainda sobe no fim (+0.019 de F1 na segunda metade): mais exemplos tendem a reduzir a variância restante.
