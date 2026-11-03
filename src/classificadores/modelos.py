"""Pré-processamento e os 4 classificadores, instanciados num único dicionário.

O resto do código percorre ``criar_modelos()`` com um laço ``for``: nenhum modelo tem
código de treino ou avaliação próprio.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE, USAR_HIST_GB


def criar_preprocessador() -> ColumnTransformer:
    """Imputação + escala para numéricos; imputação + one-hot para categóricos.

    Fica dentro do Pipeline, então é ajustado só nos folds de treino (sem vazamento de dados).
    "Numérico / não numérico" cobre qualquer dtype (int8, bool, str do pandas 3...).
    """
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputacao", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), make_column_selector(dtype_include=np.number)),
            ("cat", Pipeline([
                ("imputacao", SimpleImputer(strategy="most_frequent")),
                # denso: o HistGradientBoosting não aceita matriz esparsa
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), make_column_selector(dtype_exclude=np.number)),
        ]
    )


def _pipeline(modelo) -> Pipeline:
    return Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)])


IMPUTACAO = {"prep__num__imputacao__strategy": ["mean", "median"]}


def criar_modelos(usar_hist_gb: bool = USAR_HIST_GB) -> dict[str, dict]:
    """Dicionário nome -> {"pipeline": Pipeline, "espaco": espaço de busca de hiperparâmetros}."""
    if usar_hist_gb:
        gb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        espaco_gb = {
            **IMPUTACAO,
            "modelo__max_iter": randint(50, 300),
            "modelo__learning_rate": uniform(0.01, 0.2),
            "modelo__max_depth": [3, 5, 8, None],
            "modelo__l2_regularization": [0.0, 0.1, 1.0],
        }
    else:
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        espaco_gb = {
            **IMPUTACAO,
            "modelo__n_estimators": randint(50, 150),
            "modelo__learning_rate": uniform(0.01, 0.2),
            "modelo__max_depth": randint(3, 8),
        }

    return {
        "Decision Tree": {
            "pipeline": _pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            "espaco": {
                **IMPUTACAO,
                "modelo__max_depth": [3, 5, 10, 20, None],
                "modelo__min_samples_split": [2, 5, 10],
                "modelo__min_samples_leaf": [1, 2, 5],
                "modelo__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "pipeline": _pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
            "espaco": {
                **IMPUTACAO,
                "modelo__n_estimators": randint(100, 300),
                "modelo__max_depth": [5, 10, 20, None],
                "modelo__max_features": ["sqrt", "log2"],
                "modelo__min_samples_leaf": [1, 2, 5],
                "modelo__class_weight": [None, "balanced"],
            },
        },
        "Gradient Boosting": {"pipeline": _pipeline(gb), "espaco": espaco_gb},
        "SVM": {
            "pipeline": _pipeline(SVC(cache_size=1000)),
            # gamma só existe no RBF; no kernel linear, C alto pode demorar muito para convergir
            # em dados não separáveis, por isso C <= 1 nele.
            "espaco": [
                {
                    **IMPUTACAO,
                    "modelo__kernel": ["rbf"],
                    "modelo__C": loguniform(0.1, 100),
                    "modelo__gamma": ["scale", 0.01, 0.1, 1],
                    "modelo__class_weight": [None, "balanced"],
                },
                {
                    **IMPUTACAO,
                    "modelo__kernel": ["linear"],
                    "modelo__C": loguniform(0.01, 1),
                    "modelo__class_weight": [None, "balanced"],
                },
            ],
        },
    }
