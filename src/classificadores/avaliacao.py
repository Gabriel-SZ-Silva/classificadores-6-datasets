"""Divisão treino/teste, validação cruzada automática, busca de hiperparâmetros e métricas."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize

from .config import METRICA_BUSCA, N_FOLDS, N_ITER, RANDOM_STATE, TAMANHO_TESTE

METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
NOMES_METRICAS = {
    "accuracy": "Acurácia",
    "precision": "Precisão",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC AUC",
}


# --------------------------------------------------------------------------------------
# Divisão e validação cruzada
# --------------------------------------------------------------------------------------
def dividir(X: pd.DataFrame, y: pd.Series):
    """Treino/teste 80/20, estratificado quando todas as classes têm ao menos 2 exemplos."""
    estratificar = y if y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=TAMANHO_TESTE, stratify=estratificar,
                            random_state=RANDOM_STATE)


def criar_cv(y_train: pd.Series):
    """StratifiedKFold com o maior número de folds (até N_FOLDS) que a menor classe permite."""
    menor = int(y_train.value_counts().min())
    if menor >= 2:
        return StratifiedKFold(n_splits=min(N_FOLDS, menor), shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------------------
# Pontuações contínuas e ROC AUC (funciona com predict_proba ou decision_function)
# --------------------------------------------------------------------------------------
def pontuacoes(estimador, X) -> np.ndarray:
    """Pontuação contínua por classe, na ordem de ``estimador.classes_``.

    Usa ``predict_proba`` quando existe; no SVC (sem probability=True) usa ``decision_function``,
    o que basta para a ROC, que só depende da ordenação das pontuações.
    """
    if hasattr(estimador, "predict_proba"):
        try:
            return estimador.predict_proba(X)
        except AttributeError:
            pass
    s = estimador.decision_function(X)
    if s.ndim == 1:  # binário: decision_function devolve só a pontuação da classe positiva
        s = np.column_stack([-s, s])
    return s


def roc_auc(y_true, scores: np.ndarray, classes) -> float:
    """ROC AUC binária ou, no multiclasse, média macro um-contra-todos (OvR)."""
    classes = list(classes)
    if len(classes) == 2:
        return roc_auc_score(np.asarray(y_true) == classes[1], scores[:, 1])
    Y = label_binarize(y_true, classes=classes)
    presentes = Y.sum(axis=0) > 0  # classe ausente no fold não tem ROC definida
    return roc_auc_score(Y[:, presentes], scores[:, presentes], average="macro")


def _auc_scorer(estimador, X, y):
    return roc_auc(y, pontuacoes(estimador, X), estimador.classes_)


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "roc_auc": _auc_scorer,
}


# --------------------------------------------------------------------------------------
# Resultado por modelo
# --------------------------------------------------------------------------------------
@dataclass
class ResultadoModelo:
    nome: str
    busca: RandomizedSearchCV
    tempo_s: float
    cv_treino: dict[str, float]  # média nos folds de treino da CV (melhores hiperparâmetros)
    cv_validacao: dict[str, float]  # média nos folds de validação da CV
    cv_validacao_std: dict[str, float]
    teste: dict[str, float]
    matriz_confusao: np.ndarray
    y_pred: np.ndarray
    scores_teste: np.ndarray

    @property
    def estimador(self):
        return self.busca.best_estimator_


@dataclass
class ResultadoDataset:
    nome: str
    classes: list
    n_linhas: int
    n_atributos: int
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    modelos: dict[str, ResultadoModelo] = field(default_factory=dict)
    curvas: dict = field(default_factory=dict)  # preenchido por diagnostico.curvas_aprendizado
    diagnostico: dict = field(default_factory=dict)


def metricas_teste(estimador, X_test, y_test) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    y_pred = estimador.predict(X_test)
    scores = pontuacoes(estimador, X_test)
    m = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc(y_test, scores, estimador.classes_),
    }
    cm = confusion_matrix(y_test, y_pred, labels=estimador.classes_)
    return m, cm, y_pred, scores


def curva_roc(y_test, scores: np.ndarray, classes) -> tuple[np.ndarray, np.ndarray, float]:
    """FPR, TPR e AUC. No multiclasse, curva micro-média um-contra-todos."""
    classes = list(classes)
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(np.asarray(y_test) == classes[1], scores[:, 1])
    else:
        Y = label_binarize(y_test, classes=classes)
        fpr, tpr, _ = roc_curve(Y.ravel(), scores.ravel())
    return fpr, tpr, auc(fpr, tpr)


def avaliar_modelo(nome: str, item: dict, X_train, y_train, X_test, y_test, cv,
                   n_iter: int = N_ITER) -> ResultadoModelo:
    """Busca de hiperparâmetros com CV automática e avaliação no conjunto de teste."""
    inicio = time.time()
    busca = RandomizedSearchCV(
        estimator=item["pipeline"],
        param_distributions=item["espaco"],
        n_iter=n_iter,
        cv=cv,
        scoring=SCORING,
        refit=METRICA_BUSCA,
        return_train_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        error_score="raise",
    )
    busca.fit(X_train, y_train)
    i = busca.best_index_
    r = busca.cv_results_
    teste, cm, y_pred, scores = metricas_teste(busca.best_estimator_, X_test, y_test)
    return ResultadoModelo(
        nome=nome,
        busca=busca,
        tempo_s=time.time() - inicio,
        cv_treino={m: float(r[f"mean_train_{m}"][i]) for m in METRICAS},
        cv_validacao={m: float(r[f"mean_test_{m}"][i]) for m in METRICAS},
        cv_validacao_std={m: float(r[f"std_test_{m}"][i]) for m in METRICAS},
        teste=teste,
        matriz_confusao=cm,
        y_pred=y_pred,
        scores_teste=scores,
    )


def avaliar_dataset(nome: str, X: pd.DataFrame, y: pd.Series, modelos: dict, n_iter: int = N_ITER,
                    log=print) -> ResultadoDataset:
    X_train, X_test, y_train, y_test = dividir(X, y)
    cv = criar_cv(y_train)
    res = ResultadoDataset(nome=nome, classes=sorted(y.unique().tolist(), key=str),
                           n_linhas=len(X), n_atributos=X.shape[1],
                           X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    for nome_modelo, item in modelos.items():  # o mesmo código para os 4 classificadores
        rm = avaliar_modelo(nome_modelo, item, X_train, y_train, X_test, y_test, cv, n_iter)
        res.modelos[nome_modelo] = rm
        log(f"   ✓ {nome_modelo:<18} F1 CV = {rm.cv_validacao['f1']:.4f} | "
            f"F1 teste = {rm.teste['f1']:.4f} | AUC teste = {rm.teste['roc_auc']:.4f} ({rm.tempo_s:.0f}s)")
    return res
