import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from classificadores.avaliacao import (
    METRICAS,
    avaliar_dataset,
    criar_cv,
    curva_roc,
    pontuacoes,
    roc_auc,
)
from classificadores.modelos import criar_modelos


def test_cv_respeita_a_menor_classe():
    cv = criar_cv(pd.Series(["a"] * 20 + ["b"] * 3))
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 3


def test_roc_auc_binaria_e_multiclasse():
    assert roc_auc(["a", "b", "a", "b"], np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]),
                   ["a", "b"]) == pytest.approx(1.0)
    y = ["x", "y", "z", "x", "y", "z"]
    s = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    assert roc_auc(y, s, ["x", "y", "z"]) == pytest.approx(1.0)


def test_pontuacoes_do_svm_usa_decision_function(binario):
    X, y = binario
    m = criar_modelos()["SVM"]["pipeline"].fit(X, y)
    s = pontuacoes(m, X)
    assert s.shape == (len(X), 2)
    fpr, tpr, a = curva_roc(y, s, m.classes_)
    assert 0.5 < a <= 1.0


@pytest.mark.parametrize("fixture", ["binario", "multiclasse"])
def test_avaliar_dataset_roda_os_4_modelos_com_todas_as_metricas(fixture, request):
    X, y = request.getfixturevalue(fixture)
    res = avaliar_dataset("teste", X, y, criar_modelos(), n_iter=1, log=lambda *_: None)
    assert list(res.modelos) == ["Decision Tree", "Random Forest", "Gradient Boosting", "SVM"]
    for rm in res.modelos.values():
        assert set(METRICAS) <= set(rm.teste)
        assert all(0 <= rm.teste[m] <= 1 for m in METRICAS)
        assert rm.matriz_confusao.sum() == len(res.y_test)
        assert set(METRICAS) <= set(rm.cv_treino) and set(METRICAS) <= set(rm.cv_validacao)
