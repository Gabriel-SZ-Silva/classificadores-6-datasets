import pandas as pd
import pytest

from classificadores.config import DATASETS, prep_cardiopatia, prep_eleicoes
from classificadores.dados import classes_removidas, obter_csv, preparar


def test_preparar_remove_colunas_e_linhas_sem_alvo():
    df = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "alvo": ["a", None, "b"]})
    X, y = preparar(df, {"target": "alvo", "drop": ["id", "inexistente"]})
    assert list(X.columns) == ["x"]
    assert y.tolist() == ["a", "b"]


def test_preparar_usar_seleciona_so_as_colunas_pedidas():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "nome": ["p", "q"], "alvo": [0, 1]})
    X, _ = preparar(df, {"target": "alvo", "usar": ["x", "y"]})
    assert list(X.columns) == ["x", "y"]


def test_alvo_inexistente_falha_com_mensagem_clara():
    with pytest.raises(KeyError, match="coluna-alvo 'Legendary'"):
        preparar(pd.DataFrame({"a": [1]}), {"target": "Legendary", "drop": []})


def test_min_por_classe_remove_classes_raras():
    df = pd.DataFrame({"x": range(13), "alvo": ["a"] * 10 + ["b"] * 2 + ["c"]})
    cfg = {"target": "alvo", "drop": [], "min_por_classe": 10}
    X, y = preparar(df, cfg)
    assert set(y) == {"a"}
    assert set(classes_removidas(df, cfg).index) == {"b", "c"}


def test_prep_eleicoes_deriva_alvo_e_descarta_sem_resultado():
    df = pd.DataFrame({"pct_real_rep": [55.0, 40.0, None], "pct_real_dem": [40.0, 55.0, None]})
    out = prep_eleicoes(df)
    assert out["vencedor_real"].tolist() == [1, 0]


def test_prep_cardiopatia_trata_zero_como_ausente():
    df = pd.DataFrame({"Cholesterol": [0, 200], "RestingBP": [120, 0]})
    out = prep_cardiopatia(df)
    assert out["Cholesterol"].isna().tolist() == [True, False]
    assert out["RestingBP"].isna().tolist() == [False, True]


def test_obter_csv_usa_o_cache_local(tmp_path):
    cfg = {"arquivo": "x.csv", "fonte": "kaggle", "slug": "a/b"}
    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    assert obter_csv(cfg, tmp_path).shape == (1, 2)


def test_eleicoes_nao_usa_colunas_que_entregam_o_alvo():
    drop = DATASETS["Eleições (Literary Digest)"]["drop"]
    for col in ["pct_real_rep", "pct_real_dem", "ano", "candidato_rep", "candidato_dem"]:
        assert col in drop


def test_todos_os_datasets_tem_campos_obrigatorios():
    assert len(DATASETS) == 6
    for cfg in DATASETS.values():
        assert {"slug_curto", "fonte", "slug", "arquivo", "target"} <= set(cfg)
