import pytest

from classificadores.avaliacao import avaliar_dataset
from classificadores.diagnostico import classificar, curvas_aprendizado, diagnosticar
from classificadores.modelos import criar_modelos
from classificadores.relatorio import markdown, pivot, salvar, tabela_metricas


@pytest.mark.parametrize("treino,validacao,esperado", [
    (1.00, 0.80, "variancia"),
    (0.70, 0.68, "vies"),
    (0.70, 0.50, "ambos"),
    (0.95, 0.92, "equilibrado"),
])
def test_classificar(treino, validacao, esperado):
    assert classificar(treino, validacao) == esperado


def test_diagnostico_e_relatorio_completos(binario, tmp_path):
    X, y = binario
    res = avaliar_dataset("Câncer de Mama", X, y, criar_modelos(), n_iter=1, log=lambda *_: None)
    curvas_aprendizado(res)
    d = diagnosticar(res)
    assert d["escolha_rf_gb"] in {"Random Forest", "Gradient Boosting"}
    assert "Random Forest" in d["justificativa"] and "Gradient Boosting" in d["justificativa"]

    salvar({"Câncer de Mama": res}, tmp_path)
    for arq in ["metricas.csv", "diagnostico.csv", "tabelas.md", "painel.html", "resumo.json",
                "figuras/geral/f1.png", "figuras/cancer_mama/roc.png", "figuras/cancer_mama/confusao.png",
                "figuras/cancer_mama/aprendizado.png"]:
        assert (tmp_path / arq).exists(), arq
    html = (tmp_path / "painel.html").read_text(encoding="utf-8")
    assert 'id="cancer_mama"' in html and "data:image/png;base64" in html


def test_markdown_destaca_o_maior_valor(binario):
    X, y = binario
    res = avaliar_dataset("d", X, y, criar_modelos(), n_iter=1, log=lambda *_: None)
    md = markdown(pivot(tabela_metricas({"d": res}), "F1"), destacar_max_linha=True)
    assert md.count("**") >= 2 and md.count("**") % 2 == 0
