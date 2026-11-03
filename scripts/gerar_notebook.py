"""Gera notebooks/classificadores_6_datasets.ipynb a partir do código em src/classificadores.

O notebook é autossuficiente (roda no Google Colab sem instalar este pacote): cada módulo vira
uma célula, na mesma ordem das dependências. Assim o código do notebook e o do repositório
são sempre o mesmo. Rode depois de qualquer mudança em src/:

    python scripts/gerar_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PACOTE = RAIZ / "src" / "classificadores"
SAIDA = RAIZ / "notebooks" / "classificadores_6_datasets.ipynb"

MODULOS = [
    ("config", "## 1. Configuração: constantes e os 6 datasets"),
    ("dados", "## 2. Download (com cache) e preparação dos dados"),
    ("modelos", "## 3. Pré-processamento e os 4 classificadores, num único dicionário"),
    ("avaliacao", "## 4. Divisão treino/teste, validação cruzada automática e métricas"),
    ("diagnostico", "## 5. Diagnóstico de viés e variância e a escolha Random Forest x Gradient Boosting"),
    ("graficos", "## 6. Gráficos"),
    ("relatorio", "## 7. Tabelas comparativas, arquivos de saída e painel HTML"),
    ("pipeline", "## 8. Orquestração do experimento"),
]


def _limpar(fonte: str) -> str:
    # Imports relativos viram nada (tudo está no mesmo namespace do notebook)
    fonte = re.sub(r"^from \.\S* import \([^)]*\)\n", "", fonte, flags=re.M)
    fonte = re.sub(r"^from \.\S* import .*\n", "", fonte, flags=re.M)
    fonte = re.sub(r"^from __future__ import annotations\n", "", fonte, flags=re.M)
    fonte = re.sub(r"\bgraficos\.", "", fonte)
    return fonte.strip() + "\n"


def _md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip().splitlines(keepends=True)}


def _code(texto: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": texto.strip().splitlines(keepends=True)}


def gerar() -> Path:
    cells = [_md("""
# SVM, Decision Tree, Random Forest e Gradient Boosting em 6 datasets

Notebook gerado a partir do repositório (`scripts/gerar_notebook.py`): o código abaixo é exatamente
o do pacote `src/classificadores`.

**Como usar no Colab:** *Ambiente de execução → Executar tudo*. Os CSVs são baixados da API pública
do Kaggle (sem login) e do GitHub. A execução completa leva cerca de 10 a 15 minutos.

O que o notebook faz, para cada um dos 6 datasets:

1. divide em treino (80%) e teste (20%), estratificado;
2. percorre os 4 classificadores, guardados num dicionário, com um único laço `for`;
3. otimiza os hiperparâmetros de cada um com `RandomizedSearchCV` e validação cruzada estratificada de 5 folds;
4. mede acurácia, precisão, recall, F1, ROC AUC e matriz de confusão no teste;
5. compara treino x validação e as curvas de aprendizado para diagnosticar **alto viés** ou **alta variância**,
   e justifica a escolha entre **Random Forest** e **Gradient Boosting**.
"""),
             _code("""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
from IPython.display import Markdown, display
""")]

    for nome, titulo in MODULOS:
        fonte = (PACOTE / f"{nome}.py").read_text(encoding="utf-8")
        doc = re.match(r'"""(.*?)"""', fonte, flags=re.S)
        cells.append(_md(titulo + ("\n\n" + doc.group(1).strip() if doc else "")))
        cells.append(_code(_limpar(fonte)))

    cells += [
        _md("## 9. Execução\nTreina e avalia os 4 modelos nos 6 datasets. Para um teste rápido, use `executar(n_iter=2, com_curvas=False)`."),
        _code("resultados = executar()"),
        _md("## 10. Tabelas comparativas"),
        _code("""
met, diag = tabela_metricas(resultados), tabela_diagnostico(resultados)
for rotulo in ["Acurácia", "Precisão", "Recall", "F1", "ROC AUC"]:
    display(Markdown(f"### {rotulo} no teste"))
    display(pivot(met, rotulo).style.format("{:.3f}").highlight_max(axis=1, color="#cde2fb"))
display(Markdown("### F1 na validação cruzada"))
display(pivot(met, "F1 CV (validação)").style.format("{:.3f}").highlight_max(axis=1, color="#cde2fb"))
"""),
        _md("## 11. Resultados por dataset\nMétricas, curva ROC, matrizes de confusão, curvas de aprendizado e o diagnóstico de viés e variância."),
        _code("""
for nome_ds, res in resultados.items():
    display(Markdown(f"---\\n## {nome_ds}"))
    display(met[met["Dataset"] == nome_ds].drop(columns=["Dataset"]).set_index("Modelo")
            .style.format("{:.3f}").highlight_max(axis=0, color="#cde2fb"))
    for figura in [grafico_metricas, grafico_roc, grafico_matrizes_confusao, grafico_aprendizado]:
        if figura is grafico_aprendizado and not res.curvas:
            continue
        figura(res)
        plt.show()
    display(Markdown(f"**Diagnóstico.** {res.diagnostico['justificativa']}"))
"""),
        _md("## 12. Viés e variância em todos os datasets"),
        _code("""
grafico_gaps(diag)
plt.show()
display(diag.style.format({"F1 treino (CV)": "{:.3f}", "F1 validação (CV)": "{:.3f}", "Gap": "{:.3f}"}))
for _, r in tabela_escolhas(resultados).iterrows():
    display(Markdown(f"**{r['Dataset']}**: {r['Escolha RF x GB']}. {r['Justificativa']}"))
"""),
        _md("## 13. Salvar resultados e baixar\nGera `resultados/` (CSVs, figuras, `tabelas.md` e `painel.html`) e um `.zip` para baixar."),
        _code("""
import shutil
salvar(resultados, "resultados")
shutil.make_archive("resultados", "zip", "resultados")
try:
    from google.colab import files
    files.download("resultados.zip")
except ImportError:
    print("Arquivos salvos em resultados/ e resultados.zip")
"""),
    ]

    nb = {"cells": cells,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return SAIDA


if __name__ == "__main__":
    print(f"Notebook gerado: {gerar()}")
