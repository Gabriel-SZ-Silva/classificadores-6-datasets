"""Atualiza as tabelas de resultados do README.md a partir de resultados/.

Substitui o trecho entre <!-- RESULTADOS:INICIO --> e <!-- RESULTADOS:FIM -->, para que os
números do README sejam sempre os da última execução:

    python -m classificadores executar
    python scripts/atualizar_readme.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
README = RAIZ / "README.md"
RES = RAIZ / "resultados"
INICIO, FIM = "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIM -->"


def _secoes_tabelas() -> dict[str, str]:
    texto = (RES / "tabelas.md").read_text(encoding="utf-8")
    partes = re.split(r"^## (.+)$", texto, flags=re.M)
    return {partes[i].strip(): partes[i + 1].strip() for i in range(1, len(partes), 2)}


def gerar_bloco() -> str:
    s = _secoes_tabelas()
    esc = pd.read_csv(RES / "escolha_rf_gb.csv")
    met = pd.read_csv(RES / "metricas.csv")

    linhas = [INICIO, "", "> Tabelas geradas automaticamente por `scripts/atualizar_readme.py` a partir de "
              "[`resultados/`](resultados/). Em negrito, o melhor modelo de cada dataset. Precisão, recall e F1 "
              "são médias macro.", ""]
    for titulo in ["F1 no teste", "ROC AUC no teste", "Acurácia no teste", "Precisão no teste", "Recall no teste",
                   "F1 na validação cruzada (média dos folds)"]:
        linhas += [f"#### {titulo}", "", s[titulo], ""]

    linhas += ["### Viés e variância", "",
               "Gap = F1 de treino − F1 de validação, nos folds da validação cruzada. Acima de 0.10: alta "
               "variância (overfitting). F1 de treino abaixo de 0.80: alto viés (underfitting).", "",
               s["Diagnóstico de viés e variância"], "",
               '<p align="center"><img src="resultados/figuras/geral/gaps.png" alt="Gap entre F1 de treino e de '
               'validação por modelo e dataset, com a linha do limiar de alta variância em 0.10" width="90%"></p>', ""]

    diag = pd.read_csv(RES / "diagnostico.csv")
    linhas += ["### Random Forest x Gradient Boosting", "",
               "Decide o F1 da validação cruzada; quando a diferença é menor que o desvio-padrão entre os folds "
               "(empate técnico), vence o modelo de menor gap.", "",
               "| Dataset | Problema (RF e GB) | F1 CV RF | F1 CV GB | Desvio | Gap RF | Gap GB | Escolha |",
               "|---|---|:---:|:---:|:---:|:---:|:---:|---|"]
    for _, r in esc.iterrows():
        m = met[met["Dataset"] == r["Dataset"]].set_index("Modelo")
        g = diag[diag["Dataset"] == r["Dataset"]].set_index("Modelo")["Gap"]
        desvio = max(m.loc["Random Forest", "F1 CV (desvio)"], m.loc["Gradient Boosting", "F1 CV (desvio)"])
        linhas.append(f"| {r['Dataset']} | {r['Problema (RF e GB)']} | {m.loc['Random Forest', 'F1 CV (validação)']:.3f} | "
                      f"{m.loc['Gradient Boosting', 'F1 CV (validação)']:.3f} | {desvio:.3f} | "
                      f"{g['Random Forest']:.3f} | {g['Gradient Boosting']:.3f} | **{r['Escolha RF x GB']}** |")
    linhas += ["", "<details>", "<summary>Justificativa completa de cada dataset</summary>", ""]
    for _, r in esc.iterrows():
        linhas += [f"**{r['Dataset']}.** {r['Justificativa']}", ""]
    linhas += ["</details>", "", FIM]
    return "\n".join(linhas)


def atualizar() -> None:
    readme = README.read_text(encoding="utf-8")
    if INICIO not in readme or FIM not in readme:
        raise SystemExit(f"Marcadores {INICIO} / {FIM} não encontrados no README.md")
    antes, resto = readme.split(INICIO, 1)
    _, depois = resto.split(FIM, 1)
    README.write_text(antes + gerar_bloco() + depois, encoding="utf-8")
    print("README.md atualizado.")


if __name__ == "__main__":
    atualizar()
