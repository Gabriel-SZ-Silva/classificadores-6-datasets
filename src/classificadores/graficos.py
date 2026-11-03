"""Gráficos: matrizes de confusão, curvas ROC, curvas de aprendizado e comparativos.

Cada modelo tem sempre a mesma cor em todos os gráficos (paleta categórica validada para
daltonismo); magnitudes usam uma única rampa azul, do claro ao escuro.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .avaliacao import NOMES_METRICAS, ResultadoDataset, curva_roc
from .config import LIMIAR_GAP_VARIANCIA

CORES = {
    "Decision Tree": "#2a78d6",
    "Random Forest": "#eb6834",
    "Gradient Boosting": "#1baf7a",
    "SVM": "#eda100",
}
TEXTO, TEXTO_2, GRADE, SUPERFICIE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
AZUL = LinearSegmentedColormap.from_list(
    "azul", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_2, "axes.titlecolor": TEXTO,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
        "xtick.color": TEXTO_2, "ytick.color": TEXTO_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "lines.linewidth": 2, "figure.dpi": 110,
    })


aplicar_estilo()


def grafico_matrizes_confusao(res: ResultadoDataset):
    """Uma matriz por modelo, normalizada por linha (proporção de cada classe real)."""
    n = len(res.modelos)
    muitas = len(res.classes) > 12
    lado = 6.2 if muitas else 4.2
    fig, axes = plt.subplots(1, n, figsize=(lado * n, lado + 0.6), squeeze=False)
    for ax, (nome, rm) in zip(axes[0], res.modelos.items()):
        cm = rm.matriz_confusao
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        ax.imshow(norm, cmap=AZUL, vmin=0, vmax=1)
        rotulos = [str(c) for c in rm.estimador.classes_]
        ax.set_xticks(range(len(rotulos)), rotulos, rotation=90 if muitas or len(rotulos) > 4 else 0,
                      fontsize=6 if muitas else 9)
        ax.set_yticks(range(len(rotulos)), rotulos, fontsize=6 if muitas else 9)
        ax.grid(False)
        if not muitas:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                            color="white" if norm[i, j] > 0.55 else TEXTO)
        ax.set_title(f"{nome}\nF1 {rm.teste['f1']:.3f}")
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
    fig.suptitle(f"Matrizes de confusão (teste): {res.nome}", fontsize=14, color=TEXTO, y=1.0)
    fig.tight_layout()
    return fig


def grafico_roc(res: ResultadoDataset):
    multiclasse = len(res.classes) > 2
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#9a9893", label="Aleatório (AUC 0,5)")
    for nome, rm in res.modelos.items():
        fpr, tpr, a = curva_roc(res.y_test, rm.scores_teste, rm.estimador.classes_)
        ax.plot(fpr, tpr, color=CORES.get(nome), label=f"{nome} (AUC {a:.3f})")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    sub = "micro-média um-contra-todos" if multiclasse else "classe positiva"
    ax.set_title(f"Curva ROC (teste, {sub}): {res.nome}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def grafico_aprendizado(res: ResultadoDataset):
    nomes = list(res.curvas)
    fig, axes = plt.subplots(1, len(nomes), figsize=(6 * len(nomes), 4.4), squeeze=False, sharey=True)
    for ax, nome in zip(axes[0], nomes):
        c = res.curvas[nome]
        n = np.array(c["n"])
        cor = CORES.get(nome)
        for chave, estilo, rot in [("treino", "--", "Treino"), ("validacao", "-", "Validação (CV)")]:
            m, s = np.array(c[chave]), np.array(c[f"{chave}_std"])
            ax.plot(n, m, ls=estilo, color=cor, marker="o", ms=5, label=rot)
            ax.fill_between(n, m - s, m + s, color=cor, alpha=0.12, lw=0)
        d = res.diagnostico.get("por_modelo", {}).get(nome, {})
        ax.set_title(f"{nome}: {d.get('rotulo', '')}")
        ax.set_xlabel("Exemplos de treino")
        ax.set_ylabel("F1 macro")
        ax.legend(loc="lower right")
    fig.suptitle(f"Curvas de aprendizado: {res.nome}", fontsize=14, color=TEXTO)
    fig.tight_layout()
    return fig


def grafico_metricas(res: ResultadoDataset):
    """Métricas de teste lado a lado, um grupo por métrica, uma barra por modelo."""
    metricas = list(NOMES_METRICAS)
    nomes = list(res.modelos)
    larg = 0.8 / len(nomes)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(metricas))
    for k, nome in enumerate(nomes):
        v = [res.modelos[nome].teste[m] for m in metricas]
        ax.bar(x + (k - (len(nomes) - 1) / 2) * larg, v, width=larg * 0.9, color=CORES.get(nome),
               label=nome, edgecolor=SUPERFICIE, linewidth=1)
    ax.set_xticks(x, [NOMES_METRICAS[m] for m in metricas])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Valor no teste")
    ax.set_title(f"Métricas no conjunto de teste: {res.nome}")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return fig


def grafico_mapa(tabela: pd.DataFrame, titulo: str):
    """Mapa de calor dataset x modelo, com o valor escrito em cada célula."""
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(tabela) + 1.8))
    vmin = max(0.0, float(np.nanmin(tabela.values)) - 0.05)  # escala do menor valor a 1, para contraste
    ax.imshow(tabela.values, cmap=AZUL, vmin=vmin, vmax=1, aspect="auto")
    ax.set_xticks(range(tabela.shape[1]), tabela.columns)
    ax.set_yticks(range(tabela.shape[0]), tabela.index)
    ax.grid(False)
    for i in range(tabela.shape[0]):
        melhor = np.nanargmax(tabela.values[i])
        for j in range(tabela.shape[1]):
            v = tabela.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if j == melhor else "normal",
                    color="white" if (v - vmin) / (1 - vmin + 1e-9) > 0.55 else TEXTO)
    ax.xaxis.tick_top()
    ax.set_title(titulo, pad=28)
    fig.tight_layout()
    return fig


def grafico_gaps(diag: pd.DataFrame):
    """Gap treino - validação (F1, CV) por modelo e dataset, com o limiar de alta variância."""
    datasets = diag["Dataset"].unique().tolist()
    nomes = list(CORES)
    larg = 0.8 / len(nomes)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(datasets))
    for k, nome in enumerate(nomes):
        sub = diag[diag["Modelo"] == nome].set_index("Dataset").reindex(datasets)
        ax.barh(y + (k - (len(nomes) - 1) / 2) * larg, sub["Gap"], height=larg * 0.9,
                color=CORES[nome], label=nome, edgecolor=SUPERFICIE, linewidth=1)
    ax.axvline(LIMIAR_GAP_VARIANCIA, color=TEXTO_2, ls="--", lw=1)
    ax.text(LIMIAR_GAP_VARIANCIA, len(datasets) - 0.45, " limiar de alta variância", color=TEXTO_2,
            fontsize=8, va="bottom")
    ax.set_yticks(y, datasets)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 de treino - F1 de validação (CV)")
    ax.set_title("Diagnóstico de variância: quanto o modelo perde do treino para a validação")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    return fig
