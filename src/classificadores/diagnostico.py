"""Diagnóstico de viés (underfitting) e variância (overfitting) e a escolha RF x GB.

Ideia central: Random Forest e Gradient Boosting combinam as mesmas árvores de decisão,
mas atacam problemas opostos.

* Random Forest (bagging): treina muitas árvores profundas em amostras bootstrap e tira a média.
  A média reduz a **variância**; o viés fica parecido com o de uma árvore profunda (baixo).
* Gradient Boosting (boosting): soma árvores rasas em sequência, cada uma corrigindo o erro
  residual das anteriores. Isso reduz o **viés**; a variância é controlada por learning_rate,
  profundidade e regularização.

O diagnóstico compara F1 de treino e de validação (CV) dos dois ensembles para saber qual
problema domina o dataset, usa a árvore de decisão isolada como linha de base e deixa a
validação cruzada decidir; no empate técnico, vence o modelo de menor gap (mais estável).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import learning_curve

from .avaliacao import ResultadoDataset, criar_cv
from .config import LIMIAR_GAP_VARIANCIA, LIMIAR_TREINO_VIES, RANDOM_STATE

ROTULOS = {
    "variancia": "Alta variância (overfitting)",
    "vies": "Alto viés (underfitting)",
    "ambos": "Viés e variância altos",
    "equilibrado": "Equilibrado",
}


def classificar(f1_treino: float, f1_validacao: float) -> str:
    """Classifica um modelo pelo F1 de treino e de validação da CV.

    * gap (treino - validação) > LIMIAR_GAP_VARIANCIA -> alta variância
    * F1 de treino < LIMIAR_TREINO_VIES -> alto viés (o modelo não ajusta nem o treino)
    """
    gap = f1_treino - f1_validacao
    alta_var = gap > LIMIAR_GAP_VARIANCIA
    alto_vies = f1_treino < LIMIAR_TREINO_VIES
    if alta_var and alto_vies:
        return "ambos"
    if alta_var:
        return "variancia"
    if alto_vies:
        return "vies"
    return "equilibrado"


TAMANHOS_CURVA = np.linspace(0.3, 1.0, 5)  # frações do treino usadas nas curvas de aprendizado


def curvas_aprendizado(res: ResultadoDataset, nomes=("Random Forest", "Gradient Boosting"),
                       tamanhos=TAMANHOS_CURVA) -> dict:
    """Curvas de aprendizado (F1 macro de treino e validação x tamanho do treino)."""
    cv = criar_cv(res.y_train)
    curvas = {}
    for nome in nomes:
        if nome not in res.modelos:
            continue
        n, tr, va = learning_curve(
            clone(res.modelos[nome].estimador), res.X_train, res.y_train, cv=cv,
            train_sizes=tamanhos, scoring="f1_macro", shuffle=True, random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        curvas[nome] = {
            "n": n.tolist(),
            "treino": tr.mean(axis=1).tolist(), "treino_std": tr.std(axis=1).tolist(),
            "validacao": va.mean(axis=1).tolist(), "validacao_std": va.std(axis=1).tolist(),
        }
    res.curvas = curvas
    return curvas


def _tendencia(curva: dict) -> float:
    """Quanto o F1 de validação subiu na última metade da curva (positivo = mais dados ajudam)."""
    v = curva["validacao"]
    return v[-1] - v[len(v) // 2]


ROTULOS["misto"] = "Misto (um modelo superajusta, o outro não)"


def _problema(rf: dict, gb: dict) -> str:
    """Problema predominante do dataset, olhando os dois candidatos (RF e GB)."""
    codigos = {rf["codigo"], gb["codigo"]}
    if codigos <= {"variancia", "ambos"}:
        return "ambos" if "ambos" in codigos and "variancia" not in codigos else "variancia"
    if codigos == {"vies"}:
        return "vies"
    if codigos == {"equilibrado"}:
        return "equilibrado"
    return "misto"


def diagnosticar(res: ResultadoDataset) -> dict:
    """Diagnóstico por modelo e justificativa da escolha entre Random Forest e Gradient Boosting.

    1. O problema do dataset é lido nos dois candidatos: se os dois superajustam (gap alto),
       é alta variância; se os dois não ajustam nem o treino, é alto viés.
    2. A árvore de decisão isolada serve de linha de base: mostra quanto cada ensemble
       reduziu o gap (variância) e quanto melhorou a validação.
    3. Decide a validação cruzada. Se a diferença de F1 entre RF e GB for menor que o
       desvio-padrão entre os folds (empate técnico), vence o de menor gap, isto é, o que
       generaliza de forma mais estável.
    """
    por_modelo = {}
    for nome, rm in res.modelos.items():
        tr, va = rm.cv_treino["f1"], rm.cv_validacao["f1"]
        codigo = classificar(tr, va)
        por_modelo[nome] = {"f1_treino": tr, "f1_validacao": va, "gap": tr - va,
                            "codigo": codigo, "rotulo": ROTULOS[codigo]}

    rf, gb = por_modelo.get("Random Forest"), por_modelo.get("Gradient Boosting")
    base = por_modelo.get("Decision Tree")
    if not (rf and gb):
        res.diagnostico = {"por_modelo": por_modelo, "problema": "equilibrado",
                           "problema_rotulo": ROTULOS["equilibrado"], "escolha_rf_gb": None,
                           "justificativa": ""}
        return res.diagnostico

    problema = _problema(rf, gb)
    std = max(res.modelos["Random Forest"].cv_validacao_std["f1"],
              res.modelos["Gradient Boosting"].cv_validacao_std["f1"])
    diff = rf["f1_validacao"] - gb["f1_validacao"]
    empate = abs(diff) < std
    if empate:
        escolha = "Random Forest" if rf["gap"] <= gb["gap"] else "Gradient Boosting"
    else:
        escolha = "Random Forest" if diff > 0 else "Gradient Boosting"
    outro = "Gradient Boosting" if escolha == "Random Forest" else "Random Forest"
    e, o = por_modelo[escolha], por_modelo[outro]

    def f(x):
        return f"{x:.3f}"

    partes = []
    if problema == "variancia":
        partes.append(
            f"O dataset tem **alta variância**: os dois ensembles ajustam o treino muito melhor que a validação "
            f"(gap de {f(rf['gap'])} no Random Forest e {f(gb['gap'])} no Gradient Boosting). "
            "Na teoria, é o caso do Random Forest, cuja média de árvores reduz variância."
        )
    elif problema == "vies":
        partes.append(
            f"O dataset tem **alto viés**: nem no treino os ensembles passam de F1 {f(max(rf['f1_treino'], gb['f1_treino']))}. "
            "Na teoria, é o caso do Gradient Boosting, que reduz viés corrigindo os erros em sequência."
        )
    elif problema == "ambos":
        partes.append(
            "O dataset tem **viés e variância altos**: os ensembles nem ajustam bem o treino nem generalizam. "
            "Nenhum dos dois resolve isso sozinho; faltam atributos que expliquem o alvo."
        )
    elif problema == "misto":
        sup = rf if rf["codigo"] in ("variancia", "ambos") else gb
        nome_sup = "Random Forest" if sup is rf else "Gradient Boosting"
        partes.append(
            f"O diagnóstico é **misto**: o {nome_sup} superajusta (gap {f(sup['gap'])}), o outro não "
            f"(gap {f(gb['gap'] if sup is rf else rf['gap'])})."
        )
    else:
        partes.append(
            f"O dataset está **equilibrado**: os dois ensembles generalizam bem (gap de {f(rf['gap'])} no "
            f"Random Forest e {f(gb['gap'])} no Gradient Boosting), sem sinal de viés nem de variância altos."
        )

    if base:
        partes.append(
            f"Como linha de base, a árvore de decisão isolada teve F1 de validação {f(base['f1_validacao'])} "
            f"(gap {f(base['gap'])}); o Random Forest chegou a {f(rf['f1_validacao'])} e o Gradient Boosting "
            f"a {f(gb['f1_validacao'])}."
        )

    if empate:
        partes.append(
            f"A diferença de F1 entre os dois ({f(abs(diff))}) é menor que o desvio-padrão entre os folds "
            f"({f(std)}), um empate técnico. O critério de desempate é a estabilidade: o **{escolha}** "
            f"tem o menor gap ({f(e['gap'])} contra {f(o['gap'])})."
        )
    else:
        partes.append(
            f"O **{escolha}** generalizou melhor na validação cruzada (F1 {f(e['f1_validacao'])} contra "
            f"{f(o['f1_validacao'])}, diferença maior que o desvio-padrão entre os folds, {f(std)})."
        )

    teoria = {"variancia": "Random Forest", "vies": "Gradient Boosting"}.get(problema)
    if teoria and teoria == escolha:
        partes.append("A evidência confirma a teoria.")
    elif teoria:
        partes.append(
            f"A evidência contraria a teoria: com os hiperparâmetros encontrados (taxa de aprendizado, "
            f"profundidade, regularização), o {escolha} controlou o problema melhor que o {teoria} neste dataset."
        )

    if max(rf["f1_validacao"], gb["f1_validacao"]) < 0.5:
        partes.append(
            "O F1 de validação baixo, mesmo com o treino quase perfeito, mostra que estes atributos explicam "
            "pouco do alvo: trocar de modelo não muda muito o resultado."
        )

    if res.curvas.get(escolha):
        t = _tendencia(res.curvas[escolha])
        if t > 0.01:
            partes.append(
                f"A curva de aprendizado do {escolha} ainda sobe no fim (+{f(t)} de F1 na segunda metade): "
                "mais exemplos tendem a reduzir a variância restante."
            )
        else:
            partes.append(
                f"A curva de aprendizado do {escolha} já está estável: mais dados do mesmo tipo ajudariam pouco."
            )

    res.diagnostico = {"por_modelo": por_modelo, "problema": problema, "problema_rotulo": ROTULOS[problema],
                       "escolha_rf_gb": escolha, "empate": empate, "justificativa": " ".join(partes)}
    return res.diagnostico
