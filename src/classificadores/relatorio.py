"""Tabelas comparativas, figuras em disco, tabelas em Markdown e o painel HTML."""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import graficos
from .avaliacao import METRICAS, NOMES_METRICAS, ResultadoDataset
from .config import DATASETS


# --------------------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------------------
def tabela_metricas(resultados: dict[str, ResultadoDataset]) -> pd.DataFrame:
    linhas = []
    for nome_ds, res in resultados.items():
        for nome, rm in res.modelos.items():
            linha = {"Dataset": nome_ds, "Modelo": nome}
            linha.update({NOMES_METRICAS[m]: rm.teste[m] for m in METRICAS})
            linha["F1 CV (validação)"] = rm.cv_validacao["f1"]
            linha["F1 CV (desvio)"] = rm.cv_validacao_std["f1"]
            linha["AUC CV (validação)"] = rm.cv_validacao["roc_auc"]
            linha["Tempo (s)"] = rm.tempo_s
            linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_diagnostico(resultados: dict[str, ResultadoDataset]) -> pd.DataFrame:
    linhas = []
    for nome_ds, res in resultados.items():
        for nome, d in res.diagnostico.get("por_modelo", {}).items():
            linhas.append({"Dataset": nome_ds, "Modelo": nome, "F1 treino (CV)": d["f1_treino"],
                           "F1 validação (CV)": d["f1_validacao"], "Gap": d["gap"],
                           "Diagnóstico": d["rotulo"]})
    return pd.DataFrame(linhas)


def tabela_escolhas(resultados: dict[str, ResultadoDataset]) -> pd.DataFrame:
    linhas = []
    for nome_ds, res in resultados.items():
        d = res.diagnostico
        if not d:
            continue
        melhor = max(res.modelos, key=lambda m: res.modelos[m].cv_validacao["f1"])
        linhas.append({"Dataset": nome_ds, "Problema (RF e GB)": d["problema_rotulo"],
                       "Escolha RF x GB": d["escolha_rf_gb"], "Melhor dos 4 (F1 CV)": melhor,
                       "Justificativa": d["justificativa"]})
    return pd.DataFrame(linhas)


def tabela_hiperparametros(resultados: dict[str, ResultadoDataset]) -> pd.DataFrame:
    linhas = []
    for nome_ds, res in resultados.items():
        for nome, rm in res.modelos.items():
            p = {k.replace("modelo__", "").replace("prep__num__imputacao__strategy", "imputação"): v
                 for k, v in rm.busca.best_params_.items()}
            txt = ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in sorted(p.items()))
            linhas.append({"Dataset": nome_ds, "Modelo": nome, "Hiperparâmetros": txt})
    return pd.DataFrame(linhas)


def pivot(df: pd.DataFrame, valor: str) -> pd.DataFrame:
    ordem_ds = list(dict.fromkeys(df["Dataset"]))
    ordem_m = list(dict.fromkeys(df["Modelo"]))
    return df.pivot(index="Dataset", columns="Modelo", values=valor).reindex(index=ordem_ds, columns=ordem_m)


def markdown(df: pd.DataFrame, destacar_max_linha: bool = False, casas: int = 3) -> str:
    """Tabela Markdown sem dependências; opcionalmente põe em negrito o maior valor de cada linha."""
    cab = [str(df.index.name or "")] + [str(c) for c in df.columns]
    linhas = ["| " + " | ".join(cab) + " |", "|" + "|".join(["---"] + [":---:"] * len(df.columns)) + "|"]
    for idx, row in df.iterrows():
        nums = pd.to_numeric(row, errors="coerce")
        mx = nums.max() if destacar_max_linha else None
        cels = []
        for v, n in zip(row, nums):
            if pd.notna(n) and not isinstance(v, str):
                s = f"{n:.{casas}f}"
                cels.append(f"**{s}**" if mx is not None and n == mx else s)
            else:
                cels.append(str(v))
        linhas.append("| " + " | ".join([str(idx)] + cels) + " |")
    return "\n".join(linhas)


# --------------------------------------------------------------------------------------
# Saída em disco
# --------------------------------------------------------------------------------------
def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def salvar(resultados: dict[str, ResultadoDataset], pasta: str | Path = "resultados") -> dict:
    pasta = Path(pasta)
    (pasta / "figuras").mkdir(parents=True, exist_ok=True)

    met, diag = tabela_metricas(resultados), tabela_diagnostico(resultados)
    esc, hip = tabela_escolhas(resultados), tabela_hiperparametros(resultados)
    met.to_csv(pasta / "metricas.csv", index=False)
    diag.to_csv(pasta / "diagnostico.csv", index=False)
    esc.to_csv(pasta / "escolha_rf_gb.csv", index=False)
    hip.to_csv(pasta / "hiperparametros.csv", index=False)

    figuras: dict[str, dict[str, bytes]] = {"geral": {}}
    figuras["geral"]["f1"] = _png(graficos.grafico_mapa(pivot(met, "F1"), "F1 macro no teste"))
    figuras["geral"]["auc"] = _png(graficos.grafico_mapa(pivot(met, "ROC AUC"), "ROC AUC no teste"))
    figuras["geral"]["acuracia"] = _png(graficos.grafico_mapa(pivot(met, "Acurácia"), "Acurácia no teste"))
    if len(diag):
        figuras["geral"]["gaps"] = _png(graficos.grafico_gaps(diag))
    for nome_ds, res in resultados.items():
        f = figuras.setdefault(DATASETS.get(nome_ds, {}).get("slug_curto", nome_ds), {})
        f["metricas"] = _png(graficos.grafico_metricas(res))
        f["roc"] = _png(graficos.grafico_roc(res))
        f["confusao"] = _png(graficos.grafico_matrizes_confusao(res))
        if res.curvas:
            f["aprendizado"] = _png(graficos.grafico_aprendizado(res))
    for grupo, figs in figuras.items():
        (pasta / "figuras" / grupo).mkdir(parents=True, exist_ok=True)
        for nome, conteudo in figs.items():
            (pasta / "figuras" / grupo / f"{nome}.png").write_bytes(conteudo)

    # Tabelas em Markdown, prontas para colar no README
    partes = ["# Resultados", "", f"Gerado em {datetime.now():%d/%m/%Y %H:%M}.", ""]
    for rotulo in ["Acurácia", "Precisão", "Recall", "F1", "ROC AUC"]:
        partes += [f"## {rotulo} no teste", "", markdown(pivot(met, rotulo), destacar_max_linha=True), ""]
    partes += ["## F1 na validação cruzada (média dos folds)", "",
               markdown(pivot(met, "F1 CV (validação)"), destacar_max_linha=True), ""]
    if len(diag):
        d2 = diag.copy()
        d2["Resumo"] = d2.apply(lambda r: f"{r['Diagnóstico']} (gap {r['Gap']:.3f})", axis=1)
        partes += ["## Diagnóstico de viés e variância", "",
                   markdown(d2.pivot(index="Dataset", columns="Modelo", values="Resumo")
                            .reindex(index=list(dict.fromkeys(d2["Dataset"])), columns=list(graficos.CORES))), ""]
        partes += ["## Random Forest x Gradient Boosting", ""]
        for _, r in esc.iterrows():
            partes += [f"**{r['Dataset']}**: {r['Escolha RF x GB']}. {r['Justificativa']}", ""]
    (pasta / "tabelas.md").write_text("\n".join(partes), encoding="utf-8")

    resumo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "datasets": {n: {"linhas": r.n_linhas, "atributos": r.n_atributos, "classes": len(r.classes)}
                     for n, r in resultados.items()},
        "metricas": json.loads(met.to_json(orient="records", force_ascii=False)),
        "diagnostico": json.loads(diag.to_json(orient="records", force_ascii=False)),
        "escolhas": json.loads(esc.to_json(orient="records", force_ascii=False)),
    }
    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / "painel.html").write_text(painel_html(resultados, met, diag, esc, hip, figuras), encoding="utf-8")
    return {"metricas": met, "diagnostico": diag, "escolhas": esc, "hiperparametros": hip}


# --------------------------------------------------------------------------------------
# Painel HTML (arquivo único, abre em qualquer navegador, sem servidor)
# --------------------------------------------------------------------------------------
def _img(b: bytes | None, alt: str) -> str:
    if not b:
        return ""
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(b).decode()}">'


def _tabela_html(df: pd.DataFrame, destacar: list[str] | None = None, casas: int = 3) -> str:
    destacar = destacar or []
    cab = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    maximos = {c: df[c].max() for c in destacar if c in df}
    corpo = []
    for _, row in df.iterrows():
        cels = []
        for c, v in row.items():
            if isinstance(v, float):
                cls = ' class="max"' if c in maximos and v == maximos[c] else ""
                cels.append(f'<td{cls} data-num>{v:.{casas}f}</td>')
            else:
                cels.append(f"<td>{html.escape(str(v))}</td>")
        corpo.append("<tr>" + "".join(cels) + "</tr>")
    return f'<div class="tabela"><table><thead><tr>{cab}</tr></thead><tbody>{"".join(corpo)}</tbody></table></div>'


def _md_negrito(txt: str) -> str:
    partes = html.escape(txt).split("**")
    return "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(partes))


CSS = """
:root{--sup:#fcfcfb;--sup2:#f3f2ef;--borda:#e4e3df;--t1:#0b0b0b;--t2:#52514e;--t3:#7a7975;--azul:#2a78d6;
--dt:#2a78d6;--rf:#eb6834;--gb:#1baf7a;--svm:#eda100}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
color:var(--t1);background:var(--sup)}
header{padding:28px 40px 12px;border-bottom:1px solid var(--borda)}
header h1{margin:0 0 4px;font-size:24px}header p{margin:0;color:var(--t2)}
nav{display:flex;flex-wrap:wrap;gap:6px;padding:14px 40px;border-bottom:1px solid var(--borda);
position:sticky;top:0;background:var(--sup);z-index:2}
nav button{border:1px solid var(--borda);background:var(--sup);color:var(--t1);padding:7px 14px;
border-radius:999px;font:inherit;font-size:14px;cursor:pointer}
nav button[aria-selected=true]{background:var(--t1);color:var(--sup);border-color:var(--t1)}
main{padding:24px 40px 60px;max-width:1500px}
section{display:none}section.ativa{display:block}
h2{font-size:20px;margin:8px 0 6px}h3{font-size:16px;margin:28px 0 8px}
.sub{color:var(--t2);margin:0 0 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:10px 0 18px}
.kpi{border:1px solid var(--borda);border-radius:10px;padding:14px 16px;background:#fff}
.kpi .r{color:var(--t2);font-size:13px}.kpi .v{font-size:26px;font-weight:600;margin-top:2px}
.kpi .d{color:var(--t3);font-size:12px}
.grade2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:1000px){.grade2{grid-template-columns:1fr}header,nav,main{padding-left:16px;padding-right:16px}}
img{max-width:100%;height:auto;border:1px solid var(--borda);border-radius:10px;background:#fff}
.tabela{overflow-x:auto;border:1px solid var(--borda);border-radius:10px;background:#fff}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--borda);white-space:nowrap}
th{background:var(--sup2);color:var(--t2);font-weight:600}td[data-num]{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:hover{background:#eef4fc}td.max{font-weight:700}
.diag{border-left:4px solid var(--azul);background:#fff;border-radius:0 10px 10px 0;padding:14px 18px;
border-top:1px solid var(--borda);border-right:1px solid var(--borda);border-bottom:1px solid var(--borda)}
.chip{display:inline-block;padding:2px 10px;border-radius:999px;background:var(--sup2);font-size:13px;margin-right:6px}
.leg{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--t2);margin:4px 0 14px}
.leg span:before{content:"";display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;
background:var(--c);vertical-align:-1px}
footer{color:var(--t3);font-size:13px;padding:0 40px 30px}
"""

JS = """
const bots=[...document.querySelectorAll('nav button')];
function abrir(id){bots.forEach(b=>b.setAttribute('aria-selected',b.dataset.alvo===id));
document.querySelectorAll('section').forEach(s=>s.classList.toggle('ativa',s.id===id));
if(history.replaceState)history.replaceState(null,'','#'+id);}
bots.forEach(b=>b.addEventListener('click',()=>abrir(b.dataset.alvo)));
const inicial=location.hash.slice(1);
abrir(inicial&&document.getElementById(inicial)?inicial:'geral');
window.addEventListener('hashchange',()=>{const h=location.hash.slice(1);if(document.getElementById(h))abrir(h);});
"""


def painel_html(resultados, met, diag, esc, hip, figuras) -> str:
    legenda = '<div class="leg">' + "".join(
        f'<span style="--c:{c}">{html.escape(n)}</span>' for n, c in graficos.CORES.items()) + "</div>"
    abas = [("geral", "Visão geral")] + [
        (DATASETS.get(n, {}).get("slug_curto", n), n) for n in resultados]
    nav = "".join(f'<button data-alvo="{i}" aria-selected="false">{html.escape(t)}</button>' for i, t in abas)

    g = figuras["geral"]
    melhor_por_ds = met.loc[met.groupby("Dataset", sort=False)["F1"].idxmax()]
    resumo = melhor_por_ds[["Dataset", "Modelo", "Acurácia", "Precisão", "F1", "ROC AUC"]].rename(
        columns={"Modelo": "Melhor modelo (F1 teste)"})
    secoes = [f"""
<section id="geral"><h2>Visão geral</h2>
<p class="sub">{len(resultados)} datasets, 4 classificadores, busca de hiperparâmetros com validação cruzada
estratificada e avaliação num conjunto de teste separado (20%).</p>
<h3>Melhor modelo por dataset</h3>{_tabela_html(resumo, ["F1", "ROC AUC"])}
<div class="grade2"><div><h3>F1 macro no teste</h3>{_img(g.get('f1'), 'Mapa de calor do F1 por dataset e modelo')}</div>
<div><h3>ROC AUC no teste</h3>{_img(g.get('auc'), 'Mapa de calor da ROC AUC por dataset e modelo')}</div></div>
<h3>Viés e variância</h3>{_img(g.get('gaps'), 'Gap entre F1 de treino e de validação por modelo e dataset')}
{_tabela_html(diag, [])}
<h3>Random Forest x Gradient Boosting</h3>
{''.join(f'<p class="diag"><strong>{html.escape(r["Dataset"])}</strong>: <span class="chip">{html.escape(str(r["Escolha RF x GB"]))}</span>{_md_negrito(r["Justificativa"])}</p>' for _, r in esc.iterrows())}
<h3>Todas as métricas</h3>{_tabela_html(met.drop(columns=["Tempo (s)"]), [])}
</section>"""]

    for nome_ds, res in resultados.items():
        sid = DATASETS.get(nome_ds, {}).get("slug_curto", nome_ds)
        f = figuras.get(sid, {})
        m = met[met["Dataset"] == nome_ds].drop(columns=["Dataset"])
        melhor = m.loc[m["F1"].idxmax()]
        d = res.diagnostico
        cfg = DATASETS.get(nome_ds, {})
        kpis = f"""<div class="kpis">
<div class="kpi"><div class="r">Exemplos / atributos</div><div class="v">{res.n_linhas} / {res.n_atributos}</div>
<div class="d">{len(res.classes)} classes · alvo <code>{html.escape(str(cfg.get('target', '')))}</code></div></div>
<div class="kpi"><div class="r">Melhor modelo (F1 teste)</div><div class="v">{html.escape(melhor['Modelo'])}</div>
<div class="d">F1 {melhor['F1']:.3f} · acurácia {melhor['Acurácia']:.3f}</div></div>
<div class="kpi"><div class="r">Melhor ROC AUC</div><div class="v">{m['ROC AUC'].max():.3f}</div>
<div class="d">{html.escape(m.loc[m['ROC AUC'].idxmax(), 'Modelo'])}</div></div>
<div class="kpi"><div class="r">Diagnóstico (RF e GB)</div><div class="v" style="font-size:18px">{html.escape(d.get('problema_rotulo', ''))}</div>
<div class="d">escolha RF x GB: {html.escape(str(d.get('escolha_rf_gb')))}</div></div></div>"""
        h = hip[hip["Dataset"] == nome_ds].drop(columns=["Dataset"])
        secoes.append(f"""
<section id="{sid}"><h2>{html.escape(nome_ds)}</h2><p class="sub">{html.escape(cfg.get('descricao', ''))}</p>
{kpis}{legenda}
<h3>Métricas (teste e validação cruzada)</h3>{_tabela_html(m.drop(columns=['Tempo (s)']), ['Acurácia', 'Precisão', 'Recall', 'F1', 'ROC AUC', 'F1 CV (validação)'])}
<div class="grade2"><div><h3>Comparativo</h3>{_img(f.get('metricas'), 'Barras das métricas de teste por modelo')}</div>
<div><h3>Curva ROC</h3>{_img(f.get('roc'), 'Curvas ROC dos quatro modelos')}</div></div>
<h3>Matrizes de confusão</h3>{_img(f.get('confusao'), 'Matrizes de confusão dos quatro modelos')}
<h3>Viés e variância: Random Forest x Gradient Boosting</h3>
<p class="diag">{_md_negrito(d.get('justificativa', ''))}</p>
{_img(f.get('aprendizado'), 'Curvas de aprendizado do Random Forest e do Gradient Boosting')}
<h3>Hiperparâmetros escolhidos</h3>{_tabela_html(h)}
</section>""")

    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Painel: 4 classificadores em 6 datasets</title>
<style>{CSS}</style></head><body>
<header><h1>Painel de resultados: 4 classificadores em 6 datasets</h1>
<p>SVM, Decision Tree, Random Forest e Gradient Boosting · gerado em {datetime.now():%d/%m/%Y %H:%M}</p></header>
<nav role="tablist">{nav}</nav><main>{''.join(secoes)}</main>
<footer>Gerado por <code>python -m classificadores executar</code>. Os números vêm de resultados/metricas.csv.</footer>
<script>{JS}</script></body></html>"""
