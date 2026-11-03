"""Captura as telas do painel (resultados/painel.html) para docs/assets/telas/.

Requer o Playwright:  pip install playwright && python -m playwright install chromium

    python scripts/capturar_telas.py                # painel
    python scripts/capturar_telas.py --log run.log  # também a saída do terminal
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
PAINEL = RAIZ / "resultados" / "painel.html"
DESTINO = RAIZ / "docs" / "assets" / "telas"

DESKTOP = {"width": 1440, "height": 900}
CELULAR = {"width": 390, "height": 844}


def _abrir(pagina, aba: str):
    if not pagina.url.startswith(PAINEL.resolve().as_uri()):
        pagina.goto(PAINEL.resolve().as_uri())
        pagina.wait_for_load_state("networkidle")
    pagina.evaluate("aba => { abrir(aba); window.scrollTo(0, 0); }", aba)
    pagina.wait_for_timeout(150)


def _secao(pagina, aba: str, titulo: str, arquivo: str):
    """Recorta da seção ``aba`` o trecho entre o título h3 ``titulo`` e o próximo h3."""
    _abrir(pagina, aba)
    caixa = pagina.evaluate(
        """([aba, titulo]) => {
            const s = document.getElementById(aba);
            const hs = [...s.querySelectorAll('h3')];
            const i = hs.findIndex(h => h.textContent.startsWith(titulo));
            const grade = hs[i].closest('.grade2');  // títulos lado a lado: recorta a grade inteira
            const ini = (grade || hs[i]).getBoundingClientRect().top + window.scrollY - 12;
            const prox = hs.slice(i + 1).find(h => !grade || !grade.contains(h));
            const fim = grade ? grade.getBoundingClientRect().bottom + window.scrollY + 8
                : (prox ? prox.getBoundingClientRect().top + window.scrollY
                        : s.getBoundingClientRect().bottom + window.scrollY) - 8;
            return {x: 0, y: ini, width: document.documentElement.clientWidth, height: fim - ini};
        }""", [aba, titulo])
    pagina.screenshot(path=str(DESTINO / arquivo), clip=caixa, full_page=True)


def capturar(log: Path | None = None) -> None:
    if not PAINEL.exists():
        raise SystemExit("resultados/painel.html não existe: rode `python -m classificadores executar` antes.")
    DESTINO.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        nav = p.chromium.launch()
        pg = nav.new_page(viewport=DESKTOP, device_scale_factor=1)
        _abrir(pg, "geral")
        pg.screenshot(path=str(DESTINO / "painel-geral.png"))
        _secao(pg, "geral", "Viés e variância", "painel-vies-variancia.png")
        _secao(pg, "geral", "Random Forest x Gradient Boosting", "painel-rf-x-gb.png")
        for aba in ["cancer_mama", "eleicoes", "fishmorph", "cardiopatia", "estresse", "pokemon"]:
            _abrir(pg, aba)
            pg.screenshot(path=str(DESTINO / f"painel-{aba}.png"))
        _secao(pg, "cardiopatia", "Comparativo", "painel-cardiopatia-roc.png")
        _secao(pg, "pokemon", "Matrizes de confusão", "painel-pokemon-confusao.png")
        _secao(pg, "fishmorph", "Viés e variância", "painel-fishmorph-diagnostico.png")

        cel = nav.new_page(viewport=CELULAR, device_scale_factor=2)
        _abrir(cel, "estresse")
        cel.screenshot(path=str(DESTINO / "painel-celular.png"))

        if log and log.exists():
            texto = html.escape(log.read_text(encoding="utf-8"))
            term = nav.new_page(viewport={"width": 1100, "height": 400})
            term.set_content(
                "<body style='margin:0;background:#1a1a19'><pre style='margin:0;padding:22px 26px;"
                "color:#f0efec;font:13px/1.5 ui-monospace,Consolas,monospace'>"
                f"<span style='color:#86b6ef'>$ python -m classificadores executar</span>\n{texto}</pre></body>")
            term.screenshot(path=str(DESTINO / "terminal.png"), full_page=True)
        nav.close()
    print(f"Telas salvas em {DESTINO}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--log", type=Path, help="arquivo com a saída de `executar`, para a captura do terminal")
    capturar(a.parse_args().log)
