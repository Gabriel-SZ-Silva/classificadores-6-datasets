"""Linha de comando: ``python -m classificadores <comando>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera PNGs sem precisar de tela

from .config import DATASETS, N_ITER  # noqa: E402

CACHE = ".execucao.pkl"  # resultados completos da última execução (fora do git), usados por `relatorio`
ATALHOS = {cfg["slug_curto"]: nome for nome, cfg in DATASETS.items()}


def _nomes(lista: list[str] | None) -> list[str] | None:
    if not lista:
        return None
    desconhecidos = [n for n in lista if n not in ATALHOS]
    if desconhecidos:
        sys.exit(f"Datasets desconhecidos: {desconhecidos}. Opções: {', '.join(ATALHOS)}")
    return [ATALHOS[n] for n in lista]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="classificadores",
                                description="SVM, Decision Tree, Random Forest e Gradient Boosting em 6 datasets.")
    sub = p.add_subparsers(dest="comando", required=True)

    b = sub.add_parser("baixar", help="baixa os 6 CSVs para data/raw (cache local)")
    b.add_argument("--datasets", nargs="+", metavar="NOME", help=f"subconjunto: {', '.join(ATALHOS)}")
    b.add_argument("--dados", type=Path, help="pasta do cache (padrão: data/raw)")

    e = sub.add_parser("executar", help="treina, avalia, diagnostica e gera resultados/ e o painel")
    e.add_argument("--datasets", nargs="+", metavar="NOME", help=f"subconjunto: {', '.join(ATALHOS)}")
    e.add_argument("--n-iter", type=int, default=N_ITER, help=f"combinações por modelo (padrão: {N_ITER})")
    e.add_argument("--sem-curvas", action="store_true", help="pula as curvas de aprendizado (mais rápido)")
    e.add_argument("--gb-classico", action="store_true",
                   help="usa GradientBoostingClassifier em vez de HistGradientBoostingClassifier (bem mais lento)")
    e.add_argument("--dados", type=Path, help="pasta do cache (padrão: data/raw)")
    e.add_argument("--saida", type=Path, default=Path("resultados"), help="pasta de saída (padrão: resultados)")

    r = sub.add_parser("relatorio", help="regera resultados/ a partir da última execução, sem treinar de novo")
    r.add_argument("--saida", type=Path, default=Path("resultados"), help="pasta de saída (padrão: resultados)")

    a = p.parse_args(argv)
    if a.comando == "relatorio":
        import pickle

        from .relatorio import salvar

        cache = a.saida / CACHE
        if not cache.exists():
            sys.exit(f"{cache} não existe: rode `python -m classificadores executar` antes.")
        salvar(pickle.loads(cache.read_bytes()), a.saida)
        print(f"Resultados regerados em {a.saida}/")
        return 0

    if a.comando == "baixar":
        from .dados import caminho_local, obter_csv

        for nome in _nomes(a.datasets) or list(DATASETS):
            cfg = DATASETS[nome]
            df = obter_csv(cfg, a.dados)
            print(f"✓ {nome}: {df.shape[0]} linhas -> {caminho_local(cfg, a.dados)}")
        return 0

    from .pipeline import executar
    from .relatorio import salvar

    resultados = executar(_nomes(a.datasets), n_iter=a.n_iter, com_curvas=not a.sem_curvas,
                          usar_hist_gb=not a.gb_classico, pasta_dados=a.dados)
    if not resultados:
        print("Nenhum dataset foi processado.")
        return 1
    salvar(resultados, a.saida)
    import pickle

    (a.saida / CACHE).write_bytes(pickle.dumps(resultados))
    print(f"\nResultados em {a.saida}/: metricas.csv, diagnostico.csv, tabelas.md, figuras/ e painel.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
