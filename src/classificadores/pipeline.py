"""Orquestra o experimento completo: dados -> 4 modelos -> métricas -> diagnóstico -> relatórios."""

from __future__ import annotations

import time
from pathlib import Path

from .avaliacao import ResultadoDataset, avaliar_dataset
from .config import DATASETS, N_ITER, USAR_HIST_GB
from .dados import carregar_dataset, classes_removidas, obter_csv
from .diagnostico import curvas_aprendizado, diagnosticar
from .modelos import criar_modelos


def executar(nomes: list[str] | None = None, n_iter: int = N_ITER, com_curvas: bool = True,
             usar_hist_gb: bool = USAR_HIST_GB, pasta_dados: Path | None = None,
             log=print) -> dict[str, ResultadoDataset]:
    nomes = nomes or list(DATASETS)
    modelos = criar_modelos(usar_hist_gb)
    resultados: dict[str, ResultadoDataset] = {}
    inicio = time.time()
    for nome in nomes:  # o mesmo laço para os 6 datasets
        cfg = DATASETS[nome]
        log(f"\n{'=' * 60}\n{nome}\n{'=' * 60}")
        try:
            raras = classes_removidas(obter_csv(cfg, pasta_dados), cfg)
            X, y = carregar_dataset(nome, pasta_dados)
        except Exception as e:  # noqa: BLE001
            log(f"   ✗ não foi possível carregar: {e}")
            continue
        if len(raras):
            log(f"   ↳ {len(raras)} classes com < {cfg['min_por_classe']} exemplos removidas: {list(raras.index)}")
        log(f"   {X.shape[0]} linhas x {X.shape[1]} atributos | alvo '{cfg['target']}' com {y.nunique()} classes")
        res = avaliar_dataset(nome, X, y, modelos, n_iter=n_iter, log=log)
        if com_curvas:
            curvas_aprendizado(res)
        d = diagnosticar(res)
        log(f"   → diagnóstico (RF e GB): {d['problema_rotulo']} | escolha RF x GB: {d['escolha_rf_gb']}")
        resultados[nome] = res
    log(f"\nConcluído em {(time.time() - inicio) / 60:.1f} min.")
    return resultados
