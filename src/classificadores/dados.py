"""Download (com cache local) e preparação dos datasets."""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import DATASETS, GITHUB_RAW, KAGGLE_API

PASTA_DADOS = Path(os.environ.get("CLASSIFICADORES_DADOS", "data/raw"))


def _baixar(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "classificadores-6-datasets"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        conteudo = resp.read()
    if conteudo[:2] == b"PK":  # o Kaggle às vezes entrega o CSV dentro de um .zip
        with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
            conteudo = z.read(z.namelist()[0])
    return conteudo


def caminho_local(cfg: dict, pasta: Path | None = None) -> Path:
    return Path(pasta or PASTA_DADOS) / cfg["arquivo"]


def obter_csv(cfg: dict, pasta: Path | None = None) -> pd.DataFrame:
    """Lê o CSV do cache local; se não existir, baixa e guarda no cache.

    Ordem de tentativa: cache local -> download direto (API pública do Kaggle ou GitHub raw)
    -> kagglehub. Se tudo falhar, a mensagem diz onde colocar o arquivo manualmente.
    """
    local = caminho_local(cfg, pasta)
    if local.exists():
        return pd.read_csv(local)

    url = (GITHUB_RAW if cfg["fonte"] == "github" else KAGGLE_API).format(**cfg)
    erros = []
    try:
        conteudo = _baixar(url)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(conteudo)
        return pd.read_csv(local)
    except Exception as e:  # noqa: BLE001
        erros.append(f"download direto ({url}): {e}")

    if cfg["fonte"] == "kaggle":
        try:
            import kagglehub

            pasta_kh = kagglehub.dataset_download(cfg["slug"])
            return pd.read_csv(Path(pasta_kh) / cfg["arquivo"])
        except Exception as e:  # noqa: BLE001
            erros.append(f"kagglehub: {e}")

    raise RuntimeError(
        f"Não consegui obter {cfg['arquivo']}. Baixe-o manualmente e salve em {local}. "
        f"Detalhes: {erros}"
    )


def preparar(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica o pré-tratamento do dataset e separa X (atributos) e y (alvo)."""
    if cfg.get("prep"):
        df = cfg["prep"](df)

    alvo = cfg["target"]
    if alvo not in df.columns:
        raise KeyError(f"coluna-alvo '{alvo}' não encontrada. Colunas disponíveis: {list(df.columns)}")
    df = df.dropna(subset=[alvo])  # linhas sem rótulo não servem para treino nem teste

    minimo = cfg.get("min_por_classe")
    if minimo:
        contagem = df[alvo].value_counts()
        df = df[df[alvo].isin(contagem[contagem >= minimo].index)]

    if "usar" in cfg:
        faltando = [c for c in cfg["usar"] if c not in df.columns]
        if faltando:
            raise KeyError(f"colunas de entrada não encontradas: {faltando}")
        X = df[cfg["usar"]]
    else:
        X = df.drop(columns=[alvo] + [c for c in cfg.get("drop", []) if c in df.columns])
    return X.reset_index(drop=True), df[alvo].reset_index(drop=True)


def classes_removidas(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Classes descartadas por terem menos exemplos que ``min_por_classe``."""
    minimo = cfg.get("min_por_classe")
    if not minimo:
        return pd.Series(dtype=int)
    contagem = df[cfg["target"]].value_counts()
    return contagem[contagem < minimo]


def carregar_dataset(nome: str, pasta: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    cfg = DATASETS[nome]
    return preparar(obter_csv(cfg, pasta), cfg)
