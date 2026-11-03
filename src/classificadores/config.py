"""Configuração geral: constantes do experimento e descrição dos 6 datasets.

Cada dataset declara explicitamente de onde vem, qual coluna é o alvo e quais colunas
entram (``usar``) ou saem (``drop``). Nada é adivinhado: se o arquivo ou a coluna não
existir, o carregamento falha com uma mensagem clara.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TAMANHO_TESTE = 0.20  # 80% treino / 20% teste, estratificado
N_FOLDS = 5  # folds da validação cruzada estratificada
N_ITER = 10  # combinações de hiperparâmetros sorteadas por modelo no RandomizedSearchCV
METRICA_BUSCA = "f1"  # F1 macro: critério de escolha dos hiperparâmetros (todas as classes pesam igual)

# GradientBoostingClassifier "clássico" leva ~1 min por ajuste no Fishmorph (uma árvore por classe a
# cada iteração). HistGradientBoostingClassifier é o mesmo algoritmo com histogramas, ~20x mais rápido.
USAR_HIST_GB = True

# Limiares do diagnóstico de viés e variância (seção "Metodologia" do README)
LIMIAR_GAP_VARIANCIA = 0.10  # F1 de treino - F1 de validação acima disso = alta variância
LIMIAR_TREINO_VIES = 0.80  # F1 de treino abaixo disso (e gap pequeno) = alto viés
LIMIAR_GAP_VIES = 0.05

KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/download/{slug}/{arquivo}"
GITHUB_RAW = "https://raw.githubusercontent.com/{slug}/main/{arquivo}"


def prep_eleicoes(df: pd.DataFrame) -> pd.DataFrame:
    """Alvo: o republicano venceu de fato no estado? (1 = sim).

    1932 quase não tem resultado real no arquivo e cai fora aqui.
    """
    df = df.dropna(subset=["pct_real_rep", "pct_real_dem"]).copy()
    df["vencedor_real"] = (df["pct_real_rep"] > df["pct_real_dem"]).astype(int)
    return df


def prep_cardiopatia(df: pd.DataFrame) -> pd.DataFrame:
    """Colesterol e pressão iguais a 0 significam "não medido": viram NaN para o imputador."""
    df = df.copy()
    for col in ["Cholesterol", "RestingBP"]:
        df[col] = df[col].replace(0, np.nan)
    return df


DATASETS: dict[str, dict] = {
    "Câncer de Mama": {
        "slug_curto": "cancer_mama",
        "fonte": "kaggle",
        "slug": "ahmedesso/brest-cancer",
        "arquivo": "data.csv",
        "target": "diagnosis",  # M = maligno, B = benigno
        "classe_positiva": "M",
        "drop": ["id", "Unnamed: 32"],  # identificador + coluna vazia que vem no CSV
        "descricao": "Breast Cancer Wisconsin (Diagnostic): 30 medidas de núcleos celulares.",
    },
    "Eleições (Literary Digest)": {
        "slug_curto": "eleicoes",
        "fonte": "github",
        "slug": "Gabriel-SZ-Silva/LiteraryDigestElections",
        "arquivo": "dataset_literary_digest_consolidado_1924_1936.csv",
        "target": "vencedor_real",
        "classe_positiva": 1,
        "drop": [
            # Resultado real: o alvo é derivado dele (seria vazamento de dados)
            "votos_real_rep", "votos_real_dem", "total_real", "pct_real_rep", "pct_real_dem",
            "vies_republicano_pp",
            # Ano e candidatos: em cada eleição um partido venceu quase todos os estados,
            # então essas colunas entregariam a resposta sem olhar para a pesquisa
            "ano", "candidato_rep", "candidato_dem",
        ],
        "prep": prep_eleicoes,
        "descricao": "Pesquisa do Literary Digest (1924-1936) por estado; prevê o vencedor real.",
    },
    "Fishmorph (Ordem)": {
        # Vertente "Ordem". Para a vertente "Família" (197 classes), use
        # arquivo="FISHMORPH_Family_Dataset.csv" e target="Family".
        "slug_curto": "fishmorph",
        "fonte": "kaggle",
        "slug": "menegidio/fishmorph-dataset",
        "arquivo": "FISHMORPH_Order_Dataset.csv",
        "target": "Order",
        "drop": [],
        "min_por_classe": 10,  # ordens com < 10 espécies inviabilizam treino/teste/CV estratificados
        "descricao": "10 medidas morfológicas de 8342 espécies de peixes de água doce; prevê a ordem.",
    },
    "Cardiopatia": {
        "slug_curto": "cardiopatia",
        "fonte": "kaggle",
        "slug": "fedesoriano/heart-failure-prediction",
        "arquivo": "heart.csv",
        "target": "HeartDisease",
        "classe_positiva": 1,
        "drop": [],
        "prep": prep_cardiopatia,
        "descricao": "Heart Failure Prediction: 11 atributos clínicos de 918 pacientes.",
    },
    "Estresse Estudantil": {
        "slug_curto": "estresse",
        "fonte": "kaggle",
        "slug": "rxnach/student-stress-factors-a-comprehensive-analysis",
        "arquivo": "StressLevelDataset.csv",
        "target": "stress_level",  # 0 = baixo, 1 = médio, 2 = alto
        "drop": [],
        "descricao": "20 fatores psicológicos, físicos, ambientais e acadêmicos de 1100 estudantes.",
    },
    "Ultimate Pokémon": {
        "slug_curto": "pokemon",
        "fonte": "kaggle",
        "slug": "darkmatternet/ultimate-pokmon-dataset-2025",
        "arquivo": "pokemon_complete_2025.csv",
        "target": "type_1",  # tipo primário (18 classes)
        # Só atributos numéricos de batalha e biologia. Ficam de fora nome, descrição, sprite,
        # habilidades e type_2, que praticamente entregam o tipo.
        "usar": [
            "num_types", "hp", "attack", "defense", "sp_attack", "sp_defense", "speed",
            "base_stat_total", "height_m", "weight_kg", "base_experience", "capture_rate",
            "base_happiness", "hatch_counter", "gender_rate", "bmi", "attack_defense_ratio",
            "physical_total", "special_total", "offensive_total", "defensive_total",
        ],
        "descricao": "1025 Pokémon com atributos de batalha e biologia; prevê o tipo primário.",
    },
}
