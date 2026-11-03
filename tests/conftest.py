import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def binario():
    """Dataset binário sintético com uma coluna categórica e valores ausentes."""
    rng = np.random.default_rng(0)
    n = 160
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "cat": rng.choice(["x", "y", "z"], n),
    })
    y = pd.Series(np.where(X["a"] + 0.5 * X["b"] + rng.normal(0, 0.5, n) > 0, "M", "B"))
    X.loc[::17, "b"] = np.nan
    return X, y


@pytest.fixture
def multiclasse():
    rng = np.random.default_rng(1)
    n = 240
    y = pd.Series(rng.choice(["c1", "c2", "c3", "c4"], n, p=[0.4, 0.3, 0.2, 0.1]))
    centros = {"c1": 0, "c2": 1.5, "c3": 3, "c4": 4.5}
    X = pd.DataFrame({"a": y.map(centros) + rng.normal(0, 1, n), "b": rng.normal(size=n)})
    return X, y
