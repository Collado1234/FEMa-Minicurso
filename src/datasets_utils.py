"""
datasets_utils.py
-----------------
Carregadores de dataset padronizados: sempre devolvem (X_train, X_test, y_train, y_test)
já normalizados (z-score), prontos para qualquer modelo baseado em distância.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_classification_dataset(name, test_size=0.3, random_state=42, scale=True):
    """name em {'breast_cancer', 'digits', 'iris', 'wine'}"""
    loaders = {"breast_cancer": load_breast_cancer, "digits": load_digits, "iris": load_iris, "wine": load_wine}
    data = loaders[name]()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, data.target_names if hasattr(data, "target_names") else None


def load_regression_dataset(name="diabetes", test_size=0.3, random_state=42, scale=True):
    """name em {'diabetes', 'synthetic_nonlinear'}

    'california_housing' exigiria baixar dados da internet (fetch_california_housing),
    o que não funciona neste ambiente sandbox sem acesso à rede — por isso o segundo
    dataset é sintético (gerado localmente), mas serve igualmente bem para comparar
    KNN vs. Regressão Linear em características diferentes (aqui, uma relação não-linear
    entre X e y, onde o KNN tende a se sair relativamente melhor que o modelo linear).
    Se você tiver acesso à internet no seu próprio ambiente, pode trocar por
    `fetch_california_housing()` sem mudar o resto do notebook.
    """
    if name == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
    elif name == "synthetic_nonlinear":
        rng = np.random.RandomState(random_state)
        X, y_linear = make_regression(
            n_samples=600, n_features=6, n_informative=4, noise=8.0, random_state=random_state
        )
        # adiciona uma componente não-linear proposital (interação + termo quadrático)
        y = y_linear + 15 * np.sin(X[:, 0]) + 5 * (X[:, 1] ** 2) + rng.normal(0, 3, size=len(y_linear))
    else:
        raise ValueError("dataset de regressão desconhecido")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def load_wine_as_dataframe():
    """Devolve o dataset Wine completo como um pandas DataFrame (features + coluna 'target'),
    útil para a etapa de EDA do estudo de caso — antes de qualquer split/normalização."""
    import pandas as pd
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["target_name"] = df["target"].map(dict(enumerate(data.target_names)))
    return df, data


def to_2d(X, method="pca", random_state=42):
    """Reduz X para 2 dimensões (só para poder desenhar fronteira de decisão)."""
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(X)
    raise ValueError("método de redução desconhecido")


def make_scale_mismatch_demo(n=40, random_state=7):
    """Dataset sintético 2D com escalas bem diferentes (ex: idade x renda),
    usado no notebook de normalização."""
    rng = np.random.RandomState(random_state)
    idade_a = rng.normal(30, 5, n // 2)
    idade_b = rng.normal(45, 5, n // 2)
    renda_a = rng.normal(3000, 800, n // 2)
    renda_b = rng.normal(9000, 1500, n // 2)
    X = np.column_stack([
        np.concatenate([idade_a, idade_b]),
        np.concatenate([renda_a, renda_b]),
    ])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    perm = rng.permutation(n)
    return X[perm], y[perm]
