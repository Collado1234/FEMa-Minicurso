import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_classification_dataset(name, test_size=0.3, random_state=42, scale=True):
    """name em {'breast_cancer', 'digits', 'iris'}"""
    loaders = {"breast_cancer": load_breast_cancer, "digits": load_digits, "iris": load_iris}

    data = loaders[name]()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state= random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data.target_names if hasattr(data, "target_names") else None


def load_regression_dataset(name="diabetes", test_size=0.3, random_state=42, scale=True):
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
        raise ValueError("dataset de regressao desconhecido")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    return X_train, X_test, y_train, y_test