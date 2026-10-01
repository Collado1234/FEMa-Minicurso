"""
fema.py
-------
FEMa (Finite Element Machine) — versão enxuta.

Ideia: igual ao KNN ponderado do Dia 1 (peso = 1/distância), mas a FORMA do peso
agora é escolhida entre 5 funções de base diferentes — essa escolha é o próprio
objeto de pesquisa do pôster do CIC.

    peso = funcao_de_base(distancia, **parametros)
    previsão = média dos rótulos dos k vizinhos, ponderada pelo peso

Só busca por força bruta (compara com todos os pontos de treino) — é a mesma
limitação do KNN. Estruturas espaciais (KD-Tree/Ball-Tree) ficariam aqui como
próximo passo, acelerando essa busca sem mudar mais nada no resto do código.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

EPS = 1e-8  # evita divisão por zero quando a distância é 0

# As 5 funções de base investigadas no pôster do CIC.
# Cada uma recebe um array de distâncias e devolve os pesos correspondentes.
BASIS_FUNCTIONS = {
    "shepard": lambda d, z=2.0: 1.0 / (d ** z + EPS),
    "wendland_c2": lambda d, h=5.0: (np.clip(1 - d / h, 0, None) ** 4) * (4 * d / h + 1),
    "laplacian": lambda d, eps=1.0: np.exp(-eps * d),
    "inverse_multiquadratic": lambda d, c=1.0: 1.0 / np.sqrt(d ** 2 + c ** 2),
    "radial": lambda d, z=1.0: np.exp(-(d ** 2) / (2 * z ** 2 + EPS)),
}


def pairwise_distance(X, Y):
    """Distância euclidiana entre todos os pontos de X e todos os pontos de Y."""
    X2 = np.sum(X ** 2, axis=1)[:, None]
    Y2 = np.sum(Y ** 2, axis=1)[None, :]
    return np.sqrt(np.maximum(X2 - 2 * X @ Y.T + Y2, 0))


class _FEMaBase(BaseEstimator):
    def __init__(self, basis="shepard", basis_kwargs=None, k=15):
        self.basis = basis
        self.basis_kwargs = basis_kwargs
        self.k = k

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _neighbor_weights(self, X):
        """Para cada ponto de X, acha os k vizinhos mais próximos (força bruta)
        e devolve (pesos, índices_dos_vizinhos)."""
        X = np.asarray(X, dtype=float)
        D = pairwise_distance(X, self.X_train_)
        k = D.shape[1] if self.k is None else min(self.k, D.shape[1])
        idx = np.argsort(D, axis=1)[:, :k]
        dist = np.take_along_axis(D, idx, axis=1)

        basis_fn = BASIS_FUNCTIONS[self.basis]
        kwargs = self.basis_kwargs or {}
        pesos = basis_fn(dist, **kwargs)
        return pesos, idx


class FEMaClassifier(_FEMaBase, ClassifierMixin):
    def predict_proba(self, X):
        pesos, idx = self._neighbor_weights(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            for cls, w in zip(self.y_train_[idx[i]], pesos[i]):
                proba[i, np.where(self.classes_ == cls)[0][0]] += w
            soma = proba[i].sum()
            proba[i] = proba[i] / soma if soma > 0 else 1.0 / len(self.classes_)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class FEMaRegressor(_FEMaBase, RegressorMixin):
    def predict(self, X):
        pesos, idx = self._neighbor_weights(X)
        preds = np.zeros(len(X))
        for i in range(len(X)):
            soma_pesos = pesos[i].sum()
            preds[i] = np.average(self.y_train_[idx[i]], weights=pesos[i]) if soma_pesos > 0 else self.y_train_[idx[i]].mean()
        return preds
