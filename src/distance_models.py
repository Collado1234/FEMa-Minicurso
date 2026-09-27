import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy import stats


def euclidean_distance(a,b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.sqrt(np.sum((a - b)**2))


def knn_majority_predict_one(X_train, y_train, x_query, k):
    """KNN classico (voto majoritario por consulta), todos k vizinhos possuem peso igual """

    distancias = np.array([euclidean_distance(x_query, xi) for xi in X_train])

    indices_k = np.argsort(distancias)[:k]
    rotulos_vizinhos = y_train[indices_k]

    valores, contagens = np.unique(rotulos_vizinhos, return_counts=True)
    return valores[np.argmax(contagens)]


def knn_weighted_predict_one(X_train, y_train, x_query, k, p=1, eps=1e-8):
    """ Knn Ponderado pela distância para um único ponto de consulta"""
    distancias = np.array([euclidean_distance(x_query, xi) for xi in X_train])
    indices_k = np.argsort(distancias)[:k]
    pesos = 1.0 / (distancias[indices_k] ** p + eps)

    votos = {}
    for idx, peso in zip(indices_k, pesos):
        classe = y_train[idx]
        votos[classe] = votos.get(classe, 0.0) + peso
    return max(votos, key=votos.get)

def _pairwise_distances(X, Y, metric="euclidean"):
    """Matriz de distâncias (n_X, n_Y) entre duas matrizes de pontos."""
    if metric == "euclidean":
        # (a-b)^2 = a^2 - 2ab + b^2, evita loop explícito
        X2 = np.sum(X ** 2, axis=1)[:, None]
        Y2 = np.sum(Y ** 2, axis=1)[None, :]
        d2 = X2 - 2 * X @ Y.T + Y2
        return np.sqrt(np.maximum(d2, 0))
    elif metric == "manhattan":
        return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    else:
        raise ValueError(f"Métrica desconhecida: {metric}")


class BaseDistanceModel(BaseEstimator):
    """Classe-base comum a todo modelo baseado em distância/interpolação.

    Guarda o essencial (dados de treino, métrica) e define o "contrato" que
    qualquer modelo baseado em vizinhança/interpolação deve seguir:

        fit(X, y)          -> guarda os dados de treino (nenhum ajuste de peso aqui)
        _weights(D)         -> dado um vetor de distâncias, devolve os pesos de cada vizinho
        predict(X)          -> usa os pesos para combinar os rótulos/vizinhos

    O FEMa é, na essência, um outro jeito de calcular `_weights` (via função de base
    de interpolação, não só 1/distância) — por isso a base já foi desenhada em torno
    desse método. Uma futura `FEMaClassifier` deve poder herdar de BaseDistanceModel,
    reimplementar apenas `_weights`/`predict`, e reaproveitar todo o resto.
    """

    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _kneighbors(self, X):
        X = np.asarray(X, dtype=float)
        D = _pairwise_distances(X, self.X_train_, metric=self.metric)
        idx = np.argsort(D, axis=1)[:, : self.k]
        dist = np.take_along_axis(D, idx, axis=1)
        return dist, idx


class KNNClassifier(BaseDistanceModel, ClassifierMixin):
    """KNN de classificação, com dois modos de voto.

    weighting='majority' -> votação majoritária (todos os vizinhos pesam igual)
    weighting='distance' -> ponderado por 1/distancia**p
    """

    def __init__(self, k=5, weighting="majority", p=1, eps=1e-8, metric="euclidean"):
        super().__init__(k=k, metric=metric)
        self.weighting = weighting
        self.p = p
        self.eps = eps

    def predict(self, X):
        dist, idx = self._kneighbors(X)
        preds = []
        for row_dist, row_idx in zip(dist, idx):
            vizinhos_y = self.y_train_[row_idx]
            if self.weighting == "majority":
                valores, contagens = np.unique(vizinhos_y, return_counts=True)
                preds.append(valores[np.argmax(contagens)])
            elif self.weighting == "distance":
                pesos = 1.0 / (row_dist ** self.p + self.eps)
                votos = {}
                for cls, w in zip(vizinhos_y, pesos):
                    votos[cls] = votos.get(cls, 0.0) + w
                preds.append(max(votos, key=votos.get))
            else:
                raise ValueError("weighting deve ser 'majority' ou 'distance'")
        return np.array(preds)

    def predict_proba(self, X):
        """Necessário para AUC-ROC — estima probabilidade pela fração/peso dos vizinhos."""
        dist, idx = self._kneighbors(X)
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i, (row_dist, row_idx) in enumerate(zip(dist, idx)):
            vizinhos_y = self.y_train_[row_idx]
            if self.weighting == "majority":
                pesos = np.ones_like(row_dist)
            else:
                pesos = 1.0 / (row_dist ** self.p + self.eps)
            for cls, w in zip(vizinhos_y, pesos):
                c_idx = np.where(self.classes_ == cls)[0][0]
                proba[i, c_idx] += w
            proba[i] /= proba[i].sum()
        return proba


class KNNRegressor(BaseDistanceModel, RegressorMixin):
    """KNN de regressão: média (majoritário) ou média ponderada pela distância."""

    def __init__(self, k=5, weighting="majority", p=1, eps=1e-8, metric="euclidean"):
        super().__init__(k=k, metric=metric)
        self.weighting = weighting
        self.p = p
        self.eps = eps

    def predict(self, X):
        dist, idx = self._kneighbors(X)
        preds = []
        for row_dist, row_idx in zip(dist, idx):
            vizinhos_y = self.y_train_[row_idx]
            if self.weighting == "majority":
                preds.append(np.mean(vizinhos_y))
            else:
                pesos = 1.0 / (row_dist ** self.p + self.eps)
                preds.append(np.average(vizinhos_y, weights=pesos))
        return np.array(preds)
