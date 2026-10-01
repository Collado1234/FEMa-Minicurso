"""
distance_models.py
-------------------
Implementações "do zero" de classificadores/regressores baseados em distância.

Arquitetura pensada para o FEMa entrar depois com o mínimo de atrito:
- Toda classe herda de BaseEstimator (+ ClassifierMixin/RegressorMixin) do scikit-learn.
- Isso significa que GridSearchCV, RandomizedSearchCV, cross_val_score, e as funções
  de plot de fronteira de decisão em viz.py funcionam automaticamente com QUALQUER
  classe daqui, sem precisar adaptar nada.
- Quando formos implementar o FEMa, basta criar `FEMaClassifier(BaseEstimator, ClassifierMixin)`
  com a mesma assinatura de fit/predict — todos os notebooks (tuning, métricas, comparação)
  vão funcionar sem alteração, só trocando o modelo usado.

As funções "puras" (knn_majority_predict_one, knn_weighted_predict_one) ficam expostas
separadamente porque são as que aparecem no slide "Implementar o KNN do zero" — são
mais fáceis de ler/explicar ao vivo do que a versão vetorizada usada dentro das classes.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy import stats


# ---------------------------------------------------------------------------
# 1) Funções "puras", ponto a ponto — para mostrar/explicar ao vivo no minicurso
# ---------------------------------------------------------------------------

def euclidean_distance(a, b):
    """Distância euclidiana entre dois vetores 1D."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2))


def knn_majority_predict_one(X_train, y_train, x_query, k):
    """KNN clássico (votação majoritária) para um único ponto de consulta.

    Todos os k vizinhos mais próximos pesam igual no voto.
    """
    distancias = np.array([euclidean_distance(x_query, xi) for xi in X_train])
    indices_k = np.argsort(distancias)[:k]
    rotulos_vizinhos = y_train[indices_k]
    valores, contagens = np.unique(rotulos_vizinhos, return_counts=True)
    return valores[np.argmax(contagens)]


def knn_weighted_predict_one(X_train, y_train, x_query, k, p=1, eps=1e-8):
    """KNN ponderado pela distância para um único ponto de consulta.

    Cada vizinho vota com peso 1/distancia**p — vizinhos mais próximos
    pesam mais que vizinhos mais distantes.
    """
    distancias = np.array([euclidean_distance(x_query, xi) for xi in X_train])
    indices_k = np.argsort(distancias)[:k]
    pesos = 1.0 / (distancias[indices_k] ** p + eps)

    votos = {}
    for idx, peso in zip(indices_k, pesos):
        classe = y_train[idx]
        votos[classe] = votos.get(classe, 0.0) + peso
    return max(votos, key=votos.get)


# ---------------------------------------------------------------------------
# 2) Versões vetorizadas + wrappers estilo scikit-learn (usadas no resto do curso)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3) Onde o FEMa vai entrar (deixe este bloco como referência / esqueleto)
# ---------------------------------------------------------------------------
#
# class FEMaClassifier(BaseDistanceModel, ClassifierMixin):
#     """Esqueleto para o FEMa — mesma interface do KNNClassifier acima.
#
#     A diferença central: em vez de pesos 1/distancia**p, o FEMa usa uma
#     função de base (Shepard, Wendland C2, Laplaciana, Inverse Multiquadratic,
#     Radial...) para ponderar a influência de cada ponto de treino.
#     """
#
#     def __init__(self, basis="shepard", **basis_kwargs):
#         super().__init__(k=None, metric="euclidean")  # FEMa tipicamente usa todos os pontos
#         self.basis = basis
#         self.basis_kwargs = basis_kwargs
#
#     def _weights(self, dist):
#         # aqui entra a função de base escolhida (Shepard, Wendland C2, etc.)
#         raise NotImplementedError
#
#     def predict(self, X):
#         raise NotImplementedError
#
# Assim que essa classe existir, ela já pode ser usada em:
#   - 03_tuning_grid_random.ipynb   (GridSearchCV/RandomizedSearchCV)
#   - 04_metricas.ipynb             (accuracy, F1, MAE, RMSE, etc.)
#   - 05_comparacao_modelos_lineares.ipynb (comparação com modelos lineares)
# sem precisar mudar mais nada além de adicionar "FEMa": FEMaClassifier() nos dicts de modelos.
