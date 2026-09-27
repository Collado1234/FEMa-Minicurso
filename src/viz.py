"""
viz.py
------
Funções de visualização reutilizadas em todos os notebooks. Funcionam com
QUALQUER modelo que implemente `.fit(X, y)` / `.predict(X)` no padrão
scikit-learn — incluindo o futuro FEMaClassifier.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PALETTE = ["#1C7293", "#D64550", "#E8871E", "#5B6B85"]


def plot_decision_boundary(model, X, y, ax=None, title=None, resolution=180, alpha=0.25):
    """Treina `model` em (X, y) [2 features] e desenha a fronteira de decisão.

    Usa um número FIXO de pontos por eixo (`resolution`), em vez de um passo fixo,
    para que o grid funcione tanto com dados padronizados (escala ~[-3, 3]) quanto
    com dados em escala bruta (ex: renda em milhares) sem estourar memória.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    model.fit(X, y)

    margin_x = (X[:, 0].max() - X[:, 0].min()) * 0.05 + 1e-6
    margin_y = (X[:, 1].max() - X[:, 1].min()) * 0.05 + 1e-6
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid)
    classes = np.unique(y)
    class_to_int = {c: i for i, c in enumerate(classes)}
    Z_int = np.array([class_to_int[z] for z in Z]).reshape(xx.shape)

    cmap_bg = ListedColormap(PALETTE[: len(classes)])
    ax.contourf(xx, yy, Z_int, alpha=alpha, cmap=cmap_bg, levels=np.arange(len(classes) + 1) - 0.5)

    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=PALETTE[i], edgecolor="white", s=45, linewidth=0.8, label=str(c))

    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return ax


def plot_weighted_scatter(X, y, query_point, k, weights=None, ax=None, title=None):
    """Mostra o ponto de consulta, seus vizinhos e (opcionalmente) o peso de cada um
    como o tamanho do marcador — ilustra a diferença majoritário vs. ponderado."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    classes = np.unique(y)
    for i, c in enumerate(classes):
        mask = y == c
        sizes = 60 if weights is None else 60 + 300 * (weights[mask] / weights.max())
        ax.scatter(X[mask, 0], X[mask, 1], c=PALETTE[i], s=sizes, edgecolor="white", linewidth=0.8, label=str(c))

    ax.scatter(*query_point, c="#8A94A6", s=140, edgecolor="black", linewidth=1.5, marker="o", zorder=5, label="consulta")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return ax


def plot_confusion_matrix(cm, class_names, ax=None, title="Matriz de Confusão"):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title(title, fontsize=12, fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax


def plot_residuals(y_true, y_pred, ax=None, title="Resíduos"):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    order = np.argsort(y_true)
    yt, yp = np.asarray(y_true)[order], np.asarray(y_pred)[order]
    x = np.arange(len(yt))
    ax.plot(x, yt, color="#E8871E", linewidth=2, label="valor real (ordenado)")
    ax.scatter(x, yp, color=PALETTE[0], s=25, label="previsão", zorder=3)
    for xi, a, b in zip(x, yt, yp):
        ax.plot([xi, xi], [a, b], color="#D64550", linewidth=0.8, alpha=0.6)
    ax.set_xticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return ax
