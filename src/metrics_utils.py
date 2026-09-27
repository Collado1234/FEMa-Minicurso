"""
metrics_utils.py
----------------
Wrappers finos em torno do sklearn.metrics, retornando tudo já organizado
em um dicionário/tabela — usados no notebook 04 e na comparação do notebook 05.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)


def classification_report_dict(y_true, y_pred, y_proba=None, average="binary", pos_label=None):
    out = {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Acurácia balanceada": balanced_accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                out["AUC-ROC"] = roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                out["AUC-ROC"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except ValueError:
            out["AUC-ROC"] = np.nan
    return out


def regression_report_dict(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R²": r2_score(y_true, y_pred),
    }


def confusion(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return confusion_matrix(y_true, y_pred, labels=labels), labels


def results_table(results_dict):
    """results_dict: {"nome_do_modelo": {"Métrica": valor, ...}, ...} -> DataFrame arrumado."""
    return pd.DataFrame(results_dict).T.round(4)
