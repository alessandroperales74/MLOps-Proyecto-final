"""
Etapa 6: Métricas de evaluación del modelo.
"""
from __future__ import annotations

import logging
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

log = logging.getLogger("EVALUATE")


def evaluar_modelo(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    umbral: float = 0.5
) -> dict:

    log.info('Iniciando evaluación del modelo')

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    log.info('Probabilidades generadas')

    y_train_pred = (y_train_prob > umbral).astype(int)
    y_test_pred = (y_test_prob > umbral).astype(int)

    log.info('Predicciones binarias generadas con umbral %.2f', umbral)

    metrics = {
        "accuracy_train": accuracy_score(y_train, y_train_pred),
        "accuracy_test": accuracy_score(y_test, y_test_pred),

        "recall_train": recall_score(
            y_train,
            y_train_pred,
            zero_division=0
        ),

        "recall_test": recall_score(
            y_test,
            y_test_pred,
            zero_division=0
        ),

        "precision_train": precision_score(
            y_train,
            y_train_pred,
            zero_division=0
        ),

        "precision_test": precision_score(
            y_test,
            y_test_pred,
            zero_division=0
        ),

        "f1_train": f1_score(
            y_train,
            y_train_pred,
            zero_division=0
        ),

        "f1_test": f1_score(
            y_test,
            y_test_pred,
            zero_division=0
        ),

        "auc_train": roc_auc_score(
            y_train,
            y_train_prob
        ),

        "auc_test": roc_auc_score(
            y_test,
            y_test_prob
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            y_test_pred,
            labels=[0, 1]
        ).tolist(),
    }

    log.info(
        'Modelo evaluado correctamente | AUC TEST: %.4f | F1 TEST: %.4f',
        metrics['auc_test'],
        metrics['f1_test']
    )

    return metrics


def metricas_a_dataframe(metrics: dict) -> pd.DataFrame:

    log.info('Convirtiendo métricas a DataFrame')

    clean = {
        k: v
        for k, v in metrics.items()
        if k != "confusion_matrix"
    }

    return pd.DataFrame([clean])