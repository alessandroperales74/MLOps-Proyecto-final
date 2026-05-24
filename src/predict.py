"""
Etapa 5: Predicción usando el artifact entrenado.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import logging
import pandas as pd

log = logging.getLogger("PREDICT")

def cargar_artifact(
    model_path: str | Path = "artifacts/model.joblib"
) -> dict:

    """
    Carga artifact serializado del modelo.
    """

    path = Path(model_path)

    log.info(
        'Cargando artifact desde: %s',
        path
    )

    if not path.exists():

        log.error(
            'No existe el artifact: %s',
            path
        )

        raise FileNotFoundError(
            f"No existe el modelo entrenado: {path}"
        )

    artifact = joblib.load(path)

    log.info(
        'Artifact cargado correctamente'
    )

    return artifact


def predecir(
    df_raw: pd.DataFrame,
    model_path: str | Path = "artifacts/model.joblib"
) -> pd.DataFrame:

    """
    Genera predicciones usando el modelo entrenado.
    """

    log.info(
        'Iniciando proceso de predicción'
    )

    artifact = cargar_artifact(model_path)

    fe = artifact["feature_engineer"]

    selected_features = artifact[
        "selected_features"
    ]

    model = artifact["model"]

    threshold = artifact.get(
        "threshold",
        0.5
    )

    log.info(
        'Aplicando feature engineering'
    )

    df_features = fe.transform(df_raw)

    log.info(
        'Seleccionando features finales'
    )

    X = df_features[selected_features]

    log.info(
        'Generando probabilidades'
    )

    proba = model.predict_proba(X)[:, 1]

    log.info(
        'Aplicando threshold %.2f',
        threshold
    )

    pred = (
        proba > threshold
    ).astype(int)

    resultado = pd.DataFrame({
        "probabilidad_ataque_cardiaco": proba,
        "prediccion_ataque_cardiaco": pred,
    })

    log.info(
        'Predicciones generadas correctamente | Registros: %d',
        resultado.shape[0]
    )

    return resultado