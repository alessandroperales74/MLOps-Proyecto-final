"""
drift.py — Generación de reportes de drift con Evidently AI.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
from evidently import Report
from evidently.presets import (DataDriftPreset,DataSummaryPreset)

log = logging.getLogger("DRIFT")

DATA_PATH = Path("data/2_DS_train_enf_corazon.csv")
REPORTS_DIR = Path("reportes")

FEATURES = [
    "Edad",
    "Promedio_nivel_glucosa",
    "IMC",
    "Flag_hipertension",
    "Flag_problem_cardiaco",
]


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Divide el dataset en referencia vs producción simulada.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH,sep=';')

    log.info("Dataset cargado | Shape: %s",df.shape)

    # referencia histórica
    df_ref = df.iloc[:30000].copy()

    # producción simulada
    df_prod = df.iloc[30000:].copy()

    # inducir drift artificial ya que no tenemos un dataset de nuevos datos
    df_prod["Edad"] = df_prod["Edad"] + 10

    log.info(
        "Reference: %d filas | Current: %d filas",
        len(df_ref),
        len(df_prod)
    )

    return df_ref, df_prod


def generar_reporte_drift(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame
) -> dict:

    """
    Genera reporte principal de drift.
    """

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        reference_data=df_ref[FEATURES],
        current_data=df_prod[FEATURES],
    )

    REPORTS_DIR.mkdir(
        exist_ok=True
    )

    output_path = (
        REPORTS_DIR /
        "01_data_drift.html"
    )

    snapshot.save_html(
        str(output_path)
    )

    log.info(
        "Reporte drift generado: %s",
        output_path
    )

    resultado = snapshot.dict()

    resumen = resultado["metrics"][0]["value"]

    drift_info = {
        "drift_detectado":
            resumen.get("count", 0) > 0,

        "features_con_drift":
            int(resumen.get("count", 0)),

        "share_drifted":
            round(resumen.get("share", 0.0), 4),

        "total_features":
            len(FEATURES),
    }

    return drift_info


def generar_reporte_calidad(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame
):

    """
    Genera reporte de calidad.
    """

    report = Report(
        metrics=[
            DataSummaryPreset()
        ]
    )

    snapshot = report.run(
        reference_data=df_ref[FEATURES],
        current_data=df_prod[FEATURES],
    )

    output_path = (
        REPORTS_DIR /
        "02_data_quality.html"
    )

    snapshot.save_html(str(output_path))

    log.info(
        "Reporte calidad generado: %s",
        output_path
    )


def ejecutar_monitoreo():

    """
    Pipeline principal de monitoreo.
    """

    REPORTS_DIR.mkdir(exist_ok=True)

    log.info("Iniciando monitoreo batch")

    df_ref, df_prod = cargar_datos()

    drift_info = generar_reporte_drift(
        df_ref,
        df_prod
    )

    generar_reporte_calidad(df_ref,df_prod)

    resumen_path = (
        REPORTS_DIR /
        "drift_summary.json"
    )

    with open(
        resumen_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            drift_info,
            f,
            indent=4,
            ensure_ascii=False
        )

    log.info(
        "Resumen drift guardado: %s",
        resumen_path
    )

    log.info(
        "Drift detectado: %s",
        drift_info["drift_detectado"]
    )

    log.info(
        "Features con drift: %d/%d",
        drift_info["features_con_drift"],
        drift_info["total_features"]
    )

    log.info(
        "Monitoreo finalizado correctamente"
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | DRIFT | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ejecutar_monitoreo()