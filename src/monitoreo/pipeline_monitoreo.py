"""
Pipeline batch de monitoreo.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from src.monitoreo.drift import (cargar_datos,generar_reporte_drift,generar_reporte_calidad)

log = logging.getLogger("MONITOR")

REPORTS_DIR = Path("reportes")

DRIFT_THRESHOLD = 0.30


def ejecutar_monitoreo():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    log.info("Iniciando pipeline monitoreo")

    # cargar datasets
    df_ref, df_prod = cargar_datos()

    # drift
    drift_info = generar_reporte_drift(
        df_ref,
        df_prod
    )

    # calidad
    generar_reporte_calidad(df_ref,df_prod)

    # estado general
    estado = (
        "ALERTA"
        if drift_info["share_drifted"] > DRIFT_THRESHOLD
        else "OK"
    )

    resumen = {
        "timestamp": timestamp,
        "estado": estado,
        **drift_info,
    }

    REPORTS_DIR.mkdir(exist_ok=True)

    output_json = (REPORTS_DIR / f"{timestamp}_drift_summary.json")

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            resumen,
            f,
            indent=4,
            ensure_ascii=False
        )

    log.info("Resumen guardado: %s",output_json)

    if estado == "ALERTA":
        log.warning("Drift detectado sobre umbral")
    else:
        log.info("Monitoreo OK")

    return resumen


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | MONITOR | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ejecutar_monitoreo()