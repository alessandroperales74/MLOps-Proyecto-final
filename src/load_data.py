"""
Etapa 1: Carga y validación del dataset de prevención de ataque cardiaco.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("DATA")


TARGET = "Ataque_cardiaco"

ID_COL = "ID"


COLUMNAS_REQUERIDAS = {
    "ID",
    "Genero",
    "Edad",
    "Flag_hipertension",
    "Flag_problem_cardiaco",
    "Estados_civil",
    "Tipo_trabajo",
    "Zona_residencia",
    "Promedio_nivel_glucosa",
    "IMC",
    "Flag_fumador",
    "Ataque_cardiaco"
}


def cargar_datos(
    ruta: str | Path,
    sep: str = ";"
) -> pd.DataFrame:

    """
    Carga el CSV y valida columnas mínimas.
    """

    ruta = Path(ruta)

    log.info(
        'Iniciando carga de datos'
    )

    if not ruta.exists():

        log.error(
            'Archivo no encontrado: %s',
            ruta
        )

        raise FileNotFoundError(
            f"No existe el archivo: {ruta}"
        )

    log.info(
        'Leyendo archivo desde: %s',
        ruta
    )

    df = pd.read_csv(
        ruta,
        sep=sep
    )

    log.info(
        'Dataset cargado correctamente | Shape: %s',
        df.shape
    )

    validar_columnas(df)

    log.info(
        'Validación de columnas completada'
    )

    resumen = resumen_calidad(df)

    log.info(
        'Duplicados encontrados: %d',
        resumen['duplicados']
    )

    log.info(
        'Distribución target: %s',
        resumen['target_distribution']
    )

    log.info(
        'Carga de datos finalizada correctamente'
    )

    return df


def validar_columnas(df: pd.DataFrame) -> None:

    """
    Valida que el dataframe tenga las columnas esperadas.
    """

    log.info(
        'Validando columnas requeridas'
    )

    faltantes = (
        COLUMNAS_REQUERIDAS
        - set(df.columns)
    )

    if faltantes:

        log.error(
            'Faltan columnas requeridas: %s',
            faltantes
        )

        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    log.info(
        'Todas las columnas requeridas están presentes'
    )


def resumen_calidad(
    df: pd.DataFrame
) -> dict:

    """
    Retorna un resumen básico de calidad.
    """

    log.info(
        'Calculando resumen de calidad'
    )

    resumen = {
        "filas": int(df.shape[0]),

        "columnas": int(df.shape[1]),

        "duplicados": int(
            df.duplicated().sum()
        ),

        "nulos_por_columna": df.isna().sum().to_dict(),

        "target_distribution": df[
            TARGET
        ].value_counts(
            dropna=False
        ).to_dict(),
    }

    log.info(
        'Resumen de calidad generado'
    )

    return resumen