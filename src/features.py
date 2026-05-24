"""
Etapa 2: Feature engineering tomado del notebook original.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("FEATURES")

RAW_FEATURES = [
    "Genero",
    "Edad",
    "Flag_hipertension",
    "Flag_problem_cardiaco",
    "Estados_civil",
    "Tipo_trabajo",
    "Zona_residencia",
    "Promedio_nivel_glucosa",
    "IMC",
    "Flag_fumador"
]


TIPO_TRABAJO_MAP = {
    "cuidar_ninos": 1,
    "Nunca_trabajo": 1,
    "Empresa_privada": 2,
    "En_gobierno": 2,
    "Emprendedor": 3,
}


COLS_A_ESCALAR = [
    "Edad",
    "Promedio_nivel_glucosa",
    "IMC"
]


def categorizar_edad(edad: float) -> int:

    if edad < 35:
        return 1

    if edad < 50:
        return 2

    if edad < 66:
        return 3

    return 4


def categorizar_imc(imc: float) -> int:

    if imc < 25:
        return 1

    if imc < 30:
        return 2

    return 4


class FeatureEngineer(BaseEstimator, TransformerMixin):

    """
    Transformador sklearn compatible con Pipeline.
    """

    def __init__(self):

        self.mediana_imc_: float | None = None

        self.clip_limits_: dict[str, tuple[float, float]] = {}

        self.scaler_: StandardScaler | None = None

    def fit(self, X: pd.DataFrame, y=None):

        log.info('Entrenando FeatureEngineer')

        df = X.copy()

        log.info('Calculando mediana de IMC')

        self.mediana_imc_ = float(
            df["IMC"].median()
        )

        log.info(
            'Mediana IMC calculada: %.2f',
            self.mediana_imc_
        )

        df["IMC"] = df["IMC"].fillna(
            self.mediana_imc_
        )

        log.info('Calculando límites de clipping')

        for col in [
            "Promedio_nivel_glucosa",
            "IMC"
        ]:

            self.clip_limits_[col] = (
                float(df[col].quantile(0.05)),
                float(df[col].quantile(0.95)),
            )

            df[col] = df[col].clip(
                *self.clip_limits_[col]
            )

            log.info(
                'Clipping %s | min=%.2f | max=%.2f',
                col,
                self.clip_limits_[col][0],
                self.clip_limits_[col][1]
            )

        log.info('Entrenando StandardScaler')
        self.scaler_ = StandardScaler()
        self.scaler_.fit(df[COLS_A_ESCALAR])
        log.info('Scaler entrenado para columnas: %s',COLS_A_ESCALAR)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        log.info(
            'Aplicando transformaciones de features'
        )

        if (
            self.mediana_imc_ is None
            or self.scaler_ is None
        ):

            raise RuntimeError(
                'FeatureEngineer debe ejecutarse con fit antes de transform.'
            )

        df = X.copy()

        log.info('Eliminando ID')

        df = df.drop(
            columns=["ID"],
            errors="ignore"
        )

        log.info(
            'Imputando valores nulos'
        )

        df["IMC"] = df["IMC"].fillna(
            self.mediana_imc_
        )

        df["Flag_fumador"] = df[
            "Flag_fumador"
        ].fillna("No se sabe")

        log.info('Aplicando clipping de outliers')

        for col, limits in self.clip_limits_.items():

            df[col] = df[col].clip(*limits)

        log.info('Generando variables logarítmicas y sqrt')

        df["Edad_log"] = np.log1p(df["Edad"])
        df["Edad_sqrt"] = np.sqrt(df["Edad"])

        df["IMC_log"] = np.log1p(df["IMC"])
        df["IMC_sqrt"] = np.sqrt(df["IMC"])

        df["Promedio_nivel_glucosa_log"] = np.log1p(
            df["Promedio_nivel_glucosa"]
        )

        df["Promedio_nivel_glucosa_sqrt"] = np.sqrt(
            df["Promedio_nivel_glucosa"]
        )

        log.info('Generando variables categóricas')

        df["Estados_civil_cat"] = np.where(
            df["Estados_civil"] == "Si",
            1,
            0
        )

        df["Zona_residencia_Urbano"] = np.where(
            df["Zona_residencia"] == "Urbano",
            1,
            0
        )

        df["Zona_residencia_Rural"] = np.where(
            df["Zona_residencia"] == "Rural",
            1,
            0
        )

        log.info('Generando variables de fumador')

        df["Flag_Fumador_Nunca_fuma"] = np.where(
            df["Flag_fumador"] == "Nunca_fuma",
            1,
            0
        )

        df["Flag_Fumador_Antes_fumaba"] = np.where(
            df["Flag_fumador"] == "antes_fumaba",
            1,
            0
        )

        df["Flag_Fumador_Fumador"] = np.where(
            df["Flag_fumador"] == "fumador",
            1,
            0
        )

        df["Flag_Fumador_No_se_sabe"] = np.where(
            df["Flag_fumador"] == "No se sabe",
            1,
            0
        )

        log.info('Generando variables de género y trabajo')

        df["Genero_M"] = np.where(
            df["Genero"] == "Hombre",
            1,
            0
        )

        df["Genero_F"] = np.where(
            df["Genero"] != "Hombre",
            1,
            0
        )

        df["Tipo_trabajo_cat"] = df[
            "Tipo_trabajo"
        ].replace(
            TIPO_TRABAJO_MAP
        ).infer_objects(copy=False)

        df["Tipo_trabajo_cat"] = pd.to_numeric(
            df["Tipo_trabajo_cat"],
            errors="coerce"
        ).fillna(0).astype(int)

        log.info('Generando variables categorizadas')
        df["Edad_cat"] = df["Edad"].apply(categorizar_edad)
        df["IMC_cat"] = df["IMC"].apply(categorizar_imc)

        df["Promedio_nivel_glucosa_cat"] = np.where(
            df["Promedio_nivel_glucosa"] < 165,
            0,
            1
        )

        log.info('Aplicando StandardScaler')

        scaled = self.scaler_.transform(df[COLS_A_ESCALAR])

        scaled_df = pd.DataFrame(
            scaled,
            columns=[
                f"{c}_scaled"
                for c in COLS_A_ESCALAR
            ],
            index=df.index,
        )

        df = pd.concat([df, scaled_df],axis=1)

        log.info('Feature engineering completado | Shape final: %s',df.shape)

        return df