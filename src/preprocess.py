"""
Etapa 3: Preparación de datos: split, balanceo y selección de variables.
"""
from __future__ import annotations

import logging
import pandas as pd

from sklearn.feature_selection import (
    SelectKBest,
    f_classif
)

from sklearn.model_selection import (
    train_test_split
)

from src.load_data import TARGET
from src.features import FeatureEngineer

log = logging.getLogger("PREPROCESS")


DEFAULT_RANDOM_STATE = 42


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE
):

    log.info(
        'Realizando train test split | test_size=%.2f',
        test_size
    )

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    log.info(
        'Split completado | Train: %s | Test: %s',
        X_train.shape,
        X_test.shape
    )

    return X_train, X_test, y_train, y_test


def balancear_oversampling(
    df_train: pd.DataFrame,
    random_state: int = 1206
) -> pd.DataFrame:

    """
    Balancea el train replicando la clase minoritaria.
    """

    log.info(
        'Aplicando oversampling'
    )

    df_class_0 = df_train[
        df_train[TARGET] == 0
    ]

    df_class_1 = df_train[
        df_train[TARGET] == 1
    ]

    if df_class_0.empty or df_class_1.empty:

        log.error(
            'No se puede balancear: clase vacía'
        )

        raise ValueError(
            'No se puede balancear: alguna clase está vacía.'
        )

    log.info(
        'Distribución antes del balanceo | Clase 0: %d | Clase 1: %d',
        len(df_class_0),
        len(df_class_1)
    )

    df_class_1_over = df_class_1.sample(
        len(df_class_0),
        random_state=random_state,
        replace=True
    )

    df_balanceado = pd.concat(
        [df_class_0, df_class_1_over],
        axis=0
    ).sample(
        frac=1,
        random_state=random_state
    )

    log.info(
        'Oversampling completado | Shape final: %s',
        df_balanceado.shape
    )

    return df_balanceado


def balancear_undersampling(
    df_train: pd.DataFrame,
    random_state: int = 1206
) -> pd.DataFrame:

    """
    Balancea el train reduciendo la clase mayoritaria.
    """

    log.info(
        'Aplicando undersampling'
    )

    df_class_0 = df_train[
        df_train[TARGET] == 0
    ]

    df_class_1 = df_train[
        df_train[TARGET] == 1
    ]

    if df_class_0.empty or df_class_1.empty:

        log.error(
            'No se puede balancear: clase vacía'
        )

        raise ValueError(
            'No se puede balancear: alguna clase está vacía.'
        )

    df_class_0_under = df_class_0.sample(
        len(df_class_1),
        random_state=random_state
    )

    df_balanceado = pd.concat(
        [df_class_0_under, df_class_1],
        axis=0
    ).sample(
        frac=1,
        random_state=random_state
    )

    log.info(
        'Undersampling completado | Shape final: %s',
        df_balanceado.shape
    )

    return df_balanceado


def seleccionar_features(
    df_train_balanceado: pd.DataFrame,
    k: int = 5
) -> list[str]:

    """
    Selección de variables usando SelectKBest.
    """

    log.info(
        'Iniciando selección de features | k=%d',
        k
    )

    cols_prueba = [
        c
        for c in df_train_balanceado.columns
        if c.endswith("_cat")
    ]

    cols_prueba += [
        "Flag_hipertension",
        "Flag_problem_cardiaco"
    ]

    cols_prueba = [
        c
        for c in cols_prueba
        if c in df_train_balanceado.columns
    ]

    log.info(
        'Variables candidatas: %s',
        cols_prueba
    )

    X = df_train_balanceado[
        cols_prueba
    ]

    y = df_train_balanceado[
        TARGET
    ]

    k = min(k, X.shape[1])

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    selector.fit(X, y)

    selected = list(
        X.columns[
            selector.get_support()
        ]
    )

    log.info(
        'Features seleccionadas: %s',
        selected
    )

    return selected


def preparar_datasets(
    df: pd.DataFrame,
    balanceo: str = "over"
):

    """
    Split + feature engineering + balanceo + selección de variables.
    """

    log.info(
        'Iniciando preparación de datasets'
    )

    X_train_raw, X_test_raw, y_train, y_test = split_train_test(df)

    log.info(
        'Entrenando FeatureEngineer'
    )

    fe = FeatureEngineer()

    train_features = fe.fit_transform(
        X_train_raw
    )

    test_features = fe.transform(
        X_test_raw
    )

    log.info(
        'Feature engineering completado'
    )

    train_full = pd.concat(
        [
            train_features.reset_index(drop=True),
            y_train.reset_index(drop=True)
        ],
        axis=1
    )

    test_full = pd.concat(
        [
            test_features.reset_index(drop=True),
            y_test.reset_index(drop=True)
        ],
        axis=1
    )

    log.info(
        'Aplicando estrategia de balanceo: %s',
        balanceo
    )

    if balanceo == "over":

        train_model = balancear_oversampling(
            train_full
        )

    elif balanceo == "under":

        train_model = balancear_undersampling(
            train_full
        )

    elif balanceo == "none":

        train_model = train_full

    else:

        log.error(
            'Balanceo inválido: %s',
            balanceo
        )

        raise ValueError(
            "balanceo debe ser: 'over', 'under' o 'none'."
        )

    selected_features = seleccionar_features(
        train_model
    )

    X_train = train_model[
        selected_features
    ]

    y_train = train_model[
        TARGET
    ]

    X_test = test_full[
        selected_features
    ]

    y_test = test_full[
        TARGET
    ]

    log.info(
        'Datasets preparados correctamente'
    )

    log.info(
        'X_train shape: %s | X_test shape: %s',
        X_train.shape,
        X_test.shape
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        fe,
        selected_features
    )