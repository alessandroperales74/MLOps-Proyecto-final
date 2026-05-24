"""
Etapa 4: Entrenamiento del modelo final con MLflow + Optuna.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import optuna
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.load_data import cargar_datos
from src.preprocess import preparar_datasets
from src.evaluate import evaluar_modelo


log = logging.getLogger("TRAIN")


def construir_modelo(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    random_state: int = 42
) -> AdaBoostClassifier:

    """
    Construye el modelo AdaBoost.
    """

    log.info('Construyendo modelo AdaBoost')

    base_tree = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )

    try:

        model = AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    except TypeError:

        log.warning('Compatibilidad sklearn antigua detectada')

        model = AdaBoostClassifier(
            base_estimator=base_tree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    log.info('Modelo construido correctamente')

    return model


def objective(trial,X_train,y_train,X_test,y_test):

    """
    Función objetivo para Optuna.
    """

    n_estimators = trial.suggest_int(
        'n_estimators',
        50,
        300
    )

    learning_rate = trial.suggest_float(
        'learning_rate',
        0.01,
        0.3
    )

    max_depth = trial.suggest_int(
        'max_depth',
        2,
        10
    )

    model = construir_modelo(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )

    model.fit(X_train,y_train)

    metrics = evaluar_modelo(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    return metrics['auc_test']


def entrenar(ruta_data: str,artifact_dir: str = "artifacts") -> dict:

    """
    Ejecuta pipeline completo de entrenamiento.
    """

    log.info('Iniciando pipeline de entrenamiento')
    df = cargar_datos(ruta_data)
    log.info('Preparando datasets')

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_engineer,
        selected_features
    ) = preparar_datasets(
        df,
        balanceo="over"
    )

    log.info('Datasets preparados correctamente')
    log.info('Features seleccionadas: %s',selected_features)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True,exist_ok=True)

    log.info('Directorio de artifacts creado: %s',artifact_dir)

    mlflow.set_experiment('Prevencion_Ataque_Cardiaco')

    with mlflow.start_run():

        log.info('Iniciando optimización con Optuna')

        study = optuna.create_study(direction='maximize')

        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                X_test,
                y_test
            ),
            n_trials=10
        )

        best_params = study.best_params

        log.info('Mejores parámetros encontrados: %s',best_params)

        mlflow.log_params(best_params)

        model = construir_modelo(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth']
        )

        log.info('Entrenando modelo final')
        model.fit(X_train,y_train)
        log.info('Modelo entrenado correctamente')
        log.info('Evaluando modelo final')

        metrics = evaluar_modelo(
            model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        log.info('Registrando métricas en MLflow')

        metrics_sin_matriz = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        mlflow.log_metrics(metrics_sin_matriz)
        log.info('Registrando modelo en MLflow')
        mlflow.sklearn.log_model(model,"model")

        artifact = {
            "feature_engineer": feature_engineer,
            "selected_features": selected_features,
            "model": model,
            "threshold": 0.5,
        }

        model_path = artifact_dir / "model.joblib"
        metrics_path = artifact_dir / "metrics.json"

        log.info('Guardando modelo entrenado')

        joblib.dump(artifact,model_path)

        log.info('Modelo guardado en: %s',model_path)
        log.info('Guardando métricas')

        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8"
        )

        log.info('Métricas guardadas en: %s',metrics_path)
        log.info('Pipeline de entrenamiento finalizado correctamente')

        return {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "selected_features": selected_features,
            "metrics": metrics,
            "best_params": best_params,
        }