# Proyecto Final - MLOps: Riesgo de Ataque Cardiaco

Pipeline completo de Machine Learning orientado a predicción de ataque cardíaco, siguiendo buenas prácticas de MLOps: ingestión de datos, feature engineering, entrenamiento, evaluación y despliegue automatizado en Cloud Run.

# Estructura del Proyecto

```
.
├── artifacts/
│   └── metrics.json
│   └── model.joblib
├── data/
│   └── 2_DS_train_enf_corazon.csv.dvc
├── notebooks/
│   └── ataque_cardiaco.ipynb
├── src/
│   ├── load_data.py
│   ├── features.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_pipeline.py
├── requirements.txt
├── Dockerfile
├── Makefile
├── main.py
├── deploy.sh
├── params.yaml
└── README.md
```

# Objetivo del Proyecto

Este proyecto busca demostrar un flujo end-to-end de MLOps aplicando:

- Automatización
- Reproducibilidad
- Modularidad
- Buenas prácticas de despliegue
- Exposición de modelos mediante API

# 🌐 API Desplegada

La API se encuentra desplegada en Google Cloud Run:

- https://ataque-cardiaco-api-57413793137.us-central1.run.app

## Swagger UI

Documentación interactiva:

- https://ataque-cardiaco-api-57413793137.us-central1.run.app/docs