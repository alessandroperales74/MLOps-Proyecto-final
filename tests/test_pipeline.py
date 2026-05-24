import pandas as pd
from src.features import FeatureEngineer


def test_feature_engineer_crea_columnas():
    df = pd.DataFrame({
        "Genero": ["Hombre", "Mujer"],
        "Edad": [30, 70],
        "Flag_hipertension": [0, 1],
        "Flag_problem_cardiaco": [0, 1],
        "Estados_civil": ["No", "Si"],
        "Tipo_trabajo": ["Empresa_privada", "Emprendedor"],
        "Zona_residencia": ["Urbano", "Rural"],
        "Promedio_nivel_glucosa": [90.0, 190.0],
        "IMC": [None, 31.0],
        "Flag_fumador": [None, "fumador"],
    })
    fe = FeatureEngineer()
    out = fe.fit_transform(df)
    assert "Edad_cat" in out.columns
    assert "IMC_cat" in out.columns
    assert "Promedio_nivel_glucosa_cat" in out.columns
    assert out.isna().sum().sum() == 0
