import pandas as pd
from src.load_data import validar_columnas


def test_validar_columnas_minimas():
    df = pd.DataFrame(columns=[
        "ID", "Genero", "Edad", "Flag_hipertension", "Flag_problem_cardiaco",
        "Estados_civil", "Tipo_trabajo", "Zona_residencia",
        "Promedio_nivel_glucosa", "IMC", "Flag_fumador", "Ataque_cardiaco"
    ])
    validar_columnas(df)
