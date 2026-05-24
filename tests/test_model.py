from src.train import construir_modelo


def test_construir_modelo():
    model = construir_modelo()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")
