#test_pipeline.py
import pytest
import pandas as pd
from pathlib import Path
from joblib import load

from src.preprocessing.a_00_data_cleaning import preprocess_data
from src.preprocessing.a_01_feature_engineering import features_engineer
from src.preprocessing.a_02_encoding import encode_data
from src.preprocessing.a_03_split_data import prepare_all_splits
from src.models.a_05_train import TES
from src.utils.config_manager import load_paths, load_params
from src.utils.helpers import directs

#rutas centralizdas
PATHS = load_paths()
PARAMS = load_params()["test_models"]

@pytest.fixture
def processed_data():
    directs()
    df, _ = preprocess_data(PATHS["files"]["raw_data"])
    df = features_engineer(df)
    return encode_data(df)

#Modelos a testear
MODELS_TO_TEST = PARAMS["selected_models"]

def test_categories():
    """Verifica que no haya categorías duplicadas."""
    df, _ = preprocess_data(PATHS["files"]["raw_data"])
    #categorical_cols = ['vehicle_type', 'fuel_type', 'gearbox']
    
    for col in ['vehicle_type', 'gearbox', 'fuel_type']:
        if col in df.columns:
            assert isinstance(df[col].dtype, pd.CategoricalDtype), (
                f"{col} debería ser categórica. Tipo actual: {df[col].dtype}\n"
                f"Valores únicos: {df[col].unique()}\n"
                f"Ejemplo: {df[col].iloc[0]} (tipo {type(df[col].iloc[0])})"
            )
            assert not df[col].isnull().any(), f"{col} contiene nulos"
            assert set(df[col].cat.categories) == set(df[col].unique()), \
                f"{col} tiene categorías inconsistentes"
    df, _ = preprocess_data(PATHS["files"]["raw_data"])
    df = features_engineer(df)
    
    assert not df.empty, "DataFrame vacío después de preprocesamiento"
    assert "vehicle_age" in df.columns, "Falta feature engineering (vehicle_age)"
    assert df["price"].min() >= 100, "Filtrado de precio falló"

def test_encoding():
    """Verifica que el encoding no introduzca nulos."""
    df, _ = preprocess_data(PATHS["files"]["raw_data"])
    df = features_engineer(df)
    df_encoded = encode_data(df)
    
    encoded_cols = [col for col in df_encoded.columns if not col.startswith('vehicle_type_')]
    assert df_encoded[encoded_cols].isnull().sum().sum() == 0, "Encoding introdujo nulos"
    assert "log_price" in df_encoded.columns, "Falta transformación logarítmica"

@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_model_training_and_saving(model_name):
    """Test individual para cada modelo."""
    # 1. Prepara datos
    df, _ = preprocess_data(PATHS["files"]["raw_data"])
    df = features_engineer(df)
    df_encoded = encode_data(df)
    splits = prepare_all_splits(df, df_encoded)
    
    # 2. Entrena TODOS los modelos (o usa model_name para uno específico)
    results = TES(splits, models_to_train=[model_name])
    
    # 3. Verificaciones por modelo
    assert model_name in results, f"Modelo {model_name} no fue entrenado"
    model_path = Path(PATHS["dirs"]["selected_models"]) / f"{model_name}.joblib"
    assert model_path.exists(), f"Modelo {model_name} no se guardó"
    
    # 4. Verifica que el modelo pueda predecir
    model = load(model_path)
    split_key = 'LGBM_log' if model_name.endswith('_log') else 'LGBM'  # Para LightGBM
    if model_name in ['XGBoost_log', 'RF_log', 'DT_log']:
        split_key = 'Reg_log'  # Para otros modelos"""
    X_test = splits[model_name][1]  # Asume [X_train, X_test, y_train, y_test]
    y_pred = model.predict(X_test)
    
    assert len(y_pred) == len(X_test), f"Número incorrecto de predicciones en {model_name}"
    assert not any(pd.isnull(y_pred)), f"Predicciones NaN en {model_name}"
    
    # 5. Verifica métricas (opcional)
    assert "rmse" in results[model_name], f"No hay métricas para {model_name}"