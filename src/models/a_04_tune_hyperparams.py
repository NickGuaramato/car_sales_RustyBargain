# a_04_tune_hyperparams.py
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV
from src.utils.config_manager import load_params, load_paths
from src.utils.helpers import directs
from src.preprocessing.a_03_split_data import prepare_all_splits

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Configuración
PARAMS = load_params()
PATHS = load_paths()
logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("artifacts/logs/hyperparameter_tuning.log"),
            logging.StreamHandler()
        ]
    )

def tune_model(estimator, params: Dict[str, Any], split_data: tuple, model_name: str) -> Dict[str, Any]:
    """Optimiza hiperparámetros para un modelo específico."""
    X_train, _, y_train, _ = split_data
    
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    return {
        "model_name": model_name,
        "best_params": grid.best_params_,
        "best_rmse": -grid.best_score_,
        "best_estimator": grid.best_estimator_
    }

def save_results(results: Dict[str, Any]):
    filename = f"best_{results['model_name']}_params.joblib"
    joblib.dump(results, Path(PATHS["dirs"]["selected_models"]) / filename)
    logger.info(f"Guardados parámetros para {results['model_name']}")

def hyperparameter_tune(df_engineered: pd.DataFrame, df_encoded: pd.DataFrame):
    """Ejecuta la optimización para todos los modelos definidos en prepare_all_splits."""
    directs()
    setup_logging()
    
    # Obtiene todos los splits de una vez
    splits = prepare_all_splits(df_engineered, df_encoded)
    selected_models = PARAMS["test_models"]["selected_models"]
    
    # Configuración de modelos
    models_config = {
    "LGBM": (LGBMRegressor(verbose=-1, random_state=12345), PARAMS["lightgbm"]),
    "LGBM_log": (LGBMRegressor(verbose=-1, random_state=12345), PARAMS["lightgbm"]),
    "XGBoost_log": (XGBRegressor(random_state=12345), PARAMS["xgboost"]),
    "RF_log": (RandomForestRegressor(random_state=12345), PARAMS["random_forest"]),
    "DT_log": (DecisionTreeRegressor(random_state=12345), PARAMS["decision_tree"]),
    "CatBoost": (
        CatBoostRegressor(verbose=False, random_seed=12345, loss_function='RMSE'), 
        PARAMS["catboost"]
    ),
    "CatBoost_log": (
        CatBoostRegressor(verbose=False, random_seed=12345, loss_function='Tweedie:variance_power=0'), 
        PARAMS["catboost"]
    )
}

    # Optimiza cada modelo
    results = {}
    for model_name in selected_models:
        if model_name in splits and model_name in models_config:
            estimator, params = models_config[model_name]
            result = tune_model(estimator, params, splits[model_name], model_name)
            save_results(result)
            results[model_name] = result
        else:
            logger.warning(f"Modelo {model_name} no configurado o sin splits")
    
    return results

if __name__ == "__main__":
    # Ejemplo de uso (requiere cargar los datos primero)
    from src.preprocessing.a_00_data_cleaning import preprocess_data
    from src.preprocessing.a_01_feature_engineering import features_engineer
    from src.preprocessing.a_02_encoding import encode_data
    
    # Carga y preprocesa datos (igual que en main.py)
    df, _ = preprocess_data(PATHS["files"]["raw_data"])
    df_engineered = features_engineer(df)
    df_encoded = encode_data(df_engineered)
    
    # Ejecuta tuning
    hyperparameter_tune(df_engineered, df_encoded)