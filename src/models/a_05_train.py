#a_05_train
#MODELOS SELECCIONADOS DEL GRIDSEARCH
import pandas as pd
import numpy as np

from src.utils.config_manager import load_paths

from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.base import clone

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from joblib import dump
from pathlib import Path

PATHS = load_paths()["dirs"]

#MEJORES Hiperparametros obtenidos del monolito (también se puede ubicar en módulo a_04_tune_hyperparams.py) 
PARAMS = {
    'LGBM': {
        'n_estimators': 300,
        'learning_rate': 0.1,
        'num_leaves': 30,
        'max_depth': 10,
        'subsample': 0.6,
        'random_state': 12345
    },
    'LGBM_log': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 30,
        'max_depth': 5,
        'subsample': 0.6,
        'random_state': 12345
    },
    'XGBoost_log': {
        'random_state':12345,
        'max_depth': 4, 
        'n_estimators': 100, 
        'learning_rate': 0.1, 
        'subsample': 0.8
    },
    'RF_log': {
        'random_state': 12345, 
        'max_depth': 6, 
        'min_samples_leaf': 8, 
        'min_samples_split': 6, 
        'n_estimators': 10
    },
    'DT_log': {
        'random_state': 12345, 
        'max_depth': 6, 
        'min_samples_split': 8, 
        'min_samples_leaf': 6
    }
}

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str) -> Any:
    
    models = {
        'LGBM': LGBMRegressor(**PARAMS['LGBM']),
        'LGBM_log': LGBMRegressor(**PARAMS['LGBM_log']),
        'XGBoost_log': XGBRegressor(**PARAMS['XGBoost_log']),
        'RF_log': RandomForestRegressor(**PARAMS['RF_log']),
        'DT_log': DecisionTreeRegressor(**PARAMS['DT_log'])
    }
    
    if model_type not in models:
        raise ValueError(f"No soportado: {model_type}")
    
    model = clone(models[model_type])
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    is_log_target: bool = False
) -> Tuple[np.ndarray, float]:
    #Evalúa el modelo y devuelve predicciones + RMSE (en espacio original si es logarítmico)
    y_pred = model.predict(X_test)
    
    if is_log_target:
        y_pred_original = np.expm1(y_pred)
        y_test_original = np.expm1(y_test)
        rmse = mean_squared_error(y_test_original, y_pred_original, squared=False)
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return (y_pred, rmse)

def TES(splits: Dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]],
        models_to_train: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    available_models = {
        'LGBM': {'split_key': 'LGBM', 'is_log': False},
        'LGBM_log': {'split_key': 'LGBM_log', 'is_log': True},
        'XGBoost_log': {'split_key': 'XGBoost_log', 'is_log': True},
        'RF_log': {'split_key': 'RF_log', 'is_log': True},
        'DT_log': {'split_key': 'DT_log', 'is_log': True}
    }
    
    #por si se específica alguna lista
    models_to_run = available_models if models_to_train is None else {
        name: config for name, config in available_models.items()
        if name in models_to_train
    }

    results = {}
    for model_name, config in models_to_run.items():
        #Carga conjunto
        X_train, X_test, y_train, y_test = splits[config['split_key']]
        #entrena
        model = train_model(X_train, y_train, model_name)

        #evalúa
        y_pred, rmse = evaluate_model(model, X_test, y_test, config['is_log'])

        # Guardado de modelos seleccionados
        dump(model, Path(PATHS["selected_models"]) /  f"{model_name}.joblib")
        results[model_name] = {'model': model, 'predictions': y_pred, 'rmse': rmse}

    return results