#a_03_train_test_split.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

#ENTRENAMIENTO DE MODELO
#Características y Objetivo
#Dividiendo el Dataset y verificando conjunto
def preparing_split(df: pd.DataFrame,
                    target_col: str = 'price',
                    test_size: float = 0.25,
                    random_state: int = 12345,
                    convert_categories: bool = False,
                    log_target: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    #selección de objetivo
    if log_target and f'log_{target_col}' not in df.columns:
        df = df.copy()
        df[f'log_{target_col}'] = np.log1p(df[target_col])

    X = df.drop(columns=[target_col, f'log_{target_col}'], errors='ignore')
    y_log = f'log_{target_col}' if log_target else target_col
    y = df[y_log]
    #convierte en caso de ser necesario
    if convert_categories:
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_columns:
            X.loc[:, col] = X.loc[:, col].astype('category')
    
    #Split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def prepare_all_splits(df_engineered: pd.DataFrame,#datos originales (pre-encoding)
                       df_encoded: pd.DataFrame,#datos encodificados (post-encoding)
                       ) -> Dict[str, Tuple]:
    categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type', 'brand', 'model']
    for col in categorical_cols:
        if col in df_engineered.columns:
            df_engineered[col] = df_engineered[col].astype('category')
    return {
        #Modelos que no requieren encodificación
        'LGBM': preparing_split(df_engineered, convert_categories=True),
        'LGBM_log': preparing_split(df_engineered, log_target=True, convert_categories=True),

        #Otros modelos que requieren encodificado
        'XGBoost_log': preparing_split(df_encoded, log_target=True),
        'RF_log': preparing_split(df_encoded, log_target=True),
        'DT_log': preparing_split(df_encoded, log_target=True)
        }