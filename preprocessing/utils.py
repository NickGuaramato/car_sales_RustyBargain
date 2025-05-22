#utils.py

import pandas as pd
import numpy as np
from typing import Any, Dict
from pathlib import Path

def directs():
    dirs = [
        "outputs/models",
        "outputs/plots", 
        "outputs/reports",
        "outputs/preprocessed",
        "outputs/logs"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


#Normalizando formato de columnas para una mejor exploración de datos
def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.replace(r'([A-Z])', r'_\1', regex=True)
        .str.strip('_')
        .str.lower()
    )
    return df#también return df.rename(columns=lambda x: (x.replace(' ', '_').lower()))

#Calcula la moda de una serie, manejando series vacías.
def mode_f(series: pd.Series) -> Any:
    return series.mode().iloc[0] if not series.empty else np.nan

def fill_missing_values(df: pd.DataFrame, groups_cols: list[str], target_col:str, method: str = 'mode') -> pd.DataFrame:
    df = df.copy()
    if method not in ['mode', 'median']:
        raise ValueError("Método debe ser 'mode' o 'median'")  #Validación básica
    #rellena NaN en target_col usando moda o mediana

    agg_func = mode_f if method == 'mode' else lambda x: x.median()

    value_dict = (
        df.dropna(subset=[target_col] + groups_cols)
        .groupby(groups_cols)[target_col]
        .agg(agg_func)
        .to_dict()
)

    df[target_col] = df.apply(
        lambda row: value_dict.get(tuple(row[col] for col in groups_cols), np.nan)
        if pd.isna(row[target_col]) else row[target_col],
        axis=1)
    
    return df.dropna(subset=[target_col])