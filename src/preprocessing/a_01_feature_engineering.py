#a_01_feature_engineering
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional

from src.utils.helpers import directs
from src.utils.config_manager import load_paths

#INGENIERÍA DE CARACTERÍSTICAS
def vehicle_age(df: pd.DataFrame, current_year: int = 2024, fixed_outlier: Optional[int] = 114) -> pd.DataFrame:
    df = df.copy()
    df['vehicle_age'] = current_year - df['registration_year']
    
    # Eliminación de TODOS los outliers mayores o iguales a 114 años
    if fixed_outlier is not None and fixed_outlier in df['vehicle_age'].values:
        df = df[df['vehicle_age'] < fixed_outlier].reset_index(drop=True)

    return df

def mileage_per_year(df: pd.DataFrame, fill_na: bool = True) -> pd.DataFrame:
    df = df.copy()
    if 'mileage' in df.columns and 'vehicle_age' in df.columns:
        df['mileage_per_year'] = np.where(
            df['vehicle_age'] > 0,
            df['mileage'] / df['vehicle_age'],
            np.where(df['vehicle_age'] == 0, 0, np.nan)
        )
        if fill_na:
            df['mileage_per_year'] = df['mileage_per_year'].fillna(0)
    return df

def features_engineer(df: pd.DataFrame) -> pd.DataFrame:
    directs()
    PATHS = load_paths()

    df = vehicle_age(df)
    df= mileage_per_year(df)
    #último ajuste
    df = df.drop('registration_month', axis=1, errors='ignore')
    #estadistica de dataset final
    df.describe(include='all').to_csv(Path(PATHS["dirs"]["metrics"]) / "final_stats_data.csv", 
                                      index=True
                                      )

    #guardado con nuevas caracteristicas (datos procesados)
    df.to_parquet(PATHS["files"]["processed_data"])

    return df