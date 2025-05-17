#a_01_features_eng
import pandas as pd
import numpy as np
from typing import Optional

#INGENIERÍA DE CARACTERÍSTICAS
def vehicle_age(df: pd.DataFrame, current_year: int = 2024, fixed_outlier: Optional[int] = 114) -> pd.DataFrame:
    df = df.copy()
    df['vehicle_age'] = current_year - df['registration_year']
    
    # Eliminación FIJADA de 114 años
    if 114 in df['vehicle_age'].values:
        df = df[df['vehicle_age'] != 114].reset_index(drop=True)
    
    return df

def mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    if 'mileage' in df.columns:
        df = df.copy()
        df['mileage_per_year'] = (df['mileage'] / df['vehicle_age']).mask(df['vehicle_age'] == 0, np.nan)
        df['mileage_per_year'] = df['mileage_per_year'].fillna(0)  # Rellena con 0

    return df

def features_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = vehicle_age(df)
    df= mileage_per_year(df)
    #último ajuste
    df = df.drop('registration_month', axis=1, errors='ignore')
    #estadistica de dataset final
    df.describe(include='all').to_csv('outputs/reports/final_statistics_data.csv', index=True)

    #guardado con nuevas caracteristicas
    df.to_csv('outputs/preprocessed/prepro_data_eng_charact.csv', index=False)

    return df