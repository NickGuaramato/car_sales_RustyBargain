#a_00_data_cleaning

import pandas as pd
import numpy as np

from pathlib import Path

from typing import Tuple

from src.utils.helpers import fill_missing_values, normalize_names, directs
from src.utils.config_manager import load_paths
from src.utils.logging_config import setup_logging

PATHS = load_paths()["dirs"]
logger = setup_logging(module='data_cleaning')

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_names(df)

#Cambiando tipo de datos
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fechas
    date_cols = ['date_crawled', 'date_created', 'last_seen']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Categóricas - ¡Versión mejorada!
    categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df

#Eliminando columnas no necesarias
def unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['number_of_pictures',
                       'date_crawled', 
                       'date_created', 
                       'last_seen', 
                       'postal_code']
    return df.drop(columns_to_drop, axis=1)

#Tratando valores duplicados
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

#Filtrando y acotando datos para columna del año de registro (registration_year)
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.query('1900 <= registration_year <= 2024')
#filtrando precio para valores mayores o iguales a 100 (price)
    df = df.query('price >= 100')
#Limitando valores en la columna de potencia  y sustituyendo valores atípicos con valores NaN(power) 
    df = df.query('power <= 2000')
#reemplazo
    df.loc[df['power'] < 45, 'power'] = np.nan

#Tratando valores atípicos. Mes 0 (registration_month)
#Reemplanzando con la mediana
    df = df.copy()
    df.loc[df['registration_month'] == 0, 'registration_month'] = df['registration_month'].median()
    df.loc[:, 'registration_month'] = df['registration_month'].astype('int')

    return df

#Función moda. Usando en columnas de datos categóricos faltantes
def process_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    #vehicle_type
    df = fill_missing_values(
        df,
        groups_cols=['brand', 'model'],
        target_col='vehicle_type'
    )
    #gearbox
    df = fill_missing_values(
        df,
        groups_cols=['brand', 'model'],
        target_col='gearbox'
    )
    #fuel_type
    if 'fuel_type' in df.columns:
        df['fuel_type'] = df['fuel_type'].replace('petrol', 'gasoline')
        df = fill_missing_values(df, ['model'], 'fuel_type')

    #model
    df = fill_missing_values(
        df,
        ['model'],
        'fuel_type'
    )
    #model
    df = fill_missing_values(
        df,
        groups_cols=['brand', 'registration_year'],
        target_col='model'
    )
    #filtrando a power
    df['power'] = pd.to_numeric(df['power'], errors='coerce')#cambio tipo de dato a númerico
    df = fill_missing_values(
        df,
        ['model'],
        'power',
        method='median'
    )

    if 'not_repaired' in df.columns:
        df['not_repaired'] = df['not_repaired'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)  # ← Conversión directa a binario
    
    # Eliminar nulos como en monolito
    df = df.dropna(subset=['not_repaired', 'power'])

    categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type']
    for col in categorical_cols:
        if col in df.columns and col in df.columns:
            df[col] = df[col].astype('category')  # Forzar tipo
    
    return df

def analyze_data(df: pd.DataFrame) -> dict:
    directs()
    #Distribución. Variable objetivo 'price' (logarítmico)
    log_price = np.log1p(df['price'])
    #Resumen de estadística descriptiva del precio original respecto a la transformación
#Estadísticas descriptivas del precio original
    stats = {
        "price_stats": pd.DataFrame({
            "Precio Original": df['price'].describe(),
            "Precio Transformado (Log)": log_price.describe()
        }),
        #Análisis de distribución de columnas categóricas
        
        "categorical_columns" : {
            col:df.groupby(col, observed=True)['price'].mean().sort_values(ascending=False)#se agregó observed=True
            for col in ['vehicle_type', 'fuel_type', 'gearbox', 'not_repaired', 'brand']
        },
        #Análisis de distribución de columnas númericas
        
        "numeric_columns": df[['registration_year', 'power', 'registration_month']].describe(),
        "cleaned_stats": df.describe(include='all')        
        }
    #luego de limipieza(estadísticas)
    stats["cleaned_stats"].to_csv(Path(PATHS["metrics"]) / " preprocessed_data_statistics.csv", 
                                          index=True
                                          )
    
    stats["price_stats"].to_csv(Path(PATHS["metrics"]) / "stats_price(ori)_price(log).csv", 
                                          index=True
                                          )

    return stats

def preprocess_data(input_path: str) -> pd.DataFrame:
    directs()

    logger.info(f"🚀 Iniciando preprocesamiento desde: {input_path}")

    #estadisticas de dataset original
    df_raw = pd.read_csv(input_path)
    
    logger.info(f"📊 Dataset original cargado. Shape: {df_raw.shape}")

    df_raw.describe(include='all').to_csv(Path(PATHS["metrics"]) / "original_data_statistics.csv", 
                                          index=True
                                          )
    #ejecución de todo el pipeline
    df = load_data(input_path)
    df = convert_data_types(df)
    df = unnecessary_columns(df)

    logger.debug(f"🔄 Después de limpieza básica. Columnas: {len(df.columns)}")

    df= remove_duplicates(df)

    logger.info(f"🧹 Duplicados eliminados. Filas: {df.shape[0]}")
    
    df.to_pickle(Path(PATHS["metrics"]) / "unduplicated_data.pkl")
    df.describe(include='all').to_csv(Path(PATHS["metrics"]) / "unduplicated_data_stats.csv", 
                                      index=True
                                      )

    df = filter_data(df)

    logger.info(f"🎯 Después de filtros. Filas: {df.shape[0]}")

    df = process_missing_values(df)

    logger.info(f"🔧 Valores faltantes procesados. ¿Nulos?: {df.isnull().sum().sum() == 0}")

    assert df.isnull().sum().sum() == 0, "¡Quedan valores nulos!"

    logger.info("✅ Validación pasada: Cero valores nulos")

    df.to_pickle(Path(PATHS["metrics"]) / "preprocessed_data.pkl")
    df.describe(include='all').to_csv(Path(PATHS["metrics"]) / "preprocessed_stats.csv", 
                                      index=True
                                      )#datos preprocesados (antes de features_engineer)

    logger.debug("💾 Datos preprocesados guardados")

    stats = analyze_data(df)

    logger.info("🎉 Preprocesamiento completado exitosamente")

    return df, stats