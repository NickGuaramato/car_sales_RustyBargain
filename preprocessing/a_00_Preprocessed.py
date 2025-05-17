#a_00_preprocesado

import pandas as pd
import numpy as np
from typing import Tuple
from utils import fill_missing_values, normalize_names

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_names(df)#df = pd.read_csv("dataset/car_data.csv")

#Cambiando tipo de datos
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    date_col = df[['date_crawled', 'date_created', 'last_seen']] 
    category_col = df[['vehicle_type', 'fuel_type', 'gearbox']] 
    
    for col in date_col:
        df[col] = pd.to_datetime(df[col], format= '%d/%m/%Y %H:%M') 
    for col in category_col:
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
    df.loc[df['registration_month'] == 0, 'registration_month'] = df['registration_month'].median()
    df['registration_month'] = df['registration_month'].astype('int')

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
    df['fuel_type'] = df['fuel_type'].replace('petrol', 'gasoline')
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
    return df.dropna(subset=['not_repaired'])

def analyze_data(df: pd.DataFrame) -> dict:
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
            col:df.groupby(col)['price'].mean().sort_values(ascending=False)
            for col in ['vehicle_type', 'fuel_type', 'gearbox', 'not_repaired', 'brand']
        },
        #Análisis de distribución de columnas númericas
        
        "numeric_columns": df[['registration_year', 'power', 'registration_month']].describe()
        }
    #luego de limipieza(estadísticas)
    stats["cleaned_stats"] = df.describe(include='all')
    stats["cleaned_stats"].to_csv('outputs/reports/preprocessed_data_statistics.csv')
    
    stats.to_csv('outputs/reports/stats_price(ori)_price(log).csv', index=True)

    return stats

def preprocess_data(input_path: str) -> pd.DataFrame:
    #estadisticas de dataset original
    df_raw = pd.read_csv(input_path)
    df_raw.describe(include='all').to_csv('outputs/reports/original_data_statistics.csv', index=True)
    #ejecución de todo el pipeline
    df = load_data(input_path)
    df = convert_data_types(df)
    df = unnecessary_columns(df)
    df= remove_duplicates(df)
    
    df.describe(include='all').to_csv('outputs/reports/unduplicated_data_statistics.csv', index=True)

    df = filter_data(df)
    df = process_missing_values(df)

    stats = analyze_data(df)

    df.to_csv('outputs/preprocessed/preprocessed_data.csv', index=False)#guardo con datos preprocesados
    return df, stats