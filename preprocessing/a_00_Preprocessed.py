#Preprocesado

import pandas as pd
import numpy as np

df = pd.read_csv("dataset/car_data.csv")

#Cambiando formato de columnas para una mejor exploración de datos
df.columns = df.columns.str.replace(r'([A-Z])', r'_\1', regex=True).str.strip('_').str.lower()

#Cambiando tipo de datos
date_col = df[['date_crawled', 'date_created', 'last_seen']]
category_col = df[['vehicle_type', 'fuel_type', 'gearbox']]
for col in date_col:
    df[col] = pd.to_datetime(df[col], format= '%d/%m/%Y %H:%M')
    
for col in category_col:
    df[col] = df[col].astype('category')

#Eliminando columnas no necesarias
df_new = df.drop(['number_of_pictures','date_crawled', 'date_created', 'last_seen', 'postal_code'], axis=1)

#Tratando valores duplicados
#Chequeamos la duplicidad de los mismos para saber si eliminarlos o no
dup_row = df_new[df_new.duplicated(keep=False)]
#keep=False asegura que todas las instancias duplicadas se marquen, no solo las adicionales.

dup_row_sorted = dup_row.sort_values(list(df_new.columns))

#Eliminando duplicados
df_new.drop_duplicates(inplace= True)

#Filtrando y acotando datos para columna del año de registro (registration_year)
df_new_filt = df_new.query('1900 <= registration_year <= 2024')

#filtrando precio para valores mayores o iguales a 100 (price)
df_new_filt = df_new_filt.query('price >= 100')

#Limitando valores en la columna de potencia  y sustituyendo valores atípicos con valores NaN(power)
df_new_filt = df_new_filt.query('power <= 2000')
#reemplazo
df_new_filt.loc[df_new_filt['power'] < 45, 'power'] = np.nan

#Tratando valores atípicos. Mes 0 (registration_month)
#Reemplanzando con la mediana
df_new_filt.loc[df_new_filt['registration_month'] == 0, 'registration_month'] = df_new_filt['registration_month'].median()
df_new_filt['registration_month'] = df_new_filt['registration_month'].astype('int')

#Función moda para usar en columnas de datos categóricos faltantes
def mode_f(var):#creo una función para la moda
    return var.mode().iloc[0] if len(var) > 0 else np.nan

vehicle_values = df_new_filt.dropna(subset=['vehicle_type', 'model'])

vehicle_model = vehicle_values.groupby(['brand','model'])['vehicle_type'].agg(mode_f).reset_index()
vehicle_model_dict = vehicle_model.set_index(['brand', 'model'])['vehicle_type'].to_dict()

def filling_v(row):
    model = row['model']
    vehicle = row['vehicle_type']
    brand = row['brand']

    if pd.isna(vehicle):
        return vehicle_model_dict.get((brand, model), np.nan)
    return vehicle

#aplico función al Dataset y almaceno en columna
df_new_filt['vehicle_type'] = df_new_filt.apply(filling_v, axis=1)

#Compruebo existencia de valores ausentes luego de la función
no_value_v = df_new_filt[df_new_filt['vehicle_type'].isna()]

#elimino los que no pudieron ser rellenados
df_new_filt = df_new_filt.dropna(subset=['vehicle_type'])
no_nan_v = df_new_filt[df_new_filt['vehicle_type'].isna()]

#Filtrando columnas
gearbox_values = df_new_filt.dropna(subset=['gearbox', 'model'])

gearbox_model = gearbox_values.groupby(['brand', 'model'])['gearbox'].agg(mode_f).reset_index()
gearbox_model_dict = gearbox_model.set_index(['brand', 'model'])['gearbox'].to_dict()

def filling_g(row):#creo función que llenará la columna con los valores faltantantes
    model = row['model']
    gearbox = row['gearbox']
    brand = row['brand']

    if pd.isna(gearbox):
        return gearbox_model_dict.get((brand, model), np.nan)
    return gearbox

#aplico función a Dataset y guardo en columna del mismo
df_new_filt['gearbox'] = df_new_filt.apply(filling_g, axis=1)

#Comprobando si existen valores sin rellenar
no_value_g = df_new_filt[df_new_filt['gearbox'].isna()]

df_new_filt = df_new_filt.dropna(subset=['gearbox'])
no_nan_g = df_new_filt[df_new_filt['gearbox'].isna()]

#Reemplazo
df_new_filt['fuel_type'] = df_new_filt['fuel_type'].replace('petrol', 'gasoline')

#Filtro
fuel_values = df_new_filt.dropna(subset=['fuel_type', 'model'])

fuel_model = fuel_values.groupby(['model'])['fuel_type'].agg(mode_f).reset_index()
fuel_model_dict = fuel_model.set_index(['model'])['fuel_type'].to_dict()

def filling_f(row):#creo función
    model = row['model']
    fuel = row['fuel_type']
    
    if pd.isna(fuel):
        return fuel_model_dict.get(model, np.nan)
    return fuel

#aplico función al Dataset y almaceno en columna
df_new_filt['fuel_type'] = df_new_filt.apply(filling_f, axis=1)

#Compruebo existencia de valores ausentes luego de la función
no_value_f = df_new_filt[df_new_filt['fuel_type'].isna()]

df_new_filt = df_new_filt.dropna(subset=['fuel_type'])
no_nan_f = df_new_filt[df_new_filt['fuel_type'].isna()]

model_values = df_new_filt.dropna(subset=['model'])

model_brand = model_values.groupby(['brand','registration_year'])['model'].agg(mode_f).reset_index()
model_brand_dict = model_brand.set_index(['brand', 'registration_year'])['model'].to_dict()

def filling_m(row):
    model = row['model']
    year = row['registration_year']
    brand = row['brand']

    if pd.isna(model):
        return model_brand_dict.get((brand, model), np.nan)
    return model

#aplico función al Dataset y almaceno en columna
df_new_filt['model'] = df_new_filt.apply(filling_m, axis=1)

#Compruebo existencia de valores ausentes luego de la función
no_value_m = df_new_filt[df_new_filt['model'].isna()]

df_new_filt = df_new_filt.dropna(subset=['model'])
no_nan_m = df_new_filt[df_new_filt['model'].isna()]

#filtrando
df_new_filt['power'] = pd.to_numeric(df_new_filt['power'], errors='coerce')#cambio tipo de dato a númerico
power_values = df_new_filt.dropna(subset=['power'])

power_model = power_values.groupby(['model'])['power'].median().reset_index()
power_model_dict = power_model.set_index(['model'])['power'].to_dict()

def filling_p(row):
    model = row['model']
    power = row['power']

    if pd.isna(power):
        return model_brand_dict.get(model, np.nan)
    return power

#Aplico y almaceno
df_new_filt['power'] = df_new_filt.apply(filling_p, axis=1)

#compruebo existencia de ausentes luego de aplicada la función
no_value_p = df_new_filt[df_new_filt['power'].isna()]

df_new_filt = df_new_filt.dropna(subset=['power'])
no_nan_p = df_new_filt[df_new_filt['power'].isna()]

#elimino
df_new_filt = df_new_filt.dropna(subset=['not_repaired'])
no_nan_not_r = df_new_filt[df_new_filt['power'].isna()]

#Nuevo DataFrame
df_new_filt.reset_index(drop=True, inplace=True)

#Guardando dataset con datos preprocesados.
df_new_filt.to_csv('outputs/preprocessed/preprocessed_data.csv', index=False)

#Distribución. Variable objetivo 'price' (logarítmico)
log_price = np.log1p(df_new_filt['price'])

#Resumen de estadística descriptiva del precio original respecto a la transformación
#Estadísticas descriptivas del precio original

original_price = df_new_filt['price'].describe()

#Estadísticas descriptivas del precio transformado
log_price_stats = log_price.describe()

#guardar estadísticas
estadisticas = pd.DataFrame({
    "Precio Original": original_price,
    "Precio Transformado (Log)": log_price_stats
})

#Análisis de distribución de columnas categóricas
categorical_columns = ['vehicle_type', 'fuel_type', 'gearbox', 'not_repaired', 'brand']

for col in categorical_columns:
    mean_price = df_new_filt.groupby(col)['price'].mean().sort_values(ascending=False)

#Análisis de distribución de columnas númericas
numeric_columns = ['registration_year', 'power', 'registration_month']