#rusty_bargain_project

#PREPARACIÓN DE DATOS
#librerías
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from joblib import dump

#carpetas para guardado de gráficos, modelos, métricas y reportes de clasificación
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/preprocessed', exist_ok=True)

#PREPROCESAMIENTO DE DATOS

df = pd.read_csv("dataset/car_data.csv")
print(df.info())

print()
print('Estadística descriptiva de DataFrame:')
print(df.describe())

#Cambiando formato de columnas para una mejor exploración de datos
df.columns = df.columns.str.replace(r'([A-Z])', r'_\1', regex=True).str.strip('_').str.lower()
print(df.columns)#mostrando cambios

#Cambiando tipo de datos
date_col = df[['date_crawled', 'date_created', 'last_seen']]
category_col = df[['vehicle_type', 'fuel_type', 'gearbox']]
for col in date_col:
    df[col] = pd.to_datetime(df[col], format= '%d/%m/%Y %H:%M')
    
for col in category_col:
    df[col] = df[col].astype('category')
print(df.info())

#Eliminando columnas no necesarias
df_new = df.drop(['number_of_pictures','date_crawled', 'date_created', 'last_seen', 'postal_code'], axis=1)
print(df_new.columns)#observando cambios

#Checando existencia de valores ausentes y duplicados
print(df_new.isna().sum())
print()
print('Valores duplicado:', df_new.duplicated().sum())

#Tratando valores duplicados
#Chequeamos la duplicidad de los mismos para saber si eliminarlos o no
dup_row = df_new[df_new.duplicated(keep=False)]
#keep=False asegura que todas las instancias duplicadas se marquen, no solo las adicionales.

print(f'Total de registros duplicados: {len(dup_row)}')

dup_row_sorted = dup_row.sort_values(list(df_new.columns))
#Muestro filas duplicadas
print(dup_row_sorted)

#Eliminando duplicados
df_new.drop_duplicates(inplace= True)
#Muestro la existencia de los cambios
print(f'Valores duplicados: {df_new.duplicated().sum()}')

#Observando mínimos y máximos en columna con datos sospechosamente errados
columns = [
    ('registration_year', 'Año'),
    ('power', 'Potencia'),
    ('registration_month', 'Mes')
]

for col, description in columns:
    print(f'{description} más bajo registrado: {df_new[col].min()}')
    print(f'{description} más alto registrado: {df_new[col].max()}')


#Filtrando y acotando datos para columna del año de registro (registration_year)
df_new_filt = df_new.query('1900 <= registration_year <= 2024')
print(df_new_filt['registration_year'].describe())#compruebo cambios

#filtrando precio para valores mayores o iguales a 100 (price)
df_new_filt = df_new_filt.query('price >= 100')
print(df_new_filt['price'].describe())#compruebo

#Limitando valores en la columna de potencia  y sustituyendo valores atípicos con valores NaN(power)
df_new_filt = df_new_filt.query('power <= 2000')
#reemplazo
df_new_filt.loc[df_new_filt['power'] < 45, 'power'] = np.nan
print(df_new_filt['power'].describe())#compruebo

#Tratando valores atípicos. Mes 0 (registration_month)
#Reemplanzando con la mediana
df_new_filt.loc[df_new_filt['registration_month'] == 0, 'registration_month'] = df_new_filt['registration_month'].median()
df_new_filt['registration_month'] = df_new_filt['registration_month'].astype('int')
print(df_new_filt['registration_month'].value_counts(normalize=True).sort_index())#compruebo valores únicos

#Observo el DF actualizado respecto a los valores ausentes
print(df_new_filt.isna().sum())

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
print(f'Cantidad de valores ausentes luego de aplicar función: {len(no_value_v)}')

#elimino los que no pudieron ser rellenados
df_new_filt = df_new_filt.dropna(subset=['vehicle_type'])
no_nan_v = df_new_filt[df_new_filt['vehicle_type'].isna()]
print(f'Cantidad de valores ausentes luego de eliminar filas: {len(no_nan_v)}')

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
print(f'Cantidad de valores ausentes luego de aplicar función: {len(no_value_g)}')

df_new_filt = df_new_filt.dropna(subset=['gearbox'])
no_nan_g = df_new_filt[df_new_filt['gearbox'].isna()]
print(f'Cantidad de valores ausentes luego de eliminar filas: {len(no_nan_g)}')

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
print(f'Cantidad de valores ausentes luego de aplicar función: {len(no_value_f)}')

df_new_filt = df_new_filt.dropna(subset=['fuel_type'])
no_nan_f = df_new_filt[df_new_filt['fuel_type'].isna()]
print(f'Cantidad de valores ausentes luego de eliminar filas: {len(no_nan_f)}')

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
print(f'Cantidad de valores ausentes luego de aplicar función: {len(no_value_m)}')

df_new_filt = df_new_filt.dropna(subset=['model'])
no_nan_m = df_new_filt[df_new_filt['model'].isna()]
print(f'Cantidad de valores ausentes luego de eliminar filas: {len(no_nan_m)}')

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
print(f'Cantidad de valores ausentes luego de aplicar función: {len(no_value_p)}')

df_new_filt = df_new_filt.dropna(subset=['power'])
no_nan_p = df_new_filt[df_new_filt['power'].isna()]
print(f'Cantidad de valores ausentes luego de eliminar filas: {len(no_nan_p)}')

#Observó cantidad de valores ausentes
print(df_new_filt.isna().sum())
#elimino
df_new_filt = df_new_filt.dropna(subset=['not_repaired'])
no_nan_not_r = df_new_filt[df_new_filt['power'].isna()]
print(f'Cantidad de valores ausentes luego de eliminar filas: {len(no_nan_not_r)}')

#Nuevo DataFrame
df_new_filt.reset_index(drop=True, inplace=True)
print(df_new_filt.info())

#Observo correlación
print(df_new_filt.corr())



#ENTRENAMIENTO DEL MODELO
actual_year = 2024
df_new_filt['vehicle_age'] = actual_year - df_new_filt['registration_year']
df_new_filt.rename(columns={'registration_year' : 'vehicle_age'})
print(df_new_filt)
print()
print(df_new_filt['vehicle_age'].describe())

#Último ajuste
df_new_filt = df_new_filt.drop(['registration_month'], axis=1)
#observo cambios
print(df_new_filt.columns)

#Codificación y Escalado
categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type', 'brand', 'model', 'not_repaired']
numerical_cols = ['power', 'mileage', 'vehicle_age']

#Asigno a variable que luego será usada
df_new_filt_OHE = df_new_filt.copy()

#Codifico frecuencia para columnas categóricas
#Asigno proporción de su aparición en las respectivas columnas
for col in categorical_cols:
    encod_frequency = df_new_filt_OHE[col].value_counts() / len(df_new_filt_OHE)
    df_new_filt_OHE[col] = df_new_filt_OHE[col].map(encod_frequency)

#Escalo características númericas
scaler = StandardScaler()
df_new_filt_OHE[numerical_cols] = scaler.fit_transform(df_new_filt_OHE[numerical_cols])

#Observo
print('Luego de codificado de frecuencia y escalado:')
print(df_new_filt_OHE.head())

#Características y Objetivo
#Dividiendo el Dataset y verificando conjunto
X = df_new_filt_OHE.drop('price', axis=1)
y = df_new_filt_OHE['price']

#entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)

#verificando forma
print(X_train.shape, X_test.shape)

#REGRESION LINEAL
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predict = lr_model.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_predict)**0.5
print(f'RMSE de Regresión Lineal: {lr_rmse}')

#ÁRBOL DE DECISIÓN
#hiperparámetros de árbol de decisión
dt_params = {
    'max_depth': [1, 2, 3, 4, 5, 6] ,
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [2, 4, 6, 8],
}

#GridSearchCV e hiperparámetros establecidos.
#Metríca y valor para validación cruzada
dt_grid = GridSearchCV(
    estimator=DecisionTreeRegressor(),
    param_grid=dt_params,
    scoring='neg_root_mean_squared_error',
    cv=3)

#entrenamos para encontrar mejores hiperparametros
dt_grid.fit(X_train, y_train)
#Buscamos mejores hiperparametros que devuelven más bajo RSME
max_score = dt_grid.cv_results_['mean_test_score'].max()
index_max_score = np.where(dt_grid.cv_results_['mean_test_score'] == max_score)[0][0]

best_set_of_params = dt_grid.cv_results_['params'][index_max_score]
print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')

#Entrenamiento de modelo
dt_model = DecisionTreeRegressor(random_state=12345, max_depth=6, min_samples_split=2, min_samples_leaf=2)
dt_model.fit(X_train, y_train)
dt_predict = dt_model.predict(X_test)
dt_rmse = mean_squared_error(y_test, dt_predict)**0.5
print(f'RMSE de Árbol de Decisión: {dt_rmse}')

#BOSQUE ALEATORIO
rf_params = {
    'n_estimators' : [10, 20, 40],
    'max_depth': [1, 2, 3, 4, 5, 6] 
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=rf_params,
    scoring='neg_root_mean_squared_error',
    cv=3)

#entrenamos para encontrar mejores hiperparametros
rf_grid.fit(X_train, y_train)
#Buscamos mejores hiperparametros que devuelven más bajo RSME
max_score = rf_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(rf_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_set_of_params = rf_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')

#Entrenamiento de modelo
rf_model = RandomForestRegressor(random_state=12345, max_depth=6, n_estimators=20)
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_predict)**0.5
print(f'RMSE de Bosque Aleatorio: {rf_rmse}')

#CATBOOST
#Características y objetivos antes de OHE
features = df_new_filt.drop('price', axis=1)
target = df_new_filt['price']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=12345)

#Selecciono sólo las columnas cuyas características son categóricas
categorical_columns = features_train.select_dtypes(include=['object']).columns.tolist()

#Tipo 'category'
for column in categorical_columns:
    features_train.loc[:, column] = features_train.loc[:, column].astype('category')
    features_test.loc[:, column] = features_test.loc[:, column].astype('category')

#Hiperparámetros a ajustar
cb_params = {
    'depth': [4, 6, 10],
    'learning_rate': [0.1, 0.2, 0.5]
}

#Estimador
cb_est = CatBoostRegressor(iterations=100, cat_features=categorical_columns, verbose=False, loss_function='RMSE')

cb_grid = GridSearchCV(
    estimator=cb_est,
    param_grid=cb_params,
    scoring='neg_root_mean_squared_error',
    cv=3
)

cb_grid.fit(features_train, target_train)

max_score = cb_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(cb_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = cb_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')

cb_model = CatBoostRegressor(random_state=12345, iterations=100, depth=10, learning_rate=0.5, loss_function='RMSE', cat_features=categorical_columns, verbose=False)
cb_model.fit(features_train, target_train)
cb_predict = cb_model.predict(features_test)
cb_rmse = mean_squared_error(target_test, cb_predict)**0.5
print(f'RMSE de CatBoost: {cb_rmse}')

#XGBOOOST
#Hiperparametros
xgb_params = {'max_depth': [4, 6, 8], 
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.6, 0.8]
}

xgb_est = XGBRegressor()

xgb_grid = GridSearchCV(estimator=xgb_est, param_grid=xgb_params, scoring='neg_root_mean_squared_error', cv=3)

"""#Buscamos los mejores hiperparametros
xgb_grid.fit(X_train, y_train)
max_score = xgb_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(xgb_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = xgb_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')"""

#Entrenamos modelo con hiperparametros
xgb_model = XGBRegressor(random_state=12345, max_depth=8, n_estimators=100, learning_rate=0.1, subsample=0.8)
xgb_model.fit(X_train, y_train)
xgb_predict = xgb_model.predict(X_test)
xgb_rmse = mean_squared_error(y_test, xgb_predict)**0.5
print(f'RMSE de XGBoost: {xgb_rmse}')

#LIGHTGBM
#Igual que catboost, tomo el conjunto para antes de OHE
for col in categorical_columns:
    df_new_filt[col] = df_new_filt[col].astype('category')

X_LGBM = df_new_filt.drop('price', axis=1)
y_LGBM = df_new_filt['price']

X_LGBM_train, X_LGBM_test, y_LGBM_train, y_LGBM_test = train_test_split(X_LGBM, y_LGBM, test_size=0.25, random_state=12345)

LGBM_params = {
    'n_estimators': [100, 150, 300],
    'learning_rate': [0.1, 0.2, 0.5],
    'num_leaves': [10, 20, 30],
    'max_depth': [5, 8, 10],
    'subsample': [0.6, 0.7, 0.8]
}

LGBM_est = LGBMRegressor()

LGBM_grid = GridSearchCV(
    estimator=LGBM_est,
    param_grid=LGBM_params,
    scoring='neg_root_mean_squared_error',
    cv=3  
)

#Entrenamos para hallar los mejores hiperparametros
"""LGBM_grid.fit(X_LGBM_train, y_LGBM_train)

max_score = LGBM_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(LGBM_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = LGBM_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RECM: {-max_score}')"""

LGBM_model = LGBMRegressor(n_estimators=300, learning_rate=0.2, num_leaves=30, max_depth=10, subsample=0.6, random_state=12345)
LGBM_model.fit(X_LGBM_train, y_LGBM_train)
LGBM_predict = LGBM_model.predict(X_LGBM_test)
LGBM_rmse = mean_squared_error(y_LGBM_test, LGBM_predict)**0.5
print(f'RMSE de LightGBM: {LGBM_rmse}')

#ANÁLISIS DE MODELOS
#Ordenando resultado en un DataFrame para mejor visualización
data_models = {
    'modelo': ['Regresión Lineal', 'Árbol de Decisión', 'Bosque Aleatorio', 'CatBoost', 'XGBoost', 'LightGBM'],
    'tiempo_ajuste_hiperparámetros': [0.0423, 0.307, 4.39, 21.8, 39.4, 4.96],
    'tiempo_de_entrenamiento': [0.0419, 0.305, 4.39, 21.8, 39.8, 4.96],
    'tiempo_de_prueba': [0.0357, 0.0711, 0.0778, 0.0855, 0.418, 1.51],
    'RMSE': [3155.4428640331175, 2328.080497374315, 2259.878030906163, 1615.3803898078384, 1615.4275074653872, 1575.5393829849745]
}

models_table = pd.DataFrame(data_models)

print(models_table)

#GRAFICA
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
fig.suptitle('Análisis de velocidad y calidad por modelo', fontsize=19)
plt.subplots_adjust(hspace=0.4)

# Gráfico 1: Tiempo de ajuste de hiperparámetros
axs[0, 0].barh(models_table['modelo'], models_table['tiempo_ajuste_hiperparámetros'], color='b')
axs[0, 0].set_xlabel('Tiempo de Ajuste de Hiperparámetros (s)')
axs[0, 0].set_title('Tiempo de Ajuste de Hiperparámetros por Modelo')

# Gráfico 2: Tiempo de entrenamiento
axs[0, 1].barh(models_table['modelo'], models_table['tiempo_de_entrenamiento'], color='g')
axs[0, 1].set_xlabel('Tiempo de Entrenamiento (s)')
axs[0, 1].set_title('Tiempo de Entrenamiento por Modelo')

# Gráfico 3: Tiempo de prueba
axs[1, 0].barh(models_table['modelo'], models_table['tiempo_de_prueba'], color='orange')
axs[1, 0].set_xlabel('Tiempo de Prueba (s)')
axs[1, 0].set_title('Tiempo de Prueba por Modelo')

# Gráfico 4: RMSE
axs[1, 1].barh(models_table['modelo'], models_table['RMSE'], color='r')
axs[1, 1].set_xlabel('RMSE')
axs[1, 1].set_title('RMSE por Modelo')

# Mostrar el gráfico
plt.show()
