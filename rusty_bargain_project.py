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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

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

#Exportanción de estadísticas descriptiva en datos originales
df_stats = df.describe(include='all')
df_stats.to_csv('outputs/reports/original_data_statistics.csv', index=True)


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

#exportación de datos luego de eliminación de duplicados
df_stats = df_new.describe(include='all')
df_stats.to_csv('outputs/reports/unduplicated_data_statistics.csv', index=True)


#VISUALIZANDO DATOS

#Análisis de distribución de columnas categóricas
categorical_columns = ['vehicle_type', 'fuel_type', 'gearbox', 'not_repaired', 'brand']

for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue=col, palette='viridis', legend=False)
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{col}_distribution.png')
    plt.show()

#Distribución. Primeros 50
top50_models = df['model'].value_counts().head(50).index
plt.figure(figsize=(12, 6))
sns.countplot(data=df[df['model'].isin(top50_models)], x='model', order=top50_models)
plt.title('Distribución de Modelos (Primeros 50)')
plt.xlabel('Modelos')
plt.ylabel('Frecuencia')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'outputs/plots/top50_distribution.png')
plt.show()

#Observando mínimos y máximos en columna con datos sospechosamente errados
columns = [
    ('registration_year', 'Año'),
    ('power', 'Potencia'),
    ('registration_month', 'Mes')
]

for col, description in columns:
    print(f'{description} más bajo registrado: {df_new[col].min()}')
    print(f'{description} más alto registrado: {df_new[col].max()}')

#Diagrama de caja
numeric_columns = ['registration_year', 'power', 'registration_month']
         
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=np.log1p(df_new[col]))  #np.log1p(x) aplica log(1+x) y evita log(0)
    plt.title(f'Distribución de {col} (en escala logarítmica)')
    plt.xlabel(f'{col} (log)')
    plt.savefig(f'outputs/plots/boxplot_log_{col}.png')
    plt.show()

#Estadísticas adicionales
for col in numeric_columns:
    Q1 = df_new[col].quantile(0.25)
    Q3 = df_new[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_new[(df_new[col] < Q1 - 1.5 * IQR) | (df_new[col] > Q3 + 1.5 * IQR)]
    print(f'{col}: {len(outliers)} valores atípicos detectados.')

outliers_month = df_new[(df_new['registration_month'] < 1)]
print(f"registration_month: {len(outliers_month)} valores atípicos fuera del rango esperado (1-12).")

#Valores inusuales que no son detectados estadísticamente
plt.figure(figsize=(10, 5))
sns.histplot(df_new['registration_month'], bins=12, kde=False)
plt.title('Distribución atípica registration_month')
plt.xlabel('Mes')
plt.ylabel('Frecuencia')
plt.savefig(f'outputs/plots/non-typical_value_hist.png')
plt.show()


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

#Estadística descriptiva luego de limpieza de datos. Guardando
df_stats = df_new_filt.describe(include='all')
df_stats.to_csv('outputs/reports/preprocessed_data_statistics.csv', index=True)

#Guardando dataset con datos preprocesados.
df_new_filt.to_csv('outputs/preprocessed/preprocessed_data.csv', index=False)



#Distribución. Variable objetivo 'price'
log_price = np.log1p(df_new_filt['price'])

plt.figure(figsize=(10, 5))
sns.histplot(log_price, kde=True)
plt.title('Distribución de precios (log)')
plt.xlabel('Log(Price)')
plt.ylabel('Frecuencia')
plt.savefig('outputs/plots/log_price_distribution.png')
plt.show()

#Resumen de estadística descriptiva del precio original respecto a la transformación
#Estadísticas descriptivas del precio original
original_price = df_new_filt['price'].describe()

#Estadísticas descriptivas del precio transformado
log_price_stats = log_price.describe()

print("Estadísticas descriptivas del precio original:")
print(original_price)

print("\nEstadísticas descriptivas del precio transformado (log):")
print(log_price_stats)

#guardar estadísticas
estadisticas = pd.DataFrame({
    "Precio Original": original_price,
    "Precio Transformado (Log)": log_price_stats
})
estadisticas.to_csv('outputs/reports/stats_price(ori)_price(log).csv', index=True)

#Visualización de ambas distribuciones
plt.figure(figsize=(12, 6))

#Precio original
plt.subplot(1, 2, 1)
sns.histplot(df_new_filt['price'], kde=True, color='blue')
plt.title('Distribución de precios originales')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')

#Precio transformado (log)
plt.subplot(1, 2, 2)
sns.histplot(log_price, kde=True, color='green')
plt.title('Distribución de precios (log)')
plt.xlabel('Log(Precio)')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.savefig('outputs/plots/distribution_price(orig)_price(log).png')
plt.show()


#Relación: Precio/variables categóricas
for col in categorical_columns:
    mean_price = df_new_filt.groupby(col)['price'].mean().sort_values(ascending=False)
    print(f'Precio promedio por {col}:\n', mean_price)

#histograma
plt.figure(figsize=(12, 6))

for col in categorical_columns:
    mean_price = df_new_filt.groupby(col)['price'].mean().sort_values(ascending=False)
    mean_price_df = mean_price.reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=mean_price_df, x=col, y='price', palette='viridis')

    plt.title(f'Precio promedio por {col}', fontsize=16)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Precio promedio', fontsize=14)
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(f'outputs/plots/bar_price_vs_{col}.png')
    plt.show()

#boxplot
plt.figure(figsize=(12, 6))

for col in categorical_columns:
    #gráfico de cajas
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_new_filt, x=col, y='price', palette='viridis')
    plt.title(f'Distribución del precio por {col}', fontsize=16)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Precio', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/boxplot_price_vs_{col}.png')
    plt.show()


#Observo correlación
numeric_df = df_new_filt.select_dtypes(include=[float, int])
print(numeric_df.corr())

#Graficamos
correlation_matrix = numeric_df.corr()

#Configuramos el tamaño de la figura
plt.figure(figsize=(10, 10))

#heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Matriz de Correlación")
plt.savefig(f'outputs/plots/corr_matrix.png')
plt.show()



#INGENIERÍA DE CARACTERÍSTICAS
actual_year = 2024
df_new_filt['vehicle_age'] = actual_year - df_new_filt['registration_year']
df_new_filt.rename(columns={'registration_year' : 'vehicle_age'})
print(df_new_filt)
print()
print(df_new_filt['vehicle_age'].describe())

#vehicle_age es igual al máximo encontrado
max_age = df_new_filt['vehicle_age'].max()
outliers = df_new_filt[df_new_filt['vehicle_age'] == max_age]
print(outliers)

#Elimino al ser un dato practicamente superfluo en nuestro análisis
df_new_filt = df_new_filt[df_new_filt['vehicle_age'] != 114]
df_new_filt.reset_index(drop=True, inplace=True)
print(df_new_filt.loc[df_new_filt['vehicle_age'] == 114])
print(df_new_filt['vehicle_age'].describe())

#mileage_per_year. Kilometraje por año
if 'mileage' in df_new.columns:
    df_new_filt['mileage_per_year'] = df_new_filt['mileage'] / df_new_filt['vehicle_age']

print(df_new_filt)
print()
print(df_new_filt['mileage_per_year'].describe())

#Último ajuste
df_new_filt = df_new_filt.drop(['registration_month'], axis=1)
#observo cambios
print(df_new_filt.columns)

#Estadística descriptiva para dataset final
df_stats = df_new_filt.describe(include='all')
df_stats.to_csv('outputs/reports/final_statistics_data.csv', index=True)

#Guardo dataset con nuevas características
df_new_filt.to_csv('outputs/preprocessed/prepro_data_eng_charact.csv', index=False)



#ENTRENAMMIENTO DE MODELO

#Codificación y Escalado
categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type', 'brand', 'model', 'not_repaired']
numerical_cols = ['power', 'mileage', 'vehicle_age']

#Asigno a variable que luego será usada
df_new_filt_OHE = df_new_filt.copy()

# Separar columnas con pocas categorías
low_cardinality_cols = ['vehicle_type', 'gearbox', 'fuel_type', 'not_repaired']
high_cardinality_cols = ['brand', 'model']

#Aplicar One-Hot Encoding con prefijos
encoder = OneHotEncoder(sparse_output=False, drop='first')
ohe_encoded = encoder.fit_transform(df_new_filt_OHE[low_cardinality_cols])

#Generar nombres de columnas con prefijos claros
ohe_column_names = [f"{col}_{cat}" for col, categories in zip(low_cardinality_cols, encoder.categories_)
                    for cat in categories[1:]]  # Excluye la primera categoría por 'drop=first'

#Crear DataFrame con nombres ajustados
ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_column_names)

#Reemplazar y concatenar con nombres personalizados
df_new_filt_OHE = df_new_filt_OHE.drop(low_cardinality_cols, axis=1)
df_new_filt_OHE = pd.concat([df_new_filt_OHE, ohe_df], axis=1)

#Codificar por frecuencia para columnas con alta cardinalidad
for col in high_cardinality_cols:
    freq_encoding = df_new_filt_OHE[col].value_counts() / len(df_new_filt_OHE)
    df_new_filt_OHE[col] = df_new_filt_OHE[col].map(freq_encoding)

#Escalar características numéricas
scaler = StandardScaler()
df_new_filt_OHE[numerical_cols] = scaler.fit_transform(df_new_filt_OHE[numerical_cols])

print('Luego de codificación híbrida y escalado:')
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

#Guardado de modelos
models = {
    'LGBM_model': LGBM_model,
    'CatBoost_model': cb_model
}

for name, model in models.items():
    dump(model, f'outputs/models/{name}.joblib')
    print(f"Modelo {name} guardado como outputs/models/{name}.joblib")


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
plt.savefig(f'outputs/plots/models_analysis.png')
# Mostrar el gráfico
plt.show()