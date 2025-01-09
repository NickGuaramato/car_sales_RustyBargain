#EDA_description

import pandas as pd


df = pd.read_csv("dataset/car_data.csv")
print(df.info())

print()
print('Estadística descriptiva de DataFrame:')
print(df.describe())

#Exportanción de estadísticas descriptiva en datos originales
df_stats = df.describe(include='all')
df_stats.to_csv('outputs/reports/original_data_statistics.csv', index=True)

#Observando mínimos y máximos en columna con datos sospechosamente errados
columns = [
    ('registration_year', 'Año'),
    ('power', 'Potencia'),
    ('registration_month', 'Mes')
]

for col, description in columns:
    print(f'{description} más bajo registrado: {df_new[col].min()}')
    print(f'{description} más alto registrado: {df_new[col].max()}')

#Exportanción de estadísticas descriptiva en eliminación de duplicados
df_stats = df_new.describe(include='all')
df_stats.to_csv('outputs/reports/unduplicated_data_statistics.csv', index=True)

#Estadística descriptiva luego de limpieza de datos
df_stats = df_new_filt.describe(include='all')
df_stats.to_csv('outputs/reports/preprocessed_data_statistics.csv', index=True)

#Estadística descriptiva precio original vs logarítmico
estadisticas.to_csv('outputs/reports/stats_price(ori)_price(log).csv', index=True)

#Estadística descriptiva para dataset final
df_stats = df_new_filt.describe(include='all')
df_stats.to_csv('outputs/reports/final_statistics_data.csv', index=True)

#Guardo la tabla de los resultados de los modelos
models_table.to_csv('outputs/reports/models_table_result.csv', index=True)


#ANÁLISIS DE MODELOS
#Ordenando resultado en un DataFrame para mejor visualización
data_models = {
    'modelo': [
        'Regresión Lineal', 'Regresión Lineal (log)', 
        'Árbol de Regresión', 'Árbol de Regresión (log)', 
        'Bosque Aleatorio', 'Bosque Aleatorio (log)', 
        'CatBoost', 'CatBoost (log)', 
        'XGBoost', 'XGBoost (log)', 
        'LightGBM', 'LightGBM (log)'
    ],
    'tiempo_entrenamiento_s': [
        0.372, 0.711, 
        0.771, 0.723, 
        10.9, 4.59, 
        23.9, 12, 
        7.57, 3.67, 
        10.6, 5.19
    ],
    'tiempo_prediccion_s': [
        0.0647, 0.0507, 
        0.171, 0.0562, 
        0.150, 0.0503, 
        0.137, 0.117, 
        0.384, 0.198, 
        1.21, 0.296
    ],
    'RMSE': [
        3060.6223, 5.762930581907668e-11, 
        2335.5843, 120.1818, 
        2258.3020, 118.2869, 
        1622.3060, 5001.0582, 
        1605.2631, 43.7402, 
        1569.2482, 21.0746
    ],
    'logaritmico': [
        False, True, 
        False, True, 
        False, True, 
        False, True, 
        False, True, 
        False, True
    ]
}

models_table = pd.DataFrame(data_models)

#Guardo la tabla
models_table.to_csv('outputs/reports/models_table_result.csv', index=True)