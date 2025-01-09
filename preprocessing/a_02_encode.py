#encode
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from a_00_Preprocessed import log_price
from a_01_features_eng import df_new_filt

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

#Crear nueva columna con transformación logarítmica para 'price'
df_new_filt_OHE['log_price'] = log_price