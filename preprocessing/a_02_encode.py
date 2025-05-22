#a_02_encode
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Convertir 'not_repaired' a binario (1/0) ANTES de cualquier encoding
    if 'not_repaired' in df.columns:
        df['not_repaired'] = df['not_repaired'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    
    # 2. Frequency encoding para 'brand' y 'model' (como ya lo haces)
    for col in ['brand', 'model']:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[col + '_freq'] = df[col].map(freq)  # Nueva columna
            df.drop(col, axis=1, inplace=True)  # Elimina la original
    
    # 3. One-Hot Encoding para categóricas (¡asegúrate de que existen!)
    categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type']
    existing_cats = [col for col in categorical_cols if col in df.columns]
    
    if existing_cats:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        ohe_data = encoder.fit_transform(df[existing_cats])
        ohe_cols = encoder.get_feature_names_out(existing_cats)
        df_ohe = pd.DataFrame(ohe_data, columns=ohe_cols, index=df.index)
        df = pd.concat([df.drop(existing_cats, axis=1), df_ohe], axis=1)
    
    # 4. Escalado numérico (verifica columnas)
    numerical_cols = [col for col in ['power', 'mileage', 'vehicle_age'] if col in df.columns]
    if numerical_cols:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # 5. Transformación logarítmica (opcional)
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])
    
    # Verificación final: ¿Todas las columnas son numéricas?
    non_numeric = df.select_dtypes(exclude=['number']).columns
    if not non_numeric.empty:
        raise ValueError(f"Columnas no numéricas después de encoding: {non_numeric}")
    
    return df