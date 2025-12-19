#a_02_encode
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.config_manager import load_paths
from src.utils.helpers import directs
from src.utils.logging_config import setup_logging

from pathlib import Path

logger = setup_logging(module='encoding')

def encode_data(df: pd.DataFrame, save_encoders: bool = True) -> pd.DataFrame:
    directs()
    PATHS = load_paths()
    df = df.copy()

    logger.info(f"🚀 Iniciando encoding. Shape: {df.shape}, Columnas: {list(df.columns)}")
    
    if save_encoders:
        brand_freq_to_save = {}
        model_freq_to_save = {}
        
        if 'brand' in df.columns:
            brand_freq_to_save = df['brand'].value_counts(normalize=True).to_dict()
        if 'model' in df.columns:
            model_freq_to_save = df['model'].value_counts(normalize=True).to_dict()

    # 1. Convertir 'not_repaired' a binario (1/0) ANTES de cualquier encoding
    if 'not_repaired' in df.columns:
        df['not_repaired'] = df['not_repaired'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    logger.debug("✅ not_repaired convertido a binario")

    # 2. Frequency encoding para 'brand' y 'model
    for col in ['brand', 'model']:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[col + '_freq'] = df[col].map(freq)  # Nueva columna
            df.drop(col, axis=1, inplace=True)  # Elimina la original
    logger.debug("✅ Frequency encoding aplicado a brand/model")
    
    # 3. One-Hot Encoding para categóricas
    categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type']
    existing_cats = [col for col in categorical_cols if col in df.columns]
    
    if existing_cats:
        logger.debug(f"✅ One-Hot Encoding aplicado a: {existing_cats}")
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        ohe_data = encoder.fit_transform(df[existing_cats])
        ohe_cols = encoder.get_feature_names_out(existing_cats)
        df_ohe = pd.DataFrame(ohe_data, columns=ohe_cols, index=df.index)
        df = pd.concat([df.drop(existing_cats, axis=1), df_ohe], axis=1)
    
    # 4. Escalado numérico (verifica columnas)
    numerical_cols = [col for col in ['power', 'mileage', 'vehicle_age'] if col in df.columns]
    if numerical_cols:
        logger.debug(f"✅ Escalado aplicado a: {numerical_cols}")
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # 5. Transformación logarítmica (opcional)
    if 'price' in df.columns:
        logger.debug("✅ Transformación logarítmica aplicada a price")
        df['log_price'] = np.log1p(df['price'])
    
    # Verificación final: ¿Todas las columnas son numéricas?
    non_numeric = df.select_dtypes(exclude=['number']).columns
    if not non_numeric.empty:
        raise ValueError(f"Columnas no numéricas después de encoding: {non_numeric}")
    
    #guardado con encodificado (data final)
    logger.info(f"💾 Dataset codificado guardado. Shape final: {df.shape}, Columnas numéricas: {len(df.select_dtypes(include=['number']).columns)}")
    df.to_parquet(PATHS["files"]["final_data"])
    

 # GUARDAR ENCODERS SI SE SOLICITA (solo durante entrenamiento)
    if save_encoders:
        logger.info("💾 Encoders guardados en artifacts/encoders/")
        from pathlib import Path
        import joblib
        
        encoders_dir = Path("artifacts/encoders/")
        encoders_dir.mkdir(exist_ok=True)
        
        # 1. Guardar frequency mappings
        joblib.dump(brand_freq_to_save, encoders_dir / "brand_freq.joblib")
        joblib.dump(model_freq_to_save, encoders_dir / "model_freq.joblib")
        
        # 2. Guardar OneHotEncoder
        joblib.dump(encoder, encoders_dir / "onehot_encoder.joblib")
        
        # 3. Guardar StandardScaler (Sólo si existe)
        if 'scaler' in locals():  # ← Verifica si scaler fue definido
            joblib.dump(scaler, encoders_dir / "standard_scaler.joblib")
        
        # 4. Guardar lista de features esperadas
        expected_features = list(df.columns)
        joblib.dump(expected_features, encoders_dir / "expected_features.joblib")
    
    return df