# predict_encoder.py
import pandas as pd
import joblib
from pathlib import Path

class PredictEncoder:
    def __init__(self, encoders_dir: Path = Path("artifacts/encoders/")):
        self.encoders_dir = encoders_dir
        self.load_encoders()
    
    def load_encoders(self):
        """Carga todos los encoders guardados"""
        self.brand_freq = joblib.load(self.encoders_dir / "brand_freq.joblib")
        self.model_freq = joblib.load(self.encoders_dir / "model_freq.joblib")
        self.ohe_encoder = joblib.load(self.encoders_dir / "onehot_encoder.joblib")
        self.scaler = joblib.load(self.encoders_dir / "standard_scaler.joblib")
        self.expected_features = joblib.load(self.encoders_dir / "expected_features.joblib")
        
        # OBTENER LAS COLUMNAS QUE EL ENCODER ESPERA
        self.ohe_columns = list(self.ohe_encoder.feature_names_in_)
        print(f"🔍 DEBUG - OHE encoder espera columns: {self.ohe_columns}")
    
    def encode_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica un solo registro usando encoders pre-entrenados"""
        df = df.copy()
        
        print(f"🔍 DEBUG encode_single - Input dtypes: {df.dtypes.to_dict()}")
        print(f"🔍 DEBUG encode_single - Input columns: {list(df.columns)}")
        
        # 1. Frequency encoding (con default 0 si no existe)
        # Asegurar que brand y model sean strings
        if 'brand' in df.columns:
            df['brand'] = df['brand'].astype(str)
        if 'model' in df.columns:
            df['model'] = df['model'].astype(str)
        
        df['brand_freq'] = df['brand'].map(self.brand_freq).fillna(0)
        df['model_freq'] = df['model'].map(self.model_freq).fillna(0)
        
        # 2. OneHotEncoding - USAR SOLO LAS COLUMNAS QUE EL ENCODER ESPERA
        print(f"🔍 DEBUG - OHE columns esperadas: {self.ohe_columns}")
        
        # Asegurar que todas las columnas categóricas existan
        for col in self.ohe_columns:
            if col not in df.columns:
                print(f"⚠️  Columna {col} no encontrada, añadiendo con valor 'unknown'")
                df[col] = 'unknown'
            df[col] = df[col].astype(str)
            print(f"🔍 DEBUG - {col} value: {df[col].iloc[0]}")
        
        # Transformar con OHE - SOLO las columnas que el encoder conoce
        try:
            ohe_data = self.ohe_encoder.transform(df[self.ohe_columns])
            ohe_cols = self.ohe_encoder.get_feature_names_out(self.ohe_columns)
            df_ohe = pd.DataFrame(ohe_data, columns=ohe_cols, index=df.index)
        except ValueError as e:
            print(f"❌ ERROR en OHE transform: {e}")
            print(f"❌ Valores categóricos: {df[self.ohe_columns].iloc[0].to_dict()}")
            raise
        
        # 3. Eliminar originales y añadir OHE
        cols_to_drop = self.ohe_columns + ['brand', 'model']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(cols_to_drop, axis=1)
        df = pd.concat([df, df_ohe], axis=1)
        
        # 4. Scaling - solo si tenemos las columnas
        numerical_cols = ['power', 'mileage', 'vehicle_age']
        existing_numerical = [col for col in numerical_cols if col in df.columns]
        if existing_numerical:
            df[existing_numerical] = self.scaler.transform(df[existing_numerical])
        
        # 5. Asegurar todas las features (rellenar faltantes con 0)
        print(f"🔍 DEBUG - Expected features: {len(self.expected_features)}")
        missing_features = []
        for feat in self.expected_features:
            if feat not in df.columns:
                df[feat] = 0
                missing_features.append(feat)
        
        if missing_features:
            print(f"🔍 DEBUG - Features añadidas (valor 0): {missing_features}")
        
        print(f"🔍 DEBUG - Final columns before reorder: {list(df.columns)}")
        
        return df[self.expected_features]  # Orden exacto