# predict.py
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import logging
from typing import Dict, Union

from src.utils.config_manager import load_paths
from src.utils.helpers import directs, normalize_names
from src.preprocessing.a_00_data_cleaning import convert_data_types
from src.preprocessing.a_01_feature_engineering import vehicle_age, mileage_per_year
from src.models.predict_encoder import PredictEncoder

# Configuración
PATHS = load_paths()
MODEL_DIR = PATHS["dirs"]["selected_models"]
logger = logging.getLogger(__name__)

class CarPricePredictor:
    def __init__(self, model_name: str = "LGBM"):
        directs()
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.features_required = list(self.model.feature_names_in_)  # Features exactas
        self.model_type = self._detect_model_type()

        from src.preprocessing.a_00_data_cleaning import preprocess_data
        from src.preprocessing.a_01_feature_engineering import features_engineer

        # Cargar dataset de entrenamiento para obtener categorías REALES
        df, _ = preprocess_data(PATHS["files"]["raw_data"])
        df_engineered = features_engineer(df)

        self.category_mappings = {}
        self.categorical_cols = ['vehicle_type', 'gearbox', 'fuel_type', 'brand', 'model', 'not_repaired']
        
        # IMPORTANTE: Extraer categorías EXACTAMENTE como se usaron en entrenamiento
        for col in self.categorical_cols:
            if col in df_engineered.columns:
                # Para LightGBM: las categorías deben ser EXACTAMENTE las mismas
                if col == 'not_repaired':
                    # not_repaired se entrenó como int (0/1) según tu código
                    self.category_mappings[col] = [0, 1]
                else:
                    # Obtener categorías ORIGINALES del dataset de entrenamiento
                    categories = list(df_engineered[col].astype('category').cat.categories)
                    self.category_mappings[col] = categories
                print(f"  Categorías para {col}: {self.category_mappings[col][:5]}...")

        print("\n🔍 DEBUG - Features del modelo:")
        for i, feat in enumerate(self.features_required):
            print(f"  {i}: {feat}")

        # Inicializar encoder solo para modelos encoded
        if self.model_type == 'encoded_model':
            self.encoder = PredictEncoder()
        else:
            self.encoder = None
            
        logger.info(f"Modelo {model_name} cargado. Tipo: {self.model_type}")
        logger.info(f"Features requeridas: {len(self.features_required)}")
    
    def _load_model(self, model_name: str):
        """Carga el modelo"""
        model_path = MODEL_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo {model_name} no encontrado")
        return load(model_path)
    
    def _detect_model_type(self) -> str:
        """Detecta si el modelo usa features originales o encoded"""
        # Si tiene 'vehicle_type' en features, es modelo tree (categóricas originales)
        if 'vehicle_type' in self.features_required:
            return 'tree_model'  # LightGBM/CatBoost
        else:
            return 'encoded_model'  # XGBoost/RF/DT
    
    def preprocess_input(self, input_data: Dict) -> pd.DataFrame:
        """Preprocesa entrada según tipo de modelo"""
        # 1. Crear DataFrame
        df = pd.DataFrame([input_data])
        print(f"\n🔍 DEBUG - Input columns: {list(df.columns)}")
        
        # 2. Preprocesamiento básico
        df = normalize_names(df)
        df = self._safe_remove_columns(df)

        print("🔍 DEBUG - Preprocesando para LightGBM...")
        
        # TRATAMIENTO ESPECIAL PARA LIGHTGBM (igual que en entrenamiento)
        if self.model_type == 'tree_model':
            # Convertir a CATEGORICAL EXACTAMENTE como en entrenamiento
            for col in self.categorical_cols:
                if col in df.columns:
                    if col == 'not_repaired':
                        # not_repaired como int (0/1)
                        df[col] = df[col].replace({'yes': 1, 'no': 0}).fillna(0).astype(int)
                        print(f"  {col}: {df[col].dtype} (valor: {df[col].iloc[0]})")
                    else:
                        # Convertir a string primero
                        df[col] = df[col].astype(str)
                        
                        # Verificar si el valor está en las categorías conocidas
                        val = df[col].iloc[0]
                        if col in self.category_mappings and val not in self.category_mappings[col]:
                            print(f"  ⚠️ {col}='{val}' no está en categorías, usando valor por defecto")
                            # Usar la primera categoría disponible
                            default_val = self.category_mappings[col][0] if self.category_mappings[col] else 'unknown'
                            df[col] = pd.Series([default_val], index=df.index)
                            val = default_val
                        
                        # IMPORTANTE: Convertir a CATEGORICAL con las categorías EXACTAS
                        df[col] = pd.Categorical(df[col], categories=self.category_mappings.get(col, []))
                        print(f"  {col}: {df[col].dtype} (valor: '{val}')")
        else:
            # Para otros modelos (XGBoost, etc.)
            for col in self.categorical_cols:
                if col in df.columns:
                    if col == 'not_repaired':
                        df[col] = df[col].replace({'yes': 1, 'no': 0}).fillna(0).astype(int)
                        print(f"  {col}: {df[col].dtype} (valor: {df[col].iloc[0]})")
                    else:
                        df[col] = df[col].astype(str)
                        print(f"  {col}: {df[col].dtype} (valor: {df[col].iloc[0]})")
        
        # 4. Feature engineering
        if 'vehicle_age' not in df.columns and 'registration_year' in df.columns:
            df = vehicle_age(df)
        if 'mileage_per_year' not in df.columns and all(x in df.columns for x in ['mileage', 'vehicle_age']):
            df = mileage_per_year(df, fill_na=True)
        
        # 5. Encoding según tipo
        if self.model_type == 'encoded_model' and self.encoder:
            df = self.encoder.encode_single(df)
        
        # 6. Asegurar features exactas
        return self._align_features(df)
    
    def _safe_remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = ['number_of_pictures', 'date_crawled', 'date_created', 
                       'last_seen', 'postal_code']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        return df.drop(columns=existing_columns, errors='ignore')

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asegura que df tenga exactamente las features que espera el modelo"""
        # Rellenar features faltantes con 0
        for feat in self.features_required:
            if feat not in df.columns:
                df[feat] = 0
        
        # Eliminar features sobrantes
        return df[self.features_required]
    
    def predict(self, input_data: Dict) -> float:
        """Realiza predicción"""
        processed_data = self.preprocess_input(input_data)
        print(f"🔍 DEBUG - Final features: {list(processed_data.columns)}")
        print(f"🔍 DEBUG - Final dtypes: {processed_data.dtypes.tolist()}")
        
        # Debug adicional para LightGBM
        if self.model_type == 'tree_model':
            print(f"🔍 DEBUG - Valores categóricos para LightGBM:")
            for col in self.categorical_cols:
                if col in processed_data.columns:
                    print(f"  {col}: type={type(processed_data[col].iloc[0])}, value={processed_data[col].iloc[0]}")
        
        prediction = self.model.predict(processed_data)[0]
        
        # Revertir transformación logarítmica si es necesario
        if "_log" in self.model_name.lower():
            prediction = np.expm1(prediction)
        
        return round(max(0, prediction), 2)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predice el precio de un vehículo usando modelos entrenados"
    )
    parser.add_argument(
        "-m", "--model",
        default="LGBM",
        help="Nombre del modelo a usar (LGBM, LGBM_log, XGBoost_log, RF_log, DT_log)"
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help='Datos del coche en JSON (ej: \'{"brand":"volkswagen","power":120}\')'
    )
    return parser.parse_args()

def validate_input_data(input_data: Dict) -> bool:
    required_fields = ["brand", "model", "gearbox", "power", "mileage", "registration_year"]
    return all(field in input_data for field in required_fields)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    
    try:
        car_data = json.loads(args.data.replace("'", "\""))
        
        if not validate_input_data(car_data):
            raise ValueError("Faltan campos requeridos en los datos")
        
        predictor = CarPricePredictor(args.model)
        price = predictor.predict(car_data)
        
        print(f"\n🎯 Precio estimado usando {args.model}: ${price:,.2f}\n")
    
    except json.JSONDecodeError:
        print("❌ Error: Formato JSON inválido")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()