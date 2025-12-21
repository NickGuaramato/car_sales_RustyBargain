#main.py
import warnings
import time
from pathlib import Path

from src.preprocessing.a_00_data_cleaning import preprocess_data
from src.preprocessing.a_01_feature_engineering import features_engineer
from src.preprocessing.a_02_encoding import encode_data
from src.preprocessing.a_03_split_data import prepare_all_splits
from src.models.a_05_train import TES
from src.preprocessing.a_06_save_results import save_metrics

from src.utils.config_manager import load_paths
from src.utils.helpers import directs
from src.utils.logging_config import setup_logging
warnings.filterwarnings("ignore")
PATHS_CONFIG = load_paths()
PATHS_DIRS = PATHS_CONFIG["dirs"]
PATHS_FILES = PATHS_CONFIG["files"]

logger = setup_logging(module='main')

def main():
    try:
        start_time = time.time()
        logger.info("Iniciando pipeline de entrenamiento...")

        directs()
        logger.debug("Directorios verificados")

        #Preprocesamiento
        logger.info("Fase 1/4: Preprocesamiento (Antes de features_eng)")
        df, _ = preprocess_data(Path(PATHS_FILES["raw_data"]))

        #Ingeniería de características
        logger.info("Fase 2/4: Procesamiento (con features_eng)")
        df = features_engineer(df)

        #Encodificado
        print("Columnas después de feature_engineering:", df.columns.tolist())

        logger.info("Fase 3/4: Encodificando...")
        df_encoded = encode_data(df, save_encoders=True)
        
        #Modelado
        logger.info("Fase 4/4: Entrenamiento de modelos")
        splits = prepare_all_splits(df, df_encoded)
        results = TES(splits)
        save_metrics(results)

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completado en {elapsed:.2f} segundos")

        return {
            "status": "success",
            "elapsed_time": elapsed,
            "best_model": min(results.items(), key=lambda x: x[1]['rmse'])[0],
            "metrics": {k: v['rmse'] for k, v in results.items()}
        }
    
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}", exc_info=True)
        raise

#Control de ejecución
if __name__ == "__main__":
    main()