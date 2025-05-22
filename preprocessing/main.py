#Orquestador
import os
from pathlib import Path  # Mejor práctica moderna que os.makedirs

from a_00_Preprocessed import preprocess_data
from a_01_features_eng import features_engineer
from a_02_encode import encode_data
from a_03_train_test_split import prepare_all_splits
from a_05_models import TES
from a_06_save_outputs import save_metrics

from utils import directs
from test_pipeline import CSV_PATH

import warnings
warnings.filterwarnings("ignore")

#print por logs automáticos ajustar: nivel de detalle, guardarlos y desactivar mensajes en producción

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("outputs/logs/entrenamiento.log"),
    logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    directs()
    logger.info("Iniciando pipeline de entrenamiento...")

    #Preprocesamiento
    df, _ = preprocess_data(CSV_PATH)
    df = features_engineer(df)
    df_encoded = encode_data(df)
    
    # División
    splits = prepare_all_splits(df, df_encoded)
    
    # Entrenamiento
    results = TES(splits)
    
    # Métricas
    save_metrics(results)
    logger.info("Pipeline completado")

#Control de ejecución
if __name__ == "__main__":
    main()