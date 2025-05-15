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
    logger.info("Iniciando pipeline de entrenamiento...")

    #Preprocesamiento
    logger.info("Ejecutando preprocesamiento...")
    from a_00_Preprocessed import df_new_filt

    #Ingeniería de características
    logger.info("Ejecutando ingeniería de características...")
    from a_01_features_eng import df_new_filt as df_features

    #Codificación
    logger.info("Codificando variables...")
    from a_02_encode import df_new_filt_OHE, log_price

    #División de datos
    logger.info("Dividiendo conjuntos de entrenamiento y prueba...")
    import a_03_train_test_split as split

    #Entrenamiento de modelos
    logger.info("Entrenando modelos...")
    import a_05_models as models
    
    #Guardado de métricas
    logger.info("Guardando métricas...")
    import a_06_save_outputs

    logger.info("Pipeline completado con éxito.")

#Control de ejecución
if __name__ == "__main__":
    main()