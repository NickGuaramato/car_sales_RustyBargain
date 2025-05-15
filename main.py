import warnings
warnings.filterwarnings("ignore")

def main():
    print("Iniciando pipeline de entrenamiento...")

    # 1. Preprocesamiento
    print("Ejecutando preprocesamiento...")
    from a_00_Preprocessed import df_new_filt

    # 2. Ingeniería de características
    print("Ejecutando ingeniería de características...")
    from a_01_features_eng import df_new_filt as df_features

    # 3. Codificación
    print("Codificando variables...")
    from a_02_encode import df_new_filt_OHE, log_price

    # 4. División de datos
    print("Dividiendo conjuntos de entrenamiento y prueba...")
    import a_03_train_test_split as split

    # 5. Entrenamiento de modelos
    print("Entrenando modelos...")
    import a_05_models as models

    print("Pipeline completado con éxito.")

if __name__ == "__main__":
    main()