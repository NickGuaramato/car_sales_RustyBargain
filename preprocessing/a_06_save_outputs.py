#Guardado de Outputs métricos de modelos
#a_06_save_outputs.py
import pandas as pd
from typing import Dict, Any

from utils import directs

def save_metrics(results: Dict[str, Dict[str, Any]]) -> None:
    #Guarda las métricas de los modelos en un CSV.
    directs()
    
    # Convertir resultados a DataFrame
    metrics_df = pd.DataFrame({
        'model': results.keys(),
        'rmse': [data['rmse'] for data in results.values()]
    })
    
    metrics_df.to_csv("outputs/reports/modelos_seleccionados.csv", index=False)