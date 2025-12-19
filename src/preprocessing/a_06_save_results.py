#Guardado de Outputs métricos de modelos
#a_06_save_results.py
import json

import pandas as pd
from typing import Dict, Any

from src.utils.helpers import directs
from src.utils.config_manager import load_paths

from pathlib import Path

PATHS = load_paths()["dirs"]

def save_metrics(results: Dict[str, Dict[str, Any]]) -> None:
    #Guarda las métricas de los modelos en un CSV.
    directs()
    
    # Convertir resultados a DataFrame
    metrics_df = pd.DataFrame({
        'model': results.keys(),
        'rmse': [data['rmse'] for data in results.values()]
    })
    
    metrics_df.to_csv(Path(PATHS["metrics"]) / "selected_models.csv", index=False)

    metrics_json = {
        'best_model': min(results.items(), key=lambda x: x[1]['rmse'])[0],
        'details': {model: {'rmse': data['rmse']} for model, data in results.items()}
        }
    with open(Path(PATHS['metrics']) / "selected_models.json", 'w') as f:
        json.dump(metrics_json, f, indent=4)