#Guardado de Outputs métricos de modelos
#a_06_save_outputs.py
import os

from a_05_models import data_models

os.makedirs("outputs/metrics", exist_ok=True)
data_models.to_csv("outputs/metrics/modelos_seleccionados.csv", index=False)