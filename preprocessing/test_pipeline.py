#test_pipeline.py

#Pruebas unitarias y validaciones automáticas
#Asegurando que los módulos de pipeline no funcionen mal silenciosamente

import pandas as pd
import pytest

from a_00_Preprocessed import df_new_filt
from a_02_encode import log_price
from a_05_models import data_models

def test_preprocessed_no_empty():
    assert not df_new_filt.empty, "DataFrame preprocesado está vacío."

def test_log_price_no_null():
    assert log_price.isnull().sum() == 0, "log_price contiene valores nulos."

def test_format_metrics():
    the_columns = {'modelo', 'tiempo de entrenamiento', 'tiempo de predicción', 'RMSE', 'Logarítmico'}
    assert set(data_models.columns) == the_columns, "métricas con formato incorrecto."