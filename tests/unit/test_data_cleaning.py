#test_data_cleaning
import pytest

import pandas as pd
import numpy as np

from src.preprocessing.a_00_data_cleaning import filter_data, process_missing_values

def test_filter_data():
    #Outliers (año 1899, precio 99, potencia 3000)
    data = {
        "registration_year": [2020, 1899, 2015, 2005],
        "vehicle_type": ["sedan", np.nan, "coupe", "wagon"],
        "gearbox": ["manual", np.nan, "auto", "manual"],
        "not_repaired": ["yes", "no", np.nan, "no"],
        "registration_month": [6, 0, 12, 0],
        "brand": ["vw", "bmw", "audi", "mercedes"],
        "model": ["golf", "serie3", "a4", "clase-a"],
        "price": [5000, 99, 10000, 200],
        "power": [100, 200, 3000, 40]
    }

    df = pd.DataFrame(data)
    df_filtered = filter_data(df)

    assert df_filtered["registration_year"].min() >= 1900
    assert df_filtered["price"].min() >= 100
    assert df_filtered["power"].max() <= 2000

    assert (df_filtered["registration_month"] == 0).sum() == 0
    assert 6 in df_filtered["registration_month"].values
    assert df_filtered["registration_month"].between(1, 12).all()

def test_process_missing_values():
    data = {
        "vehicle_type": ["sedan", "suv", "coupe", "wagon"],
        "gearbox": ["manual", "auto", "auto", "manual"],
        "not_repaired": ["yes", "no",np.nan, "no"],
        "brand": ["vw", "bmw", "audi", "mercedes"],
        "model": ["golf", "serie3", "a4", "clase-a"],
        "fuel_type": ["gasoline", "diesel", "diesel", "electric"],
        "registration_year": [2020, 2019, 2018, 2021],
        "power": [100, 200, 150, 180]
    }

    df = pd.DataFrame(data)
    df_processed = process_missing_values(df)
    
    assert df_processed["gearbox"].isnull().sum() == 0
    assert df_processed["not_repaired"].isin([0, 1]).all()
    assert "manual" in df_processed["gearbox"].values