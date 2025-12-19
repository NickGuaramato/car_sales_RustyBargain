#test_feature_engineering

import pytest

import pandas as pd
import numpy as np

from src.preprocessing.a_01_feature_engineering import vehicle_age, mileage_per_year

def test_vehicle_age():
    df = pd.DataFrame({"registration_year": [2010, 2000, 1910, 1850]})
    df = vehicle_age(df, current_year=2024, fixed_outlier=114)
    
    assert "vehicle_age" in df.columns
    assert df["vehicle_age"].tolist() == [14, 24]
    assert len(df) == 2

def test_mileage_per_year():
    df = pd.DataFrame({
        "mileage": [10000, 50000, 80000, 20000],
        "vehicle_age": [5, 0, 10, np.nan]
    })
    #Test con fill_na=False para ver NaN
    df = mileage_per_year(df, fill_na=False)
    
    assert "mileage_per_year" in df.columns
    assert df.loc[0, "mileage_per_year"] == 2000
    assert df.loc[1, "mileage_per_year"] == 0
    assert pd.isna(df.loc[3, "mileage_per_year"])