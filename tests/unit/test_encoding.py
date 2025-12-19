#test_encoding
import pandas as pd
import numpy as np
from src.preprocessing.a_02_encoding import encode_data

def test_encode_data():
    data = {
        "vehicle_type": ["sedan", "suv", np.nan],
        "gearbox": ["manual", "auto", "manual"],
        "brand": ["vw", "bmw", "audi"]
    }
    df = pd.DataFrame(data)
    df_encoded = encode_data(df)
    
    assert "vehicle_type" not in df_encoded.columns
    #¿Se crearon columnas OHE?
    assert any(col.startswith("vehicle_type_") for col in df_encoded.columns)
    assert "brand_freq" in df_encoded.columns