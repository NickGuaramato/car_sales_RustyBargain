#models
#MODELOS SELECCIONADOS
import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from a_03_train_test_split import X_train_log, y_train_log, X_test_log, y_test_log

from a_03_train_test_split import X_LGBM_train, y_LGBM_train, X_LGBM_test, y_LGBM_test
from a_03_train_test_split import X_LGBM_train_log, y_LGBM_train_log, X_LGBM_test_log, y_LGBM_test_log

#LGBM_model

#entrenamiento
LGBM_model = LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=30, max_depth=10, subsample=0.6, random_state=12345)
LGBM_model.fit(X_LGBM_train, y_LGBM_train)
LGBM_predict = LGBM_model.predict(X_LGBM_test)
LGBM_rmse = mean_squared_error(y_LGBM_test, LGBM_predict)**0.5

#entrenamiento logarítmico
LGBM_model_log = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=30, max_depth=5, subsample=0.6, random_state=12345)
LGBM_model_log.fit(X_LGBM_train_log, y_LGBM_train_log)
LGBM_predict_log = LGBM_model_log.predict(X_LGBM_test_log)

#revirtiendo transformación
LGBM_predict_original = np.exp(LGBM_predict_log)
#RMSE en espacio original
LGBM_rmse_original = mean_squared_error(np.exp(y_LGBM_test_log), LGBM_predict_original) ** 0.5


#XGBoost Logarítmico
#Entrenamos modelo con hiperparametros en espacio logarítmico
xgb_model_log = XGBRegressor(random_state=12345, max_depth=4, n_estimators=100, learning_rate=0.1, subsample=0.8)
xgb_model_log.fit(X_train_log, y_train_log)
xgb_predict_log = xgb_model_log.predict(X_test_log)

#revirtiendo transformación
xgb_predict_original = np.exp(xgb_predict_log)
#RMSE en espacio original
xgb_rmse_original = mean_squared_error(np.exp(y_test_log), xgb_predict_original) ** 0.5

#BOSQUE ALEATORIO
#Entrenamiento de modelo logarítmico
rf_model_log = RandomForestRegressor(random_state=12345, max_depth=6, min_samples_leaf=8, min_samples_split=6, n_estimators=10)
rf_model_log.fit(X_train_log, y_train_log)
rf_predict_log = rf_model_log.predict(X_test_log)

#revirtiendo transformación
rf_predict_original = np.exp(rf_predict_log)
#RMSE en espacio original
rf_rmse_original = mean_squared_error(np.exp(y_test_log), rf_predict_original) ** 0.5

#ÁRBOL DE REGRESIÓN
#Entrenamiento de modelo (logarítmica)
dt_model_log = DecisionTreeRegressor(random_state=12345, max_depth=6, min_samples_split=8, min_samples_leaf=6)
dt_model_log.fit(X_train_log, y_train_log)
dt_predict_log = dt_model_log.predict(X_test_log)

#revirtiendo transformación
dt_predict_original = np.exp(dt_predict_log)
#RMSE en espacio original
dt_rmse_original = mean_squared_error(np.exp(y_test_log), dt_predict_original) ** 0.5

#Guardado de modelos para uso
models = {
    'LGBM_model_log': LGBM_model_log,
    'XGBoost_model_log': xgb_model_log,
    'RF_model_log': rf_model_log,
    'DT_model_log': dt_model_log,
    'LGBM_model': LGBM_model
}

for name, model in models.items():
    dump(model, f'outputs/models/{name}.joblib')
    print(f"Modelo {name} guardado como outputs/models/{name}.joblib")