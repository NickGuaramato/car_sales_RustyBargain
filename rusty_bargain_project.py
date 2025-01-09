#rusty_bargain_project
#CÓDIGO GENERAL DE TRABAJO

#PREPARACIÓN DE DATOS
#librerías
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from joblib import dump

#carpetas para guardado de gráficos, modelos, métricas y reportes de clasificación
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/preprocessed', exist_ok=True)


#ENTRENAMIENTO DE MODELO





#ÁRBOL DE REGRESIÓN
"""#hiperparámetros de árbol de decisión
dt_params = {
    'max_depth': [1, 2, 3, 4, 5, 6] ,
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [2, 4, 6, 8],
}

#GridSearchCV e hiperparámetros establecidos.
#Metríca y valor para validación cruzada
dt_grid = GridSearchCV(
    estimator=DecisionTreeRegressor(),
    param_grid=dt_params,
    scoring='neg_root_mean_squared_error',
    cv=3)

#entrenamos para encontrar mejores hiperparametros
dt_grid.fit(X_train, y_train)
#Buscamos mejores hiperparametros que devuelven más bajo RSME
max_score = dt_grid.cv_results_['mean_test_score'].max()
index_max_score = np.where(dt_grid.cv_results_['mean_test_score'] == max_score)[0][0]

best_set_of_params = dt_grid.cv_results_['params'][index_max_score]
print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')"""


"""
#Entrenamiento de modelo
dt_model = DecisionTreeRegressor(random_state=12345, max_depth=6, min_samples_split=2, min_samples_leaf=2)
dt_model.fit(X_train, y_train)
dt_predict = dt_model.predict(X_test)
dt_rmse = mean_squared_error(y_test, dt_predict)**0.5
print(f'RMSE de Árbol de Regresión: {dt_rmse}')
"""


#ÁRBOL DE REGRESIÓN LOGARÍTMICO
"""#entrenamos para encontrar mejores hiperparametros empleando el conjunto logarítmico
dt_grid.fit(X_train_log, y_train_log)
#Buscamos mejores hiperparametros que devuelven más bajo RSME
max_score = dt_grid.cv_results_['mean_test_score'].max()
index_max_score = np.where(dt_grid.cv_results_['mean_test_score'] == max_score)[0][0]

best_set_of_params = dt_grid.cv_results_['params'][index_max_score]
print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')"""


"""
#Entrenamiento de modelo (logarítmica)
dt_model_log = DecisionTreeRegressor(random_state=12345, max_depth=6, min_samples_split=8, min_samples_leaf=6)
dt_model_log.fit(X_train_log, y_train_log)
dt_predict_log = dt_model_log.predict(X_test_log)

#revirtiendo transformación
dt_predict_original = np.exp(dt_predict_log)
#RMSE en espacio original
dt_rmse_original = mean_squared_error(np.exp(y_test_log), dt_predict_original) ** 0.5
print(f'RMSE de Árbol de Regresión en espacio original: {dt_rmse_original}')
"""


"""#BOSQUE ALEATORIO
rf_params = {
    'n_estimators' : [10, 20, 40],
    'max_depth': [1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [2, 4, 6, 8], 
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=rf_params,
    scoring='neg_root_mean_squared_error',
    cv=3)

#entrenamos para encontrar mejores hiperparametros
rf_grid.fit(X_train, y_train)
#Buscamos mejores hiperparametros que devuelven más bajo RSME
max_score = rf_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(rf_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_set_of_params = rf_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')"""

"""#Entrenamiento de modelo
rf_model = RandomForestRegressor(random_state=12345, max_depth=6, min_samples_leaf=6, min_samples_split=6, n_estimators=40)
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_predict)**0.5
print(f'RMSE de Bosque Aleatorio: {rf_rmse}')"""

"""#BOSQUE ALEATORIO LOGARÍTMICO
rf_grid.fit(X_train_log, y_train_log)
#Buscamos mejores hiperparametros que devuelven más bajo RSME
max_score = rf_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(rf_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_set_of_params = rf_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')"""

"""#Entrenamiento de modelo logarítmico
rf_model_log = RandomForestRegressor(random_state=12345, max_depth=6, min_samples_leaf=8, min_samples_split=6, n_estimators=10)
rf_model_log.fit(X_train_log, y_train_log)
rf_predict_log = rf_model_log.predict(X_test_log)

#revirtiendo transformación
rf_predict_original = np.exp(rf_predict_log)
#RMSE en espacio original
rf_rmse_original = mean_squared_error(np.exp(y_test_log), rf_predict_original) ** 0.5
print(f'RMSE de Bosque Aleatorio en espacio original: {rf_rmse_original}')
"""




"""#Selecciono sólo las columnas cuyas características son categóricas
categorical_columns = features_train.select_dtypes(include=['object']).columns.tolist()

#Tipo 'category'
for column in categorical_columns:
    features_train.loc[:, column] = features_train.loc[:, column].astype('category')
    features_test.loc[:, column] = features_test.loc[:, column].astype('category')

#Hiperparámetros a ajustar
cb_params = {
    'depth': [4, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'random_strength': [1, 5, 10]
}

#Estimador
cb_est = CatBoostRegressor(iterations=100, cat_features=categorical_columns, verbose=False, loss_function='RMSE')

cb_grid = GridSearchCV(
    estimator=cb_est,
    param_grid=cb_params,
    scoring='neg_root_mean_squared_error',
    cv=3
)

cb_grid.fit(features_train, target_train)

max_score = cb_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(cb_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = cb_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')"""


"""cb_model = CatBoostRegressor(random_state=12345, iterations=100, depth=10, learning_rate=0.2, random_strength=1, loss_function='RMSE', cat_features=categorical_columns, verbose=False)
cb_model.fit(features_train, target_train)
cb_predict = cb_model.predict(features_test)
cb_rmse = mean_squared_error(target_test, cb_predict)**0.5
print(f'RMSE de CatBoost: {cb_rmse}')"""



"""cb_grid.fit(features_train_log, target_train_log)

max_score = cb_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(cb_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = cb_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')"""


#Entrenamiento log
"""cb_model_log = CatBoostRegressor(random_state=12345, iterations=100, depth=6, learning_rate=0.1, random_strength=1, loss_function='RMSE', cat_features=categorical_columns, verbose=False)
cb_model_log.fit(features_train_log, target_train_log)
cb_predict_log = cb_model_log.predict(features_test_log)

#revirtiendo transformación
cb_predict_original = np.exp(cb_predict_log)
#RMSE en espacio original
cb_rmse_original = mean_squared_error(np.exp(target_test_log), cb_predict_original) ** 0.5
print(f'RMSE de CatBoost en espacio original: {cb_rmse_original}')"""



"""#XGBOOOST
#Hiperparametros
xgb_params = {'max_depth': [4, 6, 8], 
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.6, 0.8]
}

xgb_est = XGBRegressor()

xgb_grid = GridSearchCV(estimator=xgb_est, param_grid=xgb_params, scoring='neg_root_mean_squared_error', cv=3)

#Buscamos los mejores hiperparametros
xgb_grid.fit(X_train, y_train)
max_score = xgb_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(xgb_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = xgb_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')"""



"""#Entrenamos modelo con hiperparametros
xgb_model = XGBRegressor(random_state=12345, max_depth=8, n_estimators=100, learning_rate=0.1, subsample=0.8)
xgb_model.fit(X_train, y_train)
xgb_predict = xgb_model.predict(X_test)
xgb_rmse = mean_squared_error(y_test, xgb_predict)**0.5
print(f'RMSE de XGBoost: {xgb_rmse}')
"""


"""#XGBOOST LOGARÍTMICO
#Buscamos los mejores hiperparametros en el espacio logarítmico
xgb_grid.fit(X_train_log, y_train_log)
max_score = xgb_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(xgb_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = xgb_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')"""


"""#Entrenamos modelo con hiperparametros en espacio logarítmico
xgb_model_log = XGBRegressor(random_state=12345, max_depth=4, n_estimators=100, learning_rate=0.1, subsample=0.8)
xgb_model_log.fit(X_train_log, y_train_log)
xgb_predict_log = xgb_model_log.predict(X_test_log)

#revirtiendo transformación
xgb_predict_original = np.exp(xgb_predict_log)
#RMSE en espacio original
xgb_rmse_original = mean_squared_error(np.exp(y_test_log), xgb_predict_original) ** 0.5
print(f'RMSE de XGBoost en espacio original: {xgb_rmse_original}')
"""





"""LGBM_params = {
    'n_estimators': [100, 150, 300],
    'learning_rate': [0.1, 0.2, 0.5],
    'num_leaves': [10, 20, 30],
    'max_depth': [5, 8, 10],
    'subsample': [0.6, 0.7, 0.8]
}

LGBM_est = LGBMRegressor(verbose=-1)

LGBM_grid = GridSearchCV(
    estimator=LGBM_est,
    param_grid=LGBM_params,
    scoring='neg_root_mean_squared_error',
    cv=3  
)

#Entrenamos para hallar los mejores hiperparametros
LGBM_grid.fit(X_LGBM_train, y_LGBM_train)

max_score = LGBM_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(LGBM_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = LGBM_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RECM: {-max_score}')"""

"""LGBM_model = LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=30, max_depth=10, subsample=0.6, random_state=12345)
LGBM_model.fit(X_LGBM_train, y_LGBM_train)
LGBM_predict = LGBM_model.predict(X_LGBM_test)
LGBM_rmse = mean_squared_error(y_LGBM_test, LGBM_predict)**0.5
print(f'RMSE de LightGBM: {LGBM_rmse}')"""


#hiperparametros LGBM log
"""LGBM_grid.fit(X_LGBM_train_log, y_LGBM_train_log)

max_score = LGBM_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(LGBM_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = LGBM_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')"""

#Entrenamiento de modelo log LGBM
"""LGBM_model_log = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=30, max_depth=5, subsample=0.6, random_state=12345)
LGBM_model_log.fit(X_LGBM_train_log, y_LGBM_train_log)
LGBM_predict_log = LGBM_model_log.predict(X_LGBM_test_log)

#revirtiendo transformación
LGBM_predict_original = np.exp(LGBM_predict_log)
#RMSE en espacio original
LGBM_rmse_original = mean_squared_error(np.exp(y_LGBM_test_log), LGBM_predict_original) ** 0.5
print(f'RMSE de LightGBM en espacio original: {LGBM_rmse_original}')"""