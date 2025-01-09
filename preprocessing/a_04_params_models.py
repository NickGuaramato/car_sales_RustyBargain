#params_models

#ÁRBOL DE REGRESIÓN
#hiperparámetros de árbol de decisión
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
print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')

#ÁRBOL DE REGRESIÓN LOGARÍTMICO
#entrenamos para encontrar mejores hiperparametros empleando el conjunto logarítmico
dt_grid.fit(X_train_log, y_train_log)
#Buscamos mejores hiperparametros que devuelven más bajo RSME
max_score = dt_grid.cv_results_['mean_test_score'].max()
index_max_score = np.where(dt_grid.cv_results_['mean_test_score'] == max_score)[0][0]

best_set_of_params = dt_grid.cv_results_['params'][index_max_score]
print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')


#BOSQUE ALEATORIO
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

print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')

#BOSQUE ALEATORIO LOGARÍTMICO
rf_grid.fit(X_train_log, y_train_log)
#Buscamos mejores hiperparametros que devuelven más bajo RSME
max_score = rf_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(rf_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_set_of_params = rf_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_set_of_params} | Mejor RSME: {-max_score}')



#CATBOOST
#Selecciono sólo las columnas cuyas características son categóricas
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

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')


#CATBOOST LOGARÍTMICO
cb_grid.fit(features_train_log, target_train_log)

max_score = cb_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(cb_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = cb_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')


#XGBOOOST
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

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')

#XGBOOST LOGARÍTMICO
#Buscamos los mejores hiperparametros en el espacio logarítmico
xgb_grid.fit(X_train_log, y_train_log)
max_score = xgb_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(xgb_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = xgb_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')


#LGBM
LGBM_params = {
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

print(f'Hiperparámetros recomendados: {best_params} | Mejor RECM: {-max_score}')

#LGBM LOGARÍTMICO

LGBM_grid.fit(X_LGBM_train_log, y_LGBM_train_log)

max_score = LGBM_grid.cv_results_["mean_test_score"].max()
index_max_score = np.where(LGBM_grid.cv_results_["mean_test_score"] == max_score)[0][0]

best_params = LGBM_grid.cv_results_["params"][index_max_score]

print(f'Hiperparámetros recomendados: {best_params} | Mejor RSME: {-max_score}')