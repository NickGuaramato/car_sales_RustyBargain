#models
#LOS MODELOS ESTUDIADOS

#REGRESION LINEAL
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predict = lr_model.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_predict)**0.5
print(f'RMSE de Regresión Lineal: {lr_rmse}')

#REGRESION LINEAL LOGARITMICA
lr_model_log = LinearRegression()
lr_model_log.fit(X_train_log, y_train_log)
lr_predict_log = lr_model_log.predict(X_test_log)

#revirtiendo transformación
lr_predict_original = np.exp(lr_predict_log)
#RMSE en espacio original
lr_rmse_original = mean_squared_error(np.exp(y_test_log), lr_predict_original) ** 0.5
print(f'RMSE de Regresión Lineal es espacio original: {lr_rmse_original}')

#ÁRBOL DE REGRESIÓN
#Entrenamiento de modelo
dt_model = DecisionTreeRegressor(random_state=12345, max_depth=6, min_samples_split=2, min_samples_leaf=2)
dt_model.fit(X_train, y_train)
dt_predict = dt_model.predict(X_test)
dt_rmse = mean_squared_error(y_test, dt_predict)**0.5
print(f'RMSE de Árbol de Regresión: {dt_rmse}')

dt_model_log = DecisionTreeRegressor(random_state=12345, max_depth=6, min_samples_split=8, min_samples_leaf=6)
dt_model_log.fit(X_train_log, y_train_log)
dt_predict_log = dt_model_log.predict(X_test_log)

#revirtiendo transformación
dt_predict_original = np.exp(dt_predict_log)
#RMSE en espacio original
dt_rmse_original = mean_squared_error(np.exp(y_test_log), dt_predict_original) ** 0.5
print(f'RMSE de Árbol de Regresión en espacio original: {dt_rmse_original}')


#BOSQUE ALEATORIO
rf_model = RandomForestRegressor(random_state=12345, max_depth=6, min_samples_leaf=6, min_samples_split=6, n_estimators=40)
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_predict)**0.5
print(f'RMSE de Bosque Aleatorio: {rf_rmse}')

#Entrenamiento de modelo logarítmico
rf_model_log = RandomForestRegressor(random_state=12345, max_depth=6, min_samples_leaf=8, min_samples_split=6, n_estimators=10)
rf_model_log.fit(X_train_log, y_train_log)
rf_predict_log = rf_model_log.predict(X_test_log)

#revirtiendo transformación
rf_predict_original = np.exp(rf_predict_log)
#RMSE en espacio original
rf_rmse_original = mean_squared_error(np.exp(y_test_log), rf_predict_original) ** 0.5
print(f'RMSE de Bosque Aleatorio en espacio original: {rf_rmse_original}')

#CatBoost
#Entrenamiento
cb_model = CatBoostRegressor(random_state=12345, iterations=100, depth=10, learning_rate=0.2, random_strength=1, loss_function='RMSE', cat_features=categorical_columns, verbose=False)
cb_model.fit(features_train, target_train)
cb_predict = cb_model.predict(features_test)
cb_rmse = mean_squared_error(target_test, cb_predict)**0.5
print(f'RMSE de CatBoost: {cb_rmse}')

#CatBoost Logarítmico
cb_model_log = CatBoostRegressor(random_state=12345, iterations=100, depth=6, learning_rate=0.1, random_strength=1, loss_function='RMSE', cat_features=categorical_columns, verbose=False)
cb_model_log.fit(features_train_log, target_train_log)
cb_predict_log = cb_model_log.predict(features_test_log)

#revirtiendo transformación
cb_predict_original = np.exp(cb_predict_log)
#RMSE en espacio original
cb_rmse_original = mean_squared_error(np.exp(target_test_log), cb_predict_original) ** 0.5
print(f'RMSE de CatBoost en espacio original: {cb_rmse_original}')

#XGBoost
#Entrenamos modelo con hiperparametros
xgb_model = XGBRegressor(random_state=12345, max_depth=8, n_estimators=100, learning_rate=0.1, subsample=0.8)
xgb_model.fit(X_train, y_train)
xgb_predict = xgb_model.predict(X_test)
xgb_rmse = mean_squared_error(y_test, xgb_predict)**0.5
print(f'RMSE de XGBoost: {xgb_rmse}')


#XGBoost Logarítmico
#Entrenamos modelo con hiperparametros en espacio logarítmico
xgb_model_log = XGBRegressor(random_state=12345, max_depth=4, n_estimators=100, learning_rate=0.1, subsample=0.8)
xgb_model_log.fit(X_train_log, y_train_log)
xgb_predict_log = xgb_model_log.predict(X_test_log)

#revirtiendo transformación
xgb_predict_original = np.exp(xgb_predict_log)
#RMSE en espacio original
xgb_rmse_original = mean_squared_error(np.exp(y_test_log), xgb_predict_original) ** 0.5
print(f'RMSE de XGBoost en espacio original: {xgb_rmse_original}')

#LGBM_model
#entrenamiento
LGBM_model = LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=30, max_depth=10, subsample=0.6, random_state=12345)
LGBM_model.fit(X_LGBM_train, y_LGBM_train)
LGBM_predict = LGBM_model.predict(X_LGBM_test)
LGBM_rmse = mean_squared_error(y_LGBM_test, LGBM_predict)**0.5
print(f'RMSE de LightGBM: {LGBM_rmse}')

#entrenamiento logarítmico
LGBM_model_log = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=30, max_depth=5, subsample=0.6, random_state=12345)
LGBM_model_log.fit(X_LGBM_train_log, y_LGBM_train_log)
LGBM_predict_log = LGBM_model_log.predict(X_LGBM_test_log)

#revirtiendo transformación
LGBM_predict_original = np.exp(LGBM_predict_log)
#RMSE en espacio original
LGBM_rmse_original = mean_squared_error(np.exp(y_LGBM_test_log), LGBM_predict_original) ** 0.5
print(f'RMSE de LightGBM en espacio original: {LGBM_rmse_original}')