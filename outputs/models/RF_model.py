#RF_model

#BOSQUE ALEATORIO
"""#Entrenamiento de modelo
rf_model = RandomForestRegressor(random_state=12345, max_depth=6, min_samples_leaf=6, min_samples_split=6, n_estimators=40)
rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_predict)**0.5
print(f'RMSE de Bosque Aleatorio: {rf_rmse}')"""

#Entrenamiento de modelo logarítmico
rf_model_log = RandomForestRegressor(random_state=12345, max_depth=6, min_samples_leaf=8, min_samples_split=6, n_estimators=10)
rf_model_log.fit(X_train_log, y_train_log)
rf_predict_log = rf_model_log.predict(X_test_log)

#revirtiendo transformación
rf_predict_original = np.exp(rf_predict_log)
#RMSE en espacio original
rf_rmse_original = mean_squared_error(np.exp(y_test_log), rf_predict_original) ** 0.5
