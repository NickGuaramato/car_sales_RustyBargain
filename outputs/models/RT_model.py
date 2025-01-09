#RT_model

#ÁRBOL DE REGRESIÓN
#Entrenamiento de modelo
dt_model = DecisionTreeRegressor(random_state=12345, max_depth=6, min_samples_split=2, min_samples_leaf=2)
dt_model.fit(X_train, y_train)
dt_predict = dt_model.predict(X_test)
dt_rmse = mean_squared_error(y_test, dt_predict)**0.5
print(f'RMSE de Árbol de Regresión: {dt_rmse}')

#Entrenamiento de modelo (logarítmica)
dt_model_log = DecisionTreeRegressor(random_state=12345, max_depth=6, min_samples_split=8, min_samples_leaf=6)
dt_model_log.fit(X_train_log, y_train_log)
dt_predict_log = dt_model_log.predict(X_test_log)

#revirtiendo transformación
dt_predict_original = np.exp(dt_predict_log)
#RMSE en espacio original
dt_rmse_original = mean_squared_error(np.exp(y_test_log), dt_predict_original) ** 0.5
print(f'RMSE de Árbol de Regresión en espacio original: {dt_rmse_original}')