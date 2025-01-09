#XGB_model

"""#XGBoost
#Entrenamos modelo con hiperparametros
xgb_model = XGBRegressor(random_state=12345, max_depth=8, n_estimators=100, learning_rate=0.1, subsample=0.8)
xgb_model.fit(X_train, y_train)
xgb_predict = xgb_model.predict(X_test)
xgb_rmse = mean_squared_error(y_test, xgb_predict)**0.5
print(f'RMSE de XGBoost: {xgb_rmse}')"""

#Entrenamos modelo con hiperparametros en espacio logarítmico
xgb_model_log = XGBRegressor(random_state=12345, max_depth=4, n_estimators=100, learning_rate=0.1, subsample=0.8)
xgb_model_log.fit(X_train_log, y_train_log)
xgb_predict_log = xgb_model_log.predict(X_test_log)

#revirtiendo transformación
xgb_predict_original = np.exp(xgb_predict_log)
#RMSE en espacio original
xgb_rmse_original = mean_squared_error(np.exp(y_test_log), xgb_predict_original) ** 0.5