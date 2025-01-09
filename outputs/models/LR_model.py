#LR_model

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
print(f'RMSE de Regresión Lineal en espacio original: {lr_rmse_original}')