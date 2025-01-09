#CB_model

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