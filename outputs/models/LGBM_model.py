#LGBM_model

from lightgbm import LGBMRegressor

"""
#entrenamiento
LGBM_model = LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=30, max_depth=10, subsample=0.6, random_state=12345)
LGBM_model.fit(X_LGBM_train, y_LGBM_train)
LGBM_predict = LGBM_model.predict(X_LGBM_test)
LGBM_rmse = mean_squared_error(y_LGBM_test, LGBM_predict)**0.5
"""

#entrenamiento logarítmico
LGBM_model_log = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=30, max_depth=5, subsample=0.6, random_state=12345)
LGBM_model_log.fit(X_LGBM_train_log, y_LGBM_train_log)
LGBM_predict_log = LGBM_model_log.predict(X_LGBM_test_log)

#revirtiendo transformación
LGBM_predict_original = np.exp(LGBM_predict_log)
#RMSE en espacio original
LGBM_rmse_original = mean_squared_error(np.exp(y_LGBM_test_log), LGBM_predict_original) ** 0.5