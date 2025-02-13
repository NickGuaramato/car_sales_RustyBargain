#train_test_split
from sklearn.model_selection import train_test_split

from a_00_Preprocessed import categorical_columns
from a_01_features_eng import df_new_filt
from a_02_encode import df_new_filt_OHE, log_price

#ENTRENAMIENTO DE MODELO
#Características y Objetivo
#Dividiendo el Dataset y verificando conjunto
X = df_new_filt_OHE.drop('price', axis=1)
y = df_new_filt_OHE['price']
y_log = df_new_filt_OHE['log_price']

#entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)

#entrenamiento y prueba (transformación logarítmica)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.25, random_state=12345)

#CATBOOST
#Características y objetivos antes de OHE
features = df_new_filt.drop('price', axis=1)
target = df_new_filt['price']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=12345)

#Selecciono sólo las columnas cuyas características son categóricas
categorical_columns = features_train.select_dtypes(include=['object']).columns.tolist()

#Tipo 'category'
for column in categorical_columns:
    features_train.loc[:, column] = features_train.loc[:, column].astype('category')
    features_test.loc[:, column] = features_test.loc[:, column].astype('category')

#CATBOOST LOGARÍTMICO
#Agrego la transformación logarítmica de la variable objetivo
df_new_filt['log_price'] = log_price
#variable objetivo logarítmica
target_log = df_new_filt['log_price']

features_train_log, features_test_log, target_train_log, target_test_log = train_test_split(features, target_log, test_size=0.25, random_state=12345)

#LIGHTGBM
#Igual que catboost, tomo el conjunto para antes de OHE
for col in categorical_columns:
    df_new_filt[col] = df_new_filt[col].astype('category')

X_LGBM = df_new_filt.drop('price', axis=1)
y_LGBM = df_new_filt['price']

X_LGBM_train, X_LGBM_test, y_LGBM_train, y_LGBM_test = train_test_split(X_LGBM, y_LGBM, test_size=0.25, random_state=12345)

#LGBM LOGARÍTMICO
#variable objetivo logarítmica
y_LGBM_log = df_new_filt['log_price']

X_LGBM_train_log, X_LGBM_test_log, y_LGBM_train_log, y_LGBM_test_log = train_test_split(X_LGBM, y_LGBM_log, test_size=0.25, random_state=12345)