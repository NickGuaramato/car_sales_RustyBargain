#features_eng
from a_00_Preprocessed import df_new_filt, df_new

#INGENIERÍA DE CARACTERÍSTICAS
actual_year = 2024
df_new_filt['vehicle_age'] = actual_year - df_new_filt['registration_year']
df_new_filt.rename(columns={'registration_year' : 'vehicle_age'})


#vehicle_age es igual al máximo encontrado
max_age = df_new_filt['vehicle_age'].max()
outliers = df_new_filt[df_new_filt['vehicle_age'] == max_age]


#Elimino al ser un dato practicamente superfluo en nuestro análisis
df_new_filt = df_new_filt[df_new_filt['vehicle_age'] != 114]
df_new_filt.reset_index(drop=True, inplace=True)


#mileage_per_year. Kilometraje por año
if 'mileage' in df_new.columns:
    df_new_filt['mileage_per_year'] = df_new_filt['mileage'] / df_new_filt['vehicle_age']

#Último ajuste
df_new_filt = df_new_filt.drop(['registration_month'], axis=1)