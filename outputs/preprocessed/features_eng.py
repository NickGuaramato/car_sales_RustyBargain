#features_eng

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

print(df_new_filt)
print()
print(df_new_filt['mileage_per_year'].describe())

#Último ajuste
df_new_filt = df_new_filt.drop(['registration_month'], axis=1)

#Guardo dataset con nuevas características
df_new_filt.to_csv('outputs/preprocessed/prepro_data_eng_charact.csv', index=False)