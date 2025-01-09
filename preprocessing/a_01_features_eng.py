#features_eng

#INGENIERÍA DE CARACTERÍSTICAS
actual_year = 2024
df_new_filt['vehicle_age'] = actual_year - df_new_filt['registration_year']
df_new_filt.rename(columns={'registration_year' : 'vehicle_age'})
print(df_new_filt)
print()
print(df_new_filt['vehicle_age'].describe())

#vehicle_age es igual al máximo encontrado
max_age = df_new_filt['vehicle_age'].max()
outliers = df_new_filt[df_new_filt['vehicle_age'] == max_age]
print(outliers)

#Elimino al ser un dato practicamente superfluo en nuestro análisis
df_new_filt = df_new_filt[df_new_filt['vehicle_age'] != 114]
df_new_filt.reset_index(drop=True, inplace=True)
print(df_new_filt.loc[df_new_filt['vehicle_age'] == 114])
print(df_new_filt['vehicle_age'].describe())

#mileage_per_year. Kilometraje por año
if 'mileage' in df_new.columns:
    df_new_filt['mileage_per_year'] = df_new_filt['mileage'] / df_new_filt['vehicle_age']

print(df_new_filt)
print()
print(df_new_filt['mileage_per_year'].describe())

#Último ajuste
df_new_filt = df_new_filt.drop(['registration_month'], axis=1)
#observo cambios
print(df_new_filt.columns)
