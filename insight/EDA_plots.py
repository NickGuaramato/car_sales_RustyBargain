#EDA_plots
#VISUALIZANDO DATOS

for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue=col, palette='viridis', legend=False)
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{col}_distribution.png')
    plt.show()

#Distribución. Primeros 50
top50_models = df['model'].value_counts().head(50).index
plt.figure(figsize=(12, 6))
sns.countplot(data=df[df['model'].isin(top50_models)], x='model', order=top50_models)
plt.title('Distribución de Modelos (Primeros 50)')
plt.xlabel('Modelos')
plt.ylabel('Frecuencia')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'outputs/plots/top50_distribution.png')
plt.show()

#Observando mínimos y máximos en columna con datos sospechosamente errados
columns = [
    ('registration_year', 'Año'),
    ('power', 'Potencia'),
    ('registration_month', 'Mes')
]

for col, description in columns:
    print(f'{description} más bajo registrado: {df_new[col].min()}')
    print(f'{description} más alto registrado: {df_new[col].max()}')

         
for col in numeric_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=np.log1p(df_new[col]))  #np.log1p(x) aplica log(1+x) y evita log(0)
    plt.title(f'Distribución de {col} (en escala logarítmica)')
    plt.xlabel(f'{col} (log)')
    plt.savefig(f'outputs/plots/boxplot_log_{col}.png')
    plt.show()

#Estadísticas adicionales
for col in numeric_columns:
    Q1 = df_new[col].quantile(0.25)
    Q3 = df_new[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_new[(df_new[col] < Q1 - 1.5 * IQR) | (df_new[col] > Q3 + 1.5 * IQR)]
    print(f'{col}: {len(outliers)} valores atípicos detectados.')

outliers_month = df_new[(df_new['registration_month'] < 1)]
print(f"registration_month: {len(outliers_month)} valores atípicos fuera del rango esperado (1-12).")

#Valores inusuales que no son detectados estadísticamente
plt.figure(figsize=(10, 5))
sns.histplot(df_new['registration_month'], bins=12, kde=False)
plt.title('Distribución atípica registration_month')
plt.xlabel('Mes')
plt.ylabel('Frecuencia')
plt.savefig(f'outputs/plots/non-typical_value_hist.png')
plt.show()


#Distribución. Variable objetivo 'price' (logarítmico)
plt.figure(figsize=(10, 5))
sns.histplot(log_price, kde=True)
plt.title('Distribución de precios (log)')
plt.xlabel('Log(Price)')
plt.ylabel('Frecuencia')
plt.savefig('outputs/plots/log_price_distribution.png')
plt.show()


#Visualización de ambas distribuciones
plt.figure(figsize=(12, 6))

#Precio original
plt.subplot(1, 2, 1)
sns.histplot(df_new_filt['price'], kde=True, color='blue')
plt.title('Distribución de precios originales')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')

#Precio transformado (log)
plt.subplot(1, 2, 2)
sns.histplot(log_price, kde=True, color='green')
plt.title('Distribución de precios (log)')
plt.xlabel('Log(Precio)')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.savefig('outputs/plots/distribution_price(orig)_price(log).png')
plt.show()


#Relación: Precio/variables categóricas
for col in categorical_columns:
    mean_price = df_new_filt.groupby(col)['price'].mean().sort_values(ascending=False)
    print(f'Precio promedio por {col}:\n', mean_price)

#histograma
plt.figure(figsize=(12, 6))

for col in categorical_columns:
    mean_price = df_new_filt.groupby(col)['price'].mean().sort_values(ascending=False)
    mean_price_df = mean_price.reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=mean_price_df, x=col, y='price', palette='viridis')

    plt.title(f'Precio promedio por {col}', fontsize=16)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Precio promedio', fontsize=14)
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(f'outputs/plots/bar_price_vs_{col}.png')
    plt.show()

#boxplot
plt.figure(figsize=(12, 6))

for col in categorical_columns:
    #gráfico de cajas
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_new_filt, x=col, y='price', palette='viridis')
    plt.title(f'Distribución del precio por {col}', fontsize=16)
    plt.xlabel(col, fontsize=14)
    plt.ylabel('Precio', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/boxplot_price_vs_{col}.png')
    plt.show()


#Observo correlación
numeric_df = df_new_filt.select_dtypes(include=[float, int])
print(numeric_df.corr())

#Graficamos
correlation_matrix = numeric_df.corr()

#Configuramos el tamaño de la figura
plt.figure(figsize=(10, 10))

#heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Matriz de Correlación")
plt.savefig(f'outputs/plots/corr_matrix.png')
plt.show()

#GRAFICA. ANÁLISIS DE MODELO
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
fig.suptitle('Análisis de Velocidad y Calidad por Modelo', fontsize=19)
plt.subplots_adjust(hspace=0.4)

#Gráfico Tiempo de entrenamiento
axs[0, 0].barh(models_table['modelo'], models_table['tiempo_entrenamiento_s'], color='blue')
axs[0, 0].set_xlabel('Tiempo de Entrenamiento (s)')
axs[0, 0].set_title('Tiempo de Entrenamiento por Modelo')

#Gráfico Tiempo de predicción
axs[0, 1].barh(models_table['modelo'], models_table['tiempo_prediccion_s'], color='green')
axs[0, 1].set_xlabel('Tiempo de Predicción/Ajuste (s)')
axs[0, 1].set_title('Tiempo de Predicción/Ajuste por Modelo')

#Gráfico RMSE
axs[1, 0].barh(models_table['modelo'], models_table['RMSE'], color='red')
axs[1, 0].set_xlabel('RMSE')
axs[1, 0].set_title('RMSE por Modelo')

#Gráfico Comparativa tiempos totales
models_table['tiempo_total_s'] = models_table['tiempo_entrenamiento_s'] + models_table['tiempo_prediccion_s']
axs[1, 1].bar(models_table['modelo'], models_table['tiempo_total_s'], color='purple')
axs[1, 1].set_ylabel('Tiempo Total (s)')
axs[1, 1].set_title('Comparativa de Tiempos Totales (Entrenamiento + Predicción)')
axs[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'outputs/plots/models_analysis.png')
plt.show()