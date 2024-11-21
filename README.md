# Proyecto: Rusty Bargain - Modelo de Predicción de Valor de Autos Usados 🚗💰

El presente repositorio contiene el proyecto de machine learning desarrollado para la aplicación de autos usados Rusty Bargain. El modelo creado para tal aplicación permite estimar el valor en el mercado de algún automóvil basandose en su historial de vida, especificaciones técnicas y versiones de equipamiento.

## Objetivo del Proyecto 🏆

El proposito del presenta fue desarrollar un modelo que pueda determinar el valor de mercado de los autos usados, optimizando los siguientes aspectos clave:
	1.- Calidad de la predicción
	2.- Velocidad de la predicción
	3.- Tiempo requerido para el entrenamiento

## Exploración y Preprocesamiento de Datos 🔍

Se realizó lo siguiente para la respectiva preparación del conjunto de datos:
	- Manejo de datos ausentes y duplicados donde se eliminó y relleno algunos valores según la relevancia de los mismos.
	- Transformación de valores atípicos ajustando los datos extremos para mantener la coherencia con el mercado.
	- Codificación y escalado en las características para la respectiva preparación del entrenamiento del modelo dado el conjunto de datos pertinente.

## Modelos Probados y Metodología ⚙️

Se implementaron varios modelos de regresión, tanto para los datos originales de la variable objetivo como en su respectiva versión logarítmica. Las configuraciones y métricas empleadas para la optimización de los modelos fueron:
	- Ajuste de hiperparámetros usando la librería GridSearchCV para optimizar las combinaciones clave que demanda cada modelo.
	- Evaluación mediante múltiples pliegues (cv) para asegurar la robustez en las métricas.
	- **Función objetivo**: Minimización de RMSE

## Conclusiones 🚀
### Desempeño de los Modelos
- Según calidad de predicción (RMSE):
	1.- LightGBM (log), XGBoost (log) y Bosque Aleatorio (log) destacan por sus valores más bajos en este puntos
	2.- Árbol de Decisión (log), el cual también posee métricas bastantes competitivas.
- Según velocidad (entrenamiento + predicción):
	1.- Modelos como LightGBM (log), XGBoost (log) y Bosque Aleatorio (log) son rápidos en ambas etapas, siendo estos los favoritos.
	2.- Árbol de Regresión (log), el cual también posee tiempos de predicción muy rápidos, se convierte en un buen candidato al igual que los anteriores.

A parte de los anteriores, LightGBM (no logarítmico) también se ha tomado como un posible candidato, uno el cual se puede observar en los outputs del presente repositorio, más precisamente, en la carpeta outputs/models. Si se desean ver otras métricas importantes, en outputs/reports se pueden hallar algunas de estas.

## Recomendaciones Finales 🏅

Luego de evaluar los 3 puntos importantes y requeridos en este proyecto, los mejores modelos para Rusty Bargain son:
- LightGBM (log): Mejor desempeño general en RMSE, aunque no es el más rápido de todo.
- XGBoost (log): Ofrece equilibrio entre calidad y velocidad.
- Bosque Aleatorio (log): Muy buen desempeño y tiempos razonables.
- Árbol de Regresión (log): Destacando por la rapidez que presenta.
- LightGBM (no logarítmico): Alternativa muy robusta para escenarios o espacio no logarítmicos.

## Uso de Repositorio 🛠️
1.- Clonar repositorio: git clone https://github.com/NickGuaramato/car_sales_RustyBargain.git

2.- Instalar dependencias: pip install -r requirements.txt

3.- Ejecución de proyecto: Se recomienda consultar los notebooks para ver exploración de datos y el entrenamiento de los modelos.

# Autor ✨
Nick A. Guaramato 
[GitHub](https://github.com/NickGuaramato) | [LinkedIn](https://www.linkedin.com/in/nick-a-guaramato)

