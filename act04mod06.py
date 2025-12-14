# 1. Carga de datos (1 punto)
# Carga el conjunto de datos proporcionado en el material complementario y realiza una exploración
# inicial de los datos. Asegúrate de revisar si existen valores faltantes o anomalías en los datos.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("datos_inmuebles.csv")
print("\n",df.head(),"\n")
print("\n",df.info(),"\n")
print("\n",df.isnull().sum(),"\n")
print("\n",df.describe(),"\n")
# 2. Aplicación de modelos de regresión (5 puntos)
# Regresión Lineal:
# • Aplica un modelo de regresión lineal para predecir el precio de los inmuebles en función de
# las características proporcionadas.
# • Evalúa el desempeño del modelo utilizando el error cuadrático medio (MSE).
# Regresión Polinómica:
# • Transforma las variables de entrada utilizando características polinómicas de grado 2.
# • Ajusta un modelo de regresión lineal sobre los datos transformados.
# • Calcula el error cuadrático medio y compara con la regresión lineal.
# Árbol de Decisión:
# • Implementa un modelo de regresión basado en árboles de decisión.
# • Evalúa el rendimiento del modelo usando MSE y analiza su capacidad de generalización
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
x=df.drop("Precio_USD", axis=1)
y=df["Precio_USD"]
columnas_categoricas=["Ubicación"]
preprocesador=ColumnTransformer(transformers=[("cat",OneHotEncoder(),columnas_categoricas)],remainder="passthrough")
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
from sklearn.linear_model import LinearRegression
modelo_lineal = Pipeline(steps=[('preprocesador', preprocesador),('regresion', LinearRegression())])
modelo_lineal.fit(x_train, y_train)
y_pred_lineal = modelo_lineal.predict(x_test)
mse_lineal = mean_squared_error(y_test, y_pred_lineal)
print("\nMSE - Regresión Lineal:", mse_lineal,"\n")
from sklearn.preprocessing import PolynomialFeatures
modelo_poli = Pipeline(steps=[('preprocesador', preprocesador),('polinomio', PolynomialFeatures(degree=2)),('regresion', LinearRegression())])
modelo_poli.fit(x_train, y_train)
y_pred_poli = modelo_poli.predict(x_test)
mse_poli = mean_squared_error(y_test, y_pred_poli)
print("\nMSE - Regresión Polinómica (grado 2):", mse_poli,"\n")
from sklearn.tree import DecisionTreeRegressor
modelo_arbol = Pipeline(steps=[('preprocesador', preprocesador),('arbol', DecisionTreeRegressor(random_state=42))])
modelo_arbol.fit(x_train, y_train)
y_pred_arbol = modelo_arbol.predict(x_test)
mse_arbol = mean_squared_error(y_test, y_pred_arbol)
print("\nMSE - Árbol de Decisión:", mse_arbol,"\n")
# 3. Análisis de resultados (4 puntos)
# Comparación de modelos:
# • Explica las diferencias en los resultados obtenidos entre los tres modelos de regresión.
# • Compara los valores de MSE y analiza cuál modelo se ajusta mejor a los datos.
# Interpretación de predicciones:
# • Describe los patrones que observaste en la predicción de precios de inmuebles.
# • Explica si el modelo es adecuado para la toma de decisiones en la compra o venta de propiedades.
# Aplicabilidad:
# • Explica en qué casos sería recomendable utilizar regresión lineal, regresión polinómica y
# árboles de decisión para problemas de predicción similares.
print("\n--- COMPARACIÓN DE MODELOS ---")
print(f"Regresión Lineal MSE: {mse_lineal:.1f}")
print(f"Regresión Polinómica MSE: {mse_poli:.1f}")
print(f"Árbol de Decisión MSE: {mse_arbol:.1f}")
import matplotlib.pyplot as plt
plt.figure(figsize=(18, 5))
# A. Regresión Lineal
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lineal, color="skyblue", edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') 
plt.xlabel("Precio Real (USD)")
plt.ylabel("Precio Predicho (USD)")
plt.title("Regresión Lineal")
plt.grid(True)
# B. Regresión Polinómica
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_poli, color="lightgreen", edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Precio Real (USD)")
plt.ylabel("Precio Predicho (USD)")
plt.title("Regresión Polinómica (Grado 2)")
plt.grid(True)
# C. Árbol de Decisión
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_arbol, color="salmon", edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Precio Real (USD)")
plt.ylabel("Precio Predicho (USD)")
plt.title("Árbol de Decisión")
plt.grid(True)

plt.tight_layout()
plt.show()
df_ajustado=df.select_dtypes(include=[np.number])
correlation_matrix = df_ajustado.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación")
plt.show()
# Análisis de Resultados:
# - La regresión polinómica tiende a ajustar mejor los datos si hay relaciones no lineales, aunque 
# puede sobreajustar si se abusa del grado.
# - La regresión lineal es más simple y generaliza mejor si la relación entre las variables es más o 
# menos lineal.
# - Los árboles de decisión capturan relaciones complejas y no lineales, y permiten interpretabilidad, 
# pero pueden sobreajustar si no se controlan los hiperparámetros.
# Patrones:
# - El modelo con menor MSE indica mejor ajuste a los datos de prueba.
# - Los inmuebles con mayor tamaño, más habitaciones y ubicados en el "Centro" tienden a tener precios 
# más altos.
# - El año de construcción muestra una correlación moderada: propiedades más nuevas tienden a tener 
# precios mayores.
# - El mejor modelo en este caso, es el ajuste de regresión lineal
# Recomendaciones:
# - Para decisiones rápidas y explicables, usar regresión lineal.
# - Si se sospecha de relaciones no lineales: usar regresión polinómica (con cuidado).
# - Si se quiere buena precisión y se cuenta con suficientes datos: árbol de decisión o incluso modelos 
# ensamble como RandomForest o XGBoost.


