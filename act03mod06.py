# Te han contratado para evaluar un dataset aplicando los conceptos relacionados con el 
# preprocesamiento de datos, la codificación de variables categóricas, el escalamiento de datos y
# su implementación utilizando la librería Scikit-Learn. El trabajo consta de preguntas de reflexión 
# y un caso práctico.

# 1. Carga de datos (1 punto)
# • Descarga el archivo customer_data.csv proporcionado en el material complementario.
# • Carga el conjunto de datos utilizando Pandas.
# • Muestra las primeras 5 filas del dataset.

import pandas as pd
df=pd.read_csv("customer_data.csv")
print("\n",df,"\n")
print("\n",df.head(),"\n")

# 2. Preprocesamiento de datos (3 puntos)
# • Limpieza de datos:
#   o Verifica si hay valores nulos en el dataset y elimina las filas que los contengan.
#   o Elimina columnas que no sean relevantes para el análisis (por ejemplo, columnas de identificación).
# • Codificación de variables categóricas:
#   o Aplica Label Encoding a la columna Gender (Género).
#   o Aplica One-Hot Encoding a la columna City (Ciudad).
# • Escalamiento de datos:
#   o Aplica Min-Max Scaling a la columna Age (Edad).
#   o Aplica Standard Scaling a la columna Income (Ingresos).

print(df.isnull().sum())
df_cleaned=df.dropna()
df_cleaned=df_cleaned.drop(columns=["CustomerID"])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_cleaned["Gender"]=le.fit_transform(df_cleaned["Gender"]) # Male = 1, Female = 0
print("\n",df_cleaned,"\n")
df_encoded=pd.get_dummies(df_cleaned,columns=["City"],prefix="City")
print("\n",df_encoded,"\n")
from sklearn.preprocessing import MinMaxScaler
scaler_minmax=MinMaxScaler()
df_encoded["Age"]=scaler_minmax.fit_transform(df_encoded[["Age"]])
print("\n",df_encoded,"\n")
from sklearn.preprocessing import StandardScaler
scaler_standar=StandardScaler()
df_encoded["Income"]=scaler_standar.fit_transform(df_encoded[["Income"]])
print("\n",df_encoded,"\n")

# 3. Implementación de técnicas de distancia (3 puntos)
# • Calcula la Distancia Manhattan, Distancia Euclidiana y Distancia Minkowski (con p=3p=3) entre 
# los siguientes dos puntos:
#   o Punto A: [25, 50000] (Edad, Ingresos)
#   o Punto B: [30, 60000] (Edad, Ingresos)

from scipy.spatial import distance
point_A=[25,50000]
point_B=[30,60000]
manhattan=distance.cityblock(point_A,point_B)
euclidean=distance.euclidean(point_A,point_B)
minkowski=distance.minkowski(point_A,point_B)
print(f"Distancia Manhattan: {manhattan}")
print(f"Distancia Euclidiana: {euclidean}")
print(f"Distancia Minkowski: {minkowski}")

# 4. Análisis de resultados (3 puntos)
# • Comparación de técnicas de escalamiento:
#   o Explica las diferencias entre Min-Max Scaling y Standard Scaling. ¿En qué casos sería 
# recomendable usar cada una?
# • Interpretación de distancias:
#   o Describe las diferencias entre las distancias calculadas (Manhattan, Euclidiana y Minkowski). 
# ¿Qué información adicional proporciona la Distancia Minkowski con p=3p=3?
# • Aplicabilidad:
#   o Explica en qué tipo de problemas de Machine Learning sería útil aplicar Label Encoding y en 
# cuáles sería más adecuado usar One-Hot Encoding.

# Respuesta a Comparación de Técnicas de Escalamiento
# Min-Max Scaling: Escala los valores a un rango entre 0 y 1. Es util cuando queremos mantener la
# distribución de los datos pero normalizar su magnitud. Ideal para algoritmos sensibles a la escala
# como KNN, redes neuronales.
# Standar Scaling: Transforma los datos a una distribución normal (media=0, desviación estandar=1).
# Es util cuando los datos no estan en la misma escala, y se requiere normalidad, como en PCA, 
# regresión lineal o SVM.

# Respuesta a Interpretación de Distancias
# Manhattan: Suma diferencias absolutas (camino en "cuadricula").
# Euclidiana: Distancia recta entre dos puntos (pitagórica).
# Minkowski (p=3): Generalización que pondera de manera diferente la diferencia. A mayor "p", más
# penaliza diferencias grandes.
# La distancia Minkowski con p=3 da un valor intermedio entre Manhattan y Euclidiana, y se puede
# ajustar para dar más o menos peso a grandes diferencias.

# Respuesta a Aplicabilidad
# Label Encoding: Es útil cuando hay un orden implicito (ordinal). Ejemplo: "bajo","medio","alto".
# One-Hot Encoding: Ideal para variables categóricas nominales sin orden, como nombres de ciudades,
# productos, etc. Evita introducir un orden artificial que podría confundir al modelo.