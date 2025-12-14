# 1. Carga y Exploración Inicial del Dataset (1 punto)
# • Carga el archivo viviendas.csv.
# • Muestra las primeras 5 filas del DataFrame.
# • Describe la estructura del dataset (.info(), .describe()).
import pandas as pd
df=pd.read_csv("viviendas.csv")
print(f"\n",df,"\n")
print(f"\n",df.head(),"\n")
print(f"\n",df.info(),"\n")
print(f"\n",df.describe(),"\n")
# 2. Limpieza de Datos (1 punto)
# • Detecta y elimina o imputa los valores faltantes.
# • Asegúrate de que los tipos de datos estén correctamente definidos.
print(df.isnull().sum())
df=df.dropna()
df["superficie"]=df["superficie"].fillna(df["superficie"].median())
df["habitaciones"]=df["habitaciones"].fillna(df["habitaciones"].median())
print(f"\n",df.isnull().sum(),"\n")
df["barrio"]=df["barrio"].astype("category")
df["superficie"]=df["superficie"].astype("float")
df["habitaciones"]=df["habitaciones"].astype("float")
df["antiguedad"]=df["antiguedad"].astype("int")
df["precio"]=df["precio"].astype("float")
# 3. Análisis Exploratorio de Datos (2 puntos)
# • Realiza gráficos para explorar las relaciones entre:
# o Superficie y precio 
# o Número de habitaciones y precio
# • Detecta posibles outliers visualmente.
import matplotlib.pyplot as plt
import seaborn as sns
# Scatter para Superficie versus Precio
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='superficie', y='precio')
plt.title('Relación entre Superficie y Precio')
plt.xlabel('Superficie [m²]')
plt.ylabel('Precio de venta')
plt.show()
# Scatter para Habitaciones versus Precio
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='habitaciones', y='precio')
plt.title('Relación entre Número de Habitaciones y Precio')
plt.xlabel('Número de Habitaciones')
plt.ylabel('Precio de venta')
plt.show()
# Boxplot para Superficie
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='superficie')
plt.title('Boxplot de Superficie')
plt.show()
# Boxplot para Precio
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='precio')
plt.title('Boxplot de Precio')
plt.show()
# 4. Codificación de variables categóricas (1 punto)
# • Convierte la variable barrio a variables numéricas utilizando One-Hot Encoding.
df_encoded=pd.get_dummies(df,columns=["barrio"], drop_first=True)
print(f"\n",df_encoded,"\n")
print(f"\n",df_encoded.info(),"\n")
# 5. División del Dataset (1 punto)
# • Separa el dataset en un conjunto de entrenamiento (80%) y uno de prueba (20%).
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
x=df_encoded.drop('precio',axis=1)
y=df_encoded['precio']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
# 6. Entrenamiento de un Modelo de Regresión Lineal (2 puntos)
# • Entrena un modelo de regresión lineal usando scikit-learn.
# • Utiliza como variables predictoras: superficie, habitaciones, antigüedad y los barrios codificados.
# • Muestra los coeficientes del modelo.
modelo=LinearRegression()
modelo.fit(x_train,y_train)
coeficientes=pd.Series(modelo.coef_, index=x.columns)
print("\nCoefcientes del modelo de regresión lineal:")
print(coeficientes)
# 7. Evaluación del Modelo (2 puntos)
# • Calcula el Mean Squared Error (MSE) y el R² (coeficiente de determinación) sobre el conjunto de prueba.
# • Comenta brevemente si el modelo tiene un buen desempeño.
y_pred=modelo.predict(x_test)
print(f"\n",y,"\n")
print(f"\n",y_pred,"\n")
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"\nEvaluación del modelo:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R2): {r2:.2f}")