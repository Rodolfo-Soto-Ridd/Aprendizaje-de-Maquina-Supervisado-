import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Datos de ejemplo (tamaño de la casa en m2 y precio en miles de USD)
x=np.array([50,60,70,80,90,100]).reshape(-1,1)
y=np.array([150,160,180,200,220,240])
# Crear el modelo de regresión lineal y ajustarlo a los datos
model=LinearRegression()
model.fit(x,y)
# Predecir precios para nuevos datos
x_new=np.array([55,65,75,85,95]).reshape(-1,1)
y_pred=model.predict(x_new)
# Visualizar los datos y la linea de regresión
plt.scatter(x,y,color="blue", label="Datos de Entrenamiento")
plt.plot(x_new,y_pred, color="red",linestyle="--",label="Predicciones")
plt.xlabel("Tamaño de la casa [m2]")
plt.ylabel("Precio [USD]")
plt.legend()
plt.show()
from sklearn.preprocessing import PolynomialFeatures
# Crear caracteristicas polinomicas
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)
# Ajustar el modelo de regresion polinomica
model=LinearRegression()
model.fit(x_poly,y)
# Predecir precios para nuevos datos
x_new_poly=poly.transform(x_new)
y_pred_poly=model.predict(x_new_poly)
# Visualizar los datos y la curva de regresión polinómica
plt.scatter(x,y,color="blue", label="Datos de Entrenamiento")
plt.plot(x_new,y_pred_poly, color="red",linestyle="--",label="Predicciones")
plt.xlabel("Tamaño de la casa [m2]")
plt.ylabel("Precio [USD]")
plt.legend()
plt.show()
from sklearn.linear_model import LogisticRegression
# Datos de ejemplo (horas de estudio y aprobacion del examen)
x=np.array([5,10,15,20,25,30]).reshape(-1,1)
y=np.array([0,0,0,1,1,1])  # 0: no aprobado, 1: aprobado
# Crear y ajustar el modelo de regresión logística
model=LogisticRegression()
model.fit(x,y)
# Predecir probabilidades para nuevos datos
x_new=np.array([8,12,18,22,28]).reshape(-1,1)
y_pred=model.predict_proba(x_new)[:,1]
# Visualizar los datos y la curva logística
plt.scatter(x,y,color="blue", label="Datos de Entrenamiento")
plt.scatter(x_new,y_pred,color="green", label="Nuevos Datos de Entrenamiento")
plt.plot(x_new,y_pred, color="red",linestyle="--",label="Predicciones")
plt.xlabel("Horas de Estudio")
plt.ylabel("Probabilidad de Aprobar")
plt.legend()
plt.show()

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia logs de TensorFlow
warnings.filterwarnings("ignore")          # Oculta warnings
 
 
# Datos (asegura misma cardinalidad)
X = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)  # 6 muestras
y = np.array([150, 160, 180, 200, 220, 240])            # 6 muestras
X_new = np.array([55, 65, 75, 85, 95]).reshape(-1, 1)   # Predicción
 
# Modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
])
 
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=100, verbose=0)
 
# Predicción
y_pred = model.predict(X_new)
 
# Gráfico
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X_new, y_pred, 'r--', label='Predicciones')
plt.xlabel('Tamaño (m²)')
plt.ylabel('Precio (USD)')
plt.legend()
plt.show()
 