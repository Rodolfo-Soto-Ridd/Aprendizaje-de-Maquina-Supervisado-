# 1. Carga de Datos (1 punto)
# A. Descarga el archivo dataset.csv.
# B. Carga el conjunto de datos en Python y realiza una exploración inicial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv("dataset.csv")
print("\n",df.head(),"\n")
print("\n",df.info(),"\n")
print("\n",df.isnull().sum(),"\n")
print("\n",df.describe(),"\n")
# 2. Optimización del Modelo (6 puntos)
# Aplica las siguientes técnicas de optimización y analiza su impacto en el rendimiento del modelo:
# • Feature Engineering:
# o Crea nuevas características a partir de los datos existentes.
# o Aplica transformación y selección de características.
df["Tamaño_por_Habitacion"]=df["tamaño_casa"]/df["num_habitaciones"]
df["Año_Construccion"]=2025-df["antigüedad_casa"]
print(df.head())
df = pd.get_dummies(df, columns=['ubicación'], drop_first=True)
print(df.head())
# • Ajuste de Hiperparámetros:
# o Define una grilla de parámetros y realiza una búsqueda en grilla (GridSearchCV).
# o Compara los resultados con una búsqueda aleatoria (RandomizedSearchCV).
# Uso GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df.drop('precio', axis=1)
y = df['precio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)
# Definir la grilla de parámetros
model = LinearRegression()
param_grid = {
    'fit_intercept': [True, False],
    'n_jobs': [1,2,3]}
# Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train,y_train)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Mejores parámetros:", grid_search.best_params_)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae_lr = mean_absolute_error(y_test, y_pred)
mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred)
# Uso RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
# Definir la distribución de los parámetros
param_dist = {
    'fit_intercept': [True, False],
    'n_jobs': [1,2,3]}
# RandomizedSearch
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, random_state=7)
random_search.fit(X_train, y_train)
# Resultados de RandomizedSearch
print("Mejores parámetros:", random_search.best_params_)
# • Regularización:
# o Aplica regularización L1 (Lasso) y L2 (Ridge) en un modelo de regresión o clasificación.
# o Analiza su impacto en lgrid_search.fit(X_train, y_train)os coeficientes y en la predicción.
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
# Lasso (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)
print("MSE Lasso:", mean_squared_error(y_test, lasso_preds))
# Ridge (L2 regularization)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
print("MSE Ridge:", mean_squared_error(y_test, ridge_preds))
# • Balanceo de Datos:
# o Si el conjunto de datos está desbalanceado, aplica sobremuestreo (RandomOversampling) o 
# submuestreo (RandomUnderSampler).
# o Evalúa si el balanceo mejora la precisión del modelo en la clase minoritaria.
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# Oversampling
oversampler = RandomOverSampler(random_state=5)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
y_pred_oversample = model.predict(X_test)
mae_lr_over = mean_absolute_error(y_test, y_pred_oversample)
mse_lr_over = mean_squared_error(y_test, y_pred_oversample)
rmse_lr_over = np.sqrt(mse_lr)
r2_lr_over = r2_score(y_test, y_pred_oversample)
# Undersampling
undersampler = RandomUnderSampler(random_state=8)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
y_pred_undersample = model.predict(X_test)
mae_lr_under = mean_absolute_error(y_test, y_pred_undersample)
mse_lr_under = mean_squared_error(y_test, y_pred_undersample)
rmse_lr_under = np.sqrt(mse_lr)
r2_lr_under = r2_score(y_test, y_pred_undersample)
# Para cada técnica:
# 1. Divide los datos en entrenamiento y prueba.
# 2. Aplica la optimización correspondiente.
# 3. Evalúa la precisión con métricas como accuracy, F1-score y matriz de confusión.
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# Predicciones
model_pred = grid_search.best_estimator_.predict(X_test)
# Métricas de evaluación
accuracy = accuracy_score(y_test, model_pred)
f1 = f1_score(y_test, model_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, model_pred)
print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
print(f"Confusion Matrix:\n {conf_matrix}")
# Análisis de Resultados

# ¿Qué técnica de optimización tuvo mayor impacto en la mejora del modelo?
# Aquí tendrás que comparar los resultados antes y después de aplicar las distintas técnicas. La que más impacto tenga en el aumento 
# de la precisión o la reducción del error será la más efectiva.

# ¿Cómo afectó la elección de hiperparámetros al rendimiento?
# El ajuste de los hiperparámetros a menudo mejora el rendimiento, ya que optimiza el modelo para que se ajuste mejor a los datos de 
# entrenamiento, sin sobreajustarse.

# ¿Cómo influyó la regularización en la generalización del modelo?
# La regularización ayuda a reducir el sobreajuste (overfitting), lo que permite que el modelo generalice mejor en datos no vistos. 
# El efecto de Lasso (L1) y Ridge (L2) puede ser notable dependiendo de la magnitud de los coeficientes de las características.
