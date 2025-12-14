import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv('dataset.csv')

# Mostrar las primeras 5 filas del dataframe
print("Primeras 5 filas del dataset:")
print(df.head())

# Mostrar información general del dataset
print("\nInformación del dataset:")
df.info()

# Resumen estadístico
print("\nResumen estadístico del dataset:")
print(df.describe(include='all'))

# Contar valores únicos en la columna 'ubicación' para ver si los datos están desbalanceados
print("\nConteo de valores en la columna 'ubicación':")
print(df['ubicación'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Preparación de datos para la regresión
X_reg = df.drop('precio', axis=1)
y_reg = df['precio']

# Codificación one-hot para 'ubicación'
categorical_features = ['ubicación']
one_hot_encoder = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', one_hot_encoder, categorical_features)],
    remainder='passthrough')

# División de datos
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Modelo sin Feature Engineering (solo con codificación one-hot)
model_simple_reg = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])
model_simple_reg.fit(X_train_reg, y_train_reg)
y_pred_simple_reg = model_simple_reg.predict(X_test_reg)

mse_simple_reg = mean_squared_error(y_test_reg, y_pred_simple_reg)
r2_simple_reg = r2_score(y_test_reg, y_pred_simple_reg)

print("Resultados del modelo de regresión sin Feature Engineering:")
print(f"MSE: {mse_simple_reg:.2f}")
print(f"R2 Score: {r2_simple_reg:.2f}")

# Feature Engineering: Crear una nueva característica 'antiguedad_x_num_habitaciones'
df_fe = df.copy()
df_fe['antiguedad_x_num_habitaciones'] = df_fe['antigüedad_casa'] * df_fe['num_habitaciones']

X_fe_reg = df_fe.drop('precio', axis=1)
y_fe_reg = df_fe['precio']

# División de datos con la nueva característica
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe_reg, y_fe_reg, test_size=0.2, random_state=42)

# Columnas numéricas para escalar, incluyendo la nueva característica
numerical_features = ['tamaño_casa', 'num_habitaciones', 'antigüedad_casa', 'antiguedad_x_num_habitaciones']
categorical_features = ['ubicación']

preprocessor_fe = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_features)],
    remainder='passthrough')

# Modelo con Feature Engineering
model_fe_reg = Pipeline(steps=[('preprocessor', preprocessor_fe),
                             ('regressor', LinearRegression())])
model_fe_reg.fit(X_train_fe, y_train_fe)
y_pred_fe_reg = model_fe_reg.predict(X_test_fe)

mse_fe_reg = mean_squared_error(y_test_fe, y_pred_fe_reg)
r2_fe_reg = r2_score(y_test_fe, y_pred_fe_reg)

print("\nResultados del modelo de regresión con Feature Engineering:")
print(f"MSE: {mse_fe_reg:.2f}")
print(f"R2 Score: {r2_fe_reg:.2f}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Se utiliza un preprocesador simple para las características numéricas y categóricas
preprocessor_grid = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['ubicación'])],
    remainder='passthrough')

# Definir el pipeline con el preprocesador y el modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor_grid),
                           ('regressor', RandomForestRegressor(random_state=42))])

# Grilla de parámetros para GridSearchCV
param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [2, 5]
}

# Realizar GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_reg, y_reg)

print("Mejores hiperparámetros encontrados por GridSearchCV:")
print(grid_search.best_params_)
print(f"Mejor R2 score de GridSearchCV: {grid_search.best_score_:.2f}")

# Grilla de parámetros para RandomizedSearchCV
param_distributions = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [5, 10, 15, 20, None],
    'regressor__min_samples_split': [2, 5, 10]
}

# Realizar RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
random_search.fit(X_reg, y_reg)

print("\nMejores hiperparámetros encontrados por RandomizedSearchCV:")
print(random_search.best_params_)
print(f"Mejor R2 score de RandomizedSearchCV: {random_search.best_score_:.2f}")

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# Se escala los datos numéricos antes de aplicar la regularización
preprocessor_reg = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['ubicación']),
        ('scaler', StandardScaler(), ['tamaño_casa', 'num_habitaciones', 'antigüedad_casa'])
    ],
    remainder='passthrough'
)

# Pipeline para Ridge
ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor_reg),
                                 ('regressor', Ridge(alpha=1.0))])
ridge_pipeline.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge_pipeline.predict(X_test_reg)

print("Resultados del modelo Ridge (L2):")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_ridge):.2f}")
print(f"R2 Score: {r2_score(y_test_reg, y_pred_ridge):.2f}")
print("Coeficientes del modelo Ridge:")
print(ridge_pipeline.named_steps['regressor'].coef_)

# Pipeline para Lasso
lasso_pipeline = Pipeline(steps=[('preprocessor', preprocessor_reg),
                                 ('regressor', Lasso(alpha=1.0))])
lasso_pipeline.fit(X_train_reg, y_train_reg)
y_pred_lasso = lasso_pipeline.predict(X_test_reg)

print("\nResultados del modelo Lasso (L1):")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_lasso):.2f}")
print(f"R2 Score: {r2_score(y_test_reg, y_pred_lasso):.2f}")
print("Coeficientes del modelo Lasso:")
print(lasso_pipeline.named_steps['regressor'].coef_)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Preparación de datos para la clasificación
X_cls = df.drop('ubicación', axis=1)
y_cls = df['ubicación']

# Codificación one-hot para 'ubicación'
preprocessor_cls = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['num_habitaciones', 'antigüedad_casa']) # Solo para demostrar
    ],
    remainder='passthrough'
)

# División de datos
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Modelo sin balanceo
model_cls = RandomForestClassifier(random_state=42)
model_cls.fit(X_train_cls, y_train_cls)
y_pred_cls = model_cls.predict(X_test_cls)

print("Resultados del modelo de clasificación sin balanceo:")
print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_cls):.2f}")
print(f"F1-Score (macro): {f1_score(y_test_cls, y_pred_cls, average='macro'):.2f}")

cm_cls = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cls, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_cls), yticklabels=np.unique(y_cls))
plt.title('Matriz de Confusión sin Balanceo')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()

# Aplicar sobremuestreo con RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_cls, y_train_cls)

# Entrenar modelo con datos balanceados
model_resampled = RandomForestClassifier(random_state=42)
model_resampled.fit(X_resampled, y_resampled)
y_pred_resampled = model_resampled.predict(X_test_cls)

print("\nResultados del modelo con sobremuestreo (RandomOverSampler):")
print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_resampled):.2f}")
print(f"F1-Score (macro): {f1_score(y_test_cls, y_pred_resampled, average='macro'):.2f}")

cm_resampled = confusion_matrix(y_test_cls, y_pred_resampled)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_resampled, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_cls), yticklabels=np.unique(y_cls))
plt.title('Matriz de Confusión con Sobremuestreo')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()

# Análisis de Resultados
# ¿Qué técnica de optimización tuvo mayor impacto en la mejora del modelo?
# Para determinar qué técnica de optimización tuvo el mayor impacto, es crucial comparar los 
# resultados de las métricas clave (R2 para regresión y F1-score/Accuracy para clasificación) antes 
# y después de aplicar cada técnica
# Feature Engineering: Si el R2 Score del modelo con las características nuevas es significativamente 
# más alto que el del modelo simple, indica que esta técnica tuvo un gran impacto. La creación de la 
# característica antiguedad_x_num_habitaciones podría capturar una relación no lineal que el modelo 
# simple no podía, mejorando así su capacidad predictiva.
# Ajuste de Hiperparámetros: El ajuste de hiperparámetros a menudo conduce a mejoras significativas, 
# especialmente en modelos complejos como RandomForest. Si el R2 score del modelo optimizado por 
# GridSearchCV o RandomizedSearchCV es notablemente superior al de un modelo con hiperparámetros por 
# defecto, esta técnica sería una de las más influyentes.
# Regularización: La regularización no siempre aumenta la precisión, pero su principal impacto es en 
# la generalización. Si los coeficientes del modelo Lasso son escasos (algunos son cero) y el modelo 
# Ridge reduce la magnitud de los coeficientes, esto indica que la regularización está funcionando 
# para prevenir el sobreajuste.
# Balanceo de Datos: El balanceo de datos impacta directamente la capacidad del modelo para predecir 
# la clase minoritaria. Un aumento en el F1-score para la clase minoritaria y un cambio en la matriz 
# de confusión (menos falsos negativos para esa clase) indicarían un impacto positivo.

# ¿Cómo afectó la elección de hiperparámetros al rendimiento?
# La elección de hiperparámetros tiene un impacto directo en el sesgo y la varianza del modelo.
# Un n_estimators más alto en RandomForest generalmente reduce la varianza y mejora la precisión.
# Un max_depth más bajo evita el sobreajuste al restringir la complejidad del árbol.
# min_samples_split controla el crecimiento del árbol y previene que se ajuste demasiado a los datos 
# de entrenamiento.
# GridSearchCV busca exhaustivamente la mejor combinación, lo que puede resultar en una mejor precisión 
# pero es computacionalmente costoso. RandomizedSearchCV explora de manera más eficiente el espacio de 
# parámetros, lo que a menudo permite encontrar una buena solución en mucho menos tiempo, con un 
# rendimiento comparable o incluso mejor en grillas de gran tamaño.

# ¿Cómo influyó la regularización en la generalización del modelo?
# La regularización es una técnica clave para mejorar la generalización del modelo, es decir, su 
# capacidad para funcionar bien en datos nuevos y no vistos.
# Regularización L2 (Ridge): Añade una penalización igual a la suma de los cuadrados de los coeficientes. 
# Esto evita que los coeficientes se vuelvan muy grandes, lo que reduce la complejidad del modelo y 
# previene el sobreajuste.
# Regularización L1 (Lasso): Añade una penalización igual a la suma de los valores absolutos de los 
# coeficientes. Esto puede llevar a que algunos coeficientes se conviertan en cero, lo que efectivamente 
# realiza una selección de características y produce un modelo más simple y más interpretable.
# El impacto de la regularización en la generalización se puede observar si el modelo regularizado tiene 
# un rendimiento similar en los conjuntos de entrenamiento y prueba, a diferencia de un modelo no 
# regularizado que podría tener una alta precisión en el entrenamiento pero una baja precisión en la prueba 
# (indicando sobreajuste).