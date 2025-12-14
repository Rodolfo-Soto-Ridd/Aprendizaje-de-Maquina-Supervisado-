# 1. Carga y exploración de datos (1 punto)
# • Carga el dataset proporcionado, que contiene información sobre temperatura media, cambio en las 
# precipitaciones, frecuencia de sequías y producción agrícola en distintos países.
# • Analiza la distribución de las variables y detecta posibles valores atípicos o tendencias.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga del dataset
df = pd.read_csv('cambio_climatico_agricultura.csv')

# Visualización de las primeras filas del dataset
print("Primeras 5 filas del dataset:")
print(df.head())

# Información general del dataframe
print("\nInformación del dataset:")
df.info()

# Resumen estadístico de las variables numéricas
print("\nResumen estadístico del dataset:")
print(df.describe())

# Análisis de distribución de las variables
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df['Temperatura_promedio'], kde=True, ax=axes[0, 0]).set_title('Distribución de Temperatura_promedio')
sns.histplot(df['Cambio_lluvias'], kde=True, ax=axes[0, 1]).set_title('Distribución de Cambio_lluvias')
sns.histplot(df['Frecuencia_sequías'], kde=True, ax=axes[1, 0]).set_title('Distribución de Frecuencia_sequías')
sns.histplot(df['Producción_alimentos'], kde=True, ax=axes[1, 1]).set_title('Distribución de Producción_alimentos')
plt.tight_layout()
plt.show()

# Detección de valores atípicos (outliers) con boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x=df['Temperatura_promedio'], ax=axes[0, 0]).set_title('Boxplot de Temperatura_promedio')
sns.boxplot(x=df['Cambio_lluvias'], ax=axes[0, 1]).set_title('Boxplot de Cambio_lluvias')
sns.boxplot(x=df['Frecuencia_sequías'], ax=axes[1, 0]).set_title('Boxplot de Frecuencia_sequías')
sns.boxplot(x=df['Producción_alimentos'], ax=axes[1, 1]).set_title('Boxplot de Producción_alimentos')
plt.tight_layout()
plt.show()

# 2. Preprocesamiento y escalamiento de datos (2 puntos)
# • Aplica técnicas de normalización o estandarización a las variables numéricas.
# • Codifica correctamente cualquier variable categórica si fuera necesario.
# • Divide los datos en conjunto de entrenamiento y prueba (80%-20%).

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definir las variables predictoras (features) y la variable objetivo
X_reg = df[['Temperatura_promedio', 'Cambio_lluvias', 'Frecuencia_sequías']]
y_reg = df['Producción_alimentos']

# Dividir los datos en entrenamiento y prueba (80%-20%)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 3. Aplicación de modelos de aprendizaje supervisado (4 puntos)
# • Regresión:
# o Entrena un modelo de regresión lineal para predecir la producción de alimentos.
# o Evalúa el modelo usando métricas como MAE, MSE y R2.
# o Compara con otros modelos de regresión (árbol de decisión, random forest).

# Estandarización de las variables numéricas
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

print("Dimensiones de los datos de entrenamiento:", X_train_reg_scaled.shape)
print("Dimensiones de los datos de prueba:", X_test_reg_scaled.shape)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modelos de regresión
models_reg = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42)
}

# Entrenamiento y evaluación de modelos de regresión
results_reg = {}
for name, model in models_reg.items():
    if name == 'Linear Regression':
        model.fit(X_train_reg_scaled, y_train_reg)
        y_pred = model.predict(X_test_reg_scaled)
    else:
        model.fit(X_train_reg, y_train_reg) # Estos modelos no necesitan escalamiento
        y_pred = model.predict(X_test_reg)
        
    mae = mean_absolute_error(y_test_reg, y_pred)
    mse = mean_squared_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)
    results_reg[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

print("\nResultados de los modelos de regresión:")
for name, metrics in results_reg.items():
    print(f"\nModelo: {name}")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  MSE: {metrics['MSE']:.2f}")
    print(f"  R2 Score: {metrics['R2']:.2f}")

# • Clasificación:
# o Crea una nueva variable categórica que clasifique los países en "Bajo", "Medio" y "Alto" impacto climático en la producción agrícola.
# o Entrena modelos de clasificación como K-Nearest Neighbors, Árbol de Decisión y Support Vector Machine. 
# o Evalúa el desempeño usando matriz de confusión, precisión, sensibilidad y curva ROC-AUC.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

# Crear la variable categórica 'impacto_climatico'
# Se usa los cuartiles para clasificar el impacto
df['impacto_climatico'] = pd.qcut(df['Producción_alimentos'], 3, labels=['Alto', 'Medio', 'Bajo'])

# Preparación de datos para la clasificación
X_cls = df[['Temperatura_promedio', 'Cambio_lluvias', 'Frecuencia_sequías']]
y_cls = df['impacto_climatico']

# Dividir los datos en entrenamiento y prueba
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

# Estandarización de las variables para los modelos que lo necesitan
scaler_cls = StandardScaler()
X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
X_test_cls_scaled = scaler_cls.transform(X_test_cls)

# Modelos de clasificación
models_cls = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42, probability=True)
}

# Entrenamiento y evaluación de modelos de clasificación
results_cls = {}
for name, model in models_cls.items():
    if name in ['K-Nearest Neighbors', 'Support Vector Machine']:
        model.fit(X_train_cls_scaled, y_train_cls)
        y_pred = model.predict(X_test_cls_scaled)
    else:
        model.fit(X_train_cls, y_train_cls)
        y_pred = model.predict(X_test_cls)

    results_cls[name] = {
        'Accuracy': accuracy_score(y_test_cls, y_pred),
        'Precision': precision_score(y_test_cls, y_pred, average='macro'),
        'Recall (Sensitivity)': recall_score(y_test_cls, y_pred, average='macro'),
        'Confusion Matrix': confusion_matrix(y_test_cls, y_pred)
    }

print("\nResultados de los modelos de clasificación:")
for name, metrics in results_cls.items():
    print(f"\nModelo: {name}")
    print(f"  Accuracy: {metrics['Accuracy']:.2f}")
    print(f"  Precision (macro): {metrics['Precision']:.2f}")
    print(f"  Recall (macro): {metrics['Recall (Sensitivity)']:.2f}")
    print("  Matriz de Confusión:\n", metrics['Confusion Matrix'])
    
    # Visualización de la matriz de confusión
    plt.figure(figsize=(6, 4))
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Alto', 'Medio', 'Bajo'], yticklabels=['Alto', 'Medio', 'Bajo'])
    plt.title(f'Matriz de Confusión - {name}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

# Curva ROC-AUC (solo para modelos que lo soportan y para clasificación binaria, o 'one-vs-rest')
# Dada la clasificación multiclass, se requiere un enfoque 'one-vs-rest' o 'macro'
# Se usará SVC por su capacidad de estimar probabilidades
svc_model = SVC(random_state=42, probability=True)
svc_model.fit(X_train_cls_scaled, y_train_cls)
y_pred_proba = svc_model.predict_proba(X_test_cls_scaled)
roc_auc = roc_auc_score(y_test_cls, y_pred_proba, multi_class='ovr')
print(f"\nROC-AUC (ovr) para Support Vector Machine: {roc_auc:.2f}")

# 4. Optimización de modelos (2 puntos)
# • Ajusta hiperparámetros utilizando validación cruzada y búsqueda en grilla.
# • Aplica técnicas de regularización y analiza su impacto en los modelos.

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge

# Optimización del RandomForestRegressor
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train_reg, y_train_reg)

print("\nMejores hiperparámetros para Random Forest Regressor:")
print(grid_search_rf.best_params_)
print(f"Mejor R2 score: {grid_search_rf.best_score_:.2f}")

# Optimización del Support Vector Machine
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid_search_svc = GridSearchCV(SVC(random_state=42), param_grid_svc, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_svc.fit(X_train_cls_scaled, y_train_cls)

print("\nMejores hiperparámetros para Support Vector Machine:")
print(grid_search_svc.best_params_)
print(f"Mejor Accuracy score: {grid_search_svc.best_score_:.2f}")

# Aplicación de regularización en regresión lineal
# Entrenamiento de Ridge (L2) y Lasso (L1)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_ridge = ridge_model.predict(X_test_reg_scaled)

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_lasso = lasso_model.predict(X_test_reg_scaled)

print("\nResultados de modelos con regularización:")
print(f"Ridge Regressor R2 score: {r2_score(y_test_reg, y_pred_ridge):.2f}")
print(f"Lasso Regressor R2 score: {r2_score(y_test_reg, y_pred_lasso):.2f}")

print("\nCoeficientes de Ridge:", ridge_model.coef_)
print("Coeficientes de Lasso:", lasso_model.coef_)

# 5. Análisis de resultados y conclusiones (1 punto)
# • Compara los modelos utilizados y justifica cuál ofrece mejores resultados para la 
# predicción y clasificación.
# • Relaciona los hallazgos con posibles implicaciones en la seguridad alimentaria
# global

# Análisis de Resultados y Conclusiones

# Comparación de Modelos y Justificación

# Regresión: Generalmente, los modelos basados en árboles (RandomForestRegressor, DecisionTreeRegressor)
# son más robustos y pueden capturar relaciones no lineales en los datos, a menudo superando a la 
# LinearRegression. El RandomForestRegressor, en particular, suele ofrecer el mejor rendimiento al 
# promediar múltiples árboles para reducir la varianza. El ajuste de hiperparámetros con GridSearchCV 
# probablemente mejoró aún más su precisión.

# Clasificación: El modelo de Support Vector Machine (SVC) y el K-Nearest Neighbors (KNN) son sensibles 
# al escalamiento de los datos, por lo que es crucial preprocesar las variables. El Decision Tree no lo 
# requiere. Un análisis de la matriz de confusión revela qué clases son mejor predichas. Si el modelo 
# con mejor Precision y Recall para la clase 'Bajo impacto' (países con baja producción) es el SVC 
# optimizado, se podría considerar el mejor modelo de clasificación para esta tarea, ya que identificar 
# a los países vulnerables es un objetivo clave.

# Implicaciones en la Seguridad Alimentaria Global

# Factores Climáticos: Los modelos de regresión nos permiten cuantificar el impacto de cada variable 
# climática (Temperatura_promedio, Cambio_lluvias, Frecuencia_sequías) en la Producción_alimentos. 
# Los coeficientes de la regresión lineal o las importancias de las características del Random Forest 
# pueden indicar cuáles son los factores más influyentes. Por ejemplo, si el coeficiente de 
# Frecuencia_sequías es muy negativo, sugiere que este es un factor crítico que reduce la producción.

# Vulnerabilidad de Países: La clasificación de países en categorías de "Bajo", "Medio" y "Alto" impacto 
# es crucial para la toma de decisiones. Los países clasificados como "Alto impacto" son los más 
# vulnerables y podrían requerir intervenciones urgentes, como la implementación de tecnologías agrícolas 
# adaptadas al clima o programas de ayuda alimentaria. Un modelo de clasificación preciso puede ayudar a 
# los gobiernos y organizaciones a asignar recursos de manera más eficiente y a desarrollar políticas 
# dirigidas para mitigar los riesgos del cambio climático en la producción agrícola.