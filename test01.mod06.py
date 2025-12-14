# modelo_viviendas.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ========================
# 1. CARGA Y EXPLORACIÓN
# ========================

# Cargar dataset
df = pd.read_csv('viviendas.csv')

# Mostrar las primeras 5 filas
print("Primeras 5 filas del dataset:")
print(df.head())

# Estructura del dataset
print("\nInformación del dataset:")
print(df.info())

# Estadísticas generales
print("\nEstadísticas descriptivas:")
print(df.describe())

# ========================
# 2. LIMPIEZA DE DATOS
# ========================

# Revisar valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# Imputar valores faltantes numéricos con la mediana
df['superficie'].fillna(df['superficie'].median(), inplace=True)
df['habitaciones'].fillna(df['habitaciones'].median(), inplace=True)

# Eliminar filas con valores faltantes en 'barrio' o 'precio'
df.dropna(subset=['barrio', 'precio'], inplace=True)

# Verificación de tipos
print("\nTipos de datos después de limpieza:")
print(df.dtypes)

# ========================
# 3. ANÁLISIS EXPLORATORIO
# ========================

# Relación superficie vs precio
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='superficie', y='precio')
plt.title('Relación Superficie vs Precio')
plt.savefig("superficie_vs_precio.png")
plt.close()

# Relación habitaciones vs precio
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='habitaciones', y='precio')
plt.title('Relación Habitaciones vs Precio')
plt.savefig("habitaciones_vs_precio.png")
plt.close()

# Boxplots para outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df[['superficie', 'habitaciones', 'antiguedad', 'precio']])
plt.title('Boxplots de variables numéricas')
plt.savefig("boxplots_variables.png")
plt.close()

# ========================
# 4. ONE-HOT ENCODING
# ========================

# Codificar la variable 'barrio'
df = pd.get_dummies(df, columns=['barrio'], drop_first=True)

# ========================
# 5. DIVISIÓN DEL DATASET
# ========================

# Variables predictoras y objetivo
X = df.drop('precio', axis=1)
y = df['precio']

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ========================
# 6. MODELO DE REGRESIÓN
# ========================

# Instanciar y entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Mostrar coeficientes
coeficientes = pd.Series(modelo.coef_, index=X.columns)
print("\nCoeficientes del modelo de regresión lineal:")
print(coeficientes)

# ========================
# 7. EVALUACIÓN DEL MODELO
# ========================

# Predicciones
y_pred = modelo.predict(X_test)

# Métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluación del modelo:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")