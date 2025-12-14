# Ejercicio 1
import pandas as pd
customer = pd.read_csv("customer_churn.csv")
customer.head()
# SELECCIÓN DE LOS DATOS
X = customer.drop("Churn", axis=1)
y = customer.Churn 
# Split into train and test
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# ENTRENAMOS EL MODELO
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42))])
pipeline.fit(X_train, y_train)
# EVALUACIÓN DEL MODELO
from sklearn.metrics import classification_report
# Make prediction on the testing data
y_pred = pipeline.predict(X_test)
# Classification Report
print(classification_report(y_pred, y_test))

from sklearn.ensemble import BaggingClassifier
# Create a bagging classifier with the decision tree pipeline
bagging_classifier = BaggingClassifier(estimator=pipeline, n_estimators=50, random_state=42)
# Train the bagging classifier on the training data
bagging_classifier.fit(X_train, y_train) 
# Evaluamos este modelo
# Make prediction on the testing data
y_pred = bagging_classifier.predict(X_test)
# Classification Report
print(classification_report(y_pred, y_test))

# Ejercicio 2
# Importar librerías
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Cargar el conjunto de datos Iris
data = load_iris()
X = data.data     # Características
y = data.target   # Etiquetas
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Crear el modelo de Random Forest (Bagging)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)   
# Entrenar el modelo
rf_model.fit(X_train, y_train)
# Hacer predicciones
y_pred = rf_model.predict(X_test)
# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo de Random Forest (Bagging): {accuracy:.2f}")

# Importar librerías
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
 
# Cargar el conjunto de datos Iris
data = load_iris()
X = data.data     # Características
y = data.target   # Etiquetas
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)   
# Crear el modelo de Gradient Boosting (Boosting)
gb_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42)
# Entrenar el modelo
gb_model.fit(X_train, y_train)
# Hacer predicciones
y_pred = gb_model.predict(X_test)
# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo de Gradient Boosting (Bagging): {accuracy:.2f}")

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Generar datos sintéticos
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)  # Tamaño de la casa (0 a 10 m²)
y = (3 * X**2 + 2 * X + np.random.randn(100, 1) * 10)  # Precio de la casa con ruido
y = y.ravel()  # Convertir a array unidimensional
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Crear y entrenar el modelo de Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train) 
# Hacer predicciones
y_pred = gb_model.predict(X_test)
# Visualizar las predicciones
plt.scatter(X_test, y_test, color="blue", label="Datos reales")
plt.scatter(X_test, y_pred, color="red", label="Predicciones")
plt.xlabel("Tamaño de la casa (m²)")
plt.ylabel("Precio de la casa (USD)")
plt.legend()
plt.show() 