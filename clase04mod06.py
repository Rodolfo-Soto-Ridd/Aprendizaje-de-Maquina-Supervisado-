# Importar Librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Paso 2: Generar datos sintéticos
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 10
 
plt.scatter(X, y, color="blue", label="Datos reales")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Paso 3: División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regresión Lineal
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_test_pred_lin = lin_reg.predict(X_test)
error_lin = mean_squared_error(y_test, y_test_pred_lin)
print("\n",y_test)
print("\n",y_test_pred_lin)
print(f"\nError de Regresión Lineal: {error_lin:.4f}")

# Paso 4: Aplicación de diferentes modelos de regresión
# Regresión Polinómica
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
 
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_test_pred_poly = poly_reg.predict(X_test_poly)
error_poly = mean_squared_error(y_test, y_test_pred_poly)
print("\n",y_test)
print("\n",y_test_pred_poly)
print(f"\nError de Regresión Polinómica: {error_poly:.4f}")

# Árbol de Decisión

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)
error_tree = mean_squared_error(y_test, y_pred_tree)
print("\n",y_test)
print("\n",y_pred_tree)
print(f"\nError de Regresión con Árbol de Decisión: {error_tree}")

# Paso 5: Comparación de modelos
plt.scatter(X_test, y_test, color="blue", label="Datos Reales")
plt.scatter(X_test, y_test_pred_lin, color="red", label="Regresión Lineal")
plt.scatter(X_test, y_test_pred_poly, color="green", label="Regresión Polinómica")
plt.scatter(X_test, y_pred_tree, color="purple", label="Árbol de Decisión")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

errores = { 
            "Regresión Lineal": error_lin,
            "Regresión Polinómica": error_poly,
            "Árbol de Decisión": error_tree,
} 
mejor_modelo = min(errores, key=errores.get)
print(f"\nEl mejor modelo es: {mejor_modelo}") 