from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
# 1. Generar datos sintéticos
X, y = make_regression(n_samples=1000, n_features=20, noise=0.5, random_state=42)
 
# 2. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# 3. Aplicar Ridge Regression (L2)
ridge = Ridge(alpha=1.0)  # alpha es el parámetro de regularización (lambda)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Error de Ridge Regression:", mean_squared_error(y_test, y_pred_ridge))
 
# 4. Aplicar Lasso Regression (L1)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("Error de Lasso Regression:", mean_squared_error(y_test, y_pred_lasso))
 
# 5. Aplicar Elastic Net
elastic_net = ElasticNet(
    alpha=1.0, l1_ratio=0.5  # l1_ratio controla la mezcla entre L1 y L2
)
elastic_net.fit(X_train, y_train)
y_pred_elastic = elastic_net.predict(X_test)
print("Error de Elastic Net:", mean_squared_error(y_test, y_pred_elastic))

## OPTIMIZACIÓN DE HIPERPARÁMETROS EN RANDOMFORESTCLASSIFIER
 
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
# 1. Cargar datos de ejemplo (Iris dataset)
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
 
# 2. Modelo base
model = RandomForestClassifier(random_state=42)
 
# ─────────────────────────────
# 3. Grid Search
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 5, 10]
}
 
grid_search = GridSearchCV(
    model, param_grid, cv=3, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Mejor configuración con Grid Search:", grid_search.best_params_)
 
# ─────────────────────────────
# 4. Random Search
param_dist = {
    "n_estimators": np.arange(10, 200, 10),
    "max_depth": [None] + list(np.arange(3, 20, 3))
}
 
random_search = RandomizedSearchCV(
    model, param_dist, n_iter=10, cv=3, scoring="accuracy",
    n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)
print("Mejor configuración con Random Search:", random_search.best_params_)
 
# ─────────────────────────────
# 5. Optimización Bayesiana
bayes_search = BayesSearchCV(
    model,
    {
        "n_estimators": (10, 200),
        "max_depth": (3, 20)
    },
    n_iter=10,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)
bayes_search.fit(X_train, y_train)
print("Mejor configuración con Optimización Bayesiana:", bayes_search.best_params_)
