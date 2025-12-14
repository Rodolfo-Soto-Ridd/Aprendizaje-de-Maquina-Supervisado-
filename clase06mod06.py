# Importar librerías necesarias
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
 
# Valores reales y predichos
y_true = np.array([3, 5, 7, 9, 11])
y_pred = np.array([2.8, 5.1, 7.2, 8.9, 10.5])
 
# Calcular MSE
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse}")
 
# Calcular MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae}")
 
# Calcular RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse}")
 
# Calcular R²
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2}")

