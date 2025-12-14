# 1. Carga de datos (1 punto)
# • Descarga los conjuntos de datos proporcionados en el material complementario.
# • Carga los datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_reg= pd.read_csv("datos_regresion.csv")
print("\n",df_reg.head(),"\n")
print("\n",df_reg.info(),"\n")
print("\n",df_reg.isnull().sum(),"\n")
print("\n",df_reg.describe(),"\n")

# 2. Evaluación de un modelo de regresión (4 puntos)
# • Datos de regresión: Utiliza el conjunto de datos proporcionado para evaluar un modelo de
# regresión que predice el valor de ventas de productos en función de características como el
# precio, la categoría y la antigüedad del producto.
# • Métricas a calcular:
# o MAE (Error Absoluto Medio).
# o MSE (Error Cuadrático Medio).
# o RMSE (Raíz del Error Cuadrático Medio).
# o R² (Coeficiente de Determinación).
# • Tareas:
# 1. Calcula las métricas mencionadas utilizando las predicciones del modelo y los
# valores reales.
from sklearn.model_selection import train_test_split
df_reg = pd.get_dummies(df_reg, columns=["Categoria"])  # codificar variables categóricas
print("\n",df_reg,"\n")
x = df_reg.drop('Valor_Ventas', axis=1)
y = df_reg['Valor_Ventas']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
# Regresión Lineal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("\n",y_pred,"\n")
print("\n",y,"\n")
# Métricas
promedio_valor_ventas=df_reg["Valor_Ventas"].mean()
mae_lr = mean_absolute_error(y_test, y_pred)
mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred)
print(f"Promedio Valor Ventas:{promedio_valor_ventas:.2f}[$]")
print(f"MAE - Regresión Lineal: {mae_lr:.2f}[$]")
print(f"MSE - Regresión Lineal: {mse_lr:.2f}[$]")
print(f"RMSE - Regresión Lineal: {rmse_lr:.2f}")
print(f"R² - Regresión Lineal: {r2_lr:.2f}","\n")

# Modelo Polinomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
poly_reg = LinearRegression()
poly_reg.fit(x_train_poly, y_train)
y_pred_poly = poly_reg.predict(x_test_poly)
# Métricas
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Promedio Valor Ventas:{promedio_valor_ventas:.2f}[$]")
print(f"MAE - Regresión Polinomial: {mae_poly:.2f}[$]")
print(f"MSE - Regresión Polinomial: {mse_poly:.2f}[$]")
print(f"RMSE - Regresión Polinomial: {rmse_poly:.2f}")
print(f"R² - Regresión Polinomial: {r2_poly:.2f}","\n")

# Regresión Logistica
#from sklearn.linear_model import LogisticRegression
#log_reg = LogisticRegression(max_iter=200)
#log_reg.fit(x_train,y_train)
#y_pred_log_reg = log_reg.predict(x_test)
# Métricas
#mae_log_reg = mean_absolute_error(y_test, y_pred_log_reg)
#mse_log_reg = mean_squared_error(y_test, y_pred_log_reg)
#rmse_log_reg = np.sqrt(mse_log_reg)
#r2_log_reg = r2_score(y_test, y_pred_log_reg)
#print(f"MAE - Regresión Logistica: {mae_log_reg:.2f}")
#print(f"MSE - Regresión Logistica: {mse_log_reg:.2f}")
#print(f"RMSE - Regresión Logistica: {rmse_log_reg:.2f}")
#print(f"R² - Regresión Logistica: {r2_log_reg:.2f}")

# Modelo Vecinos K-enesimos
from sklearn.preprocessing import StandardScaler
# Normalización de los datos
scaler = StandardScaler()
x_train1 = scaler.fit_transform(x_train)
x_test1 = scaler.transform(x_test)
from sklearn.neighbors import KNeighborsRegressor
k_values = [1,3,5,7,9]
for k in k_values:
    model_knn = KNeighborsRegressor(n_neighbors=k) # buscar kneighbors regression
    model_knn.fit(x_train1, y_train)
    y_pred_knn = model_knn.predict(x_test1)
# Métricas
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print(f"Promedio Valor Ventas:{promedio_valor_ventas:.2f}[$]")
print(f"MAE - Regresión KNN: {mae_knn:.2f}[$]")
print(f"MSE - Regresión KNN: {mse_knn:.2f}[$]")
print(f"RMSE - Regresión KNN: {rmse_knn:.2f}")
print(f"R² - Regresión KNN: {r2_knn:.2f}","\n")

# Modelo Arbol de Decisión
from sklearn.tree import DecisionTreeRegressor
model_dt = DecisionTreeRegressor(random_state=5) # buscar regresion
model_dt.fit(x_train1, y_train)
y_pred_dt = model_dt.predict(x_test1)
# Métricas
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Promedio Valor Ventas:{promedio_valor_ventas:.2f}[$]")
print(f"MAE - Regresión Arbol de Decisión: {mae_dt:.2f}[$]")
print(f"MSE - Regresión Arbol de Decisión: {mse_dt:.2f}[$]")
print(f"RMSE - Regresión Arbol de Decisión: {rmse_dt:.2f}")
print(f"R² - Regresión Arbol de Decisión: {r2_dt:.2f}","\n")

# Modelo Random Forest
from sklearn.ensemble import RandomForestRegressor # buscar regresion
model_rf = RandomForestRegressor(n_estimators=75, random_state=5)
model_rf.fit(x_train1, y_train)
y_pred_rf = model_rf.predict(x_test1)
# Métricas
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Promedio Valor Ventas:{promedio_valor_ventas:.2f}[$]")
print(f"MAE - Regresión Random Forest: {mae_rf:.2f}[$]")
print(f"MSE - Regresión Random Forest: {mse_rf:.2f}[$]")
print(f"RMSE - Regresión Random Forest: {rmse_rf:.2f}")
print(f"R² - Regresión Random Forest: {r2_rf:.2f}","\n")

# Modelo Máquina de Vector de Soporte (SVM)
from sklearn.svm import SVR # SVC es el clasiffier y SVR es el regressor
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    model_svm = SVR(kernel=kernel, C=100, epsilon=0.1)
    model_svm.fit(x_train1, y_train)
    y_pred_svm = model_svm.predict(x_test1)
# Métricas
mae_svm = mean_absolute_error(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)
rmse_svm = np.sqrt(mse_svm)
r2_svm = r2_score(y_test, y_pred_svm)
print(f"Promedio Valor Ventas:{promedio_valor_ventas:.2f}[$]")
print(f"MAE - Regresión Máquina de Vector de Soporte: {mae_svm:.2f}[$]")
print(f"MSE - Regresión Máquina de Vector de Soporte: {mse_svm:.2f}[$]")
print(f"RMSE - Regresión Máquina de Vector de Soporte: {rmse_svm:.2f}")
print(f"R² - Regresión Máquina de Vector de Soporte: {r2_svm:.2f}","\n")
# 2. La metrica que indica un buen ajuste es R2, en este caso el mejor ajuste lo dio la regresión lineal. El error 
# promedio de las predicciones lo entrega el RMSE que para el caso lineal es de 8.70.

# 3. Evaluación de un modelo de clasificación (5 puntos) 
# • Datos de clasificación: Utiliza el conjunto de datos proporcionado para evaluar un modelo de clasificación que predice si un 
# cliente comprará o no un producto en función de características como la edad, el ingreso y el historial de compras. 
# • Métricas a calcular: 
# o Matriz de confusión. 
# o Precisión y Exactitud. 
# o Sensibilidad y Especificidad. 
# o Curva ROC y AUC (Área Bajo la Curva ROC). 
# • Tareas: 1. Genera la matriz de confusión y calcula las métricas mencionadas. 
# 2. Grafica la Curva ROC y calcula el AUC. 

df_clf= pd.read_csv("datos_clasificacion.csv")
print("\n",df_clf.head(),"\n")
print("\n",df_clf.info(),"\n")
print("\n",df_clf.isnull().sum(),"\n")
print("\n",df_clf.describe(),"\n")
x1 = df_clf.drop('Comprara', axis=1)
y1 = df_clf['Comprara']
x_train2, x_test2, y_train2, y_test2 = train_test_split(x1, y1, test_size=0.2, random_state=5)
from sklearn.metrics import (confusion_matrix,precision_score,accuracy_score,recall_score,roc_curve,roc_auc_score)
# Clasificación por Regresión Logística

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
lr = LogisticRegression()
lr.fit(x_train2, y_train2)
y_pred_lr = lr.predict(x_test2) # x_test2: datos originales y x_test3: datos normalizados
print("Exactitud Clasificación por Logística:", accuracy_score(y_test2, y_pred_lr))
print("Precisión Clasificación por Logística:", precision_score(y_test2, y_pred_lr))
print("Sensibilidad Clasificación por Logística:", recall_score(y_test2, y_pred_lr))
print("Especificidad Clasificación por Logística:", recall_score(y_test2, y_pred_lr,pos_label=0))
y_prob_lr=lr.predict_proba(x_test2)[:,1]
fpr,tpr,thresholds=roc_curve(y_test2,y_prob_lr)
auc_lr=roc_auc_score(y_test2,y_prob_lr)
print(f"AUC:{auc_lr:.2f}")
plt.plot(fpr,tpr,label=f"Curav ROC (AUC={auc_lr:.2f})")
plt.plot([0,1],[0,1],linestyle="--",label="Clasificador Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Regresión Logistica")
plt.legend()
plt.show()
print("Matriz de confusión:\n", confusion_matrix(y_test2, y_pred_lr),"\n")

# Clasificación por Vecinos K-enesimos

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
scaler = StandardScaler()
x_train3 = scaler.fit_transform(x_train2)
x_test3 = scaler.transform(x_test2)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train3, y_train2)
y_pred_knn1 = knn.predict(x_test3) # x_test2: datos originales y x_test3: datos normalizados
print("Exactitud Clasificación por KNN:", accuracy_score(y_test2, y_pred_knn1))
print("Precisión Clasificación por KNN:", precision_score(y_test2, y_pred_knn1))
print("Sensibilidad Clasificación por KNN:", recall_score(y_test2, y_pred_knn1))
print("Especificidad Clasificación por KNN:", recall_score(y_test2, y_pred_knn1,pos_label=0),"\n")
y_prob_knn=knn.predict_proba(x_test2)[:,1]
fpr,tpr,thresholds=roc_curve(y_test2,y_prob_knn)
auc_knn=roc_auc_score(y_test2,y_prob_knn)
print(f"AUC:{auc_knn:.2f}")
plt.plot(fpr,tpr,label=f"Curva ROC (AUC={auc_knn:.2f})")
plt.plot([0,1],[0,1],linestyle="--",label="Clasificador Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Regresión KNN")
plt.legend()
plt.show()
print("Matriz de confusión:\n", confusion_matrix(y_test2, y_pred_knn1),"\n")
# Clasificación por Arbol de Decisión

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(max_depth=4, random_state=5)
dtc.fit(x_train3, y_train2)
y_pred_dtc = dtc.predict(x_test3) # x_test2: datos originales y x_test3: datos normalizados
print("Exactitud Clasificación por Arbol de Decisión:", accuracy_score(y_test2, y_pred_dtc))
print("Precisión Clasificación por Arbol de Decisión:", precision_score(y_test2, y_pred_dtc))
print("Sensibilidad Clasificación por Arbol de Decisión:", recall_score(y_test2, y_pred_dtc))
print("Especificidad Clasificación por Arbol de Decisión:", recall_score(y_test2, y_pred_dtc,pos_label=0),"\n")
y_prob_dtc=dtc.predict_proba(x_test2)[:,1]
fpr,tpr,thresholds=roc_curve(y_test2,y_prob_dtc)
auc_dtc=roc_auc_score(y_test2,y_prob_dtc)
print(f"AUC:{auc_dtc:.2f}")
plt.plot(fpr,tpr,label=f"Curav ROC (AUC={auc_dtc:.2f})")
plt.plot([0,1],[0,1],linestyle="--",label="Clasificador Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Regresión Arbol de Decisión")
plt.legend()
plt.show()
print("Matriz de confusión:\n", confusion_matrix(y_test2, y_pred_dtc),"\n")
# Clasificación por Bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
base_clf = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=69)
bagging_clf.fit(x_train3, y_train2)
y_pred_bagging = bagging_clf.predict(x_test3) # x_test2: datos originales y x_test3: datos normalizados
print(f"Exactitud Clasificación por Bagging:",accuracy_score(y_test2, y_pred_bagging))
print("Precisión Clasificación por Bagging:", precision_score(y_test2, y_pred_bagging))
print("Sensibilidad Clasificación por Bagging:", recall_score(y_test2, y_pred_bagging))
print("Especificidad Clasificación por Bagging:", recall_score(y_test2, y_pred_bagging,pos_label=0),"\n")
y_prob_bagging=bagging_clf.predict_proba(x_test2)[:,1]
fpr,tpr,thresholds=roc_curve(y_test2,y_prob_bagging)
auc_bagging=roc_auc_score(y_test2,y_prob_bagging)
print(f"AUC:{auc_bagging:.2f}")
plt.plot(fpr,tpr,label=f"Curav ROC (AUC={auc_bagging:.2f})")
plt.plot([0,1],[0,1],linestyle="--",label="Clasificador Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Bagging")
plt.legend()
plt.show()
print("Matriz de confusión:\n", confusion_matrix(y_test2, y_pred_bagging),"\n")
# Clasificación por SVC

from sklearn.svm import SVC
svc = SVC(kernel="rbf", gamma="scale", random_state=17,probability=True)
svc.fit(x_train3, y_train2)
y_pred_svc = svc.predict(x_test3) # x_test2: datos originales y x_test3: datos normalizados
print(f"Exactitud Clasificación por SVC:",accuracy_score(y_test2, y_pred_svc))
print("Precisión Clasificación por SVC:", precision_score(y_test2, y_pred_svc))
print("Sensibilidad Clasificación por SVC:", recall_score(y_test2, y_pred_svc))
print("Especificidad Clasificación por SVC:", recall_score(y_test2, y_pred_svc,pos_label=0),"\n")
y_prob_svc=svc.predict_proba(x_test2)[:,1]
fpr,tpr,thresholds=roc_curve(y_test2,y_prob_svc)
auc_svc=roc_auc_score(y_test2,y_prob_svc)
print(f"AUC:{auc_svc:.2f}")
plt.plot(fpr,tpr,label=f"Curav ROC (AUC={auc_svc:.2f})")
plt.plot([0,1],[0,1],linestyle="--",label="Clasificador Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Regresión SVC")
plt.legend()
plt.show()
print("Matriz de confusión:\n", confusion_matrix(y_test2, y_pred_svc),"\n")

# 3. Interpreta los resultados. ¿Qué tan bien está clasificando el modelo? ¿Cuál es la tasa de falsos positivos y falsos negativos? 
# ¿El modelo es mejor que un clasificador aleatorio?
# Ninguno de los modelos de clasificación propuestos ajustan bien, en la gran mayoría presentan una tasa de predicción del 0.5;
# lo que no es distinto del azar, asi que habría que probar otra clase de modelos, dado que aun moviendo hiperparametros de los ajustes
# testeados, no se sube significativamente ni la matriz de confusión, ni los otros parametros de predictibilidad.