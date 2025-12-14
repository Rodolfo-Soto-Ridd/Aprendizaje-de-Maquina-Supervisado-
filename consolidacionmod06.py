# 1. Carga y exploración de datos (1 punto)
# • Carga el dataset proporcionado, que contiene información sobre temperatura media, cambio en las 
# precipitaciones, frecuencia de sequías y producción agrícola en distintos países.
# • Analiza la distribución de las variables y detecta posibles valores atípicos o tendencias.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv("cambio_climatico_agricultura.csv")
print("\n",df.head(),"\n")
print("\n",df.describe(),"\n")
print("\n",df.isnull().sum(),"\n")
print("\n",df,"\n")
df_limpio=df.drop("País",axis=1)  # eliminación columna pais, dado que no aporta información de valor para el analisis
print("\n",df_limpio,"\n")
# Visualización por Histograma de las Variables Numéricas
df_limpio.hist(bins=10, figsize=(10,8))
plt.suptitle("Distribución de las Variables Numéricas")
plt.show()
# Visualización de Boxplot para las variables numéricas
fig, axes = plt.subplots(1, 4, figsize=(12, 5))
# Crear los boxplots para cada variable numérica en subgráficos separados
sns.boxplot(data=df_limpio['Temperatura_promedio'], ax=axes[0])
axes[0].set_title("Temperatura Promedio")
axes[0].set_ylabel("Temperatura")
sns.boxplot(data=df_limpio['Cambio_lluvias'], ax=axes[1])
axes[1].set_title("Cambio en Lluvias")
axes[1].set_ylabel("Cambio en Precipitaciones")
sns.boxplot(data=df_limpio['Frecuencia_sequías'], ax=axes[2])
axes[2].set_title("Frecuencia de Sequías")
axes[2].set_ylabel("Frecuencia de Sequías")
sns.boxplot(data=df_limpio['Producción_alimentos'], ax=axes[3])
axes[3].set_title("Producción de Alimentos")
axes[3].set_ylabel("Producción de Alimentos")
plt.tight_layout()
plt.show()
# Scatter Plot contra Producción de Alimentos
variables = df_limpio[["Temperatura_promedio","Cambio_lluvias","Frecuencia_sequías"]]
plt.figure(figsize=(15, 4))
for i, var in enumerate(variables, 1):
    plt.subplot(1, 3, i)
    plt.scatter(df[var], df['Producción_alimentos'], color='green', alpha=0.7)
    plt.title(f'{var} vs Producción')
    plt.xlabel(var)
    plt.ylabel('Producción de alimentos')
plt.tight_layout()
plt.show()
# Visualización de la Relación entre las Variables utilizando un Mapa de Calor
correlation_matrix = df_limpio.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de Correlación entre Variables")
plt.show()
# Se aprecia que hay alguna relación entre temperatura promedio y frecuencia sequias,
# y hay una relación leve entre cambio de lluvias y produccion de alimentos

# 2. Preprocesamiento y escalamiento de datos (2 puntos)
# • Aplica técnicas de normalización o estandarización a las variables numéricas.
# • Codifica correctamente cualquier variable categórica si fuera necesario.
# • Divide los datos en conjunto de entrenamiento y prueba (80%-20%).
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Separar las variables independientes (X) y la dependiente (y)
X = df_limpio.drop(columns=["Producción_alimentos"],axis=1)
y = df_limpio["Producción_alimentos"]
print("\n",X,"\n")
print("\n",y,"\n")
# Dividir en conjunto de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Aplicación de modelos de aprendizaje supervisado (4 puntos)
# • Regresión:
# o Entrena un modelo de regresión lineal para predecir la producción de alimentos.
# o Evalúa el modelo usando métricas como MAE, MSE y R2.
# o Compara con otros modelos de regresión (árbol de decisión, random forest).
# Regresión Lineal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n",y_pred,"\n")
print("\n",y.head(),"\n")
# Métricas
promedio_prod_alim=df_limpio["Producción_alimentos"].mean()
mae_lr = mean_absolute_error(y_test, y_pred)
mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred)
print(f"Promedio Producción Alimentos:{promedio_prod_alim:.2f}[TONS]")
print(f"MAE - Regresión Lineal: {mae_lr:.2f}[TONS]")
print(f"MSE - Regresión Lineal: {mse_lr:.2f}[TONS]")
print(f"RMSE - Regresión Lineal: {rmse_lr:.2f}")
print(f"R² - Regresión Lineal: {r2_lr:.2f}","\n")
# Arbol de Decisión
# Modelo Arbol de Decisión
from sklearn.tree import DecisionTreeRegressor
model_dt = DecisionTreeRegressor(random_state=16) 
model_dt.fit(X_train_scaled, y_train)
y_pred_dt = model_dt.predict(X_test_scaled)
# Métricas
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Promedio Producción Alimentos:{promedio_prod_alim:.2f}[TONS]")
print(f"MAE - Regresión Arbol de Decisión: {mae_dt:.2f}[TONS]")
print(f"MSE - Regresión Arbol de Decisión: {mse_dt:.2f}[TONS]")
print(f"RMSE - Regresión Arbol de Decisión: {rmse_dt:.2f}")
print(f"R² - Regresión Arbol de Decisión: {r2_dt:.2f}","\n")
# Random Forest
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=75, random_state=5)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
# Métricas
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Promedio Valor Ventas:{promedio_prod_alim:.2f}[TONS]")
print(f"MAE - Regresión Random Forest: {mae_rf:.2f}[TONS]")
print(f"MSE - Regresión Random Forest: {mse_rf:.2f}[TONS]")
print(f"RMSE - Regresión Random Forest: {rmse_rf:.2f}")
print(f"R² - Regresión Random Forest: {r2_rf:.2f}","\n")

# • Clasificación:
# o Crea una nueva variable categórica que clasifique los países en "Bajo", "Medio" y "Alto" impacto climático en la producción agrícola.
# o Entrena modelos de clasificación como K-Nearest Neighbors, Árbol de Decisión y Support Vector Machine. 
# o Evalúa el desempeño usando matriz de confusión, precisión, sensibilidad y curva ROC-AUC.

# Crear categorías: bajo, medio, alto ------> CONSULTAR SI ESTO ESTA BIEN, VARIABLES DEPENDIENTES E INDEPENDIENTES
df['Impacto_agricola'] = pd.cut(df['Producción_alimentos'],bins=3,labels=['bajo', 'medio', 'alto'])
print("\n",df[['País','Producción_alimentos','Impacto_agricola']],"\n")
print("\n",df,"\n")
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
categorias_codificadas = encoder.fit_transform(df[['Impacto_agricola',"País"]])
nombres_columnas = encoder.get_feature_names_out(['Impacto_agricola',"País"])
df_codificado = pd.DataFrame(categorias_codificadas, columns=nombres_columnas)
df_final = pd.concat([df.drop('Impacto_agricola', axis=1), df_codificado], axis=1)
df_final1= df_final.drop("País",axis=1)
print("\n",df_final1,"\n")
from sklearn.metrics import (confusion_matrix,precision_score,accuracy_score,recall_score,roc_curve,roc_auc_score)
# Modelo Clasificación KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
X1 = df_final1.drop(columns=["Impacto_agricola_bajo","Impacto_agricola_medio","Impacto_agricola_alto"])  # ojo aqui
y1 = df_final1[["Impacto_agricola_bajo","Impacto_agricola_medio","Impacto_agricola_alto"]]               # ojo aqui
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.25, random_state=7)
scaler1 = StandardScaler()
X_train_scaled1 = scaler1.fit_transform(X_train1)
X_test_scaled1 = scaler1.transform(X_test1)
knn = KNeighborsClassifier(n_neighbors=3)
modelo_knn = MultiOutputClassifier(knn) # PREGUNTAR POR ESTO
modelo_knn.fit(X_train_scaled1, y_train1)
y_pred_knn = modelo_knn.predict(X_test_scaled1)
print("Exactitud Clasificación por KNN:", accuracy_score(y_test1, y_pred_knn))
print("Precisión Clasificación por KNN:", precision_score(y_test1, y_pred_knn,average="weighted"))
print("Sensibilidad Clasificación por KNN:", recall_score(y_test1, y_pred_knn,average="weighted"))
print("Especificidad Clasificación por KNN:", recall_score(y_test1, y_pred_knn,pos_label=0,average="weighted"),"\n")
y_prob_knn=modelo_knn.predict_proba(X_test_scaled1)
y_test_array=y_test1.values
y_true_label0=y_test_array[:,0]
y_prob_label0=probs[0][:,1] # PEDIR AYUDA AQUI
fpr,tpr,thresholds=roc_curve(y_true_label0,y_prob_label0)
auc_knn=roc_auc_score(y_true_label0,y_prob_label0)
print(f"AUC:{auc_knn:.2f}")
plt.plot(fpr,tpr,label=f"Curva ROC (AUC={auc_knn:.2f})")
plt.plot([0,1],[0,1],linestyle="--",label="Clasificador Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Regresión KNN")
plt.legend()
plt.show()
print("Matriz de confusión KNN:\n", confusion_matrix(y_test1, y_pred_knn),"\n")

# Modelo Clasificación Arbol de Decisión
from sklearn.tree import DecisionTreeClassifier
modelo_dtc = DecisionTreeClassifier(max_depth=4, random_state=69)
modelo_dtc.fit(X_train_scaled1, y_train1)
y_pred_dtc = modelo_dtc.predict(X_test_scaled1)
print("Exactitud Clasificación por DTC:", accuracy_score(y_test1, y_pred_dtc))
print("Precisión Clasificación por DTC:", precision_score(y_test1, y_pred_dtc))
print("Sensibilidad Clasificación por DTC:", recall_score(y_test1, y_pred_dtc))
print("Especificidad Clasificación por DTC:", recall_score(y_test1, y_pred_dtc,pos_label=0),"\n")
y_prob_dtc=modelo_dtc.predict_proba(X_test_scaled1)[:,1]
fpr,tpr,thresholds=roc_curve(y_test1,y_prob_dtc)
auc_dtc=roc_auc_score(y_test1,y_prob_dtc)
print(f"AUC:{auc_dtc:.2f}")
plt.plot(fpr,tpr,label=f"Curva ROC (AUC={auc_dtc:.2f})")
plt.plot([0,1],[0,1],linestyle="--",label="Clasificador Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Regresión DTC")
plt.legend()
plt.show()
print("Matriz de confusión DTC:\n", confusion_matrix(y_test1, y_pred_dtc),"\n")

# Modelo Clasificación VSMC
from sklearn.svm import SVC
svc = SVC(kernel="rbf", gamma="scale", random_state=58)
svc.fit(X_train_scaled1, y_train1)
y_pred_svc = svc.predict(X_test_scaled1) 
print("Exactitud Clasificación por SVC:",accuracy_score(y_test1, y_pred_svc))
print("Precisión Clasificación por SVC:", precision_score(y_test1, y_pred_svc))
print("Sensibilidad Clasificación por SVC:", recall_score(y_test1, y_pred_svc))
print("Especificidad Clasificación por SVC:", recall_score(y_test1, y_pred_svc,pos_label=0),"\n")
y_prob_svc=svc.predict_proba(X_test_scaled1)[:,1]
fpr,tpr,thresholds=roc_curve(y_test1,y_prob_svc)
auc_svc=roc_auc_score(y_test1,y_prob_svc)
print(f"AUC:{auc_svc:.2f}")
plt.plot(fpr,tpr,label=f"Curav ROC (AUC={auc_svc:.2f})")
plt.plot([0,1],[0,1],linestyle="--",label="Clasificador Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC Regresión SVC")
plt.legend()
plt.show()
print("Matriz de confusión SVC:\n", confusion_matrix(y_test1, y_pred_svc),"\n")

# 4. Optimización de modelos (2 puntos)
# • Ajusta hiperparámetros utilizando validación cruzada y búsqueda en grilla.
# • Aplica técnicas de regularización y analiza su impacto en los modelos.
