# 1. Carga de datos (1 punto)
# ● Descarga el archivo clientes.csv.
# ● Carga el conjunto de datos utilizando Python y realiza una exploración inicial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("clientes.csv")
print("\n",df.head(),"\n")
print("\n",df.info(),"\n")
print("\n",df.isnull().sum(),"\n")
print("\n",df.describe(),"\n")
# 2. Aplicación de modelos de clasificación (6 puntos)
# Implementa los siguientes modelos de clasificación y evalúa su desempeño:
# ● Regresión logística: Implementa el modelo y analiza cómo la función sigmoidea afecta la
# clasificación.
# ● K-Nearest Neighbors (K-NN): Prueba distintos valores de k y analiza su impacto en la exactitud.
# ● Árbol de decisión: Ajusta los hiperparámetros del modelo y analiza las medidas de impureza
# de los nodos.
# ● Bosques aleatorios: Aplica bagging para mejorar la predicción y analiza la importancia de
# las variables.
# ● Support Vector Machine (SVM): Experimenta con distintos tipos de kernel y compara los resultados.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = df.drop("Contratara_Servicio", axis=1) 
y = df["Contratara_Servicio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
# Normalización de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Modelo de Regresión Logística
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
# Predicción y Evaluación del Modelo
y_pred_lr = model_lr.predict(X_test)
print("Precisión de la regresión logística:", accuracy_score(y_test, y_pred_lr))

# Modelo Vecinos K-enesimos
from sklearn.neighbors import KNeighborsClassifier
k_values = [1, 3, 5, 7, 9]
for k in k_values:
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    print(f"Precisión para k={k}: {accuracy_score(y_test, y_pred_knn)}")

# Modelo Arbol de Decisión
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state=5)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
print("Precisión del árbol de decisión:", accuracy_score(y_test, y_pred_dt))

# Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=3)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
print("Precisión del Random Forest:", accuracy_score(y_test, y_pred_rf))
importances = model_rf.feature_importances_
print("Importancia de las variables:", importances)

# Modelo Máquina de Vector de Soporte (SVM)
from sklearn.svm import SVC
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    model_svm = SVC(kernel=kernel, random_state=42)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)
    print(f"Precisión con kernel={kernel}: {accuracy_score(y_test, y_pred_svm)}")

# 3. Análisis de resultados (3 puntos) 
# ● Comparación de modelos: Explica cuál modelo tuvo mejor desempeño y por qué.
# ● Impacto de los hiperparámetros: Describe cómo afectaron los hiperparámetros en cada modelo.
# ● Aplicabilidad: Indica en qué casos sería recomendable utilizar cada algoritmo

accuracies = {
    'Regresión Logística': accuracy_score(y_test, y_pred_lr),
    'K-NN (k=3)': accuracy_score(y_test, y_pred_knn),  # Asegúrate de tener el mejor valor de k
    'Árbol de Decisión': accuracy_score(y_test, y_pred_dt),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'SVM (kernel=rbf)': accuracy_score(y_test, y_pred_svm)  # Ajusta el mejor kernel
}
plt.bar(accuracies.keys(), accuracies.values())
plt.ylabel('Precisión')
plt.title('Comparación de Modelos de Clasificación')
plt.xticks(rotation=45)
plt.show()
# Comparación de Modelos
# Los mejores modelos de acuerdo a la precisión son el arbol de decisión, random forest y SVM, en este 
# caso es indistinto cual se escoja, hay que hacer nuevas pruebas variando los k-vecinos y los kernels
# para detectar variaciones tendenciales de estos modelos.

# Impacto de los Hiperparamétros
# Regresión Logística:
# Hiperparámetros principales:
# C (Regularización inversa):
# Descripción: Este parámetro controla la regularización del modelo. Una C alta significa poca 
# regularización (modelo más flexible, puede sobreajustarse), mientras que una C baja significa más 
# regularización (modelo más rígido, pero puede generalizar mejor).
# Impacto: Si el modelo tiene demasiados parámetros o está sobreajustado, una C más baja ayudará a 
# prevenirlo. Si la C es demasiado alta, el modelo puede sobreajustarse al conjunto de datos de 
# entrenamiento.

# K-Nearest Neighbors
# Hiperparámetros principales:
# k (Número de vecinos):ç
# Descripción: El número de vecinos más cercanos que se consideran para realizar una clasificación. 
# Valores pequeños de k pueden hacer que el modelo sea muy sensible al ruido, mientras que valores 
# grandes pueden suavizar demasiado la frontera de decisión.
# Impacto: Si k es pequeño (por ejemplo, 1), el modelo puede estar muy influenciado por puntos atípicos
# (sobreajuste). Si k es grande, el modelo puede perder capacidad para distinguir entre clases 
# (subajuste).

# Arbol de Decisión
# Hiperparámetros principales:
# max_depth (Profundidad máxima):
# Descripción: Este parámetro limita la profundidad del árbol. Si no se limita, el árbol puede crecer 
# hasta abarcar todo el conjunto de datos, lo que podría llevar a un sobreajuste.
# Impacto: Limitar la profundidad puede mejorar la generalización. Un árbol muy profundo tiende a 
# sobreajustarse.
# min_samples_split (Mínimos de muestras para dividir):
# Descripción: El número mínimo de muestras requeridas para dividir un nodo. Si es demasiado pequeño, 
# el árbol puede sobreajustarse.
# Impacto: Aumentar min_samples_split puede hacer que el árbol sea más general y menos sensible al 
# ruido.
# criterion (Criterio de división):
# Descripción: Es el método para medir la calidad de una división:
# 'gini': Índice de Gini (utilizado por defecto).
# 'entropy': Entropía de la información.
# Impacto: La diferencia entre estos dos criterios generalmente no es grande, pero puede cambiar 
# ligeramente el comportamiento del modelo.


# Random Forest
# Hiperparámetros principales:
# n_estimators (Número de árboles):
# Descripción: El número de árboles en el bosque. Un número mayor tiende a mejorar el rendimiento, 
# pero también aumenta el tiempo de entrenamiento.
# Impacto: Aumentar el número de árboles mejora la precisión hasta cierto punto, pero más allá de un 
# umbral puede resultar en un tiempo de cómputo innecesario.
# max_depth (Profundidad máxima):
# Descripción: Similar al árbol de decisión, limita la profundidad de los árboles individuales en el 
# bosque.
# Impacto: Limitar la profundidad ayuda a evitar que cada árbol se sobreajuste.
# min_samples_split (Mínimos de muestras para dividir):
# Descripción: Al igual que en el árbol de decisión, controla el número mínimo de muestras requeridas 
# para dividir un nodo.
# Impacto: Evitar que los árboles crezcan demasiado, lo que puede reducir el sobreajuste.

# Support Vector Machine
# Hiperparámetros principales:
# C (Regularización):
# Descripción: Similar al de la regresión logística, controla el margen de separación. 
# Un C alto significa que el modelo tratará de ajustar los puntos de entrenamiento a la perfección 
# (potencialmente sobreajustado).
# Impacto: Un C bajo permite un margen más amplio, pero puede dejar algunos errores en los datos de 
# entrenamiento.
# kernel:
# Descripción: Determina la forma del margen de decisión:
# 'linear': Margen lineal, adecuado para datos lineales.
# 'poly': Kernel polinómico, adecuado para fronteras no lineales.
# 'rbf': Kernel radial (Gaussiano), es uno de los más utilizados y funciona bien con datos no lineales.
# 'sigmoid': Similar a las redes neuronales, menos utilizado.
# Impacto: Cambiar el kernel es útil para capturar diferentes tipos de relaciones entre las clases en 
# el espacio de características.
# gamma:
# Descripción: Es un parámetro del kernel. Controla la influencia de cada punto de entrenamiento. 
# Un valor alto puede hacer que el modelo se ajuste demasiado a los puntos de entrenamiento 
# (sobreajuste), mientras que un gamma bajo hace que el margen.

# Aplicabilidad
# Regresión logística: Ideal para casos donde la relación entre las variables es lineal y se busca 
# interpretabilidad.
# K-NN: Bueno cuando no hay una clara relación lineal y los datos no son tan grandes.
# Árboles de decisión y Random Forest: Útiles para datos con interacciones complejas y no lineales.
# SVM: Excelente cuando se tienen márgenes claros entre clases, pero puede ser computacionalmente 
# costoso.
