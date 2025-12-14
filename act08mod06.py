# 1. Carga de datos (1 punto)
# • Descarga el archivo clientes.csv.
# • Carga el conjunto de datos utilizando Python y realiza una exploración inicial.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("clientes.csv")
print("\n",df.head(),"\n")
print("\n",df.describe(),"\n")
print("\n",df.isnull().sum(),"\n")
print("\n",df.duplicated(),"\n")
print("\n",df.drop_duplicates(),"\n")
# 2. Aplicación de modelos de Boosting y Gradient Boosting (6 puntos)
# Implementa los siguientes modelos de clasificación y evalúa su desempeño:
# • AdaBoost: Implementa el modelo utilizando AdaBoostClassifier y analiza cómo el número de estimadores (n_estimators) 
# afecta la clasificación.
# • Gradient Boosting: Implementa el modelo utilizando GradientBoostingClassifier y ajusta hiperparámetros como n_estimators, 
# learning_rate, y max_depth.
# • Random Forest (Bagging): Implementa un modelo de RandomForest utilizando RandomForestClassifier para comparar su rendimiento 
# con los modelos de boosting.
# Para cada modelo:
# • Divide los datos en entrenamiento y prueba.
# • Normaliza los datos si es necesario.
# • Evalúa la precisión con métricas como accuracy y matriz de confusión.
# Separar características (X) y variable objetivo (y)
X = df.drop("Contratara_Servicio", axis=1)
y = df["Contratara_Servicio"]
# División del dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
# Normalización de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Función para entrenar y evaluar modelos
def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, nombre):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{nombre} - Accuracy: {acc:.4f}")
    print("Matriz de Confusión:")
    print(cm)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    return acc, cm
# AdaBoost
modelo_adaboost = AdaBoostClassifier(n_estimators=5, random_state=8)
acc_ada, cm_ada = evaluar_modelo(modelo_adaboost, X_train_scaled, X_test_scaled, y_train, y_test, "AdaBoost")

# Gradient Boosting
modelo_gb = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=2, random_state=5)
acc_gb, cm_gb = evaluar_modelo(modelo_gb, X_train, X_test, y_train, y_test, "Gradient Boosting")

# Random Forest
modelo_rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=1)
acc_rf, cm_rf = evaluar_modelo(modelo_rf, X_train, X_test, y_train, y_test, "Random Forest")

# Visualización de matrices de confusión
fig, axs = plt.subplots(1, 3, figsize=(12, 5))
for ax, cm, title in zip(axs, [cm_ada, cm_gb, cm_rf], ["AdaBoost", "Gradient Boosting", "Random Forest"]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
plt.tight_layout()
plt.show()

# 2. Impacto de Hiperparámetros
# n_estimators: Aumentar este valor suele mejorar la precisión hasta cierto punto, pero puede causar sobreajuste.
# learning_rate: En Gradient Boosting, reduce la tasa de aprendizaje, lo que requiere más estimadores pero mejora la generalización.
# max_depth: Controla la complejidad de cada árbol. Un valor muy alto puede sobreajustar.

# 3. Aplicabilidad de cada modelo
# Modelo	            Fortalezas      	            Cuándo usarlo
# AdaBoost	            Rápido, mejora modelos débiles	Cuando necesitas simplicidad y velocidad.
# Gradient Boosting	    Alta precisión y control	    Cuando buscas máximo rendimiento y puedes ajustar hiperparámetros.
# Random Forest	        Robusto, fácil de usar	        Ideal para datos con ruido o cuando el sobreajuste es un problema.