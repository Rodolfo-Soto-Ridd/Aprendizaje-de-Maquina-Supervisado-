from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
# Datos de ejemplo: [puntaje_matematica, puntaje_lenguaje]
X = [[450, 420], [480, 460], [500, 490], [520, 510],
    [550, 520], [600, 580], [610, 600], [620, 610]]
y = [0, 0, 0, 0, 1, 1, 1, 1]  # 0 = no admitido, 1 = admitido
# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Crear el modelo de regresión logística
modelo = LogisticRegression()
# Entrenar el modelo
modelo.fit(X_train, y_train)
# Predecir con el modelo
y_pred = modelo.predict(X_test)
# Evaluar el modelo
print("Exactitud RL:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Datos de ejemplo
X1 = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Variables independientes
y1 = [0, 0, 1, 1]  # Variable dependiente (clases)
# Dividir los datos en entrenamiento y prueba
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.25, random_state=42)
# Escalar los datos
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.transform(X_test1)
# Crear el modelo K-NN (por ejemplo, k=2)
modelo = KNeighborsClassifier(n_neighbors=2)
# Entrenar el modelo
modelo.fit(X_train1, y_train1)
# Predecir con el modelo
y_pred1 = modelo.predict(X_test1)
# Ver los datos escalados
print("X_train escalado:", X_train1)
print("X_test escalado:", X_test1)
# Ver las etiquetas reales y predichas
print("Etiqueta real (y_test):", y_test1)
print("Etiqueta predicha (y_pred):", y_pred1)
# Evaluar el modelo
print("Exactitud K-NN:", accuracy_score(y_test1, y_pred1))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Cargar el conjunto de datos Iris
data = load_iris()
X2 = data.data  # Características
y2 = data.target  # Etiquetas
# Dividir los datos en entrenamiento y prueba
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.25, random_state=42)
# Crear el modelo de árbol de decisión
modelo = DecisionTreeClassifier(max_depth=6, random_state=42)
# Entrenar el modelo
modelo.fit(X_train2, y_train2)
# Predecir con el modelo
y_pred2 = modelo.predict(X_test2)
# Evaluar el modelo
print("Exactitud Arbol Decisión:", accuracy_score(y_test2, y_pred2))
# Visualizar el árbol
plt.figure(figsize=(12, 8))
plot_tree(modelo, 
          feature_names=data.feature_names, 
          class_names=data.target_names, 
          filled=True, 
          rounded=True)
plt.title("Árbol de decisión - Iris")
plt.show()

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Crear un conjunto de datos de ejemplo
X3, y3 = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=5)
# Dividir el conjunto de datos en entrenamiento y prueba
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
# Crear un clasificador base (por ejemplo, un árbol de decisión)
base_clf = DecisionTreeClassifier()
# Implementar Bagging con el clasificador base
bagging_clf = BaggingClassifier(estimator=base_clf, n_estimators=10, random_state=42)
# Entrenar el modelo de Bagging
bagging_clf.fit(X_train3, y_train3)
# Realizar predicciones
y_pred3 = bagging_clf.predict(X_test3)
# Calcular la precisión
accuracy = accuracy_score(y_test3, y_pred3)
print(f"Precisión del modelo Bagging: {accuracy:.2f}")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Cargar el conjunto de datos Iris
data = load_iris()
X4 = data.data  # Características
y4 = data.target  # Etiquetas
# Dividir los datos en entrenamiento y prueba
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.25, random_state=42)
# Crear el modelo SVM con kernel RBF
modelo = SVC(kernel="rbf", gamma="scale", random_state=42)
# Entrenar el modelo
modelo.fit(X_train4, y_train4)
# Predecir con el modelo
y_pred4 = modelo.predict(X_test4)
# Evaluar el modelo
print("Exactitud SVM:", accuracy_score(y_test4, y_pred4))