from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target
 
# Paso 1: Separar en conjunto de entrenamiento+validación y conjunto de prueba
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Paso 2: Separar el conjunto temporal en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
# 0.25 de 80% = 20%, entonces queda: 60% train, 20% val, 20% test
 
# Modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
 
# Predicciones
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
 
# Errores
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)
 
print("Accuracy en entrenamiento:", train_acc)
print("Accuracy en validación:   ", val_acc)
print("Accuracy en prueba:       ", test_acc)

