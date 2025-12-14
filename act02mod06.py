# 1. Niveles de ajuste de un modelo: Explica con tus propias palabras qué es el sobreajuste,
# el subajuste y el ajuste apropiado de un modelo. Proporciona un ejemplo práctico para cada caso.

# Un SOBREAJUSTE, es un modelo que predice casi con completa certeza los datos de entrenamiento, pero si
# dichos datos son pocos, y/o no incluyen datos reales complejos, el modelo será disfuncional cuando
# se use en contextos reales. Ejemplo: Usar datos de comportamiento de flujo de fluidos, y entrenar solo 
# con fluidos newtonianos (tipo agua), y cuando se usa en fluidos alimentarios, son fluidos no newtonianos
# (ejemplo, manjar, ketchup, mayonesa, etc) y las predicciones fallen.

# Un SUBAJUSTE, es un modelo que es entrenado con muy pocos datos, esto genera dos problema, por los 
# pocos datos utilizados, existe la posibilidad que el modelamiento sea incorrecto (dos puntos generan 
# una recta, pero 10 puntos pueden generar una curva por ejemplo); y por otro lado, si es modelo es
# incorrecto, cuando se le pasen datos fuera del entrenamiento, es muy probable que las predicciones 
# disten mucho de la realidad que se desea anticipar. Ejemplo: construir un modelo de predicción de 
# tiempos de congelacion de alimentos, con datos de tiempo versus temperatura cada 5 minutos. En la 
# realidad hay una serie de fenomenos que ocurren en muy poco tiempo, luego un predictor fallará porque no
# "ve" el comportamiento real del fenomeno observado (hay que medir tiempo y temperatura cada 5 a 10 seg).

# un AJUSTE APROPIADO DE UN MODELO, es un modelamiento que combina adecuadamente tanto un 
# comportamiento de tendencia general versus ciertos elementos de variabilidad de datos reales. 
# Ejemplo: Tomar datos de un perfil de flujo tanto de fluidos newtonianos y no newtonianos cada 1 min, 
# y hacer lo mismo, variando en ocasiones el caudal y la temperatura del fluidos, en una tubería determinada

# 2. Trade-off entre sesgo y varianza:
# o ¿Qué es el sesgo (bias) y la varianza en el contexto de modelos de aprendizaje automático?
# o ¿Cómo se relacionan con el sobreajuste y el subajuste?
# o Proporciona un ejemplo de cómo equilibrar el sesgo y la varianza en un modelo

# un SESGO, es un supuesto sobre alguna condicion de la experimentación, que no representa a la realidad
# e introduce un error en lo que sea que se contruya a partir de esto.

# la VARIANZA en modelos de aprendizaje automático, es la sensibilidad del modelo a pequeñas fluctuaciones en
# los datos de entrenamiento 

# Los sesgos son directamente proporcionales tanto a subajuste como sobreajustes; por otro lado, una varianza 
# alta tiende al sobreajuste, y una varianza baja, tiende al subajuste
# Subajuste: sesgo alto, varianza baja.
# sobreajuste: sesgo bajo, varianza alta.

# Existe una serie de técnica para poder compensar tanto los sesgos como la varianza en modelos de aprendizaje
# automático; una alternativa es utilizar la estrategia de random forest o validaciones cruzadas

# 3. 3. Validación cruzada:
# o Explica qué es la validación cruzada y por qué es importante en la evaluación de modelos.
# o Describe brevemente las siguientes técnicas de validación cruzada:
#   ▪ Método de retención (Hold-Out).
#   ▪ Validación cruzada de k-iteraciones (k-Fold).
#   ▪ Validación cruzada aleatoria (Random Subsampling).
#   ▪ Validación cruzada dejando uno afuera (Leave-One-Out). 

# Una VALIDACIÓN CRUZADA, consiste en dividir los datos disponibles en varios subconjuntos, para entrenar el 
# modelo con algunos datos, mientras se valida con otro grupo de datos, la idea es hacer multiples validaciones
# mediante iteraciones.

# METODO DE RETENCIÓN, Consiste en dividir el conjunto de datos en dos partes: un conjunto de
# entrenamiento (usado para ajustar el modelo) y un conjunto de prueba (usado para evaluar su rendimiento).  

# una VALIDACION CRUZADA K - FOLD, o En este método, el conjunto de datos se divide en k subconjuntos (o "folds") de
# tamaño aproximadamente igual.
# o El modelo se entrena k veces, utilizando en cada iteración k−1 folds para entrenamiento y el fold restante para 
# evaluación.
# o Finalmente, se calcula el promedio de las métricas de rendimiento obtenidas en cada iteración

# un RANDOM SAMPLING, Implica dividir repetidamente los datos en conjuntos de entrenamiento y validación de forma 
# aleatoria y evaluar el modelo en varias iteraciones.

# una VALIDACIÓN CRUZADA LEAVE - ONE - OUT, Es un caso especial de k-Fold donde k es igual al número de muestras en 
# el conjunto de datos. En cada iteración, se deja fuera una única muestra para evaluación y se entrena el modelo 
# con el resto de los datos.

# 4. 4. Implementación con Scikit-Learn:
# o ¿Cómo se implementa la validación cruzada utilizando la librería Scikit-Learn?
# o Menciona al menos dos funciones o clases de Scikit-Learn que se utilizan para este propósito.

# Usando funciones como cross_val_score() o GridSearchCV, que automatizan el proceso de división de datos y 
# evaluación de métricas.
# cross_val_score(estimator, X, y, cv=k): Evalúa el modelo con validación cruzada.
# GridSearchCV(estimator, param_grid, cv=k): Busca los mejores hiperparámetros mediante validación cruzada.

# Caso:
# Un equipo de científicos de datos está desarrollando un modelo de predicción para estimar el precio de 
# viviendas en una ciudad. Han notado que el modelo tiene un rendimiento inconsistente: en algunos casos predice 
# muy bien, pero en otros falla significativamente. Sospechan que el modelo puede estar sufriendo de sobreajuste 
# o subajuste.

# Preguntas:
# 1. ¿Cómo podrían diagnosticar si el modelo está sobreajustado o subajustado? Describe los pasos que seguirías 
# para identificar el problema.

# Comparar el error en entrenamiento y prueba:
# Si el error en entrenamiento es bajo, y el error en prueba es alto → sobreajuste.
# Si ambos errores son altos → subajuste.
# Pasos:
# Evaluar el modelo con datos de prueba.
# Medir métricas como MSE o R² en entrenamiento y prueba.

# 2. ¿Qué técnica de validación cruzada recomendarías para evaluar el rendimiento del modelo y por qué? 

# Recomendación: k-Fold Cross Validation (por ejemplo, k=5 o k=10).
# Razón:
# Proporciona un buen equilibrio entre precisión y costo computacional. Permite evaluar el rendimiento promedio 
# del modelo sobre diferentes divisiones del conjunto de datos, lo que lo hace más robusto que Hold-Out.

# 3. ¿Cómo podrían utilizar la validación cruzada para ajustar los hiperparámetros del modelo y mejorar su 
# rendimiento?

# Se puede usar GridSearchCV o RandomizedSearchCV:
# Definir un rango de valores para cada hiperparámetro.
# Utilizar GridSearchCV con k-Fold para probar todas las combinaciones posibles.
# El modelo se entrena y valida k veces por combinación.
# Se seleccionan los hiperparámetros con mejor puntuación promedio.

# 4. ¿Qué beneficios y desafíos podrían enfrentar al implementar la validación cruzada en este caso? 

# Beneficios:
# Mejora la estimación del rendimiento real del modelo.
# Ayuda a detectar sobreajuste o subajuste.
# Permite elegir hiperparámetros más adecuados.
# Desafíos:
# Costo computacional alto, especialmente con grandes conjuntos de datos o modelos complejos.
# Tiempo de entrenamiento mayor, ya que el modelo se entrena múltiples veces.
# Puede requerir una buena estrategia de partición si los datos tienen desequilibrios o temporalidad.


