import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# Cargar los datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/winequality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Mostrar las primeras filas
print(data.head())
# Estadísticas descriptivas
print(data.describe())

# Distribución de la variable objetivo (calidad)
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=data)
plt.title('Distribución de la Calidad del Vino')
plt.show()

# Correlación entre variables
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# Boxplots para algunas variables vs calidad
plt.figure(figsize=(12, 6))
sns.boxplot(x='quality', y='alcohol', data=data)
plt.title('Relación entre Alcohol y Calidad')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='quality', y='volatile acidity', data=data)
plt.title('Relación entre Acidez Volátil y Calidad')
plt.show()
# Dividir en características (X) y variable objetivo (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Como la calidad es un valor entre 3 y 8, podemos convertirlo en un problema de clasificación binaria
# Por ejemplo, vinos "buenos" (calidad >= 7) y "no buenos" (calidad < 7)
y = y.apply(lambda x: 1 if x >= 7 else 0)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Crear y entrenar el modelo
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_dt = dt.predict(X_test)

# Evaluar el modelo
print("Árbol de Decisión - Reporte de Clasificación:")
print(classification_report(y_test, y_pred_dt))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_dt))
print("\nPrecisión:", accuracy_score(y_test, y_pred_dt))

# Visualizar el árbol (opcional)
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['No Bueno', 'Bueno'], filled=True, rounded=True)
plt.title("Árbol de Decisión")
plt.show()
# Crear y entrenar el modelo
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_rf = rf.predict(X_test)

# Evaluar el modelo
print("Random Forest - Reporte de Clasificación:")
print(classification_report(y_test, y_pred_rf))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nPrecisión:", accuracy_score(y_test, y_pred_rf))

# Ajuste de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Mejores parámetros
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)

# Mejor modelo
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("\nRandom Forest Optimizado - Reporte de Clasificación:")
print(classification_report(y_test, y_pred_best_rf))
print("\nPrecisión Optimizada:", accuracy_score(y_test, y_pred_best_rf))
# Obtener importancia de características
feature_importances = best_rf.feature_importances_
features = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
features = features.sort_values(by='Importance', ascending=False)

# Visualizar importancia
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features)
plt.title('Importancia de Características en el Random Forest')
plt.tight_layout()
plt.show()

# Mostrar las características más importantes
print("\nCaracterísticas más importantes:")
print(features)
