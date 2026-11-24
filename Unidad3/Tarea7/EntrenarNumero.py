import joblib
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("1. Descargando dataset MNIST...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print("2. Entrenando red neuronal...")
modelo = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42, verbose=True)
modelo.fit(X_train, y_train)

precision = accuracy_score(y_test, modelo.predict(X_test))
print(f"\nPrecisión en prueba: {precision * 100:.2f}%")

joblib.dump(modelo, "digit_mlp.joblib")
print("Modelo guardado como 'digit_mlp.joblib'")