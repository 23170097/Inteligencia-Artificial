import numpy as np
from sklearn.datasets import fetch_california_housing


def ajustar_regresion_lineal(x, y):
    """Calcula el intercepto (b0) y la pendiente (b1) por mínimos cuadrados."""
    prom_x = np.mean(x)
    prom_y = np.mean(y)

    numerador = np.sum((x - prom_x) * (y - prom_y))
    denominador = np.sum((x - prom_x) ** 2)

    pendiente = numerador / denominador
    intercepto = prom_y - (pendiente * prom_x)
    return intercepto, pendiente


def evaluar_modelo(y_real, y_estimado):
    """Calcula MSE, RMSE y el coeficiente de determinación R^2."""
    residuos = y_real - y_estimado
    mse = np.mean(residuos ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum(residuos ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mse, rmse, r2


# 1. Cargar el dataset California Housing
dataset = fetch_california_housing(as_frame=True)

# 2. Seleccionar la variable independiente (MedInc) y la dependiente (precio)
n_muestras = 100  # Puedes cambiarlo a 50 o quitar [:n_muestras] para usar todo el dataset
x = dataset.data["MedInc"].to_numpy()[:n_muestras]
y = dataset.target.to_numpy()[:n_muestras]

# 3. Entrenar el modelo manualmente
b0, b1 = ajustar_regresion_lineal(x, y)
print(f"Modelo lineal: y = {b0:.4f} + {b1:.4f}x")

# 4. Generar predicciones y evaluar
y_pred = b0 + b1 * x
mse, rmse, r2 = evaluar_modelo(y, y_pred)

print(f"MSE  (Error Cuadrático Medio): {mse:.4f}")
print(f"RMSE (Raíz del MSE):           {rmse:.4f}")
print(f"R^2  (Ajuste del modelo):      {r2:.4f}")

# 5. Predicción de prueba
ingreso_ejemplo = 5.0
precio_estimado = b0 + b1 * ingreso_ejemplo
print(f"\nPredicción para MedInc = {ingreso_ejemplo}: ${precio_estimado * 100000:,.2f} USD")