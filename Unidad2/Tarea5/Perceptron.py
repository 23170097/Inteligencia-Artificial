# Datos de entrenamiento: compuerta lógica AND
entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
salidas_esperadas = [0, 0, 0, 1]

# Inicialización de pesos, sesgo y tasa de aprendizaje
pesos = [0.0, 0.0]
sesgo = 0.0
tasa_aprendizaje = 0.5


def activar(entrada):
    suma = entrada[0] * pesos[0] + entrada[1] * pesos[1] + sesgo
    return 1 if suma >= 0 else 0


# Ciclo de entrenamiento
for epoca in range(1, 15):
    total_errores = 0
    for x, esperado in zip(entradas, salidas_esperadas):
        estimado = activar(x)
        error = esperado - estimado

        if error != 0:
            pesos[0] += tasa_aprendizaje * error * x[0]
            pesos[1] += tasa_aprendizaje * error * x[1]
            sesgo += tasa_aprendizaje * error
            total_errores += 1

    print(f"Época {epoca:2d} | Errores: {total_errores} | Pesos: {pesos} | Sesgo: {sesgo:.2f}")
    if total_errores == 0:
        break

# Verificación final
print("\n=== Pruebas Compuerta AND ===")
for x in entradas:
    print(f"{x} -> {activar(x)}")