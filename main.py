import numpy as np

X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

escala = np.array([100, 50])

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def predecir_probabilidad(entradas, pesos, sesgo):
    entradas = np.atleast_2d(np.asarray(entradas, dtype=float))
    return sigmoide((entradas / escala) @ pesos + sesgo)


def entrenar(tasa_aprendizaje, epocas, mostrar=False):
    rng = np.random.default_rng(42)
    pesos = rng.normal(size=(2, 1))
    sesgo = 0.0

    for epoca in range(1, epocas + 1):
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)
        error = probabilidad - y

        gradiente_z = error * probabilidad * (1 - probabilidad)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos = pesos - tasa_aprendizaje * gradiente_pesos
        sesgo = sesgo - tasa_aprendizaje * gradiente_sesgo

        if mostrar and (epoca == 1 or epoca % (epocas // 10) == 0):
            print(f"Época {epoca:>6} | Error cuadrático medio = {np.mean(error ** 2):.6f}")

    probabilidad = sigmoide(X_normalizado @ pesos + sesgo)
    error_final = np.mean((probabilidad - y) ** 2)
    return pesos, sesgo, error_final


print("\n=== Entrenamiento base: tasa_aprendizaje = 0.5, epocas = 10000 ===")
pesos, sesgo, error_final = entrenar(0.5, 10000, mostrar=True)

probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
respuestas = (probabilidades >= 0.5).astype(int)

print(f"\nPeso humedad     : {pesos[0, 0]:.4f}")
print(f"Peso temperatura : {pesos[1, 0]:.4f}")
print(f"Sesgo            : {sesgo:.4f}")
print(f"Error final      : {error_final:.6f}")

print("\nCaso | Humedad | Temperatura | Probabilidad | Respuesta | Esperada")
for i in range(len(X)):
    print(f"{i + 1:>4} | {X[i, 0]:>6.0f}% | {X[i, 1]:>9.0f}°C | "
          f"{probabilidades[i, 0]:>12.4f} | {respuestas[i, 0]:>9} | {int(y[i, 0]):>8}")

nuevos = np.array([[75, 30], [45, 34], [25, 22], [50, 25], [30, 40]], dtype=float)
probabilidades_nuevos = predecir_probabilidad(nuevos, pesos, sesgo)

print("\n=== Pruebas con condiciones nuevas ===")
print("Humedad | Temperatura | Probabilidad | Decisión")
for i in range(len(nuevos)):
    decision = (probabilidades_nuevos[i, 0] >= 0.5).astype(int)
    print(f"{nuevos[i, 0]:>6.0f}% | {nuevos[i, 1]:>9.0f}°C | "
          f"{probabilidades_nuevos[i, 0]:>12.4f} | {decision}")

experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

print("\n=== Experimentos con los parámetros ===")
print(f"{'Prueba':<20}{'Épocas':>7}{'Tasa':>6}{'Error final':>13}{'Correctas':>11}{'P(45%, 34°C)':>14}")
for nombre, epocas_exp, tasa_exp in experimentos:
    w, b, error_exp = entrenar(tasa_exp, epocas_exp)
    prediccion = (sigmoide(X_normalizado @ w + b) >= 0.5).astype(int)
    correctas = int(np.sum(prediccion == y))
    p = predecir_probabilidad([45, 34], w, b)[0, 0]
    print(f"{nombre:<20}{epocas_exp:>7}{tasa_exp:>6}{error_exp:>13.6f}{correctas:>8}/10{p:>14.4f}")

print("\n=== Prueba con umbrales (sin volver a entrenar) ===")
todos = np.vstack([X, nuevos])
probabilidades_todos = predecir_probabilidad(todos, pesos, sesgo)
print("Humedad | Temperatura | Probabilidad | U=0.4 | U=0.5 | U=0.6")
for i in range(len(todos)):
    p = probabilidades_todos[i, 0]
    print(f"{todos[i, 0]:>6.0f}% | {todos[i, 1]:>9.0f}°C | {p:>12.4f} | "
          f"{int(p >= 0.4):>5} | {int(p >= 0.5):>5} | {int(p >= 0.6):>5}")
