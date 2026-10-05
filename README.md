# Neurona artificial para el riego de plantas

Talle IA Generativa. Una neurona artificial con **dos entradas, dos pesos, un sesgo y activación sigmoide** decide si una planta necesita riego (`1` = regar, `0` = no regar).


## Problema

| Variable | Descripción | Escala |
|---|---|---|
| x1 | Humedad del suelo (%) | 0 – 100 |
| x2 | Temperatura ambiental (°C) | 0 – 50 aprox. |
| y | 1 = regar, 0 = no regar | binaria |

Se entrena con 10 ejemplos. Las entradas se normalizan dividiendo por `escala = [100, 50]`; **la misma escala** se usa para los datos nuevos. `X_normalizado` se usa en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente (`X_normalizado.T @ gradiente_z`). `X` se conserva para mostrar los valores originales.

## Ejecución

```bash
uv sync
uv run main.py
```

Requiere Python >= 3.12 y NumPy. Los resultados son reproducibles: los pesos iniciales usan la semilla 42.

## Resultado del entrenamiento base (tasa 0.5, 10000 épocas)

- Peso de la humedad: **-33.2571**
- Peso de la temperatura: **0.1955**
- Sesgo: **14.8320**
- Error cuadrático medio final: **0.005361**

| Caso | Humedad | Temp. | Probabilidad | Respuesta | Esperada |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0.0000 | 0 | 0 |
| 2 | 70 % | 22 °C | 0.0002 | 0 | 0 |
| 3 | 65 % | 28 °C | 0.0013 | 0 | 0 |
| 4 | 55 % | 25 °C | 0.0335 | 0 | 0 |
| 5 | 50 % | 32 °C | 0.1582 | 0 | 0 |
| 6 | 40 % | 30 °C | 0.8384 | 1 | 1 |
| 7 | 35 % | 25 °C | 0.9641 | 1 | 1 |
| 8 | 30 % | 32 °C | 0.9932 | 1 | 1 |
| 9 | 20 % | 35 °C | 0.9998 | 1 | 1 |
| 10 | 10 % | 38 °C | 1.0000 | 1 | 1 |

**Signo de los pesos:** el peso de la humedad es negativo: a más humedad, menor `z` y menor probabilidad de regar. El peso de la temperatura es positivo (y pequeño): a más calor, algo más de probabilidad de regar.

## Predicciones con condiciones nuevas (evidencia)

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0000 | 0 (no regar) |
| 45 % | 34 °C | 0.4998 | 0 (no regar) |
| 25 % | 22 °C | 0.9986 | 1 (regar) |
| 50 % | 25 °C | 0.1546 | 0 (no regar) |
| 30 % | 40 °C | 0.9934 | 1 (regar) |

El caso 45 % / 34 °C está prácticamente en la frontera de decisión (0.4998): la neurona está "indecisa" y la respuesta depende del umbral.

## Experimentos con los parámetros

Solo se cambia el parámetro indicado (misma semilla y mismos datos).

| Prueba | Épocas | Tasa | Error final (MSE) | Correctas /10 | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Base | 10000 | 0.5 | 0.005361 | 10 | 0.4998 | Rápido |
| Pocas épocas | 100 | 0.5 | 0.089490 | 10 | 0.5451 | Lento / incompleto |
| Intermedia | 1000 | 0.5 | 0.030038 | 10 | 0.5603 | Lento |
| Más épocas | 20000 | 0.5 | 0.002795 | 10 | 0.4856 | Rápido |
| Tasa pequeña | 10000 | 0.01 | 0.066192 | 10 | 0.5621 | Lento |
| Tasa moderada | 10000 | 0.1 | 0.019328 | 10 | 0.5411 | Lento |
| Tasa alta | 10000 | 1.0 | 0.002795 | 10 | 0.4857 | Rápido |
| Tasa muy alta | 10000 | 2.0 | 0.001396 | 10 | 0.4713 | Rápido |


## Umbral (sin reentrenar)

Con umbrales 0.4, 0.5 y 0.6, los 10 casos de entrenamiento y 4 de los 5 casos nuevos no cambian. **Solo cambia 45 % / 34 °C** (probabilidad 0.4998): con 0.4 se decide regar (1) y con 0.5 y 0.6 no regar (0). Cambiar el umbral solo modifica cómo se interpreta la probabilidad ya calculada; los pesos y el sesgo salen del entrenamiento y no se tocan.

## Análisis

1. **Normalizar:** humedad (hasta 100) y temperatura (hasta 50) tienen escalas distintas. Sin normalizar, `z` sería enorme, la sigmoide se saturaría (gradientes casi cero) y la variable con números más grandes dominaría el aprendizaje.
2. **Uso de `X_normalizado` y `X`:** `X_normalizado` se usa en la suma ponderada y en el gradiente de los pesos. `X` se conserva para mostrar los valores originales (% y °C) y poder interpretar los resultados.
3. **100 épocas:** clasificó bien los 10 casos, pero el error seguía alto (0.0895) y las probabilidades eran poco seguras; la neurona no había terminado de aprender (pesos aún lejos de los finales).
4. **¿Más épocas siempre mejoran?** No de forma importante. Pasar de 100 a 1000 mejoró mucho (0.0895 → 0.0300), de 1000 a 10000 bastante (→ 0.0054), pero de 10000 a 20000 solo bajó 0.0054 → 0.0028. Las 10 respuestas ya eran correctas desde 100 épocas: el error sigue bajando, pero con rendimientos decrecientes.
5. **Tasa pequeña (0.01):** aprendizaje lento; tras 10000 épocas el error seguía en 0.066, peor que la prueba base con solo 1000 épocas.
6. **Tasa alta (1.0) y muy alta (2.0):** aprendieron más rápido y llegaron a menor error (0.0028 y 0.0014). Con estos datos y esta red tan pequeña no hubo inestabilidad. Con otros datos o más capas, una tasa así podría hacer oscilar o divergir el error. Además, 1.0 con 10000 épocas dio casi lo mismo que 0.5 con 20000: lo que importa es aproximadamente tasa × épocas. También hay un costo: los pesos crecen más (humedad ≈ -47), lo que vuelve la neurona más "tajante".
7. **Signo del peso de la humedad (negativo):** más humedad → menos necesidad de riego. Coincide con la lógica del problema.
8. **Signo del peso de la temperatura:** en la prueba base es positivo (+0.20), coherente con "más calor, más riego", pero es muy pequeño y **no es estable**: con 20000 épocas o tasas altas pasa a ser negativo (-0.70, -1.66). Los datos casi se separan solo con la humedad, así que la temperatura aporta poca información y su signo no es confiable con solo 10 ejemplos.
9. **Umbral:** la sigmoide da una probabilidad continua entre 0 y 1, pero la decisión de riego es binaria (se riega o no). El umbral (0.5 por defecto) convierte una en otra y puede ajustarse según el costo de regar de más o de menos.
10. **Limitaciones:** solo 10 datos didácticos; solo dos variables (no considera tipo de planta, tipo de suelo, lluvia, hora, viento, etapa de crecimiento); una neurona solo separa con una línea recta; sin validación con datos de prueba reales; el 45 %/34 °C muestra que cerca de la frontera la respuesta no es confiable; y los datos no vienen de un criterio agronómico real.
