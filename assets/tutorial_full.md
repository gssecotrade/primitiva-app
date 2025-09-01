
## `assets/tutorial_full.md`
```md
# 📘 Tutorial completo

## 1) Filosofía del modelo
Buscamos **maximizar probabilidad relativa** dentro de un juego esencialmente aleatorio, evitando sesgos obvios:
- **Ventana reciente** (24 sorteos) para evitar “memorias largas” irrelevantes.
- **Decay temporal** (half-life = 60 días).
- **Mezcla por día** (para Primitiva) por posibles sesgos de operación.
- **Penalización de popularidad** (fechas, secuencias, décadas/unidades concentradas, sumas “redondas”).
- **Diversidad** entre A2 para no “canibalizar” probabilidad.

## 2) Parámetros clave (ajustables en código)
- `WINDOW_DRAWS = 24` — histórico activo.
- `HALF_LIFE_DAYS = 60` — decaimiento.
- `DAY_BLEND_ALPHA = 0.30` — mezcla día/global.
- `ALPHA_DIR = 0.30` — suavizado Dirichlet.
- `MIN_DIV = 0.60` — solape máximo con A1.
- `LAMBDA_DIVERSIDAD = 0.60` — penalización entre A2.
- `THRESH_N` — tabla z→n (nº de A2/boletos sugeridos según señal y volatilidad).

## 3) Flujo de uso recomendado
1. **Introduce** el último sorteo del día (Primitiva: Lun/Jue/Sáb; Bonoloto: diario).
2. **Valida** que el histórico se actualiza (ver tabla “Últimos sorteos cargados”).
3. **Ejecuta** cálculo y revisa métricas:
   - Señal z (intensidad).
   - Prob. “teórica proxy” relativa por combinación (no absoluta).
   - Diversidad/solape entre A2.
4. **Apuesta escalonada**:
   - **Volatilidad Low**: 1–2 A2 si z baja.
   - **Medium**: 2–4 A2 si z media.
   - **High**: 4–6 A2 cuando z alta (y bank lo permite).

## 4) Métricas mostradas
- **z-score** sobre pesos dinámicos (mayor = mejor señal).
- **Score** = suma de logs de pesos + penalización de popularidad.
- **Solape** de cada A2 con A1/A2 previas (queremos bajo).
- **Proxy p(≥k aciertos)** con aproximación hipergeométrica (orientativa, no prob. real de la lotería completa; sirve para comparar combinaciones entre sí).

## 5) Preguntas típicas
- **¿Por qué cambia A1 por día?** A1 es ancla fija calibrada por día para evitar sesgos de operación.
- **¿Se guarda todo lo que tecleo?** Solo si marcas “Guardar en histórico (si es nuevo)”.
- **¿Por qué no siempre recomienda Joker?** Se activa cuando la señal supera umbral y tu banco lo permite.
