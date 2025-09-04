# app.py — Recomendador Primitiva & Bonoloto (UX por tarjetas)
# -----------------------------------------------------------
# Auto-contenido para pruebas: no usa Google Sheets ni archivos externos.
# Diseño móvil-first: apuesta óptima + tarjetas por A2 (selección, k, Joker).
# -----------------------------------------------------------

from math import comb as _comb
from datetime import datetime
import numpy as np
import streamlit as st

# =============== Utilidades de probabilidad y formato ===============

def prob_base_k(k: int) -> float:
    """Probabilidad de acertar 6 con una apuesta de tamaño k (sin señal)."""
    return _comb(k, 6) / _comb(49, 6)

def one_in_x(p: float) -> str:
    """Devuelve '1 entre X' con separador de miles. Si p=0 → ∞."""
    if p <= 0:
        return "∞"
    x = int(round(1.0 / p))
    return f"{x:,}".replace(",", ".")

def lift_of_combo(combo, weights: dict) -> float:
    """Lift = media(pesos en combo) / media(pesos globales). Recorta a [0.5, 3]."""
    allW = np.array([weights.get(i, 0.0) for i in range(1, 50)], dtype=float)
    den = float(allW.mean()) if allW.mean() > 0 else 1.0
    num = float(np.mean([weights.get(n, 0.0) for n in combo])) if combo else 0.0
    lift = (num / den) if den > 0 else 1.0
    # recortes conservadores para evitar outliers
    lift = float(np.clip(lift, 0.5, 3.0))
    return round(lift, 2)

def scale_01(values: np.ndarray) -> np.ndarray:
    """Escala linealmente a [0,1]. Si todo igual → 0.5."""
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin <= 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - vmin) / (vmax - vmin)

# =============== Generación determinista de pesos y A2 ===============

def make_weights_from_last_draw(last_nums: list[int], volatility: str = "Medium") -> dict:
    """
    Genera pesos simples a partir del último sorteo:
    - sube el peso de los números cercanos a los extraídos (ventana +/-1)
    - añade una pequeña forma de campana para dar estabilidad
    - controla 'amplitud' con la volatilidad
    """
    vol_map = {"Low": 0.05, "Medium": 0.12, "High": 0.22}
    amp = vol_map.get(volatility, 0.12)
    xs = np.arange(1, 50, dtype=float)

    # Campana centrada en 25
    center_bump = np.exp(-0.5 * ((xs - 25.0) / 10.0) ** 2)

    bump = np.zeros_like(xs)
    for n in last_nums:
        bump += np.exp(-0.5 * ((xs - float(n)) / 2.0) ** 2)  # ventana estrecha

    raw = 0.6 * center_bump + 0.4 * bump
    raw = raw / (raw.mean() + 1e-12)

    # Añadimos ruido determinista suave (no aleatorio → estable)
    # función periódica suave de índice para diversificar ligeramente
    idx_noise = 0.15 * np.sin(xs / 7.0) + 0.10 * np.cos(xs / 11.0)
    raw = raw * (1.0 + amp * idx_noise)

    # Escalamos a media 1.0
    raw = raw / (raw.mean() + 1e-12)

    weights = {int(i): float(raw[i-1]) for i in range(1, 50)}
    return weights

def greedy_top_k(weights: dict, k: int, banned: set[int]) -> list[int]:
    """Selecciona k números con mayor peso evitando 'banned'."""
    nums = [n for n, w in sorted(weights.items(), key=lambda t: (-t[1], t[0])) if n not in banned]
    return nums[:k]

def diversify(prev: list[int], weights: dict, pool: list[int], k: int) -> list[int]:
    """Devuelve k números tratando de no solaparse demasiado con 'prev'."""
    prev_set = set(prev)
    # penaliza pesos de los ya elegidos para diversificar
    penalized = []
    for n in pool:
        w = weights.get(n, 0.0)
        if n in prev_set:
            w *= 0.80  # penalización
        penalized.append((n, w))
    penalized.sort(key=lambda t: (-t[1], t[0]))
    return [n for (n, _) in penalized[:k]]

def build_A2s(weights: dict, how_many: int = 3) -> list[list[int]]:
    """
    Construye hasta 3 combinaciones A2 (listas de 6) de forma determinista:
    - A2#1 = top-6 por peso.
    - A2#2 y #3 = buscan diversidad con respecto a las anteriores.
    """
    all_nums = [n for n, _ in sorted(weights.items(), key=lambda t: (-t[1], t[0]))]
    if len(all_nums) < 6:
        return []

    a2_list = []
    a2_1 = all_nums[:6]
    a2_list.append(sorted(a2_1))

    if how_many >= 2:
        cand_pool = all_nums[:18]  # pool alto para variedad
        a2_2 = diversify(a2_1, weights, cand_pool, 6)
        a2_list.append(sorted(a2_2))

    if how_many >= 3:
        # diversifica contra #1 y #2
        used = set(a2_1) | set(a2_list[1])
        cand_pool2 = [n for n in all_nums if n not in used][:24]
        a2_3 = diversify(a2_list[1], weights, cand_pool2 or all_nums, 6)
        a2_list.append(sorted(a2_3))

    # Unicidad
    uniq = []
    seen = set()
    for a in a2_list:
        t = tuple(a)
        if t not in seen:
            seen.add(t)
            uniq.append(a)
    return uniq[:how_many]

def joker_score_for_a2(a2: list[int], weights: dict) -> float:
    """
    ScoreJ ∈ [0,1] aproximado: normaliza la media de pesos del combo
    contra la distribución global.
    """
    allW = np.array([weights[i] for i in range(1, 50)], dtype=float)
    z = (np.mean([weights[n] for n in a2]) - allW.mean()) / (allW.std() + 1e-9)
    # squash a [0,1] con sigmoide
    s = 1.0 / (1.0 + np.exp(-z))
    return float(round(s, 2))

# =============== Componente UI: Builder de ticket en tarjetas ===============

def render_ticket_builder(
    juego: str,                      # "Primitiva" | "Bonoloto"
    a2_list: list,                   # [ [6 ints], ... ]
    weights: dict,
    price_simple: float,
    price_joker: float,
    suggest_joker_scores: list | None = None,
    joker_umbral: float = 0.65,
    default_k: int = 6,
):
    """
    Pinta tarjetas (una por A2) y devuelve un dict con selección y totales.
    """
    st.subheader("🎫 Ajusta tu ticket", anchor=False)
    if not a2_list:
        st.info("No hay A2 recomendadas para este sorteo.")
        return {"seleccion": [], "totales": {"n": 0, "coste": 0.0, "p_sum": 0.0}}

    ss_key = f"builder_{juego.lower()}_{datetime.now().date()}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = {}

    seleccion = []
    total_cost = 0.0
    total_p = 0.0

    for idx, a2 in enumerate(a2_list):
        lift = lift_of_combo(a2, weights)
        scJ = None
        suggest_j = False
        if juego == "Primitiva" and suggest_joker_scores and idx < len(suggest_joker_scores):
            scJ = suggest_joker_scores[idx]
            suggest_j = (scJ >= joker_umbral)

        row_key = f"{ss_key}_row{idx}"
        state = st.session_state[ss_key].get(row_key, {
            "use": (idx == 0),
            "k": default_k,
            "joker": suggest_j,
        })

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**A2 #{idx+1}** · {sorted(a2)}")
                info = [f"Lift ×{lift:.2f}"]
                if scJ is not None:
                    info.append(f"ScoreJ {scJ:.2f}")
                st.caption(" · ".join(info))
            with c2:
                state["use"] = st.toggle("Incluir", value=state["use"], key=f"{row_key}_use")

            cc1, cc2, cc3, cc4 = st.columns([1, 1, 2, 2])
            with cc1:
                state["k"] = st.selectbox("k", [6, 7, 8], index=[6, 7, 8].index(state["k"]), key=f"{row_key}_k")
            with cc2:
                if juego == "Primitiva":
                    state["joker"] = st.toggle("Joker", value=state["joker"], key=f"{row_key}_jk")
                else:
                    st.caption("Joker")
                    st.write("—")
                    state["joker"] = False

            k = int(state["k"])
            p_base = prob_base_k(k)
            p_adj = p_base * lift
            coste = _comb(k, 6) * float(price_simple) + (float(price_joker) if state["joker"] else 0.0)

            with cc3:
                st.metric("Prob. base", f"1 entre {one_in_x(p_base)}")
                st.caption("Solo por tamaño k.")
            with cc4:
                st.metric("Prob. ajustada", f"1 entre {one_in_x(p_adj)}")
                st.caption("Con Lift de esta A2.")

            if state["use"]:
                total_cost += coste
                total_p += p_adj
                seleccion.append({
                    "a2": list(sorted(a2)),
                    "k": k,
                    "joker": bool(state["joker"]),
                    "lift": float(lift),
                    "p_base": float(p_base),
                    "p_adj": float(p_adj),
                    "coste": float(coste),
                })

        st.session_state[ss_key][row_key] = state

    st.markdown("---")
    cA, cB, cC = st.columns([1, 1, 2])
    cA.metric("Apuestas", len(seleccion))
    cB.metric("Coste total (€)", f"{total_cost:,.2f}".replace(",", "."))
    cC.metric("Prob. ajustada total", f"1 entre {one_in_x(total_p)}")
    st.caption("La probabilidad total se aproxima como la **suma** de las probabilidades ajustadas de cada apuesta seleccionada.")

    return {
        "seleccion": seleccion,
        "totales": {"n": len(seleccion), "coste": float(total_cost), "p_sum": float(total_p)},
    }

# =============== Página principal ===============

st.set_page_config(
    page_title="Recomendador Primitiva & Bonoloto",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Recomendador Primitiva & Bonoloto")
st.caption("Optimización determinista · Builder por tarjetas · Lift ×N y probabilidad ajustada")

tabs = st.tabs(["La Primitiva", "Bonoloto", "📘 Tutorial"])

# -------- Datos “último sorteo” (ejemplo) --------
LAST_PRIMITIVA = [8, 9, 15, 29, 39, 46]
LAST_BONOLOTO = [5, 9, 21, 26, 36, 49]

# ======================= PRIMITIVA =======================
with tabs[0]:
    st.subheader("La Primitiva · Ticket Óptimo (EV/€)")

    c0, _ = st.columns([2, 1])
    with c0:
        mode = st.radio("Origen de datos del último sorteo", ["Usar último del histórico", "Introducir manualmente"], horizontal=True)
        if mode == "Usar último del histórico":
            nums = LAST_PRIMITIVA[:]
            st.info(f"Usando el último sorteo (demo): {sorted(nums)}", icon="ℹ️")
        else:
            cols = st.columns(6)
            nums = []
            for i, c in enumerate(cols, start=1):
                nums.append(c.number_input(f"N{i}", min_value=1, max_value=49, value=LAST_PRIMITIVA[i-1], step=1))
            nums = sorted(set(int(x) for x in nums))[:6]
            if len(nums) < 6:
                st.warning("Introduce 6 números distintos (1–49).")

    left, right = st.columns([1, 1])
    with left:
        price_simple = st.number_input("Precio por apuesta simple (€)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
        volatility = st.selectbox("Volatilidad", ["Low", "Medium", "High"], index=1)
    with right:
        use_joker = st.checkbox("Activar recomendaciones de Joker por apuesta", value=True)
        joker_thr = st.slider("Umbral para recomendar Joker", 0.0, 1.0, 0.65, 0.01)
        price_joker = st.number_input("Precio Joker (€)", min_value=0.0, max_value=2.0, value=1.0, step=0.5)

    if st.button("Calcular · Primitiva", type="primary"):
        st.session_state["calc_p"] = True
    if not st.session_state.get("calc_p"):
        st.stop()

    # Pesos y A2
    w = make_weights_from_last_draw(nums, volatility=volatility)
    A2s = build_A2s(w, how_many=3)

    # Apuesta óptima
    if A2s:
        best = A2s[0]
        lift_best = lift_of_combo(best, w)
        st.markdown("### 🏆 Apuesta Óptima (EV/€)")
        st.markdown(f"**{sorted(best)}** · `Lift ×{lift_best:.2f}`")
        p_base = prob_base_k(6)
        p_adj = p_base * lift_best
        st.caption(f"Prob. base (k=6): **1 entre {one_in_x(p_base)}** · Prob. ajustada: **1 entre {one_in_x(p_adj)}**")

    # ScoreJ sugerido
    scoreJ_list = [joker_score_for_a2(a, w) for a in A2s] if use_joker else None

    # Builder
    result = render_ticket_builder(
        juego="Primitiva",
        a2_list=A2s,
        weights=w,
        price_simple=float(price_simple),
        price_joker=float(price_joker),
        suggest_joker_scores=scoreJ_list,
        joker_umbral=float(joker_thr),
        default_k=6,
    )

    # Bitácora (simulada)
    if result["totales"]["n"] > 0 and st.toggle("Guardar este ticket en bitácora (demo)"):
        st.success("✅ Ticket guardado (demo).")

# ======================= BONOLOTO =======================
with tabs[1]:
    st.subheader("Bonoloto · Ticket Óptimo (EV/€)")

    c0, _ = st.columns([2, 1])
    with c0:
        modeb = st.radio("Origen de datos del último sorteo", ["Usar último del histórico", "Introducir manualmente"], horizontal=True, key="modeB")
        if modeb == "Usar último del histórico":
            nums_b = LAST_BONOLOTO[:]
            st.info(f"Usando el último sorteo (demo): {sorted(nums_b)}", icon="ℹ️")
        else:
            cols = st.columns(6)
            nums_b = []
            default_b = LAST_BONOLOTO[:]
            for i, c in enumerate(cols, start=1):
                nums_b.append(c.number_input(f"N{i} (Bono)", min_value=1, max_value=49, value=default_b[i-1], step=1, key=f"nb_{i}"))
            nums_b = sorted(set(int(x) for x in nums_b))[:6]
            if len(nums_b) < 6:
                st.warning("Introduce 6 números distintos (1–49).")

    left, right = st.columns([1, 1])
    with left:
        price_simple_b = st.number_input("Precio simple Bonoloto (€)", min_value=0.5, max_value=5.0, value=0.5, step=0.5)
        volatility_b = st.selectbox("Volatilidad · Bonoloto", ["Low", "Medium", "High"], index=1, key="volB")
    with right:
        st.caption("Bonoloto no tiene Joker.")
        price_joker_b = 0.0

    if st.button("Calcular · Bonoloto", type="primary", key="calcB"):
        st.session_state["calc_b"] = True
    if not st.session_state.get("calc_b"):
        st.stop()

    w_b = make_weights_from_last_draw(nums_b, volatility=volatility_b)
    A2s_b = build_A2s(w_b, how_many=3)

    if A2s_b:
        best_b = A2s_b[0]
        lift_best_b = lift_of_combo(best_b, w_b)
        st.markdown("### 🏆 Apuesta Óptima (EV/€)")
        st.markdown(f"**{sorted(best_b)}** · `Lift ×{lift_best_b:.2f}`")
        p_base_b = prob_base_k(6)
        p_adj_b = p_base_b * lift_best_b
        st.caption(f"Prob. base (k=6): **1 entre {one_in_x(p_base_b)}** · Prob. ajustada: **1 entre {one_in_x(p_adj_b)}**")

    result_b = render_ticket_builder(
        juego="Bonoloto",
        a2_list=A2s_b,
        weights=w_b,
        price_simple=float(price_simple_b),
        price_joker=float(price_joker_b),
        suggest_joker_scores=None,
        joker_umbral=0.0,
        default_k=6,
    )

    if result_b["totales"]["n"] > 0 and st.toggle("Guardar este ticket en bitácora (demo)", key="logB"):
        st.success("✅ Ticket guardado (demo).")

# ======================= TUTORIAL =======================
with tabs[2]:
    st.markdown("## 📘 Tutorial rápido")
    st.markdown("""
**Objetivo**  
Te damos una **apuesta óptima** y un **constructor de ticket** sencillo: eliges qué A2 jugar, con qué **k** (6/7/8) y si añades **Joker** (solo Primitiva).  
Verás siempre:
- **Lift ×N**: calidad relativa del combo frente al azar (×1.70 = 70% mejor que aleatorio en señal).
- **Prob. base** (solo por k) y **Prob. ajustada** (k × Lift).
- **Coste** por apuesta y **total** abajo.
- **Prob. ajustada total** ≈ suma de probabilidades ajustadas de tus apuestas seleccionadas.

**¿k ayuda a la eficiencia?**  
No. A igualdad de precio lineal por combinación, **la probabilidad por euro es constante**; subir k **aumenta varianza y coste**, no la eficiencia. Por eso el **k recomendado por defecto es 6**.

**¿Y el Joker?**  
Decisión aparte (1€). Te lo sugerimos cuando el **ScoreJ ≥ umbral**. Sube la probabilidad de “algo bueno” (otra vía de premio) pero no cambia la probabilidad de acertar 6.

**Estrategia práctica**  
1) Juega la **A2 #1** (la de mayor Lift).  
2) Si quieres más cobertura y tu bank lo permite, añade **A2 #2 y/o #3**. La probabilidad total crece **linealmente** con el coste.  
3) Activa **Joker** solo si se recomienda o lo prefieres.

**Aviso**: La lotería es aleatoria por naturaleza; estas métricas son orientativas.
    """)
