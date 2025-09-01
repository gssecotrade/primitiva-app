# app.py — Primitiva & Bonoloto · Recomendador A2 (UX + métricas + simulador)
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from math import comb
import json
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ==========================
# Configuración base de la app
# ==========================
st.set_page_config(
    page_title="Primitiva & Bonoloto · Recomendador A2",
    page_icon="🎯",
    layout="wide"
)
# Cargar estilos personalizados
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🎯 Primitiva & Bonoloto · Recomendador A2 (n dinámico)")
st.caption(
    "Ventana 24 sorteos · t½=60d · mezcla por día (30%) · antipopularidad · diversidad · antiduplicados · fuente: Google Sheets (live)"
)

# ==========================
# Parámetros del modelo (ajustables)
# ==========================
WINDOW_DRAWS    = 24           # nº sorteos recientes usados
HALF_LIFE_DAYS  = 60.0         # semivida para pesos temporales
DAY_BLEND_ALPHA = 0.30         # mezcla pesos global/día
ALPHA_DIR       = 0.30         # suavizado dirichlet
MU_PENALTY      = 1.00         # penalización "popularidad"
K_CANDIDATOS    = 3000         # candidatos a generar
MIN_DIV         = 0.60         # diversidad (1-overlap)
LAMBDA_DIVERSIDAD = 0.60       # fuerza de diversidad en greedy

# mapa z -> nº apuestas (editando umbrales de señal)
THRESH_N = [
    {"z": 0.50, "n": 6},
    {"z": 0.35, "n": 4},
    {"z": 0.20, "n": 3},
    {"z": 0.10, "n": 2},
    {"z":-999,  "n": 1},
]

# A1 fijas por día (Primitiva)
A1_FIJAS_PRIMI = {
    "Monday":    [4, 24, 35, 37, 40, 46],
    "Thursday":  [1, 10, 23, 39, 45, 48],
    "Saturday":  [7, 12, 14, 25, 29, 40],
}
REIN_FIJOS_PRIMI = {"Monday":1, "Thursday":8, "Saturday":0}

# A1 neutras por día (Bonoloto) — se calibran tras 8–12 semanas si quieres
A1_FIJAS_BONO = {
    0: [4,24,35,37,40,46],  # Mon
    1: [4,24,35,37,40,46],  # Tue
    2: [4,24,35,37,40,46],  # Wed
    3: [4,24,35,37,40,46],  # Thu
    4: [4,24,35,37,40,46],  # Fri
    5: [4,24,35,37,40,46],  # Sat
    6: [4,24,35,37,40,46],  # Sun
}

# ==========================
# Credenciales de Google (robusto)
# ==========================
def _creds_from_gcp_block():
    info = dict(st.secrets["gcp_service_account"])
    pk = info.get("private_key", "")
    if isinstance(pk, str) and "\\n" in pk:
        info["private_key"] = pk.replace("\\n", "\n")
    info["private_key"] = info["private_key"].strip()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",   # escritura/lectura
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    return Credentials.from_service_account_info(info, scopes=scopes)

def _creds_from_gcp_json():
    raw = st.secrets["gcp_json"]
    # Permite pegar JSON tal cual o en triple-comillas
    js = raw.strip()
    data = json.loads(js)
    pk = data.get("private_key", "")
    if "\\n" in pk:
        data["private_key"] = pk.replace("\\n", "\n")
    data["private_key"] = data["private_key"].strip()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    return Credentials.from_service_account_info(data, scopes=scopes)

def get_gcp_credentials():
    if "gcp_service_account" in st.secrets:
        return _creds_from_gcp_block()
    if "gcp_json" in st.secrets:
        return _creds_from_gcp_json()
    st.error("Falta el bloque [gcp_service_account] o la clave gcp_json en **Settings → Secrets**.")
    st.stop()

def get_secret_key(name, group="gcp_service_account"):
    try:
        if name in st.secrets:
            return st.secrets[name]
        if group in st.secrets and name in st.secrets[group]:
            return st.secrets[group][name]
    except Exception:
        pass
    return None

# ==========================
# Lectura / escritura Google Sheets
# ==========================
def open_ws(sheet_id, worksheet_name):
    creds = get_gcp_credentials()
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    return ws

@st.cache_data(ttl=600, show_spinner=True)
def load_sheet_df_generic(sheet_id_key: str, worksheet_key: str, default_ws: str):
    sid = get_secret_key(sheet_id_key)
    wsn = get_secret_key(worksheet_key) or default_ws
    if not sid:
        st.error(f"No encuentro `{sheet_id_key}` en Secrets. Añade el ID de la hoja.")
        return pd.DataFrame()
    try:
        ws = open_ws(sid, wsn)
    except Exception as e:
        st.error(f"No puedo abrir el Sheet/Worksheet ({sheet_id_key}/{worksheet_key}). Detalle: {e}")
        return pd.DataFrame()

    rows = ws.get_all_records(numericise_ignore=["FECHA"])
    df = pd.DataFrame(rows)
    expected = ["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en la pestaña '{wsn}': {missing}")
        return pd.DataFrame(columns=expected)

    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
    for c in ["N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def append_row_if_new(sheet_id_key, worksheet_key, dt, nums, comp, rein):
    """Añade fila si no existe ese sorteo exacto (fecha+numeración). Devuelve True si escribió."""
    try:
        sid = get_secret_key(sheet_id_key)
        wsn = get_secret_key(worksheet_key)
        if not sid or not wsn: 
            return False
        ws = open_ws(sid, wsn)
        df = load_sheet_df_generic.cache_clear() or None  # aseguramos limpieza
        # Relee fresco
        df = load_sheet_df_generic(sheet_id_key, worksheet_key, wsn)
        if df.empty:
            # Añadimos cabeceras implícitamente con append_row
            pass
        same = df["FECHA"].dt.date == pd.to_datetime(dt).date()
        if same.any():
            row = df.loc[same].tail(1)
            try:
                match = (
                    int(row["N1"].values[0])==nums[0] and int(row["N2"].values[0])==nums[1] and
                    int(row["N3"].values[0])==nums[2] and int(row["N4"].values[0])==nums[3] and
                    int(row["N5"].values[0])==nums[4] and int(row["N6"].values[0])==nums[5] and
                    int(row["Complementario"].values[0])==comp and int(row["Reintegro"].values[0])==rein
                )
            except Exception:
                match = False
            if match:
                return False  # ya está
        ws.append_row([
            pd.to_datetime(dt).strftime("%d/%m/%Y"),
            nums[0], nums[1], nums[2], nums[3], nums[4], nums[5],
            comp, rein
        ])
        # invalidar caché de lectura
        load_sheet_df_generic.clear()
        return True
    except Exception:
        return False

# Wrappers cacheados para cada juego
@st.cache_data(ttl=600, show_spinner=True)
def load_sheet_df_primi():
    return load_sheet_df_generic("sheet_id", "worksheet_historico", "Historico")

@st.cache_data(ttl=600, show_spinner=True)
def load_sheet_df_bono():
    return load_sheet_df_generic("sheet_id_bono", "worksheet_historico_bono", "HistoricoBono")


# ==========================
# Utilidades de modelo
# ==========================
def time_weight(d, ref):
    delta = max(0, (ref - d).days)
    return float(np.exp(-np.log(2)/HALF_LIFE_DAYS * delta))

def weighted_counts_nums(df_in, ref):
    w = {i:0.0 for i in range(1,50)}
    for _, r in df_in.iterrows():
        tw = time_weight(r["FECHA"], ref)
        for c in ["N1","N2","N3","N4","N5","N6"]:
            if not pd.isna(r[c]):
                w[int(r[c])] += tw
    return w

def weighted_counts_rei(df_in, ref):
    w = {i:0.0 for i in range(10)}
    if "Reintegro" in df_in.columns:
        for _, r in df_in.dropna(subset=["Reintegro"]).iterrows():
            tw = time_weight(r["FECHA"], ref)
            w[int(r["Reintegro"])] += tw
    return w

def blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA):
    return {n: alpha*w_day.get(n,0.0) + (1-alpha)*w_glob.get(n,0.0) for n in range(1,50)}

def popularity_penalty(combo):
    c = sorted(combo)
    p_dates = sum(1 for x in c if x<=31)/6.0
    consec  = sum(1 for a,b in zip(c, c[1:]) if b==a+1)
    decades = [x//10 for x in c]; units = [x%10 for x in c]
    max_dec = max(Counter(decades).values()); max_unit = max(Counter(units).values())
    s = sum(c); roundness = 1.0/(1.0 + abs(s-120)/10.0)
    return 1.2*p_dates + 0.8*consec + 0.5*(max_dec-2 if max_dec>2 else 0) + 0.5*(max_unit-2 if max_unit>2 else 0) + 0.4*roundness

def score_combo(combo, weights):
    return sum(np.log(weights.get(n,0.0) + ALPHA_DIR) for n in combo) - MU_PENALTY*popularity_penalty(combo)

def terciles_ok(combo):
    return any(1<=x<=16 for x in combo) and any(17<=x<=32 for x in combo) and any(33<=x<=49 for x in combo)

def random_combo():
    pool = list(range(1,50)); out=[]
    while len(out)<6:
        i=np.random.randint(0,len(pool)); out.append(pool.pop(i))
    return sorted(out)

def overlap_ratio(a,b): 
    return len(set(a)&set(b))/6.0

def zscore_combo(combo, weights):
    allW = np.array([weights.get(i,0.0) for i in range(1,50)], dtype=float)
    meanW = float(allW.mean()); sdW = float(allW.std()) if allW.std()!=0 else 1e-6
    comboMean = float(np.mean([weights.get(n,0.0) for n in combo])) if combo else 0.0
    return (comboMean - meanW)/sdW

def pick_n(z, bank, vol):
    adj = 0.05 if vol=="Low" else -0.05 if vol=="High" else 0.0
    for th in THRESH_N:
        if z >= th["z"] + adj:
            n = min(th["n"], int(bank))
            return max(1, n)
    return 1

def greedy_select(pool, weights, n):
    if n<=0: return []
    sorted_pool = sorted(pool, key=lambda c: score_combo(c,weights), reverse=True)
    selected = [sorted_pool[0]]
    while len(selected)<n and len(sorted_pool)>1:
        bestC=None; bestVal=-1e9
        for c in sorted_pool[1:]:
            if any(tuple(c)==tuple(s) for s in selected): continue
            div_pen = sum(overlap_ratio(c,s) for s in selected)
            val = score_combo(c,weights) - LAMBDA_DIVERSIDAD*div_pen
            if val>bestVal: bestVal=val; bestC=c
        if bestC is None: break
        selected.append(bestC)
    return selected

def to_js_day(dayname):
    return 1 if dayname=="Monday" else 4 if dayname=="Thursday" else 6 if dayname=="Saturday" else -1

# Probabilidades exactas 6/49
def hypergeom_probs_6of49():
    total = comb(49, 6)
    probs = {}
    for k in range(0, 7):
        probs[k] = comb(6, k) * comb(43, 6-k) / total
    return probs

def relative_lift_from_z(z: float) -> float:
    lift = 1.0 + 0.25 * float(z)  # heurística acotada
    return float(max(0.5, min(2.0, lift)))

def softmax_scores(pool, weights):
    if not pool:
        return {}, []
    vals = np.array([score_combo(c, weights) for c in pool], dtype=float)
    vals = vals - vals.max()
    exps = np.exp(vals)
    probs = exps / exps.sum()
    mapping = {tuple(c): float(p) for c, p in zip(pool, probs)}
    order = sorted(pool, key=lambda c: mapping[tuple(c)], reverse=True)
    return mapping, order

def metrics_table(z, pool, weights):
    lift = relative_lift_from_z(z)
    base = hypergeom_probs_6of49()
    rel_map, order = softmax_scores(pool, weights)
    best = tuple(order[0]) if order else None
    rel_prob = rel_map.get(best, 0.0)

    rows = []
    rows.append({"Métrica": "Score relativo (softmax, top A2)", "Valor": f"{rel_prob:.3f}"})
    rows.append({"Métrica": "z (señal)", "Valor": f"{z:.3f}"})
    rows.append({"Métrica": "Lift relativo vs azar", "Valor": f"{lift:.2f}×"})
    rows.append({"Métrica": "—", "Valor": "—"})
    for k in range(6, -1, -1):
        base_p = base[k]
        adj_p = min(1.0, base_p * lift)
        rows.append({
            "Métrica": f"P(exactamente {k})",
            "Valor": f"{base_p*100:.6f}%  (ajustada: {adj_p*100:.6f}%)"
        })
    return pd.DataFrame(rows)

# Simulador “¿qué cambia si muevo la volatilidad?”
def simulate_volatility(z, bank):
    res = []
    for vol in ["Low","Medium","High"]:
        adj = 0.05 if vol=="Low" else -0.05 if vol=="High" else 0.0
        n = 1
        for th in THRESH_N:
            if z >= th["z"] + adj:
                n = min(th["n"], int(bank))
                n = max(1, n); break
        res.append({"Volatilidad": vol, "n recomendado": n})
    return pd.DataFrame(res)


# ==========================
# Pestañas de la app
# ==========================
tab_primi, tab_bono = st.tabs(["La Primitiva", "Bonoloto"])

# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader("La Primitiva · Recomendador A2")
    st.caption("A1 fija por día · A2 dinámica · Joker opcional")

    # ---- Sidebar simétrico
    with st.sidebar:
        st.markdown("### Primitiva · Parámetros")
        bank = st.number_input("Banco disponible (€)", min_value=0, value=10, step=1, key="bank_primi")
        vol  = st.selectbox("Volatilidad objetivo", ["Low","Medium","High"], index=1, key="vol_primi",
                            help="Low: conservador · Medium: estándar · High: agresivo")
        st.markdown("---")
        pool_size = st.slider("Tamaño del pool de candidatos", 500, 5000, 1000, 500,
                              help="Para debug/experimentos: cuántos candidatos pasan a ranking final.")

    # ---- Carga histórico
    df_hist = load_sheet_df_primi()
    if df_hist.empty:
        st.stop()

    # ---- Formulario
    with st.form("entrada_primi"):
        c1, c2 = st.columns(2)
        last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=pd.Timestamp.today().date())
        rein = c2.number_input("Reintegro", min_value=0, max_value=9, value=2, step=1)
        comp = c2.number_input("Complementario", min_value=1, max_value=49, value=18, step=1)
        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults = [5,6,8,23,46,47]
        nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"npr{i+1}") for i in range(6)]
        save_new = st.checkbox("Guardar en histórico (Primitiva) si es nuevo", value=True)
        do_calc = st.form_submit_button("Calcular recomendaciones · Primitiva")

    # ---- Cálculo
    if do_calc:
        if len(set(nums)) != 6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        last_dt = pd.to_datetime(last_date)
        wd = last_dt.weekday()  # 0=Mon..6=Sun
        if wd==0: next_dt, next_dayname = last_dt + pd.Timedelta(days=3), "Thursday"
        elif wd==3: next_dt, next_dayname = last_dt + pd.Timedelta(days=2), "Saturday"
        elif wd==5: next_dt, next_dayname = last_dt + pd.Timedelta(days=2), "Monday"
        else:
            st.error("La fecha debe ser Lunes, Jueves o Sábado.")
            st.stop()

        st.info(f"Próximo sorteo: **{next_dt.date().strftime('%d/%m/%Y')}** ({next_dayname})")

        base = df_hist[df_hist["FECHA"] <= last_dt].sort_values("FECHA").copy()

        # Antiduplicados + guardado opcional
        def has_duplicate_row(df, last_dt, nums, comp, rein):
            if df.empty: return False, False
            same_date = df["FECHA"].dt.date == last_dt.date()
            if not same_date.any(): return False, False
            row = df.loc[same_date].tail(1)
            try:
                match = (int(row["N1"].values[0])==nums[0] and int(row["N2"].values[0])==nums[1] and
                         int(row["N3"].values[0])==nums[2] and int(row["N4"].values[0])==nums[3] and
                         int(row["N5"].values[0])==nums[4] and int(row["N6"].values[0])==nums[5] and
                         int(row["Complementario"].values[0])==comp and int(row["Reintegro"].values[0])==rein)
            except Exception:
                match = False
            return True, match

        has_date, full_match = has_duplicate_row(base, last_dt, nums, comp, rein)

        if save_new:
            wrote = append_row_if_new("sheet_id", "worksheet_historico", last_dt, nums, comp, rein)
            if wrote:
                st.success("✅ Añadido al histórico (Primitiva).")
                df_hist = load_sheet_df_primi.clear() or None
                df_hist = load_sheet_df_primi()
                base = df_hist[df_hist["FECHA"] <= last_dt].sort_values("FECHA").copy()

        # Ventana de análisis
        df_recent = base.tail(WINDOW_DRAWS).copy()
        df_recent["weekday"] = df_recent["FECHA"].dt.weekday

        # Pesos y blend
        w_glob = weighted_counts_nums(df_recent, last_dt)
        w_day  = weighted_counts_nums(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        # A1
        A1 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])

        # Candidatos y pool
        cands, seen, tries = [], set(), 0
        while len(cands)<K_CANDIDATOS and tries < K_CANDIDATOS*50:
            c = tuple(random_combo()); tries += 1
            if c in seen: continue
            seen.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1) > (1 - MIN_DIV): continue
            cands.append(c)
        cands = sorted(cands, key=lambda c: score_combo(c, w_blend), reverse=True)
        pool = cands[:pool_size]

        # Señal y n
        bestA2 = list(pool[0]) if pool else []
        zA2 = zscore_combo(bestA2, w_blend) if bestA2 else 0.0
        n = pick_n(zA2, bank, vol); n = max(1, min(6, n))
        A2s = greedy_select(pool, w_blend, max(0, n-1))

        # Reintegro sugerido
        wr_glob = weighted_counts_rei(df_recent, last_dt)
        wr_day  = weighted_counts_rei(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
        rei_scores = {r: DAY_BLEND_ALPHA*wr_day.get(r,0.0) + (1-DAY_BLEND_ALPHA)*wr_glob.get(r,0.0) for r in range(10)}
        rein_sug = max(rei_scores, key=lambda r: rei_scores[r]) if rei_scores else 0

        joker = (zA2 >= 0.35) and (bank >= n+1) and (vol!="Low")

        st.subheader("Resultados · Primitiva")
        st.write(f"**A1 (fija)** {A1}  |  **n recomendado:** {n}")
        for i, c in enumerate(A2s, start=1):
            st.write(f"**A2 #{i}** {list(c)}")
        st.write(f"**Reintegro sugerido (informativo)**: {rein_sug}  ·  **Ref. día**: {REIN_FIJOS_PRIMI.get(next_dayname,'')}")
        st.write(f"**Joker recomendado**: {'Sí' if joker else 'No'}")

        # Salidas en pestañas
        rows = [{"Tipo":"A1", "N1":A1[0],"N2":A1[1],"N3":A1[2],"N4":A1[3],"N5":A1[4],"N6":A1[5]}]
        for i, c in enumerate(A2s, start=1):
            cl = list(c)
            rows.append({"Tipo":f"A2-{i}", "N1":cl[0],"N2":cl[1],"N3":cl[2],"N4":cl[3],"N5":cl[4],"N6":cl[5]})
        df_out = pd.DataFrame(rows)

        tab_res, tab_metrics, tab_sim = st.tabs(["📋 Combinaciones", "📊 Métricas & Probabilidad", "🧪 Simulador de volatilidad"])
        with tab_res:
            st.dataframe(df_out, use_container_width=True)
            st.download_button(
                "Descargar combinaciones · Primitiva (CSV)",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="primitiva_recomendaciones.csv",
                mime="text/csv"
            )
        with tab_metrics:
            st.markdown("**Indicadores del mejor A2** (respecto a un 6/49 aleatorio)")
            dfm = metrics_table(zA2, pool, w_blend)
            st.dataframe(dfm, use_container_width=True)
            with st.expander("¿Cómo interpretarlo?"):
                st.markdown(
                    "- *Score relativo (softmax)*: probabilidad relativa del top A2 dentro del **pool** candidato.\n"
                    "- *z (señal)*: cuán por encima de la media están los pesos de ese A2.\n"
                    "- *Lift*: heurística del **plus** del modelo vs. azar. Acotada 0.5×–2.0×.\n"
                    "- *P(k)*: probabilidad exacta 6/49 de acertar **k** números; la columna ajustada multiplica por el lift."
                )
        with tab_sim:
            st.markdown("**¿Qué pasa si cambio la volatilidad?**")
            st.dataframe(simulate_volatility(zA2, bank), use_container_width=True)

# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader("Bonoloto · Recomendador A2")
    st.caption("A1 ancla inicial por día · A2 dinámica · sin Joker")

    # ---- Sidebar simétrico
    with st.sidebar:
        st.markdown("### Bonoloto · Parámetros")
        bank_b = st.number_input("Banco (€) · Bonoloto", min_value=0, value=10, step=1, key="bank_bono")
        vol_b  = st.selectbox("Volatilidad · Bonoloto", ["Low","Medium","High"], index=1, key="vol_bono")
        st.markdown("---")
        pool_size_b = st.slider("Tamaño del pool (Bonoloto)", 500, 5000, 1000, 500)

    # ---- Histórico Bonoloto
    df_bono = load_sheet_df_bono()
    if df_bono.empty:
        st.stop()

    # ---- Formulario
    with st.form("entrada_bono"):
        c1, c2 = st.columns(2)
        last_date_b = c1.date_input("Fecha último sorteo (Bonoloto)", value=pd.Timestamp.today().date())
        rein_b = c2.number_input("Reintegro (0–9)", min_value=0, max_value=9, value=2, step=1)
        comp_b = c2.number_input("Complementario (1–49)", min_value=1, max_value=49, value=18, step=1)

        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults_b = [10,13,17,30,41,44]
        nums_b = [cols[i].number_input(f"N{i+1} (Bono)", 1, 49, defaults_b[i], 1, key=f"nbo{i+1}") for i in range(6)]

        save_new_b = st.checkbox("Guardar en histórico (Bonoloto) si es nuevo", value=True)
        do_calc_b = st.form_submit_button("Calcular recomendaciones · Bonoloto")

    if do_calc_b:
        if len(set(nums_b)) != 6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        last_dt_b = pd.to_datetime(last_date_b)
        weekday = last_dt_b.weekday()
        next_dt_b = last_dt_b + pd.Timedelta(days=1)
        next_dayname_b = next_dt_b.day_name()
        st.info(f"Próximo sorteo (aprox.): **{next_dt_b.date().strftime('%d/%m/%Y')}** ({next_dayname_b})")

        base_b = df_bono[df_bono["FECHA"] <= last_dt_b].sort_values("FECHA").copy()

        # Guardado opcional
        if save_new_b:
            wrote_b = append_row_if_new("sheet_id_bono", "worksheet_historico_bono", last_dt_b, nums_b, comp_b, rein_b)
            if wrote_b:
                st.success("✅ Añadido al histórico (Bonoloto).")
                df_bono = load_sheet_df_bono.clear() or None
                df_bono = load_sheet_df_bono()
                base_b = df_bono[df_bono["FECHA"] <= last_dt_b].sort_values("FECHA").copy()

        df_recent_b = base_b.tail(WINDOW_DRAWS).copy()
        df_recent_b["weekday"] = df_recent_b["FECHA"].dt.weekday

        w_glob_b = weighted_counts_nums(df_recent_b, last_dt_b)
        w_day_b  = weighted_counts_nums(df_recent_b[df_recent_b["weekday"]==weekday], last_dt_b)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        A1b = A1_FIJAS_BONO.get((weekday+1) % 7, [4,24,35,37,40,46])

        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*50:
            c = tuple(random_combo()); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1b) > (1 - MIN_DIV): continue
            cands_b.append(c)
        cands_b = sorted(cands_b, key=lambda c: score_combo(c, w_blend_b), reverse=True)
        pool_b = cands_b[:pool_size_b]

        bestA2_b = list(pool_b[0]) if pool_b else []
        zA2_b = zscore_combo(bestA2_b, w_blend_b) if bestA2_b else 0.0

        # n recomendado
        def pick_n_b(z, bank, vol):
            adj = 0.05 if vol=="Low" else -0.05 if vol=="High" else 0.0
            for th in THRESH_N:
                if z >= th["z"] + adj:
                    n = min(th["n"], int(bank))
                    return max(1, n)
            return 1

        n_b = pick_n_b(zA2_b, bank_b, vol_b)
        n_b = max(1, min(6, n_b))

        # greedy con diversidad
        def greedy_select_b(pool,w,n):
            if n<=0: return []
            sp=sorted(pool,key=lambda c:score_combo(c,w),reverse=True)
            sel=[sp[0]]
            while len(sel)<n and len(sp)>1:
                best=None; bestv=-1e9
                for c in sp[1:]:
                    if any(tuple(c)==tuple(s) for s in sel): continue
                    pen=sum(overlap_ratio(c,s) for s in sel)
                    v=score_combo(c,w)-LAMBDA_DIVERSIDAD*pen
                    if v>bestv: bestv=v; best=c
                if best is None: break
                sel.append(best)
            return sel

        A2s_b = greedy_select_b(pool_b, w_blend_b, max(0, n_b-1))

        wr_glob_b = weighted_counts_rei(df_recent_b, last_dt_b)
        wr_day_b  = weighted_counts_rei(df_recent_b[df_recent_b["weekday"]==weekday], last_dt_b)
        rei_scores_b = {r: DAY_BLEND_ALPHA*wr_day_b.get(r,0.0) + (1-DAY_BLEND_ALPHA)*wr_glob_b.get(r,0.0) for r in range(10)}
        rein_sug_b = max(rei_scores_b, key=lambda r: rei_scores_b[r]) if rei_scores_b else 0

        st.subheader("Resultados · Bonoloto")
        st.write(f"**A1 (ancla inicial)** {A1b}  |  **n recomendado:** {n_b}")
        for i, c in enumerate(A2s_b, start=1):
            st.write(f"**A2 #{i}** {list(c)}")
        st.write(f"**Reintegro sugerido (informativo)**: {rein_sug_b}")
        st.write("**Joker**: No aplica en Bonoloto")

        rows_b = [{"Tipo":"A1","N1":A1b[0],"N2":A1b[1],"N3":A1b[2],"N4":A1b[3],"N5":A1b[4],"N6":A1b[5]}]
        for i, c in enumerate(A2s_b, start=1):
            cl = list(c)
            rows_b.append({"Tipo":f"A2-{i}","N1":cl[0],"N2":cl[1],"N3":cl[2],"N4":cl[3],"N5":cl[4],"N6":cl[5]})
        df_out_b = pd.DataFrame(rows_b)

        tab_res_b, tab_metrics_b, tab_sim_b = st.tabs(["📋 Combinaciones", "📊 Métricas & Probabilidad", "🧪 Simulador de volatilidad"])
        with tab_res_b:
            st.dataframe(df_out_b, use_container_width=True)
            st.download_button(
                "Descargar combinaciones · Bonoloto (CSV)",
                data=df_out_b.to_csv(index=False).encode("utf-8"),
                file_name="bonoloto_recomendaciones.csv",
                mime="text/csv"
            )
        with tab_metrics_b:
            st.markdown("**Indicadores del mejor A2** (respecto a un 6/49 aleatorio)")
            dfm_b = metrics_table(zA2_b, pool_b, w_blend_b)
            st.dataframe(dfm_b, use_container_width=True)
            with st.expander("¿Cómo interpretarlo?"):
                st.markdown(
                    "- *Score relativo (softmax)*: probabilidad relativa del top A2 dentro del **pool** candidato.\n"
                    "- *z (señal)*: cuán por encima de la media están los pesos de ese A2.\n"
                    "- *Lift*: heurística del **plus** del modelo vs. azar. Acotada 0.5×–2.0×.\n"
                    "- *P(k)*: probabilidad exacta 6/49 de acertar **k** números; la columna ajustada multiplica por el lift."
                )
        with tab_sim_b:
            st.markdown("**¿Qué pasa si cambio la volatilidad?**")
            st.dataframe(simulate_volatility(zA2_b, bank_b), use_container_width=True)
