# ==== PARTE 1/4 – INICIO app.py ====
# app.py — Francisco Cabrera · Predictor de La Primitiva & Bonoloto
# UI moderno + determinismo + Lift ×N + prob. base/ajustada + recomendación ÓPTIMA + Bitácora
# Incluye: Google Sheets (read/write), Joker por apuesta, Wizard asistido (ligero)

import math
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from datetime import datetime, timedelta

import gspread
from google.oauth2.service_account import Credentials

# -------------------------- ESTILO / BRANDING --------------------------
st.set_page_config(
    page_title="Francisco Cabrera · Predictor de La Primitiva & Bonoloto",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif !important; }
.block-container { padding-top: 0.8rem; }
h1, h2, h3 { font-weight: 600; }
.small-muted { color: #94a3b8; font-size: 0.85rem; }
.kpill { display:inline-block; background:#0ea5e9; color:white; padding:2px 8px; border-radius:99px; font-size:0.8rem; }
.readonly { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
### **Francisco Cabrera · Predictor de La Primitiva & Bonoloto**
<span class="small-muted">Estrategia A1/A2 con ventana móvil, mezcla por día, diversidad, selección determinista y recomendación de Joker por apuesta. Lift ×N y probabilidad ajustada. Fuente: Google Sheets.</span>
""", unsafe_allow_html=True)

# -------------------------- CONSTANTES MODELO (por defecto) --------------------------
WINDOW_DRAWS_DEF    = 24
HALF_LIFE_DAYS_DEF  = 60.0         # vida media temporal
DAY_BLEND_ALPHA_DEF = 0.30         # mezcla señal día / global
ALPHA_DIR_DEF       = 0.30         # suavizado dirichlet
MU_PENALTY_DEF      = 1.00         # penalización "popularidad"
K_CANDIDATOS        = 3000
MIN_DIV             = 0.60         # mínima diversidad vs A1
LAMBDA_DIVERSIDAD_DEF = 0.60       # penalización solapes

# Reglas para sugerir nº de A2 (si se usa modo "varias"); el óptimo solo usa 1 por defecto
THRESH_N = [
  {"z": 0.50, "n": 6},
  {"z": 0.35, "n": 4},
  {"z": 0.20, "n": 3},
  {"z": 0.10, "n": 2},
  {"z":-999,  "n": 1},
]

# A1 fijas por día (Primitiva)
A1_FIJAS_PRIMI = {
    "Monday":    [4,24,35,37,40,46],
    "Thursday":  [1,10,23,39,45,48],
    "Saturday":  [7,12,14,25,29,40],
}
REIN_FIJOS_PRIMI = {"Monday":1, "Thursday":8, "Saturday":0}

# A1 neutras por día (Bonoloto; se puede calibrar más adelante)
A1_FIJAS_BONO = {
    0: [4,24,35,37,40,46],  # Mon
    1: [4,24,35,37,40,46],  # Tue
    2: [4,24,35,37,40,46],  # Wed
    3: [4,24,35,37,40,46],  # Thu
    4: [4,24,35,37,40,46],  # Fri
    5: [4,24,35,37,40,46],  # Sat
    6: [4,24,35,37,40,46],  # Sun
}

# -------------------------- HELPERS GENERALES --------------------------
def comb(n, k):
    try:
        return math.comb(n, k)
    except Exception:
        from math import factorial
        return factorial(n)//(factorial(k)*factorial(n-k))

def dayname_to_weekday(dn: str) -> int:
    return {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}.get(dn, -1)

def weekday_to_dayname(w: int) -> str:
    return ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][w]

def time_weight(d, ref, half_life_days):
    delta = max(0, (ref - d).days)
    return float(np.exp(-np.log(2)/half_life_days * delta))

def weighted_counts_nums(df_in, ref, half_life_days):
    w = {i:0.0 for i in range(1,50)}
    for _, r in df_in.iterrows():
        tw = time_weight(r["FECHA"], ref, half_life_days)
        for c in ["N1","N2","N3","N4","N5","N6"]:
            if not pd.isna(r[c]):
                w[int(r[c])] += tw
    return w

def weighted_counts_rei(df_in, ref, half_life_days):
    w = {i:0.0 for i in range(10)}
    if "Reintegro" in df_in.columns:
        for _, r in df_in.dropna(subset=["Reintegro"]).iterrows():
            tw = time_weight(r["FECHA"], ref, half_life_days)
            w[int(r["Reintegro"])] += tw
    return w

def blend(w_day, w_glob, alpha):
    return {n: alpha*w_day.get(n,0.0) + (1-alpha)*w_glob.get(n,0.0) for n in range(1,50)}

def popularity_penalty(combo):
    c = sorted(combo)
    p_dates = sum(1 for x in c if x<=31)/6.0
    consec  = sum(1 for a,b in zip(c, c[1:]) if b==a+1)
    decades = [x//10 for x in c]; units = [x%10 for x in c]
    max_dec = max(Counter(decades).values()); max_unit = max(Counter(units).values())
    s = sum(c); roundness = 1.0/(1.0 + abs(s-120)/10.0)
    return 1.2*p_dates + 0.8*consec + 0.5*(max_dec-2 if max_dec>2 else 0) + 0.5*(max_unit-2 if max_unit>2 else 0) + 0.4*roundness

def score_combo(combo, weights, alpha_dir, mu_penalty):
    return sum(np.log(max(weights.get(n,0.0), 1e-9) + alpha_dir) for n in combo) - mu_penalty*popularity_penalty(combo)

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
    return (comboMean - meanW)/sdW, comboMean, meanW

def pick_n(z, bank, vol, thresh_table):
    adj = 0.05 if vol=="Low" else -0.05 if vol=="High" else 0.0
    for th in thresh_table:
        if z >= th["z"] + adj:
            n = min(th["n"], int(bank))
            return max(1, n)
    return 1

def greedy_select(pool, weights, n, alpha_dir, mu_penalty, lambda_div):
    if n<=0: return []
    sorted_pool = sorted(pool, key=lambda c: score_combo(c,weights,alpha_dir,mu_penalty), reverse=True)
    selected = [sorted_pool[0]]
    while len(selected)<n:
        bestC=None; bestVal=-1e9
        for c in sorted_pool:
            if any(tuple(c)==tuple(s) for s in selected): continue
            div_pen = sum(overlap_ratio(c,s) for s in selected)
            val = score_combo(c,weights,alpha_dir,mu_penalty) - lambda_div*div_pen
            if val>bestVal: bestVal=val; bestC=c
        if bestC is None: break
        selected.append(bestC)
    return selected

def expand_to_k(base6, weights, k):
    """Amplía una combinación base de 6 a k>6 con los mejores pesos restantes."""
    if k<=6:
        return list(base6[:6])
    extras = [n for n in range(1,50) if n not in base6]
    extras_sorted = sorted(extras, key=lambda x: weights.get(x,0.0), reverse=True)
    add = extras_sorted[:max(0,k-6)]
    out = sorted(list(set(base6) | set(add)))
    return out[:k]

def conf_label(z):
    if z>=0.50: return "Alta"
    if z>=0.20: return "Media"
    return "Baja"

# ---- Joker helpers ----
def minmax_norm(x, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def compute_rein_probs(df_recent, ref_dt, weekday_mask, half_life_days, alpha_day):
    wr_glob = weighted_counts_rei(df_recent, ref_dt, half_life_days)
    wr_day  = weighted_counts_rei(df_recent[df_recent["weekday"]==weekday_mask], ref_dt, half_life_days)
    rr = {r: alpha_day*wr_day.get(r,0.0) + (1-alpha_day)*wr_glob.get(r,0.0) for r in range(10)}
    return rr

def joker_score(combo, weights, rein_dict):
    """Puntuación Joker (0..1) por apuesta A2: mezcla señal (z) y 'contexto' de reintegro."""
    z,_,_ = zscore_combo(combo, weights)
    zN = minmax_norm(z, -1.5, 1.5)
    if rein_dict:
        top = max(rein_dict.values())
        reinN = minmax_norm(top, 0.0, top if top>0 else 1.0)
    else:
        reinN = 0.0
    return 0.6*zN + 0.4*reinN

# ---- Probabilidades base y Lift ----
def prob_base_k(k:int)->float:
    """Probabilidad de acertar los 6 con un boleto de tamaño k."""
    if k<6: return 0.0
    return comb(k,6)/comb(49,6)

def lift_multiplier(combo, weights):
    """Multiplicador de señal vs media: mean(weights combo) / mean(weights global)."""
    z, cm, meanW = zscore_combo(combo, weights)
    mult = (cm / (meanW if meanW>0 else 1.0))
    # evitar degenerados
    return max(0.25, float(mult))
# ==== PARTE 1/4 – FIN ====
# ==== PARTE 2/4 ====
# -------------------------- GOOGLE SHEETS --------------------------
def get_gcp_credentials():
    # Soporta tanto [gcp_service_account] (TOML) como gcp_json (JSON string)
    import json as _json
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    info = None
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        pk = info.get("private_key","")
        if isinstance(pk,str) and "\\n" in pk:
            info["private_key"]=pk.replace("\\n","\n")
    elif "gcp_json" in st.secrets:
        info = _json.loads(st.secrets["gcp_json"])
        if isinstance(info.get("private_key",""), str) and "\\n" in info["private_key"]:
            info["private_key"] = info["private_key"].replace("\\n","\n")
    else:
        raise RuntimeError("Faltan credenciales: añade [gcp_service_account] o gcp_json en Secrets.")
    return Credentials.from_service_account_info(info, scopes=scopes)

@st.cache_data(ttl=600, show_spinner=False)
def load_sheet_df(sheet_id_key: str, worksheet_key: str, default_ws: str):
    creds = get_gcp_credentials()
    gc = gspread.authorize(creds)
    sid = (st.secrets.get("gcp_service_account", {}) or {}).get(sheet_id_key) or st.secrets.get(sheet_id_key)
    wsn = (st.secrets.get("gcp_service_account", {}) or {}).get(worksheet_key, default_ws) or st.secrets.get(worksheet_key, default_ws)
    if not sid:
        return pd.DataFrame()
    try:
        sh = gc.open_by_key(sid); ws = sh.worksheet(wsn)
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    expected = ["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
    for c in expected:
        if c not in df.columns: df[c]=np.nan
    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    for c in ["N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
    return df[expected]

def append_row_if_new(sheet_id_key, worksheet_key, default_ws, row_dict):
    """Append sólo si la fila (por FECHA) no está ya con la misma combinación."""
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get("gcp_service_account", {}) or {}).get(sheet_id_key) or st.secrets.get(sheet_id_key)
        wsn = (st.secrets.get("gcp_service_account", {}) or {}).get(worksheet_key, default_ws) or st.secrets.get(worksheet_key, default_ws)
        sh = gc.open_by_key(sid)
        try:
            ws = sh.worksheet(wsn)
        except Exception:
            ws = sh.add_worksheet(title=wsn, rows=2000, cols=20)
            ws.append_row(["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"])
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
        df = pd.DataFrame(rows)
        if not df.empty:
            df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
            same = df["FECHA"].dt.date == pd.to_datetime(row_dict["FECHA"]).date()
            if same.any():
                last = df.loc[same].tail(1).to_dict("records")[0]
                keys = ["N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
                match = all(int(last[k])==int(row_dict[k]) for k in keys if not pd.isna(row_dict[k]))
                if match:
                    return False  # ya existe
        new_row = [
            pd.to_datetime(row_dict["FECHA"]).strftime("%d/%m/%Y"),
            row_dict["N1"],row_dict["N2"],row_dict["N3"],row_dict["N4"],row_dict["N5"],row_dict["N6"],
            row_dict["Complementario"],row_dict["Reintegro"]
        ]
        ws.append_row(new_row)
        return True
    except Exception:
        return False

def bitacora_append(row:dict):
    """Escribe una fila en la hoja Bitacora si está configurada en Secrets."""
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get("gcp_service_account", {}) or {}).get("sheet_id_bitacora") or st.secrets.get("sheet_id_bitacora") \
              or (st.secrets.get("gcp_service_account", {}) or {}).get("sheet_id") or st.secrets.get("sheet_id")
        if not sid:
            return False
        sh = gc.open_by_key(sid)
        try:
            ws = sh.worksheet("Bitacora")
        except Exception:
            ws = sh.add_worksheet(title="Bitacora", rows=5000, cols=30)
            ws.append_row(["ts","juego","fecha_sorteo","a1","a2","k","joker","coste","score","lift","p_base","p_ajustada","policy","decision"])
        ws.append_row([
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            row.get("juego",""),
            pd.to_datetime(row.get("fecha_sorteo")).strftime("%d/%m/%Y") if row.get("fecha_sorteo") else "",
            ", ".join(map(str,row.get("a1",[]))),
            ", ".join(map(str,row.get("a2",[]))),
            int(row.get("k",6)),
            "SI" if row.get("joker") else "NO",
            float(row.get("coste",0.0)),
            float(row.get("score",0.0)),
            float(row.get("lift",1.0)),
            float(row.get("p_base",0.0)),
            float(row.get("p_adj",0.0)),
            row.get("policy",""),
            row.get("decision","")
        ])
        return True
    except Exception:
        return False

# -------------------------- SIDEBAR PARÁMETROS --------------------------
with st.sidebar:
    st.subheader("Parámetros · Primitiva")
    bank_pr = st.number_input("Banco (€) · Primitiva", min_value=0, value=10, step=1)
    vol_pr  = st.selectbox("Volatilidad · Primitiva", ["Low","Medium","High"], index=1)
    precio_simple_pr = st.number_input("Precio por apuesta simple (€)", min_value=0.0, value=1.0, step=0.5, format="%.2f")

    st.markdown("---")
    st.subheader("Apuesta múltiple (opcional)")
    use_multi = st.checkbox("Usar apuesta múltiple (k>6)", value=True)
    k_nums    = st.slider("Números por apuesta (k)", min_value=6, max_value=8, value=7, step=1, disabled=not use_multi)

    st.markdown("---")
    st.subheader("Joker (Primitiva)")
    use_joker   = st.checkbox("Activar recomendaciones de Joker por apuesta", value=True)
    joker_thr   = st.slider("Umbral para recomendar Joker", 0.00, 1.00, 0.85, 0.01,
                             help="Recomendamos Joker en las A2 con puntuación ≥ umbral.")
    precio_joker  = st.number_input("Precio Joker (€)", min_value=1.0, value=1.0, step=1.0, format="%.2f")

    st.markdown("---")
    with st.expander("Parámetros avanzados (modelo)", expanded=False):
        st.caption("Ajustes para pruebas. La recomendación estándar usa los valores por defecto.")
        WINDOW_DRAWS    = st.slider("Ventana (nº de sorteos usados)", 12, 120, WINDOW_DRAWS_DEF, 1, key="win_pr")
        HALF_LIFE_DAYS  = float(st.slider("Vida media temporal (días)", 15, 180, int(HALF_LIFE_DAYS_DEF), 1, key="hl_pr"))
        DAY_BLEND_ALPHA = float(st.slider("Mezcla por día (α)", 0.0, 1.0, float(DAY_BLEND_ALPHA_DEF), 0.05, key="dfa_pr"))
        ALPHA_DIR       = float(st.slider("Suavizado pseudo-frecuencias (α_dir)", 0.00, 1.00, float(ALPHA_DIR_DEF), 0.01, key="ad_pr"))
        MU_PENALTY      = float(st.slider("Penalización 'popularidad'", 0.0, 2.0, float(MU_PENALTY_DEF), 0.1, key="mu_pr"))
        LAMBDA_DIVERSIDAD = float(st.slider("Peso diversidad (λ)", 0.0, 2.0, float(LAMBDA_DIVERSIDAD_DEF), 0.1, key="ld_pr"))
        st.caption("Estos parámetros aplican tanto a Primitiva como a Bonoloto.")

    st.markdown("---")
    st.subheader("Parámetros · Bonoloto")
    bank_bo = st.number_input("Banco (€) · Bonoloto", min_value=0, value=10, step=1, key="bank_bono")
    vol_bo  = st.selectbox("Volatilidad · Bonoloto", ["Low","Medium","High"], index=1, key="vol_bono")
    precio_simple_bo = st.number_input("Precio simple Bonoloto (€)", min_value=0.5, value=0.5, step=0.5, format="%.2f",
                                       help="En Bonoloto, el precio suele ser múltiplos de 0,50 € por apuesta.")

# -------------------------- TABS --------------------------
tab_primi, tab_bono, tab_tutorial = st.tabs(["La Primitiva", "Bonoloto", "📘 Tutorial"])

# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader(f"La Primitiva · Recomendador A2 · k={'múltiple' if (use_multi and k_nums>6) else '6'}")
    wizard_pr = st.toggle("🪄 Modo asistido (wizard)", value=True,
                          help="Guía paso a paso: Fuente → Calcular → 🏆 Óptima → Confirmar",
                          key="wizard_pr")

    # Carga histórico una vez para autorrellenar
    df_hist_full = load_sheet_df("sheet_id","worksheet_historico","Historico")
    last_rec = df_hist_full.tail(1) if not df_hist_full.empty else pd.DataFrame()

    fuente = st.radio("Origen de datos del último sorteo",
                      ["Usar último del histórico", "Introducir manualmente"],
                      index=0 if not df_hist_full.empty else 1, horizontal=True,
                      key="src_pr")

    if fuente == "Usar último del histórico" and not df_hist_full.empty:
        # Usar el último disponible del Sheet (sin pedir datos)
        row = last_rec.iloc[0]
        last_dt = pd.to_datetime(row["FECHA"])
        nums = [int(row["N1"]), int(row["N2"]), int(row["N3"]), int(row["N4"]), int(row["N5"]), int(row["N6"])]
        comp = int(row["Complementario"]) if not pd.isna(row["Complementario"]) else 18
        rein = int(row["Reintegro"]) if not pd.isna(row["Reintegro"]) else 0
        st.info(f"Usando el último sorteo del histórico: **{last_dt.strftime('%d/%m/%Y')}**  ·  Números: {nums}  ·  C: {comp}  ·  R: {rein}")
        save_hist = False  # no tiene sentido re-guardar
        if wizard_pr:
            st.info("**Paso 2/4** · Pulsa **Calcular** para ver la apuesta óptima.")
        do_calc = st.button("Calcular recomendaciones · Primitiva", type="primary")
    else:
        # Entrada manual
        with st.form("form_primi"):
            c1, c2, c3 = st.columns([1,1,1])
            last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=datetime.today().date())
            rein = c2.number_input("Reintegro (0-9)", min_value=0, max_value=9, value=2, step=1)
            comp = c3.number_input("Complementario (1-49)", min_value=1, max_value=49, value=18, step=1)

            st.markdown("**Números extraídos (6 distintos)**")
            cols = st.columns(6)
            defaults = [5,6,8,23,46,47]
            nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"npr{i+1}") for i in range(6)]
            save_hist = st.checkbox("Guardar en histórico (Primitiva) si es nuevo", value=True)
            do_calc = st.form_submit_button("Calcular recomendaciones · Primitiva")

        # Si la fecha existe ya en el Sheet, pisamos con los datos existentes (UX: no duplicar trabajo)
        if do_calc:
            if df_hist_full.empty:
                last_dt = pd.to_datetime(last_date)
            else:
                target = pd.to_datetime(last_date).date()
                same = df_hist_full["FECHA"].dt.date == target
                if same.any():
                    r = df_hist_full.loc[same].tail(1).iloc[0]
                    last_dt = pd.to_datetime(r["FECHA"])
                    nums = [int(r["N1"]), int(r["N2"]), int(r["N3"]), int(r["N4"]), int(r["N5"]), int(r["N6"])]
                    comp = int(r["Complementario"]) if not pd.isna(r["Complementario"]) else 18
                    rein = int(r["Reintegro"]) if not pd.isna(r["Reintegro"]) else 0
                    save_hist = False
                    st.info("La fecha ya estaba en el histórico. Se han usado los datos existentes y no se añadirá nada.")
                else:
                    last_dt = pd.to_datetime(last_date)
# ==== PARTE 2/4 – FIN ====
# ==== PARTE 3/4 ====
    if 'do_calc' in locals() and do_calc:
        if len(set(nums))!=6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        # Próximo día (Mon→Thu, Thu→Sat, Sat→Mon)
        wd = last_dt.weekday()
        if wd==0: next_dt, next_dayname = last_dt + timedelta(days=3), "Thursday"
        elif wd==3: next_dt, next_dayname = last_dt + timedelta(days=2), "Saturday"
        elif wd==5: next_dt, next_dayname = last_dt + timedelta(days=2), "Monday"
        else:
            st.error("La fecha debe ser Lunes, Jueves o Sábado.")
            st.stop()

        st.info(f"Próximo sorteo: **{next_dt.date().strftime('%d/%m/%Y')}** ({next_dayname})")

        # Base (ventana) con el histórico + entrada si fuera nueva
        base = df_hist_full[df_hist_full["FECHA"]<=last_dt].copy()
        if base.empty or not (base["FECHA"].dt.date == last_dt.date()).any():
            newrow = {
                "FECHA": last_dt, "N1": nums[0], "N2": nums[1], "N3": nums[2],
                "N4": nums[3], "N5": nums[4], "N6": nums[5],
                "Complementario": comp, "Reintegro": rein
            }
            base = pd.concat([base, pd.DataFrame([newrow])], ignore_index=True)
        base = base.sort_values("FECHA").tail(st.session_state.get('win_pr', WINDOW_DRAWS_DEF)).reset_index(drop=True)

        # Sello weekday
        base["weekday"] = base["FECHA"].dt.weekday

        # Parámetros avanzados actuales (del sidebar)
        WINDOW_DRAWS    = st.session_state.get('win_pr', WINDOW_DRAWS_DEF)
        HALF_LIFE_DAYS  = st.session_state.get('hl_pr', HALF_LIFE_DAYS_DEF)
        DAY_BLEND_ALPHA = st.session_state.get('dfa_pr', DAY_BLEND_ALPHA_DEF)
        ALPHA_DIR       = st.session_state.get('ad_pr', ALPHA_DIR_DEF)
        MU_PENALTY      = st.session_state.get('mu_pr', MU_PENALTY_DEF)
        LAMBDA_DIVERSIDAD = st.session_state.get('ld_pr', LAMBDA_DIVERSIDAD_DEF)

        # Pesos
        weekday_mask = dayname_to_weekday(next_dayname)
        w_glob = weighted_counts_nums(base, last_dt, HALF_LIFE_DAYS)
        w_day  = weighted_counts_nums(base[base["weekday"]==weekday_mask], last_dt, HALF_LIFE_DAYS)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        # Reintegros
        rein_dict = compute_rein_probs(base, last_dt, weekday_mask, HALF_LIFE_DAYS, DAY_BLEND_ALPHA)
        rein_sug_dynamic = max(rein_dict, key=lambda r: rein_dict[r]) if rein_dict else 0
        rein_sug_A1_ref  = REIN_FIJOS_PRIMI.get(next_dayname, rein_sug_dynamic)

        # A1
        A1_6 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])
        A1_k = expand_to_k(A1_6, w_blend, k_nums if (use_multi and k_nums>6) else 6)

        # Determinismo (seed estable por inputs + parámetros)
        seed_val = abs(hash(f"PRIMITIVA|{last_dt.date()}|{tuple(sorted(nums))}|{comp}|{rein}|k={k_nums}|multi={use_multi}|alpha={DAY_BLEND_ALPHA}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}")) % (2**32 - 1)
        np.random.seed(seed_val)

        # Candidatos A2 (6)
        cands, seen, tries = [], set(), 0
        while len(cands)<K_CANDIDATOS and tries < K_CANDIDATOS*60:
            c = tuple(random_combo()); tries += 1
            if c in seen: continue
            seen.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1_6) > (1 - MIN_DIV): continue
            cands.append(c)

        cands = sorted(cands, key=lambda c: score_combo(c, w_blend, ALPHA_DIR, MU_PENALTY), reverse=True)
        pool = cands[:1200]

        best6 = list(pool[0]) if pool else []
        zA2,_,_ = zscore_combo(best6, w_blend) if best6 else (0.0,0.0,0.0)
        n_sugerido = 3  # mantenemos 3 alternativas (óptima + 2 ampliaciones)

        # Selección greedily
        A2s_6 = greedy_select(pool, w_blend, n_sugerido, ALPHA_DIR, MU_PENALTY, LAMBDA_DIVERSIDAD)
        A2s_k = [expand_to_k(a2, w_blend, k_nums) if (use_multi and k_nums>6) else a2 for a2 in A2s_6]

        # Calcular métricas por A2 y decidir ÓPTIMA
        filas = []
        for i, base6 in enumerate(A2s_6, start=1):
            lift = lift_multiplier(base6, w_blend)
            scJ = joker_score(base6, w_blend, rein_dict) if use_joker else 0.0
            filas.append({
                "tipo": f"A2 #{i}",
                "base6": base6,
                "k": k_nums if (use_multi and k_nums>6) else 6,
                "lift": lift,
                "score": score_combo(base6, w_blend, ALPHA_DIR, MU_PENALTY),
                "scoreJ": scJ,
                "joker_reco": (use_joker and scJ>=joker_thr)
            })

        # -------- BONOLOTO --------
# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader(f"Bonoloto · Recomendador A2 · k={'múltiple' if (use_multi and k_nums>6) else '6'}")
    wizard_bo = st.toggle("🪄 Modo asistido (wizard)", value=True,
                          help="Guía paso a paso: Fuente → Calcular → 🏆 Óptima → Confirmar",
                          key="wizard_bo")

    # Carga histórico una vez para autorrellenar
    df_b_full = load_sheet_df("sheet_id_bono","worksheet_historico_bono","HistoricoBono")
    last_rec_b = df_b_full.tail(1) if not df_b_full.empty else pd.DataFrame()

    fuente_b = st.radio("Origen de datos del último sorteo (Bonoloto)",
                        ["Usar último del histórico", "Introducir manualmente"],
                        index=0 if not df_b_full.empty else 1, horizontal=True, key="src_b")

    if fuente_b == "Usar último del histórico" and not df_b_full.empty:
        rowb = last_rec_b.iloc[0]
        last_dt_b = pd.to_datetime(rowb["FECHA"])
        nums_b = [int(rowb["N1"]), int(rowb["N2"]), int(rowb["N3"]), int(rowb["N4"]), int(rowb["N5"]), int(rowb["N6"])]
        comp_b = int(rowb["Complementario"]) if not pd.isna(rowb["Complementario"]) else 18
        rein_b = int(rowb["Reintegro"]) if not pd.isna(rowb["Reintegro"]) else 0
        st.info(f"Usando el último sorteo del histórico (Bonoloto): **{last_dt_b.strftime('%d/%m/%Y')}**  ·  Números: {nums_b}  ·  C: {comp_b}  ·  R: {rein_b}")
        save_hist_b = False
        if wizard_bo:
            st.info("**Paso 2/4** · Pulsa **Calcular** para ver la apuesta óptima.")
        do_calc_b = st.button("Calcular recomendaciones · Bonoloto", type="primary")
    else:
        with st.form("form_bono"):
            c1, c2, c3 = st.columns([1,1,1])
            last_date_b = c1.date_input("Fecha último sorteo (Bonoloto)", value=datetime.today().date(), key="dt_b")
            rein_b = c2.number_input("Reintegro (0-9)", min_value=0, max_value=9, value=2, step=1, key="re_b")
            comp_b = c3.number_input("Complementario (1-49)", min_value=1, max_value=49, value=18, step=1, key="co_b")

            st.markdown("**Números extraídos (6 distintos)**")
            cols = st.columns(6)
            defaults_b = [5,6,8,23,46,47]
            nums_b = [cols[i].number_input(f"N{i+1} (Bono)", 1, 49, defaults_b[i], 1, key=f"nbo{i+1}") for i in range(6)]

            save_hist_b = st.checkbox("Guardar en histórico (Bonoloto) si es nuevo", value=True)
            do_calc_b = st.form_submit_button("Calcular recomendaciones · Bonoloto")

        if do_calc_b:
            if df_b_full.empty:
                last_dt_b = pd.to_datetime(last_date_b)
            else:
                target_b = pd.to_datetime(last_date_b).date()
                same_b = df_b_full["FECHA"].dt.date == target_b
                if same_b.any():
                    rb = df_b_full.loc[same_b].tail(1).iloc[0]
                    last_dt_b = pd.to_datetime(rb["FECHA"])
                    nums_b = [int(rb["N1"]), int(rb["N2"]), int(rb["N3"]), int(rb["N4"]), int(rb["N5"]), int(rb["N6"])]
                    comp_b = int(rb["Complementario"]) if not pd.isna(rb["Complementario"]) else 18
                    rein_b = int(rb["Reintegro"]) if not pd.isna(rb["Reintegro"]) else 0
                    save_hist_b = False
                    st.info("La fecha ya estaba en el histórico (Bonoloto). Se han usado los datos existentes.")
                else:
                    # ✅ Caso que faltaba: fecha manual NO existe en histórico
                    last_dt_b = pd.to_datetime(last_date_b)

    if 'do_calc_b' in locals() and do_calc_b:
        if len(set(nums_b))!=6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        # Próximo sorteo (aprox.): +1 día
        next_dt_b = last_dt_b + timedelta(days=1)
        weekday = next_dt_b.weekday()
        st.info(f"Próximo sorteo (aprox.): **{next_dt_b.date().strftime('%d/%m/%Y')}** ({next_dt_b.day_name()})")

        # Base (ventana)
        base_b = df_b_full[df_b_full["FECHA"]<=last_dt_b].copy()
        if base_b.empty or not (base_b["FECHA"].dt.date == last_dt_b.date()).any():
            new_b = {"FECHA": last_dt_b, "N1": nums_b[0], "N2": nums_b[1], "N3": nums_b[2],
                     "N4": nums_b[3], "N5": nums_b[4], "N6": nums_b[5],
                     "Complementario": comp_b, "Reintegro": rein_b}
            base_b = pd.concat([base_b, pd.DataFrame([new_b])], ignore_index=True)

        base_b = base_b.sort_values("FECHA").tail(st.session_state.get('win_pr', WINDOW_DRAWS_DEF)).reset_index(drop=True)
        base_b["weekday"] = base_b["FECHA"].dt.weekday

        # Parámetros avanzados
        WINDOW_DRAWS    = st.session_state.get('win_pr', WINDOW_DRAWS_DEF)
        HALF_LIFE_DAYS  = st.session_state.get('hl_pr', HALF_LIFE_DAYS_DEF)
        DAY_BLEND_ALPHA = st.session_state.get('dfa_pr', DAY_BLEND_ALPHA_DEF)
        ALPHA_DIR       = st.session_state.get('ad_pr', ALPHA_DIR_DEF)
        MU_PENALTY      = st.session_state.get('mu_pr', MU_PENALTY_DEF)
        LAMBDA_DIVERSIDAD = st.session_state.get('ld_pr', LAMBDA_DIVERSIDAD_DEF)

        w_glob_b = weighted_counts_nums(base_b, last_dt_b, HALF_LIFE_DAYS)
        w_day_b  = weighted_counts_nums(base_b[base_b["weekday"]==weekday], last_dt_b, HALF_LIFE_DAYS)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        A1b_6 = A1_FIJAS_BONO.get(weekday, [4,24,35,37,40,46])
        A1b_k = expand_to_k(A1b_6, w_blend_b, k_nums if (use_multi and k_nums>6) else 6)

        seed_val_b = abs(hash(f"BONOLOTO|{last_dt_b.date()}|{tuple(sorted(nums_b))}|{comp_b}|{rein_b}|k={k_nums}|multi={use_multi}|alpha={DAY_BLEND_ALPHA}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}")) % (2**32 - 1)
        np.random.seed(seed_val_b)

        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*60:
            c = tuple(random_combo()); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1b_6) > (1 - MIN_DIV): continue
            cands_b.append(c)

        cands_b = sorted(cands_b, key=lambda c: score_combo(c, w_blend_b, ALPHA_DIR, MU_PENALTY), reverse=True)
        pool_b = cands_b[:1200]

        best6_b = list(pool_b[0]) if pool_b else []
        zA2_b,_,_ = zscore_combo(best6_b, w_blend_b) if best6_b else (0.0,0.0,0.0)
        n_b = 3

        A2s_b_6 = greedy_select(pool_b, w_blend_b, n_b, ALPHA_DIR, MU_PENALTY, LAMBDA_DIVERSIDAD)
        A2s_b_k = [expand_to_k(a2, w_blend_b, k_nums) if (use_multi and k_nums>6) else a2 for a2 in A2s_b_6]

        filas_b = []
        for i, base6 in enumerate(A2s_b_6, start=1):
            lift = lift_multiplier(base6, w_blend_b)
            filas_b.append({
                "tipo": f"A2 #{i}",
                "base6": base6,
                "k": k_nums if (use_multi and k_nums>6) else 6,
                "lift": lift,
                "score": score_combo(base6, w_blend_b, ALPHA_DIR, MU_PENALTY),
            })
        opt_b = sorted(filas_b, key=lambda r: (r["lift"], r["score"]), reverse=True)[0] if filas_b else None

        k_reco_b = (k_nums if (use_multi and k_nums>6) else 6)
        p_base_b = prob_base_k(k_reco_b)
        lift_opt_b = opt_b["lift"] if opt_b else 1.0
        p_adj_b = p_base_b * lift_opt_b
        coste_b = comb(k_reco_b,6) * float(precio_simple_bo)

        subB1, subB2, subB3, subB4 = st.tabs(["🏆 Óptima", "Apuestas", "Métricas", "Ventana de referencia"])

        with subB1:
            st.metric("Boletos (A1 + A2 jugadas)", 1)
            st.metric("Coste estimado (€)", f"{coste_b:,.2f}")
            st.metric("Confianza (señal)", conf_label(zA2_b))
            if opt_b:
                st.success(f"**Apuesta Óptima (EV/€)**: {opt_b['base6'] if k_reco_b==6 else expand_to_k(opt_b['base6'], w_blend_b, k_reco_b)}")
                st.write(f"**Lift vs azar:** ×{lift_opt_b:.2f}")
                st.write(f"**Prob. base (k={k_reco_b})**: 1 entre {int(round(1/(p_base_b+1e-12))):,}")
                st.write(f"**Prob. ajustada**: 1 entre {int(round(1/(p_adj_b+1e-12))):,}")
                if st.button("✅ Marcar esta apuesta como jugada (Bitácora)", key="bit_bono"):
                    okb = bitacora_append({
                        "juego":"Bonoloto",
                        "fecha_sorteo": next_dt_b,
                        "a1": A1b_k,
                        "a2": opt_b["base6"],
                        "k": k_reco_b,
                        "joker": False,
                        "coste": coste_b,
                        "score": float(opt_b["score"]),
                        "lift": float(lift_opt_b),
                        "p_base": float(p_base_b),
                        "p_adj": float(p_adj_b),
                        "policy": f"use_multi={use_multi};k={k_reco_b};win={WINDOW_DRAWS};hl={HALF_LIFE_DAYS};alpha={DAY_BLEND_ALPHA}",
                        "decision":"aceptada"
                    })
                    st.success("Guardado en Bitácora." if okb else "No se pudo escribir en Bitácora. Revisa credenciales/hoja.")

        with subB2:
            rows_b = [{
                "Tipo":"A1","k": k_reco_b,"Simples": comb(k_reco_b,6),
                "Números": ", ".join(map(str, A1b_k if k_reco_b>6 else A1b_6)),
                "Lift":"—","Score":"—"
            }]
            for r in filas_b:
                nums = r["base6"] if k_reco_b==6 else expand_to_k(r["base6"], w_blend_b, k_reco_b)
                rows_b.append({
                    "Tipo": r["tipo"], "k": k_reco_b, "Simples": comb(k_reco_b,6),
                    "Números": ", ".join(map(str, nums)),
                    "Lift": f"×{r['lift']:.2f}", "Score": f"{r['score']:.2f}"
                })
            st.dataframe(pd.DataFrame(rows_b), use_container_width=True, height=320)

        with subB3:
            st.markdown("**Señal media A2 (z-score):** {:.3f}".format(zA2_b))
            base_wb = np.array([w_blend_b.get(i,0.0) for i in range(1,50)])
            p_normb = base_wb / (base_wb.sum() if base_wb.sum()>0 else 1.0)
            p_top6b = np.sort(p_normb)[-6:].mean()
            st.markdown(f"**Intensidad media de pesos (top-6):** {p_top6b:.3%}")
            st.caption("Métricas orientativas.")

        with subB4:
            st.dataframe(base_b[["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]].tail(min(24, len(base_b))),
                         use_container_width=True, height=280)

        if fuente_b == "Introducir manualmente" and save_hist_b:
            okb = append_row_if_new("sheet_id_bono","worksheet_historico_bono","HistoricoBono", {
                "FECHA":last_dt_b, "N1":nums_b[0], "N2":nums_b[1], "N3":nums_b[2], "N4":nums_b[3], "N5":nums_b[4], "N6":nums_b[5],
                "Complementario": comp_b, "Reintegro": rein_b
            })
            if okb: st.success("✅ Histórico (Bonoloto) actualizado.")
            else:   st.info("ℹ️ No se añadió al histórico (duplicado o acceso restringido).")
# ==== PARTE 3/4 – FIN ====
# ==== PARTE 4/4 ====
# =========================== TUTORIAL ===========================
with tab_tutorial:
    st.header("Cómo usar el recomendador (guía rápida)")
    st.markdown("""
**Objetivo:** darte **una apuesta óptima** por sorteo, clara y justificada.

- **A1**: combinación “ancla” por día (Primitiva) o neutra (Bonoloto).
- **A2**: nuestras candidatas "inteligentes". Escogemos la **Óptima (EV/€)** y mostramos 1–2 alternativas.
- **Lift ×N**: multiplicador de probabilidad vs azar. ×2.0 significa “el doble que al azar”.
- **Prob. base (k)**: chance de acertar 6 con un boleto de tamaño *k*: `C(k,6)/C(49,6)`.
- **Prob. ajustada**: `Prob. base × Lift` (aprox.).
- **Joker (Primitiva)**: vía extra de premio; lo recomendamos si `ScoreJ ≥ umbral`.

**Flujo recomendado (Wizard):**  
(1) Elige fuente (último histórico o manual) → (2) **Calcular** → (3) Revisa la **Óptima** → (4) “**Marcar como jugada**” para guardar en la **Bitácora**.

**Notas honestas**  
- Aumentar *k* no cambia la **eficiencia por €**; solo cambia coste/varianza. Por defecto, *k=6*.
- Añadir más A2 **aumenta linealmente** la probabilidad total (coste proporcional), pero no la eficiencia por €.
- La lotería es aleatoria; nuestros pesos dan señal **suave**. Usa banca y volatilidad con cabeza.
""")
# ==== PARTE 4/4 – FIN app.py ====
