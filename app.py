# app.py - Recomendador Primitiva & Bonoloto (versión robusta, UX mejorada)

import math
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from datetime import datetime, timedelta

import gspread
from google.oauth2.service_account import Credentials

# -------------------------- ESTILO / BRANDING --------------------------
st.set_page_config(page_title="Recomendador Primitiva & Bonoloto", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif !important; }
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { font-weight: 600; }
.sidebar .sidebar-content { width: 360px; }
.small-muted { color: #94a3b8; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎯 Recomendador Primitiva & Bonoloto")
st.caption("Optimización determinista · Wizard asistido · Bitácora en Google Sheets")

# -------------------------- CONSTANTES MODELO --------------------------
WINDOW_DRAWS_DEF    = 24
HALF_LIFE_DAYS_DEF  = 60.0
DAY_BLEND_ALPHA_DEF = 0.30
ALPHA_DIR_DEF       = 0.30
MU_PENALTY_DEF      = 1.00
K_CANDIDATOS        = 3000
MIN_DIV             = 0.60
LAMBDA_DIVERSIDAD_DEF = 0.60

# A1 fijas Primitiva
A1_FIJAS_PRIMI = {
    "Monday":    [4,24,35,37,40,46],
    "Thursday":  [1,10,23,39,45,48],
    "Saturday":  [7,12,14,25,29,40],
}
REIN_FIJOS_PRIMI = {"Monday":1, "Thursday":8, "Saturday":0}

# A1 neutras Bonoloto
A1_FIJAS_BONO = {d: [4,24,35,37,40,46] for d in range(7)}

# -------------------------- HELPERS GENERALES --------------------------
def comb(n, k):
    try: return math.comb(n, k)
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

def blend(w_day, w_glob, alpha):
    return {n: alpha*w_day.get(n,0.0) + (1-alpha)*w_glob.get(n,0.0) for n in range(1,50)}
# -------------------------- HELPERS ADICIONALES --------------------------
from collections import Counter

def weighted_counts_rei(df_in, ref, half_life_days):
    w = {i:0.0 for i in range(10)}
    if "Reintegro" in df_in.columns:
        for _, r in df_in.dropna(subset=["Reintegro"]).iterrows():
            tw = time_weight(r["FECHA"], ref, half_life_days)
            w[int(r["Reintegro"])] += tw
    return w

def popularity_penalty(combo):
    c = sorted(combo)
    p_dates = sum(1 for x in c if x<=31)/6.0
    consec  = sum(1 for a,b in zip(c, c[1:]) if b==a+1)
    decades = [x//10 for x in c]; units = [x%10 for x in c]
    max_dec = max(Counter(decades).values()); max_unit = max(Counter(units).values())
    s = sum(c); roundness = 1.0/(1.0 + abs(s-120)/10.0)
    return 1.2*p_dates + 0.8*consec + 0.5*(max_dec-2 if max_dec>2 else 0) + 0.5*(max_unit-2 if max_unit>2 else 0) + 0.4*roundness

def score_combo(combo, weights, alpha_dir, mu_penalty):
    return sum(np.log(weights.get(n,0.0) + alpha_dir) for n in combo) - mu_penalty*popularity_penalty(combo)

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
    """Puntuación Joker (0..1): mezcla señal (z) y contexto de reintegro."""
    z = zscore_combo(combo, weights)
    zN = minmax_norm(z, -1.5, 1.5)
    if rein_dict:
        top = max(rein_dict.values())
        reinN = minmax_norm(top, 0.0, top if top>0 else 1.0)
    else:
        reinN = 0.0
    return 0.6*zN + 0.4*reinN

# -------------------------- PROBABILIDADES & LIFT --------------------------
C496 = comb(49,6)

def prob_base_k(k: int) -> float:
    """Probabilidad de acertar 6 con un boleto de tamaño k."""
    k = int(k)
    if k < 6: return 0.0
    return comb(k, 6) / C496

def prob_to_1_in_x(p: float) -> str:
    if p <= 0: return "—"
    return f"1 entre {int(round(1.0/p, 0)):,}".replace(",", ".")

def lift_multiplier(combo6, weights, alpha_dir=ALPHA_DIR_DEF, mu_penalty=MU_PENALTY_DEF):
    """Lift ≈ exp(mean_log(w_combo+alpha) - mean_log(w_all+alpha))."""
    w = np.array([weights.get(i, 1e-12) for i in combo6], dtype=float) + float(alpha_dir)
    mean_log_combo = float(np.log(w).mean())
    allW = np.array([weights.get(i, 0.0) for i in range(1,50)], dtype=float) + float(alpha_dir)
    mean_log_all = float(np.log(allW).mean())
    L = np.exp(mean_log_combo - mean_log_all)
    return max(0.1, float(L))

def format_lift(L: float) -> str:
    return f"×{L:.2f}"

# -------------------------- CARTERA: variantes y knapsack --------------------------
def build_variants_for_a2(a2_base6, weights, alpha_dir, mu_penalty,
                          k_set=(6,7,8), price_simple=1.0,
                          allow_joker=True, scoreJ=0.0, joker_price=1.0,
                          joker_rule=(0.60, 0.45)):
    """
    Devuelve variantes de una A2: (desc, nums, k, joker, cost, p_adj, lift).
    p_adj = p_base(k)*Lift  (+ bonus suave por Joker si aplica; el Joker no cambia la prob. de 6).
    """
    L = lift_multiplier(a2_base6, weights, alpha_dir, mu_penalty)
    out = []
    for k in k_set:
        p_base = prob_base_k(k)
        p_adj = p_base * L
        cost  = comb(k,6) * float(price_simple)
        out.append({
            "desc": f"A2 k={k}",
            "nums": sorted(list(set(a2_base6))), "k": k,
            "joker": False, "cost": cost, "p_adj": p_adj, "lift": L
        })
        if allow_joker:
            tag = "no"
            if scoreJ >= joker_rule[0]: tag = "sí"
            elif scoreJ >= joker_rule[1]: tag = "opc"
            if tag != "no":
                bonus = 0.10 * (scoreJ - joker_rule[1]) if tag == "opc" else 0.20 * (scoreJ - joker_rule[0])
                bonus = max(0.0, bonus)
                out.append({
                    "desc": f"A2 k={k} + Joker",
                    "nums": sorted(list(set(a2_base6))), "k": k,
                    "joker": True, "cost": cost + float(joker_price),
                    "p_adj": p_adj + bonus, "lift": L
                })
    return out

def choose_portfolio(variants, bank_euros):
    """0/1 Knapsack: maximiza sum(p_adj) con coste ≤ bank."""
    cents = int(round(bank_euros*100))
    if cents <= 0 or not variants:
        return [], 0.0, 0.0
    V = [0.0]*(cents+1)
    keep = [[False]*len(variants) for _ in range(cents+1)]
    costs = [int(round(v["cost"]*100)) for v in variants]
    vals  = [float(v["p_adj"]) for v in variants]

    for i, (c, val) in enumerate(zip(costs, vals)):
        for b in range(cents, c-1, -1):
            if V[b-c] + val > V[b]:
                V[b] = V[b-c] + val
                keep[b] = keep[b-c].copy()
                keep[b][i] = True

    b = max(range(cents+1), key=lambda x: V[x])
    pick_idx = [i for i,flag in enumerate(keep[b]) if flag]
    chosen   = [variants[i] for i in pick_idx]
    total_cost = sum(v["cost"] for v in chosen)
    total_p    = sum(v["p_adj"] for v in chosen)
    return chosen, total_cost, total_p

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

def bitacora_append_row(row:dict):
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
            ws.append_row(["ts","juego","fecha_sorteo","nums","k","joker","coste","lift","prob_base","prob_ajustada","bank","version_modelo"])
        ws.append_row([
            row.get("ts",""),
            row.get("juego",""),
            row.get("fecha_sorteo",""),
            row.get("nums",""),
            int(row.get("k",6)),
            int(row.get("joker",0)),
            float(row.get("coste",0.0)),
            float(row.get("lift",1.0)),
            float(row.get("prob_base",0.0)),
            float(row.get("prob_ajustada",0.0)),
            float(row.get("bank",0.0)),
            row.get("version_modelo",""),
        ])
        return True
    except Exception:
        return False
# -------------------------- SIDEBAR PARÁMETROS --------------------------
with st.sidebar:
    st.subheader("Parámetros · Primitiva")
    bank_pr = st.number_input("Banco (€) · Primitiva", min_value=0, value=10, step=1, key="pr_bank")
    vol_pr  = st.selectbox("Volatilidad · Primitiva", ["Low","Medium","High"], index=1, key="pr_vol")
    precio_simple_pr = st.number_input("Precio por apuesta simple (€)", min_value=0.0, value=1.0, step=0.5, format="%.2f", key="pr_price_simple")

    st.markdown("---")
    st.subheader("Apuesta múltiple (opcional)")
    use_multi = st.checkbox("Usar apuesta múltiple (k>6)", value=True, key="pr_use_multi")
    k_nums    = st.slider("Números por apuesta (k)", min_value=6, max_value=8, value=7, step=1, disabled=not use_multi, key="pr_k")

    st.markdown("---")
    st.subheader("Joker (Primitiva)")
    use_joker   = st.checkbox("Activar recomendaciones de Joker por apuesta", value=True, key="pr_use_joker")
    joker_thr   = st.slider("Umbral para recomendar Joker", 0.00, 1.00, 0.65, 0.01,
                             help="⭐ Recomendamos Joker en las A2 con ScoreJ ≥ umbral.", key="pr_joker_thr")
    precio_joker  = st.number_input("Precio Joker (€)", min_value=1.0, value=1.0, step=1.0, format="%.2f", key="pr_price_joker")

    st.markdown("---")
    with st.expander("Parámetros avanzados (modelo)", expanded=False):
        st.caption("Ajustes para pruebas. La recomendación estándar usa los valores por defecto.")
        WINDOW_DRAWS    = st.slider("Ventana (nº de sorteos usados)", 12, 120, WINDOW_DRAWS_DEF, 1, key="pr_win")
        HALF_LIFE_DAYS  = float(st.slider("Vida media temporal (días)", 15, 180, int(HALF_LIFE_DAYS_DEF), 1, key="pr_hl"))
        DAY_BLEND_ALPHA = float(st.slider("Mezcla por día (α)", 0.0, 1.0, float(DAY_BLEND_ALPHA_DEF), 0.05, key="pr_dba"))
        ALPHA_DIR       = float(st.slider("Suavizado pseudo-frecuencias (α_dir)", 0.00, 1.00, float(ALPHA_DIR_DEF), 0.01, key="pr_alphadir"))
        MU_PENALTY      = float(st.slider("Penalización 'popularidad'", 0.0, 2.0, float(MU_PENALTY_DEF), 0.1, key="pr_mu"))
        LAMBDA_DIVERSIDAD = float(st.slider("Peso diversidad (λ)", 0.0, 2.0, float(LAMBDA_DIVERSIDAD_DEF), 0.1, key="pr_lambda"))
        st.caption("Estos parámetros aplican a ambos juegos.")

    st.markdown("---")
    st.subheader("Parámetros · Bonoloto")
    bank_bo = st.number_input("Banco (€) · Bonoloto", min_value=0, value=10, step=1, key="bo_bank")
    vol_bo  = st.selectbox("Volatilidad · Bonoloto", ["Low","Medium","High"], index=1, key="bo_vol")
    precio_simple_bo = st.number_input("Precio simple Bonoloto (€)", min_value=0.5, value=0.5, step=0.5, format="%.2f",
                                       help="En Bonoloto suele ser múltiplos de 0,50 € por apuesta.", key="bo_price_simple")

# -------------------------- TABS JUEGOS --------------------------
tab_primi, tab_bono, tab_tutorial = st.tabs(["La Primitiva", "Bonoloto", "📘 Tutorial"])

# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader(f"La Primitiva · Ticket Óptimo (EV/€)")
    wizard_pr = st.toggle("🪄 Modo asistido (wizard)", value=True,
                          help="Guía paso a paso: Datos → Calcular → 🏆 Ticket → Confirmar",
                          key="pr_wizard")

    # Carga histórico para autorrellenar
    df_hist_full = load_sheet_df("sheet_id","worksheet_historico","Historico")
    last_rec = df_hist_full.tail(1) if not df_hist_full.empty else pd.DataFrame()

    fuente = st.radio("Origen de datos del último sorteo",
                      ["Usar último del histórico", "Introducir manualmente"],
                      index=0 if not df_hist_full.empty else 1, horizontal=True,
                      key="pr_src")

    do_calc_pr = False
    if fuente == "Usar último del histórico" and not df_hist_full.empty:
        row = last_rec.iloc[0]
        last_dt = pd.to_datetime(row["FECHA"])
        nums = [int(row["N1"]), int(row["N2"]), int(row["N3"]), int(row["N4"]), int(row["N5"]), int(row["N6"])]
        comp = int(row["Complementario"]) if not pd.isna(row["Complementario"]) else 18
        rein = int(row["Reintegro"]) if not pd.isna(row["Reintegro"]) else 0
        st.info(f"Usando el último sorteo del histórico: **{last_dt.strftime('%d/%m/%Y')}** · Números: {nums} · C: {comp} · R: {rein}")
        do_calc_pr = st.button("Calcular · Primitiva", type="primary", key="pr_calc_btn")
        if wizard_pr and not do_calc_pr:
            # auto-calcular en wizard para acortar pasos
            do_calc_pr = True
    else:
        with st.form("pr_form_manual"):
            c1, c2, c3 = st.columns([1,1,1])
            last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=datetime.today().date(), key="pr_date")
            rein = c2.number_input("Reintegro (0-9)", min_value=0, max_value=9, value=2, step=1, key="pr_rein")
            comp = c3.number_input("Complementario (1-49)", min_value=1, max_value=49, value=18, step=1, key="pr_comp")
            cols = st.columns(6)
            defaults = [5,6,8,23,46,47]
            nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"pr_n{i+1}") for i in range(6)]
            do_calc_pr = st.form_submit_button("Calcular · Primitiva")

        if do_calc_pr:
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
                    st.info("La fecha ya estaba en el histórico. Se han usado los datos existentes.")
                else:
                    last_dt = pd.to_datetime(last_date)

    if do_calc_pr:
        if len(set(nums))!=6:
            st.error("Los 6 números deben ser distintos."); st.stop()

        # Próximo sorteo
        wd = last_dt.weekday()
        if wd==0: next_dt, next_dayname = last_dt + timedelta(days=3), "Thursday"
        elif wd==3: next_dt, next_dayname = last_dt + timedelta(days=2), "Saturday"
        elif wd==5: next_dt, next_dayname = last_dt + timedelta(days=2), "Monday"
        else: st.error("La fecha debe ser Lunes, Jueves o Sábado."); st.stop()
        st.info(f"Próximo sorteo: **{next_dt.date().strftime('%d/%m/%Y')}** ({next_dayname})")

        # Base (ventana)
        base = df_hist_full[df_hist_full["FECHA"]<=last_dt].copy()
        if base.empty or not (base["FECHA"].dt.date == last_dt.date()).any():
            newrow = {"FECHA": last_dt, "N1": nums[0], "N2": nums[1], "N3": nums[2],
                      "N4": nums[3], "N5": nums[4], "N6": nums[5], "Complementario": comp, "Reintegro": rein}
            base = pd.concat([base, pd.DataFrame([newrow])], ignore_index=True)
        base = base.sort_values("FECHA").tail(st.session_state.get("pr_win", WINDOW_DRAWS_DEF)).reset_index(drop=True)
        base["weekday"] = base["FECHA"].dt.weekday

        # Parámetros avanzados
        WINDOW_DRAWS    = st.session_state.get("pr_win", WINDOW_DRAWS_DEF)
        HALF_LIFE_DAYS  = st.session_state.get("pr_hl", HALF_LIFE_DAYS_DEF)
        DAY_BLEND_ALPHA = st.session_state.get("pr_dba", DAY_BLEND_ALPHA_DEF)
        ALPHA_DIR       = st.session_state.get("pr_alphadir", ALPHA_DIR_DEF)
        MU_PENALTY      = st.session_state.get("pr_mu", MU_PENALTY_DEF)
        LAMBDA_DIVERSIDAD = st.session_state.get("pr_lambda", LAMBDA_DIVERSIDAD_DEF)

        weekday_mask = dayname_to_weekday(next_dayname)
        w_glob = weighted_counts_nums(base, last_dt, HALF_LIFE_DAYS)
        w_day  = weighted_counts_nums(base[base["weekday"]==weekday_mask], last_dt, HALF_LIFE_DAYS)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        # Joker contexto
        rein_dict = compute_rein_probs(base, last_dt, weekday_mask, HALF_LIFE_DAYS, DAY_BLEND_ALPHA)
        scoreJ_map = {}

        # A1
        A1_6 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])

        # Determinismo seed
        seed_val = abs(hash(f"PRIMITIVA|{last_dt.date()}|{tuple(sorted(nums))}|{comp}|{rein}|k={k_nums}|multi={use_multi}|alpha={DAY_BLEND_ALPHA}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}")) % (2**32 - 1)
        np.random.seed(seed_val)

        # Candidatos A2
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

        # Greedy top-3
        A2s_6 = []
        if pool:
            A2s_6 = [pool[0]]
            while len(A2s_6) < 3 and len(A2s_6) < len(pool):
                bestC, bestVal = None, -1e9
                for c in pool:
                    if tuple(c) in [tuple(x) for x in A2s_6]: continue
                    div_pen = sum(overlap_ratio(c,s) for s in A2s_6)
                    val = score_combo(c, w_blend, ALPHA_DIR, MU_PENALTY) - LAMBDA_DIVERSIDAD*div_pen
                    if val>bestVal: bestVal, bestC = val, c
                if bestC is None: break
                A2s_6.append(bestC)

        # ScoreJ por A2
        if use_joker:
            for idx, c6 in enumerate(A2s_6, start=1):
                scoreJ_map[idx] = joker_score(c6, w_blend, rein_dict)

        # Render del ticket (k_set según toggle de múltiple)
        k_set = (6,7,8) if use_multi else (6,)
        if not A2s_6:
            st.warning("No se generaron A2 con las restricciones actuales. Ajusta parámetros.")
        else:
            # Renderer inline para evitar dependencias cruzadas
            import pandas as _pd
            st.markdown("### 🏆 Tu ticket recomendado")
            variants = []
            for i, a2 in enumerate(A2s_6, start=1):
                sJ = 0.0 if (scoreJ_map is None) else float(scoreJ_map.get(i, 0.0))
                variants += build_variants_for_a2(
                    a2, w_blend, ALPHA_DIR, MU_PENALTY,
                    k_set=k_set, price_simple=precio_simple_pr,
                    allow_joker=use_joker, scoreJ=sJ, joker_price=precio_joker, joker_rule=(max(0.0, min(1.0, joker_thr)), 0.45)
                )
            chosen, total_cost, total_p = choose_portfolio(variants, bank_pr)
            if not chosen:
                st.warning("No se pudo formar un ticket dentro del banco. Sube el banco o baja k/Joker.")
            else:
                best = max(chosen, key=lambda v: v["p_adj"])
                Ltxt = format_lift(best["lift"])
                pB   = prob_to_1_in_x(prob_base_k(best["k"]))
                pA   = prob_to_1_in_x(prob_base_k(best["k"])*best["lift"])
                st.success(f"**Apuesta Óptima (EV/€)**: {best['nums']} · {best['desc']} · Lift {Ltxt}")
                c1,c2,c3 = st.columns(3)
                c1.metric("Prob. base", pB)
                c2.metric("Prob. ajustada", pA)
                c3.metric("Coste", f"{best['cost']:.2f} €")
                st.caption(f"Próximo sorteo: {next_dt.strftime('%d/%m/%Y')}")

                st.markdown("#### 🎟️ Ajusta tu ticket")
                dfv = _pd.DataFrame([{
                    "Elegir": (v in chosen),
                    "Tipo": v["desc"],
                    "k": v["k"],
                    "Joker": "Sí" if v["joker"] else "No",
                    "Números": ", ".join(map(str, v["nums"])),
                    "Lift": float(v["lift"]),
                    "Prob. base": prob_to_1_in_x(prob_base_k(v["k"])),
                    "Prob. ajustada": prob_to_1_in_x(prob_base_k(v["k"])*v["lift"]),
                    "Coste (€)": round(v["cost"],2)
                } for v in variants])

                edited = st.data_editor(
                    dfv, use_container_width=True, height=360,
                    column_config={"Elegir": st.column_config.CheckboxColumn()},
                    disabled=["Tipo","k","Joker","Números","Lift","Prob. base","Prob. ajustada","Coste (€)"],
                    key=f"PRIMI_ticket_editor"
                )
                mask = edited["Elegir"].fillna(False).values
                total_cost2 = float(edited.loc[mask, "Coste (€)"].sum())
                picks_idx = [i for i,flag in enumerate(mask) if flag]
                picks = [variants[i] for i in picks_idx]
                total_p2 = sum(prob_base_k(v["k"])*v["lift"] for v in picks)
                st.info(f"**Total**: {len(picks)} apuestas · **Coste**: {total_cost2:.2f} € · **Prob. ajustada (sum)**: {total_p2:.6f} (~{prob_to_1_in_x(total_p2)})")

                mark = st.checkbox("✅ Marcar este ticket como jugado (Bitácora)", key=f"PRIMI_mark_played")
                if mark and st.button("Guardar en bitácora", type="primary", key=f"PRIMI_save_play"):
                    ok = True
                    for v in picks:
                        row = {
                            "ts": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "juego": "Primitiva",
                            "fecha_sorteo": next_dt.strftime("%d/%m/%Y"),
                            "nums": "-".join(map(str, v["nums"])),
                            "k": v["k"], "joker": int(v["joker"]),
                            "coste": float(v["cost"]),
                            "lift": float(v["lift"]),
                            "prob_base": float(prob_base_k(v["k"])),
                            "prob_ajustada": float(prob_base_k(v["k"])*v["lift"]),
                            "bank": float(bank_pr),
                            "version_modelo": "A2-2025-09-03",
                        }
                        ok = ok and bitacora_append_row(row)
                    st.success("Ticket guardado en Bitácora." if ok else "No se pudo guardar en Bitácora.")

        # Tabs internas de contexto
        base_w = np.array([w_blend.get(i,0.0) for i in range(1,50)])
        p_norm = base_w / (base_w.sum() if base_w.sum()>0 else 1.0)
        p_top6 = np.sort(p_norm)[-6:].mean()
        z_best = zscore_combo(A2s_6[0], w_blend) if A2s_6 else 0.0

        st.markdown("---")
        t1, t2 = st.tabs(["Métricas", "Ventana de referencia"])
        with t1:
            colA, colB = st.columns(2)
            colA.metric("Señal media A2 (z-score)", f"{z_best:.3f}")
            colB.metric("Intensidad media de pesos (top-6)", f"{p_top6:.3%}")
            st.caption("Métricas orientativas; la lotería es aleatoria por naturaleza.")
        with t2:
            st.dataframe(base[["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]].tail(min(24, len(base))),
                         use_container_width=True, height=280)
# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader(f"Bonoloto · Ticket Óptimo (EV/€)")
    wizard_bo = st.toggle("🪄 Modo asistido (wizard)", value=True,
                          help="Guía paso a paso: Datos → Calcular → 🏆 Ticket → Confirmar",
                          key="bo_wizard")

    df_b_full = load_sheet_df("sheet_id_bono","worksheet_historico_bono","HistoricoBono")
    last_rec_b = df_b_full.tail(1) if not df_b_full.empty else pd.DataFrame()

    fuente_b = st.radio("Origen de datos del último sorteo (Bonoloto)",
                        ["Usar último del histórico", "Introducir manualmente"],
                        index=0 if not df_b_full.empty else 1, horizontal=True, key="bo_src")

    do_calc_b = False
    if fuente_b == "Usar último del histórico" and not df_b_full.empty:
        rowb = last_rec_b.iloc[0]
        last_dt_b = pd.to_datetime(rowb["FECHA"])
        nums_b = [int(rowb["N1"]), int(rowb["N2"]), int(rowb["N3"]), int(rowb["N4"]), int(rowb["N5"]), int(rowb["N6"])]
        comp_b = int(rowb["Complementario"]) if not pd.isna(rowb["Complementario"]) else 18
        rein_b = int(rowb["Reintegro"]) if not pd.isna(rowb["Reintegro"]) else 0
        st.info(f"Usando el último sorteo del histórico (Bonoloto): **{last_dt_b.strftime('%d/%m/%Y')}** · Números: {nums_b} · C: {comp_b} · R: {rein_b}")
        do_calc_b = st.button("Calcular · Bonoloto", type="primary", key="bo_calc_btn")
        if wizard_bo and not do_calc_b:
            do_calc_b = True
    else:
        with st.form("bo_form_manual"):
            c1, c2, c3 = st.columns([1,1,1])
            last_date_b = c1.date_input("Fecha último sorteo (Bonoloto)", value=datetime.today().date(), key="bo_date")
            rein_b = c2.number_input("Reintegro (0-9)", min_value=0, max_value=9, value=2, step=1, key="bo_rein")
            comp_b = c3.number_input("Complementario (1-49)", min_value=1, max_value=49, value=18, step=1, key="bo_comp")
            cols = st.columns(6)
            defaults_b = [5,6,8,23,46,47]
            nums_b = [cols[i].number_input(f"N{i+1} (Bono)", 1, 49, defaults_b[i], 1, key=f"bo_n{i+1}") for i in range(6)]
            do_calc_b = st.form_submit_button("Calcular · Bonoloto")

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
                    st.info("La fecha ya estaba en el histórico (Bonoloto). Se han usado los datos existentes.")
                else:
                    last_dt_b = pd.to_datetime(last_date_b)

    if do_calc_b:
        if len(set(nums_b))!=6:
            st.error("Los 6 números deben ser distintos."); st.stop()

        # Próximo sorteo (casi diario)
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
        base_b = base_b.sort_values("FECHA").tail(st.session_state.get("pr_win", WINDOW_DRAWS_DEF)).reset_index(drop=True)
        base_b["weekday"] = base_b["FECHA"].dt.weekday

        # Parámetros de señal
        WINDOW_DRAWS    = st.session_state.get("pr_win", WINDOW_DRAWS_DEF)
        HALF_LIFE_DAYS  = st.session_state.get("pr_hl", HALF_LIFE_DAYS_DEF)
        DAY_BLEND_ALPHA = st.session_state.get("pr_dba", DAY_BLEND_ALPHA_DEF)
        ALPHA_DIR       = st.session_state.get("pr_alphadir", ALPHA_DIR_DEF)
        MU_PENALTY      = st.session_state.get("pr_mu", MU_PENALTY_DEF)
        LAMBDA_DIVERSIDAD = st.session_state.get("pr_lambda", LAMBDA_DIVERSIDAD_DEF)

        # Pesos y señal
        w_glob_b = weighted_counts_nums(base_b, last_dt_b, HALF_LIFE_DAYS)
        w_day_b  = weighted_counts_nums(base_b[base_b["weekday"]==weekday], last_dt_b, HALF_LIFE_DAYS)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        # A1 (neutra por día)
        A1b_6 = A1_FIJAS_BONO.get(weekday, [4,24,35,37,40,46])

        # Determinismo seed
        seed_val_b = abs(hash(f"BONOLOTO|{last_dt_b.date()}|{tuple(sorted(nums_b))}|{comp_b}|{rein_b}|k={st.session_state.get('pr_k',6)}|multi={st.session_state.get('pr_use_multi',True)}|alpha={DAY_BLEND_ALPHA}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}")) % (2**32 - 1)
        np.random.seed(seed_val_b)

        # Candidatos A2
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

        # Greedy top-3
        A2s_b_6 = []
        if pool_b:
            A2s_b_6 = [pool_b[0]]
            while len(A2s_b_6) < 3 and len(A2s_b_6) < len(pool_b):
                bestC, bestVal = None, -1e9
                for c in pool_b:
                    if tuple(c) in [tuple(x) for x in A2s_b_6]: continue
                    div_pen = sum(overlap_ratio(c,s) for s in A2s_b_6)
                    val = score_combo(c, w_blend_b, ALPHA_DIR, MU_PENALTY) - LAMBDA_DIVERSIDAD*div_pen
                    if val>bestVal: bestVal, bestC = val, c
                if bestC is None: break
                A2s_b_6.append(bestC)

        if not A2s_b_6:
            st.warning("No se generaron A2 con las restricciones actuales."); st.stop()

        # Render del ticket (sin Joker en Bonoloto)
        st.markdown("### 🏆 Tu ticket recomendado")
        variants = []
        k_set_b = (6,7,8) if st.session_state.get("pr_use_multi", True) else (6,)
        for a2 in A2s_b_6:
            variants += build_variants_for_a2(
                a2, w_blend_b, ALPHA_DIR, MU_PENALTY,
                k_set=k_set_b, price_simple=precio_simple_bo,
                allow_joker=False, scoreJ=0.0
            )
        chosen, total_cost, total_p = choose_portfolio(variants, bank_bo)
        if not chosen:
            st.warning("No se pudo formar un ticket dentro del banco. Ajusta parámetros.")
        else:
            best = max(chosen, key=lambda v: v["p_adj"])
            Ltxt = format_lift(best["lift"])
            pB   = prob_to_1_in_x(prob_base_k(best["k"]))
            pA   = prob_to_1_in_x(prob_base_k(best["k"])*best["lift"])
            st.success(f"**Apuesta Óptima (EV/€)**: {best['nums']} · {best['desc']} · Lift {Ltxt}")
            c1,c2,c3 = st.columns(3)
            c1.metric("Prob. base", pB)
            c2.metric("Prob. ajustada", pA)
            c3.metric("Coste", f"{best['cost']:.2f} €")
            st.caption(f"Próximo sorteo: {next_dt_b.strftime('%d/%m/%Y')}")

            st.markdown("#### 🎟️ Ajusta tu ticket")
            dfv_b = pd.DataFrame([{
                "Elegir": (v in chosen),
                "Tipo": v["desc"],
                "k": v["k"],
                "Números": ", ".join(map(str, v["nums"])),
                "Lift": float(v["lift"]),
                "Prob. base": prob_to_1_in_x(prob_base_k(v["k"])),
                "Prob. ajustada": prob_to_1_in_x(prob_base_k(v["k"])*v["lift"]),
                "Coste (€)": round(v["cost"],2)
            } for v in variants])

            edited_b = st.data_editor(
                dfv_b, use_container_width=True, height=360,
                column_config={"Elegir": st.column_config.CheckboxColumn()},
                disabled=["Tipo","k","Números","Lift","Prob. base","Prob. ajustada","Coste (€)"],
                key="BONO_ticket_editor"
            )
            mask_b = edited_b["Elegir"].fillna(False).values
            total_cost2_b = float(edited_b.loc[mask_b, "Coste (€)"].sum())
            picks_idx_b = [i for i,flag in enumerate(mask_b) if flag]
            picks_b = [variants[i] for i in picks_idx_b]
            total_p2_b = sum(prob_base_k(v["k"])*v["lift"] for v in picks_b)
            st.info(f"**Total**: {len(picks_b)} apuestas · **Coste**: {total_cost2_b:.2f} € · **Prob. ajustada (sum)**: {total_p2_b:.6f} (~{prob_to_1_in_x(total_p2_b)})")

            mark_b = st.checkbox("✅ Marcar este ticket como jugado (Bitácora)", key="BONO_mark_played")
            if mark_b and st.button("Guardar en bitácora", type="primary", key="BONO_save_play"):
                okb = True
                for v in picks_b:
                    row = {
                        "ts": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "juego": "Bonoloto",
                        "fecha_sorteo": next_dt_b.strftime("%d/%m/%Y"),
                        "nums": "-".join(map(str, v["nums"])),
                        "k": v["k"], "joker": 0,
                        "coste": float(v["cost"]),
                        "lift": float(v["lift"]),
                        "prob_base": float(prob_base_k(v["k"])),
                        "prob_ajustada": float(prob_base_k(v["k"])*v["lift"]),
                        "bank": float(bank_bo),
                        "version_modelo": "A2-2025-09-03",
                    }
                    okb = okb and bitacora_append_row(row)
                st.success("Ticket guardado en Bitácora." if okb else "No se pudo guardar en Bitácora.")

        # Tabs internas de contexto
        base_wb = np.array([w_blend_b.get(i,0.0) for i in range(1,50)])
        p_normb = base_wb / (base_wb.sum() if base_wb.sum()>0 else 1.0)
        p_top6b = np.sort(p_normb)[-6:].mean()
        z_best_b = zscore_combo(A2s_b_6[0], w_blend_b) if A2s_b_6 else 0.0

        st.markdown("---")
        tb1, tb2 = st.tabs(["Métricas", "Ventana de referencia"])
        with tb1:
            colA, colB = st.columns(2)
            colA.metric("Señal media A2 (z-score)", f"{z_best_b:.3f}")
            colB.metric("Intensidad media de pesos (top-6)", f"{p_top6b:.3%}")
            st.caption("Métricas orientativas.")
        with tb2:
            st.dataframe(base_b[["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]].tail(min(24, len(base_b))),
                         use_container_width=True, height=280)

# =========================== TUTORIAL ===========================
with tab_tutorial:
    st.header("📘 Tutorial rápido")
    st.markdown("""
**Qué hace este recomendador**  
- Calcula un **Lift ×N** para cada combinación A2 (señal reciente + mezcla por día).
- Convierte esa señal en **probabilidad ajustada** sobre la probabilidad base del juego.  
- Te devuelve **UN ticket recomendado** (tu mejor apuesta por €) dado tu **banco** y preferencias (k/Joker).

**Conceptos clave (sin tecnicismos)**  
- **k**: números por apuesta. Subir k **no hace más eficiente** la apuesta por euro; solo baja/redistribuye la varianza. Lo eficiente, por defecto, es **k=6**.  
- **Lift ×N**: multiplicador de tu probabilidad vs. jugar al azar. Ej.: ×1.60 ≈ 60% mejor que una combinación media.  
- **Prob. base (k)**: se muestra como “**1 entre N**”. Con k=7, N≈1.997.684 en Primitiva.  
- **Prob. ajustada**: `Prob. base × Lift` (también en “1 entre N”).  
- **Joker** (solo Primitiva): vía extra de premio (independiente del 6). Lo sugerimos con ⭐ si la señal lo respalda.

**Cómo usarlo paso a paso**  
1) Elige la **fuente** del último sorteo (Histórico o Manual) y pulsa **Calcular**.  
2) Revisa tu **Apuesta Óptima (EV/€)**: números, k, Joker, coste y probabilidades (base/ajustada).  
3) **Ajusta** el ticket (puedes añadir/quitar variantes recomendadas) según tu banco.  
4) Marca **“Guardar en Bitácora”** para registrar lo que has jugado (sirve para medir y mejorar el modelo).  

**Interpretación rápida**  
- “**Prob. base: 1 entre X**” → lo que marca la combinación por puro azar (con tu k).  
- “**Prob. ajustada: 1 entre Y**” → lo que estimamos con tu **Lift** (señal).  
- Si añades más apuestas, sube la **probabilidad total** aproximadamente de forma lineal con el coste, pero **no cambia la eficiencia por €**.

**Importante**  
- La lotería es aleatoria; este sistema **no garantiza** premios.  
- El objetivo es una **estrategia disciplinada y repetible**, con decisiones claras y registradas (Bitácora).
""")
