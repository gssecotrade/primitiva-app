# app.py — Francisco Cabrera · Predictor de La Primitiva & Bonoloto
# UI moderno + k-múltiple + determinismo + Google Sheets (read/write) + métricas + Joker por apuesta

import math
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from datetime import datetime, timedelta

import gspread
from google.oauth2.service_account import Credentials

# -------------------------- ESTILO / BRANDING --------------------------
st.set_page_config(page_title="Francisco Cabrera · Predictor de La Primitiva & Bonoloto",
                   page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif !important; }
.block-container { padding-top: 1.2rem; }
h1, h2, h3 { font-weight: 600; }
.sidebar .sidebar-content { width: 360px; }
.small-muted { color: #94a3b8; font-size: 0.85rem; }
.kpill { display:inline-block; background:#0ea5e9; color:white; padding:2px 8px; border-radius:99px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# Header (branding)
st.markdown("""
### **Francisco Cabrera · Predictor de La Primitiva & Bonoloto**
<span class="small-muted">Estrategia A1/A2 con ventana móvil, mezcla por día, diversidad, selección determinista y recomendación de Joker por apuesta. Fuente: Google Sheets.</span>
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
THRESH_N = [                       # umbrales para sugerir nº de A2
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
    z = zscore_combo(combo, weights)
    zN = minmax_norm(z, -1.5, 1.5)
    if rein_dict:
        top = max(rein_dict.values())
        reinN = minmax_norm(top, 0.0, top if top>0 else 1.0)
    else:
        reinN = 0.0
    return 0.6*zN + 0.4*reinN

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
    # Busca en bloque TOML y si no, en raíz
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
        sh = gc.open_by_key(sid); ws = sh.worksheet(wsn)
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

# -------------------------- SIDEBAR PARÁMETROS --------------------------
with st.sidebar:
    st.subheader("Parámetros · Primitiva")
    bank_pr = st.number_input("Banco (€) · Primitiva", min_value=0, value=10, step=1)
    vol_pr  = st.selectbox("Volatilidad · Primitiva", ["Low","Medium","High"], index=1)
    precio_simple = st.number_input("Precio por apuesta simple (€)", min_value=0.0, value=1.0, step=0.5, format="%.2f")

    st.markdown("---")
    st.subheader("Apuesta múltiple (opcional)")
    use_multi = st.checkbox("Usar apuesta múltiple (k>6)", value=True)
    k_nums    = st.slider("Números por apuesta (k)", min_value=6, max_value=8, value=8, step=1, disabled=not use_multi)

    st.markdown("---")
    st.subheader("Joker (Primitiva)")
    use_joker   = st.checkbox("Activar recomendaciones de Joker por apuesta", value=True)
    joker_thr   = st.slider("Umbral para recomendar Joker", 0.00, 1.00, 0.65, 0.01,
                             help="Recomendamos Joker en las A2 con puntuación ≥ umbral.")
    precio_joker  = st.number_input("Precio Joker (€)", min_value=1.0, value=1.0, step=1.0, format="%.2f")

    st.markdown("---")
    with st.expander("Parámetros avanzados (simulación)", expanded=False):
        st.caption("Ajustes para pruebas. La recomendación estándar usa los valores por defecto.")
        WINDOW_DRAWS    = st.slider("Ventana (nº de sorteos usados)", 12, 120, WINDOW_DRAWS_DEF, 1)
        HALF_LIFE_DAYS  = float(st.slider("Vida media temporal (días)", 15, 180, int(HALF_LIFE_DAYS_DEF), 1))
        DAY_BLEND_ALPHA = float(st.slider("Mezcla por día (α)", 0.0, 1.0, float(DAY_BLEND_ALPHA_DEF), 0.05))
        ALPHA_DIR       = float(st.slider("Suavizado pseudo-frecuencias (α_dir)", 0.00, 1.00, float(ALPHA_DIR_DEF), 0.01))
        MU_PENALTY      = float(st.slider("Penalización 'popularidad'", 0.0, 2.0, float(MU_PENALTY_DEF), 0.1))
        LAMBDA_DIVERSIDAD = float(st.slider("Peso diversidad (λ)", 0.0, 2.0, float(LAMBDA_DIVERSIDAD_DEF), 0.1))

    st.markdown("---")
    st.subheader("Parámetros · Bonoloto")
    bank_bo = st.number_input("Banco (€) · Bonoloto", min_value=0, value=10, step=1, key="bank_bono")
    vol_bo  = st.selectbox("Volatilidad · Bonoloto", ["Low","Medium","High"], index=1, key="vol_bono")

# -------------------------- TABS JUEGOS --------------------------
tab_primi, tab_bono = st.tabs(["La Primitiva", "Bonoloto"])

# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader(f"La Primitiva · Recomendador A2 · k={'múltiple' if (use_multi and k_nums>6) else '6'}")

    # Entrada de datos
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

    if do_calc:
        if len(set(nums))!=6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        # Próximo día (Mon→Thu, Thu→Sat, Sat→Mon)
        last_dt = pd.to_datetime(last_date)
        wd = last_dt.weekday()
        if wd==0: next_dt, next_dayname = last_dt + timedelta(days=3), "Thursday"
        elif wd==3: next_dt, next_dayname = last_dt + timedelta(days=2), "Saturday"
        elif wd==5: next_dt, next_dayname = last_dt + timedelta(days=2), "Monday"
        else:
            st.error("La fecha debe ser Lunes, Jueves o Sábado.")
            st.stop()

        st.info(f"Próximo sorteo: **{next_dt.date().strftime('%d/%m/%Y')}** ({next_dayname})")

        # Carga histórico
        df_hist = load_sheet_df("sheet_id","worksheet_historico","Historico")
        base = df_hist[df_hist["FECHA"]<=last_dt].copy()

        # Anti-duplicados + ventana (si no está, se usa la entrada actual como último sorteo para el análisis)
        if base.empty or not (base["FECHA"].dt.date == last_dt.date()).any():
            newrow = {
                "FECHA": last_dt, "N1": nums[0], "N2": nums[1], "N3": nums[2],
                "N4": nums[3], "N5": nums[4], "N6": nums[5],
                "Complementario": comp, "Reintegro": rein
            }
            base = pd.concat([base, pd.DataFrame([newrow])], ignore_index=True)

        base = base.sort_values("FECHA").tail(WINDOW_DRAWS).reset_index(drop=True)
        base["weekday"] = base["FECHA"].dt.weekday

        # Pesos
        weekday_mask = dayname_to_weekday(next_dayname)
        w_glob = weighted_counts_nums(base, last_dt, HALF_LIFE_DAYS)
        w_day  = weighted_counts_nums(base[base["weekday"]==weekday_mask], last_dt, HALF_LIFE_DAYS)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        # Reintegros: referencia A1 por día y dinámico (A2)
        rein_dict = compute_rein_probs(base, last_dt, weekday_mask, HALF_LIFE_DAYS, DAY_BLEND_ALPHA)
        rein_sug_dynamic = max(rein_dict, key=lambda r: rein_dict[r]) if rein_dict else 0
        rein_sug_A1_ref  = REIN_FIJOS_PRIMI.get(next_dayname, rein_sug_dynamic)

        # A1
        A1_6 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])
        A1_k = expand_to_k(A1_6, w_blend, k_nums if (use_multi and k_nums>6) else 6)

        # Determinismo (seed estable por inputs)
        seed_val = abs(hash(f"PRIMITIVA|{str(last_date)}|{tuple(sorted(nums))}|{comp}|{rein}|k={k_nums}|multi={use_multi}|alpha={DAY_BLEND_ALPHA}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}")) % (2**32 - 1)
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
        zA2 = zscore_combo(best6, w_blend) if best6 else 0.0
        n_sugerido = pick_n(zA2, bank_pr, vol_pr, THRESH_N)

        # Greedy y expansión a k
        A2s_6 = greedy_select(pool, w_blend, n_sugerido, ALPHA_DIR, MU_PENALTY, LAMBDA_DIVERSIDAD)
        A2s_k = [expand_to_k(a2, w_blend, k_nums) if (use_multi and k_nums>6) else a2 for a2 in A2s_6]

        # Joker por apuesta (score y recomendación individual)
        rows = []
        total_simples = 0
        joker_count = 0

        # A1 (informativo; Joker se recomienda solo para A2)
        rows.append({
            "Tipo":"A1",
            "Números": A1_k if (use_multi and k_nums>6) else A1_6,
            "k": k_nums if (use_multi and k_nums>6) else 6,
            "Simples": comb(k_nums,6) if (use_multi and k_nums>6) else 1,
            "Joker": "—",
            "ScoreJ": "—"
        })
        total_simples += (comb(k_nums,6) if (use_multi and k_nums>6) else 1)

        # A2s
        for i, a2 in enumerate(A2s_k, start=1):
            base6 = A2s_6[i-1]  # para evaluar la señal sobre la base de 6
            sc = joker_score(base6, w_blend, rein_dict) if use_joker else 0.0
            flag = (use_joker and sc >= joker_thr)
            if (use_multi and k_nums>6):
                simples = comb(k_nums,6)
                total_simples += simples
                rows.append({
                    "Tipo": f"A2 #{i} (k={k_nums})",
                    "Números": a2,
                    "k": k_nums,
                    "Simples": simples,
                    "Joker": "⭐" if flag else "—",
                    "ScoreJ": f"{sc:.2f}"
                })
            else:
                total_simples += 1
                rows.append({
                    "Tipo": f"A2 #{i}",
                    "Números": a2,
                    "k": 6,
                    "Simples": 1,
                    "Joker": "⭐" if flag else "—",
                    "ScoreJ": f"{sc:.2f}"
                })
            if flag: joker_count += 1

        # Coste total: simples × precio_simple + joker_recomendados × 1€
        coste_total = total_simples * float(precio_simple) + joker_count * float(precio_joker)

        # --------- UI (pestañas internas) ---------
        subtab1, subtab2, subtab3, subtab4 = st.tabs(["Recomendación", "Apuestas", "Métricas", "Ventana de referencia"])

        with subtab1:
            cA, cB, cC = st.columns([1,1,1])
            cA.metric("Boletos (A1 + A2)", n_sugerido)
            cB.metric("Coste estimado (€)", f"{coste_total:,.2f}")
            cC.metric("Confianza (señal)", conf_label(zA2))

            if use_multi and k_nums>6:
                st.write(f"**A1 (ancla, k={k_nums})**: {A1_k}")
            else:
                st.write(f"**A1 (ancla, 6)**: {A1_6}")

            for r in rows[1:]:
                tag = f" (k={r['k']})" if r["k"]>6 else ""
                star = " — ⭐ Joker" if r["Joker"]=="⭐" else ""
                st.write(f"**{r['Tipo']}{tag}**: {list(r['Números'])}{star}  ·  ScoreJ={r['ScoreJ']}")

            st.caption(f"Tamaño de apuesta (k): {k_nums} → {comb(k_nums,6) if (use_multi and k_nums>6) else 1} combinaciones simples por boleto.")
            st.write(f"**Reintegro A1 (referencia día)**: {rein_sug_A1_ref}  ·  **Reintegro dinámico (A2)**: {rein_sug_dynamic}")
            st.write(f"**Joker recomendados (A2)**: {joker_count} · **Umbral**: {joker_thr:.2f}")

        with subtab2:
            df_out = pd.DataFrame([{
                "Tipo":rows[0]["Tipo"], "k":rows[0]["k"], "Simples":rows[0]["Simples"],
                "Números": ", ".join(map(str, rows[0]["Números"])), "Joker": rows[0]["Joker"], "ScoreJ": rows[0]["ScoreJ"]
            }] + [{
                "Tipo":r["Tipo"], "k":r["k"], "Simples":r["Simples"],
                "Números": ", ".join(map(str, r["Números"])), "Joker": r["Joker"], "ScoreJ": r["ScoreJ"]
            } for r in rows[1:]])
            st.dataframe(df_out, use_container_width=True, height=320)
            st.download_button("Descargar combinaciones · Primitiva (CSV)",
                               data=df_out.to_csv(index=False).encode("utf-8"),
                               file_name="primitiva_recomendaciones.csv", mime="text/csv")

        with subtab3:
            st.markdown("**Señal media A2 (z-score):** {:.3f}".format(zA2))
            base_w = np.array([w_blend.get(i,0.0) for i in range(1,50)])
            p_norm = base_w / (base_w.sum() if base_w.sum()>0 else 1.0)
            p_top6 = np.sort(p_norm)[-6:].mean()
            st.markdown(f"**Intensidad media de pesos (top-6):** {p_top6:.3%}")
            st.caption("Las métricas son orientativas; la lotería es aleatoria (independiente por sorteo).")

        with subtab4:
            st.dataframe(base[["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]].tail(min(24, len(base))),
                         use_container_width=True, height=280)

        # Guardar histórico si procede
        if save_hist:
            ok = append_row_if_new("sheet_id","worksheet_historico","Historico", {
                "FECHA":last_dt, "N1":nums[0], "N2":nums[1], "N3":nums[2], "N4":nums[3], "N5":nums[4], "N6":nums[5],
                "Complementario": comp, "Reintegro": rein
            })
            if ok: st.success("✅ Histórico (Primitiva) actualizado.")
            else:  st.info("ℹ️ No se añadió al histórico (duplicado o acceso restringido).")

# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader(f"Bonoloto · Recomendador A2 · k={'múltiple' if (use_multi and k_nums>6) else '6'}")

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
        if len(set(nums_b))!=6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        last_dt_b = pd.to_datetime(last_date_b)
        next_dt_b = last_dt_b + timedelta(days=1)  # aprox (sorteo casi diario)
        weekday = next_dt_b.weekday()
        st.info(f"Próximo sorteo (aprox.): **{next_dt_b.date().strftime('%d/%m/%Y')}** ({next_dt_b.day_name()})")

        df_b = load_sheet_df("sheet_id_bono","worksheet_historico_bono","HistoricoBono")
        base_b = df_b[df_b["FECHA"]<=last_dt_b].copy()
        if base_b.empty or not (base_b["FECHA"].dt.date == last_dt_b.date()).any():
            new_b = {"FECHA": last_dt_b, "N1": nums_b[0], "N2": nums_b[1], "N3": nums_b[2],
                     "N4": nums_b[3], "N5": nums_b[4], "N6": nums_b[5],
                     "Complementario": comp_b, "Reintegro": rein_b}
            base_b = pd.concat([base_b, pd.DataFrame([new_b])], ignore_index=True)

        base_b = base_b.sort_values("FECHA").tail(WINDOW_DRAWS).reset_index(drop=True)
        base_b["weekday"] = base_b["FECHA"].dt.weekday

        w_glob_b = weighted_counts_nums(base_b, last_dt_b, HALF_LIFE_DAYS)
        w_day_b  = weighted_counts_nums(base_b[base_b["weekday"]==weekday], last_dt_b, HALF_LIFE_DAYS)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        A1b_6 = A1_FIJAS_BONO.get(weekday, [4,24,35,37,40,46])
        A1b_k = expand_to_k(A1b_6, w_blend_b, k_nums if (use_multi and k_nums>6) else 6)

        seed_val_b = abs(hash(f"BONOLOTO|{str(last_date_b)}|{tuple(sorted(nums_b))}|{comp_b}|{rein_b}|k={k_nums}|multi={use_multi}|alpha={DAY_BLEND_ALPHA}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}")) % (2**32 - 1)
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
        zA2_b = zscore_combo(best6_b, w_blend_b) if best6_b else 0.0
        n_b = pick_n(zA2_b, bank_bo, vol_bo, THRESH_N)

        A2s_b_6 = greedy_select(pool_b, w_blend_b, n_b, ALPHA_DIR, MU_PENALTY, LAMBDA_DIVERSIDAD)
        A2s_b_k = [expand_to_k(a2, w_blend_b, k_nums) if (use_multi and k_nums>6) else a2 for a2 in A2s_b_6]

        combos_por_boleto_b = comb(k_nums,6) if (use_multi and k_nums>6) else 1
        coste_b = precio_simple * (1 + len(A2s_b_k)) * combos_por_boleto_b  # A1 + A2s

        subB1, subB2, subB3, subB4 = st.tabs(["Recomendación", "Apuestas", "Métricas", "Ventana de referencia"])

        with subB1:
            cA, cB, cC = st.columns([1,1,1])
            cA.metric("Boletos (A1 + A2)", 1 + len(A2s_b_k))
            cB.metric("Coste estimado (€)", f"{coste_b:,.2f}")
            cC.metric("Confianza (señal)", conf_label(zA2_b))

            if use_multi and k_nums>6:
                st.write(f"**A1 (ancla, k={k_nums})**: {A1b_k}")
            else:
                st.write(f"**A1 (ancla, 6)**: {A1b_6}")

            for i, a2 in enumerate(A2s_b_k, start=1):
                tag = f" (k={k_nums})" if (use_multi and k_nums>6) else ""
                st.write(f"**A2 #{i}{tag}**: {list(a2)}")

            st.caption(f"Tamaño de apuesta (k): {k_nums} → {comb(k_nums,6) if (use_multi and k_nums>6) else 1} combinaciones simples por boleto.")
            st.write("**Joker**: No aplica en Bonoloto")

        with subB2:
            filas_b = [{
                "Tipo":"A1","k": k_nums if (use_multi and k_nums>6) else 6,
                "Simples": comb(k_nums,6) if (use_multi and k_nums>6) else 1,
                "Números": ", ".join(map(str, A1b_k if (use_multi and k_nums>6) else A1b_6))
            }]
            for i, a2 in enumerate(A2s_b_k, start=1):
                filas_b.append({"Tipo":f"A2-{i}","k": k_nums if (use_multi and k_nums>6) else 6,
                                "Simples": comb(k_nums,6) if (use_multi and k_nums>6) else 1,
                                "Números":", ".join(map(str,a2))})
            df_out_b = pd.DataFrame(filas_b)
            st.dataframe(df_out_b, use_container_width=True, height=320)
            st.download_button("Descargar combinaciones · Bonoloto (CSV)",
                               data=df_out_b.to_csv(index=False).encode("utf-8"),
                               file_name="bonoloto_recomendaciones.csv", mime="text/csv")

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

        if save_hist_b:
            okb = append_row_if_new("sheet_id_bono","worksheet_historico_bono","HistoricoBono", {
                "FECHA":last_dt_b, "N1":nums_b[0], "N2":nums_b[1], "N3":nums_b[2], "N4":nums_b[3], "N5":nums_b[4], "N6":nums_b[5],
                "Complementario": comp_b, "Reintegro": rein_b
            })
            if okb: st.success("✅ Histórico (Bonoloto) actualizado.")
            else:   st.info("ℹ️ No se añadió al histórico (duplicado o acceso restringido).")
