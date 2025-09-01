# app.py — Francisco Cabrera • Predictor de La Primitiva & Bonoloto (k-múltiple fijo + determinismo)
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import gspread
from google.oauth2.service_account import Credentials
import math
from datetime import datetime

# ---------- Setup página ----------
st.set_page_config(page_title="Francisco Cabrera · Predictor de La Primitiva & Bonoloto", page_icon="🎯", layout="wide")

# ---------- Parámetros de modelo (estables) ----------
WINDOW_DRAWS      = 24        # ventana móvil
HALF_LIFE_DAYS    = 60.0      # semivida temporal
DAY_BLEND_ALPHA   = 0.30      # mezcla señal por día vs global
ALPHA_DIR         = 0.30      # suavizado Dirichlet
MU_PENALTY        = 1.00      # castigo a popularidad
K_CANDIDATOS      = 3000      # tamaño muestreo de candidatos
MIN_DIV           = 0.60      # diversidad mínima vs A1
LAMBDA_DIVERSIDAD = 0.60      # fuerza de diversidad entre A2s
THRESH_N = [                  # mapeo señal → n (número de boletos)
    {"z": 0.50, "n": 6},
    {"z": 0.35, "n": 4},
    {"z": 0.20, "n": 3},
    {"z": 0.10, "n": 2},
    {"z":-999,  "n": 1},
]

# A1 fijas por día para PRIMITIVA (ancla calibrable)
A1_FIJAS_PRIMI   = {"Monday":[4,24,35,37,40,46], "Thursday":[1,10,23,39,45,48], "Saturday":[7,12,14,25,29,40]}
REIN_FIJOS_PRIMI = {"Monday":1, "Thursday":8, "Saturday":0}

# A1 base por día para BONOLOTO (arrancamos neutro; se calibrará tras rodar 8–12 semanas)
A1_FIJAS_BONO = {
    0: [4,24,35,37,40,46], 1: [4,24,35,37,40,46], 2: [4,24,35,37,40,46],
    3: [4,24,35,37,40,46], 4: [4,24,35,37,40,46], 5: [4,24,35,37,40,46], 6: [4,24,35,37,40,46],
}

# ---------- Utilidades ----------
def comb(n,k):
    if k<0 or k>n: return 0
    return math.comb(n,k)

def set_seed_from_inputs(date_str, nums, comp, rein, juego):
    """Semilla determinista: mismo input → misma salida."""
    key = f"{juego}|{date_str}|{tuple(sorted(nums))}|{comp}|{rein}"
    seed = abs(hash(key)) % (2**32 - 1)
    np.random.seed(seed)

def get_gcp_credentials():
    # Secrets recomendados: gcp_json (JSON completo) o gcp_service_account{...}
    if "gcp_json" in st.secrets:
        info = dict(st.secrets["gcp_json"])
        pk = info.get("private_key","")
        if isinstance(pk,str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n","\n")
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        return Credentials.from_service_account_info(info, scopes=scopes)

    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        pk = info.get("private_key","")
        if isinstance(pk,str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n","\n")
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        return Credentials.from_service_account_info(info, scopes=scopes)

    raise RuntimeError("Falta el bloque de credenciales en Secrets (gcp_json o gcp_service_account).")

def get_secret_key(name, group="gcp_service_account"):
    if name in st.secrets: return st.secrets[name]
    if group in st.secrets and name in st.secrets[group]:
        return st.secrets[group][name]
    return None

@st.cache_data(ttl=600, show_spinner=False)
def load_sheet_df(sheet_id_key, worksheet_key, default_ws):
    try:
        creds = get_gcp_credentials()
        gc    = gspread.authorize(creds)
        sid   = get_secret_key(sheet_id_key)
        wsn   = get_secret_key(worksheet_key) or default_ws
        if not sid:
            raise RuntimeError(f"No encuentro `{sheet_id_key}` en Secrets.")
        sh = gc.open_by_key(sid)
        ws = sh.worksheet(wsn)
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
        df = pd.DataFrame(rows)
        expected = ["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
        for c in expected:
            if c not in df.columns: df[c]=np.nan
        df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
        for c in ["N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]]
    except Exception as e:
        st.error(f"⚠️ No puedo abrir Google Sheets ({sheet_id_key}/{worksheet_key}). Detalle: {e}")
        return pd.DataFrame(columns=["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"])

def to_js_day(dayname):
    return 1 if dayname=="Monday" else 4 if dayname=="Thursday" else 6 if dayname=="Saturday" else -1

def time_weight(d, ref):
    delta = max(0, (ref - d).days)
    return float(np.exp(-np.log(2)/HALF_LIFE_DAYS * delta))

def weighted_counts_nums(df_in, ref):
    w = {i:0.0 for i in range(1,50)}
    for _, r in df_in.iterrows():
        tw = time_weight(r["FECHA"], ref)
        for c in ["N1","N2","N3","N4","N5","N6"]:
            if not pd.isna(r[c]): w[int(r[c])] += tw
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

def popularity_penalty(combo6):
    c = sorted(combo6)
    p_dates = sum(1 for x in c if x<=31)/6.0
    consec  = sum(1 for a,b in zip(c, c[1:]) if b==a+1)
    decades = [x//10 for x in c]; units = [x%10 for x in c]
    max_dec = max(Counter(decades).values()); max_unit = max(Counter(units).values())
    s = sum(c); roundness = 1.0/(1.0 + abs(s-120)/10.0)
    return 1.2*p_dates + 0.8*consec + 0.5*(max_dec-2 if max_dec>2 else 0) + 0.5*(max_unit-2 if max_unit>2 else 0) + 0.4*roundness

def score_combo6(combo6, weights):
    return sum(np.log(weights.get(n,0.0) + ALPHA_DIR) for n in combo6) - MU_PENALTY*popularity_penalty(combo6)

def terciles_ok(combo6):
    return any(1<=x<=16 for x in combo6) and any(17<=x<=32 for x in combo6) and any(33<=x<=49 for x in combo6)

def overlap_ratio(a,b): return len(set(a)&set(b))/6.0

def zscore_combo6(combo6, weights):
    allW = np.array([weights.get(i,0.0) for i in range(1,50)], dtype=float)
    meanW = float(allW.mean()); sdW = float(allW.std()) if allW.std()!=0 else 1e-6
    comboMean = float(np.mean([weights.get(n,0.0) for n in combo6])) if combo6 else 0.0
    return (comboMean - meanW)/sdW

def pick_n(z, bank, vol):
    adj = 0.05 if vol=="Low" else -0.05 if vol=="High" else 0.0
    for th in THRESH_N:
        if z >= th["z"] + adj:
            n = min(th["n"], int(bank))
            return max(1, n)
    return 1

def greedy_select(pool6, weights, n):
    if n<=0: return []
    sorted_pool = sorted(pool6, key=lambda c: score_combo6(c,weights), reverse=True)
    selected = [sorted_pool[0]]
    while len(selected)<n:
        bestC=None; bestVal=-1e9
        for c in sorted_pool:
            if any(tuple(c)==tuple(s) for s in selected): continue
            div_pen = sum(overlap_ratio(c,s) for s in selected)
            val = score_combo6(c,weights) - LAMBDA_DIVERSIDAD*div_pen
            if val>bestVal: bestVal=val; bestC=c
        if bestC is None: break
        selected.append(bestC)
    return selected

def expand_to_k(base6, weights, k):
    """Dado un base de 6 números, añadir mejores 'extras' (no presentes) hasta k."""
    if k<=6: return base6[:6]
    extras = [n for n in range(1,50) if n not in base6]
    extras_sorted = sorted(extras, key=lambda x: weights.get(x,0.0), reverse=True)
    take = extras_sorted[:max(0,k-6)]
    return sorted(list(set(base6) | set(take)))[:k]

def top_k_for_A1(A1_6, weights, k):
    return expand_to_k(A1_6, weights, k)

# ---------- Carga histórico ----------
with st.sidebar:
    st.markdown("### Parámetros · Primitiva")
    bank_pri  = st.number_input("Banco (€) · Primitiva", min_value=0, value=10, step=1, key="bank_pri")
    vol_pri   = st.selectbox("Volatilidad · Primitiva", ["Low","Medium","High"], index=1, key="vol_pri")
    precio_simple = st.number_input("Precio por apuesta simple (€)", min_value=0.0, value=1.0, step=0.5)
    st.markdown("---")
    st.markdown("### Apuesta múltiple (opcional)")
    use_multi = st.checkbox("Usar apuesta múltiple (k>6)", value=True)
    k_nums    = st.slider("Números por apuesta (k)", min_value=6, max_value=8, value=8, step=1, disabled=not use_multi)
    st.markdown("---")
    st.markdown("### Joker")
    precio_joker = st.number_input("Precio Joker (€)", min_value=0.0, value=1.0, step=0.5)

st.title("Francisco Cabrera · Predictor de La Primitiva & Bonoloto")
st.caption("Estrategia A1/A2 con ventana móvil, mezcla por día, diversidad y selección determinista. Fuente: Google Sheets.")

tab_primi, tab_bono = st.tabs(["La Primitiva", "Bonoloto"])

# =================== PRIMITIVA ===================
with tab_primi:
    st.subheader("La Primitiva · Recomendador A2 · k-múltiple")
    df_hist = load_sheet_df("sheet_id", "worksheet_historico", "Historico")
    if df_hist.empty:
        st.stop()

    with st.form("form_pri"):
        c1, c2 = st.columns(2)
        last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=pd.Timestamp.today().date())
        rein      = c2.number_input("Reintegro (0–9)", min_value=0, max_value=9, value=2, step=1)
        comp      = c2.number_input("Complementario (1–49)", min_value=1, max_value=49, value=18, step=1)
        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults = [5,6,8,23,46,47]
        nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"npr{i+1}") for i in range(6)]
        submitted = st.form_submit_button("Calcular recomendaciones · Primitiva")

    if submitted:
        if len(set(nums))!=6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        # Semilla determinista
        set_seed_from_inputs(str(last_date), nums, comp, rein, "PRIMITIVA")

        last_dt = pd.to_datetime(last_date)
        wd = last_dt.weekday()
        if   wd==0: next_dt, next_dayname = last_dt + pd.Timedelta(days=3), "Thursday"
        elif wd==3: next_dt, next_dayname = last_dt + pd.Timedelta(days=2), "Saturday"
        elif wd==5: next_dt, next_dayname = last_dt + pd.Timedelta(days=2), "Monday"
        else:
            st.error("La fecha debe ser Lunes, Jueves o Sábado.")
            st.stop()
        st.info(f"Próximo sorteo: **{next_dt.date().strftime('%d/%m/%Y')}** ({next_dayname})")

        base = df_hist[df_hist["FECHA"] <= last_dt].sort_values("FECHA").copy()

        # Ventana y pesos
        df_recent = base.tail(WINDOW_DRAWS).copy()
        df_recent["weekday"] = df_recent["FECHA"].dt.weekday
        w_glob = weighted_counts_nums(df_recent, last_dt)
        w_day  = weighted_counts_nums(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        # A1 (6) y su extensión a k
        A1_6 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])
        A1_k = top_k_for_A1(A1_6, w_blend, k_nums if use_multi else 6)

        # Generación de candidatos 6
        def random_combo6():
            pool = list(range(1,50)); out=[]
            while len(out)<6:
                i=np.random.randint(0,len(pool)); out.append(pool.pop(i))
            return sorted(out)

        cands, seen, tries = [], set(), 0
        while len(cands)<K_CANDIDATOS and tries < K_CANDIDATOS*50:
            c = tuple(random_combo6()); tries += 1
            if c in seen: continue
            seen.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1_6) > (1 - MIN_DIV): continue
            cands.append(c)
        cands = sorted(cands, key=lambda c: score_combo6(c, w_blend), reverse=True)
        pool = cands[:1000]

        bestA2 = list(pool[0]) if pool else []
        zA2 = zscore_combo6(bestA2, w_blend) if bestA2 else 0.0
        n_sugerido = pick_n(zA2, bank_pri, vol_pri)
        n_boletos  = n_sugerido
        A2s_6 = greedy_select(pool, w_blend, max(0, n_boletos))[:max(0,n_boletos)]

        # Extensión A2s a k
        if use_multi and k_nums>6:
            A2s_k = [expand_to_k(a2, w_blend, k_nums) for a2 in A2s_6]
        else:
            A2s_k = A2s_6

        # Reintegro recomendado
        wr_glob = weighted_counts_rei(df_recent, last_dt)
        wr_day  = weighted_counts_rei(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
        rei_scores = {r: DAY_BLEND_ALPHA*wr_day.get(r,0.0) + (1-DAY_BLEND_ALPHA)*wr_glob.get(r,0.0) for r in range(10)}
        rein_sug = max(rei_scores, key=lambda r: rei_scores[r]) if rei_scores else 0
        rein_ref = REIN_FIJOS_PRIMI.get(next_dayname,'')

        # Joker: automático (lock)
        joker_recomendado = (zA2 >= 0.35) and (vol_pri!="Low") and (bank_pri>=n_boletos)
        # Coste
        combos_por_boleto = comb(k_nums,6) if (use_multi and k_nums>6) else 1
        coste = precio_simple * combos_por_boleto * n_boletos + (precio_joker if joker_recomendado else 0.0)

        # ---- Presentación
        st.markdown("### Recomendación")
        m1,m2,m3 = st.columns(3)
        m1.metric("Boletos (A1 + A2)", n_boletos)
        m2.metric("Coste estimado (€)", f"{coste:,.2f}".replace(","," "))
        m3.metric("Confianza (señal)", "Alta" if zA2>=0.35 else "Media" if zA2>=0.20 else "Baja")

        if use_multi and k_nums>6:
            st.write(f"**A1 (ancla, k={k_nums})**: {A1_k}")
        else:
            st.write(f"**A1 (ancla, 6)**: {A1_6}")

        for i, a2 in enumerate(A2s_k, start=1):
            st.write(f"**A2 #{i}{' (k='+str(k_nums)+')' if (use_multi and k_nums>6) else ''}**: {list(a2)}")

        st.caption(f"Tamaño de apuesta (k): {k_nums} → {combos_por_boleto} combinaciones simples por boleto.")
        st.caption(f"Sugerencia automática de boletos según señal/presupuesto: {n_sugerido}.")
        st.caption(f"Reintegro (info): sugerido **{rein_sug}** · referencia del día **{rein_ref}**.")
        st.caption(f"Joker: **{'añadido' if joker_recomendado else 'no añadido'}** según criterio del modelo (automático).")

# =================== BONOLOTO ===================
with tab_bono:
    st.subheader("Bonoloto · Recomendador A2 · k-múltiple")
    df_bono = load_sheet_df("sheet_id_bono", "worksheet_historico_bono", "HistoricoBono")
    if df_bono.empty:
        st.stop()

    with st.form("form_bono"):
        c1, c2 = st.columns(2)
        last_date_b = c1.date_input("Fecha último sorteo (Bonoloto)", value=pd.Timestamp.today().date())
        rein_b  = c2.number_input("Reintegro (0–9)", min_value=0, max_value=9, value=2, step=1)
        comp_b  = c2.number_input("Complementario (1–49)", min_value=1, max_value=49, value=18, step=1)
        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults_b = [5,6,8,23,46,47]
        nums_b = [cols[i].number_input(f"N{i+1} (Bono)", 1, 49, defaults_b[i], 1, key=f"nbo{i+1}") for i in range(6)]
        submitted_b = st.form_submit_button("Calcular recomendaciones · Bonoloto")

    if submitted_b:
        if len(set(nums_b))!=6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        # determinismo
        set_seed_from_inputs(str(last_date_b), nums_b, comp_b, rein_b, "BONOLOTO")

        last_dt_b = pd.to_datetime(last_date_b)
        weekday = last_dt_b.weekday()
        next_dt_b = last_dt_b + pd.Timedelta(days=1) # aproximación general
        next_dayname_b = next_dt_b.day_name()
        st.info(f"Próximo sorteo (aprox.): **{next_dt_b.date().strftime('%d/%m/%Y')}** ({next_dayname_b})")

        base_b = df_bono[df_bono["FECHA"] <= last_dt_b].sort_values("FECHA").copy()
        df_recent_b = base_b.tail(WINDOW_DRAWS).copy()
        df_recent_b["weekday"] = df_recent_b["FECHA"].dt.weekday
        w_glob_b = weighted_counts_nums(df_recent_b, last_dt_b)
        w_day_b  = weighted_counts_nums(df_recent_b[df_recent_b["weekday"]==weekday], last_dt_b)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        A1b_6 = A1_FIJAS_BONO.get((weekday+1)%7, [4,24,35,37,40,46])
        A1b_k = top_k_for_A1(A1b_6, w_blend_b, k_nums if use_multi else 6)

        def random_combo6():
            pool = list(range(1,50)); out=[]
            while len(out)<6:
                i=np.random.randint(0,len(pool)); out.append(pool.pop(i))
            return sorted(out)

        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*50:
            c = tuple(random_combo6()); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1b_6) > (1 - MIN_DIV): continue
            cands_b.append(c)
        cands_b = sorted(cands_b, key=lambda c: score_combo6(c, w_blend_b), reverse=True)
        pool_b  = cands_b[:1000]

        bestA2_b = list(pool_b[0]) if pool_b else []
        zA2_b = zscore_combo6(bestA2_b, w_blend_b) if bestA2_b else 0.0
        n_b = pick_n(zA2_b, bank_pri, vol_pri)  # reutilizo política de n
        A2s_b_6 = greedy_select(pool_b, w_blend_b, max(0,n_b))[:max(0,n_b)]
        A2s_b_k = [expand_to_k(a2, w_blend_b, k_nums) for a2 in A2s_b_6] if (use_multi and k_nums>6) else A2s_b_6

        wr_glob_b = weighted_counts_rei(df_recent_b, last_dt_b)
        wr_day_b  = weighted_counts_rei(df_recent_b[df_recent_b["weekday"]==weekday], last_dt_b)
        rei_scores_b = {r: DAY_BLEND_ALPHA*wr_day_b.get(r,0.0) + (1-DAY_BLEND_ALPHA)*wr_glob_b.get(r,0.0) for r in range(10)}
        rein_sug_b = max(rei_scores_b, key=lambda r: rei_scores_b[r]) if rei_scores_b else 0

        combos_por_boleto_b = comb(k_nums,6) if (use_multi and k_nums>6) else 1
        coste_b = precio_simple * combos_por_boleto_b * n_b  # Bonoloto sin Joker

        st.markdown("### Recomendación")
        m1,m2,m3 = st.columns(3)
        m1.metric("Boletos (A1 + A2)", n_b)
        m2.metric("Coste estimado (€)", f"{coste_b:,.2f}".replace(","," "))
        m3.metric("Confianza (señal)", "Alta" if zA2_b>=0.35 else "Media" if zA2_b>=0.20 else "Baja")

        if use_multi and k_nums>6:
            st.write(f"**A1 (ancla, k={k_nums})**: {A1b_k}")
        else:
            st.write(f"**A1 (ancla, 6)**: {A1b_6}")
        for i,c in enumerate(A2s_b_k, start=1):
            st.write(f"**A2 #{i}{' (k='+str(k_nums)+')' if (use_multi and k_nums>6) else ''}**: {list(c)}")
        st.caption(f"Tamaño de apuesta (k): {k_nums} → {combos_por_boleto_b} combinaciones simples por boleto.")
        st.caption(f"Reintegro (info): sugerido **{rein_sug_b}**.")
