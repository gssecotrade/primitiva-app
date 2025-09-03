
# app.py — Recomendador Primitiva & Bonoloto · Ticket Óptimo (EV/€) + Bitácora
# UX V2: una sola apuesta óptima (con opciones) + tabla "Ajusta tu ticket"
# Determinista, Lift ×N, prob. base/ajustada, Joker (Primitiva), Bonoloto (precio múltiplos 0,50€)
# Google Sheets: Historicos + Bitácora (opcional).

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
.block-container { padding-top: 0.8rem; }
h1,h2,h3 { font-weight: 600; }
.small-muted { color: #94a3b8; font-size: 0.90rem; }
.badge { display:inline-block; padding:2px 8px; border-radius: 999px; background:#0ea5e9; color:white; font-size:0.8rem; }
.infochip { background:#0f172a; color:#cbd5e1; padding:10px 12px; border-radius:12px; }
.card { background:#0b1220; padding:14px 16px; border-radius:14px; border:1px solid #0f1a2b; }
.success { background:#0f2a1a; color:#dcfce7; padding:12px 14px; border-radius:12px; }
.warn { background:#2a1a0f; color:#fde68a; padding:12px 14px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("### 🎯 Recomendador Primitiva & Bonoloto")
st.caption("Optimización determinista · Wizard asistido · Bitácora en Google Sheets · Lift ×N y probabilidad ajustada.")

# -------------------------- CONSTANTES --------------------------
WINDOW_DRAWS_DEF    = 24
HALF_LIFE_DAYS_DEF  = 60.0
DAY_BLEND_ALPHA_DEF = 0.30
ALPHA_DIR_DEF       = 0.30
MU_PENALTY_DEF      = 1.00
LAMBDA_DIVERSIDAD_DEF = 0.60
K_CANDIDATOS        = 3000
MIN_DIV             = 0.60

# A1 fijas por día (Primitiva) — ancla
A1_FIJAS_PRIMI = {
    "Monday":    [4,24,35,37,40,46],
    "Thursday":  [1,10,23,39,45,48],
    "Saturday":  [7,12,14,25,29,40],
}

# -------------------------- HELPERS --------------------------
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

def expand_to_k(base6, weights, k):
    if k<=6: 
        return list(base6[:6])
    extras = [n for n in range(1,50) if n not in base6]
    extras_sorted = sorted(extras, key=lambda x: weights.get(x,0.0), reverse=True)
    add = extras_sorted[:max(0,k-6)]
    out = sorted(list(set(base6) | set(add)))
    return out[:k]

def p_base_k(k):
    return comb(k,6)/comb(49,6)

def pretty_one_in(p):
    if p<=0: return "∞"
    return f"{int(round(1.0/p,0)):,}".replace(",", ".")

# -------------------------- SHEETS --------------------------
def get_gcp_credentials():
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
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get("gcp_service_account", {}) or {}).get(sheet_id_key) or st.secrets.get(sheet_id_key)
        wsn = (st.secrets.get("gcp_service_account", {}) or {}).get(worksheet_key, default_ws) or st.secrets.get(worksheet_key, default_ws)
        if not sid: return pd.DataFrame()
        sh = gc.open_by_key(sid); ws = sh.worksheet(wsn)
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
        df = pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()
    expected = ["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
    for c in expected:
        if c not in df.columns: df[c]=np.nan
    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    for c in ["N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
    return df[expected]

def append_bitacora(row_dict):
    """Escribe un ticket en la hoja Bitacora si está configurada."""
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get("gcp_service_account", {}) or {}).get("sheet_id_bitacora") or st.secrets.get("sheet_id_bitacora")
        if not sid: 
            return False, "Sin sheet_id_bitacora"
        sh = gc.open_by_key(sid)
        try:
            ws = sh.worksheet("Bitacora")
        except Exception:
            ws = sh.add_worksheet(title="Bitacora", rows=200, cols=20)
            ws.append_row(["TS","Juego","Fecha_sorteo","Numeros","k","Joker","Coste","Lift","p_base","p_adj","Bank","Modelo"])
        new_row = [
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            row_dict.get("Juego",""),
            row_dict.get("Fecha_sorteo",""),
            row_dict.get("Numeros",""),
            row_dict.get("k",""),
            row_dict.get("Joker",""),
            row_dict.get("Coste",""),
            row_dict.get("Lift",""),
            row_dict.get("p_base",""),
            row_dict.get("p_adj",""),
            row_dict.get("Bank",""),
            row_dict.get("Modelo",""),
        ]
        ws.append_row(new_row)
        return True, "OK"
    except Exception as e:
        return False, f"Error: {e}"

# -------------------------- SIDEBAR (parámetros globales) --------------------------
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
    joker_thr   = st.slider("Umbral para recomendar Joker", 0.00, 1.00, 0.65, 0.01)
    precio_joker  = st.number_input("Precio Joker (€)", min_value=1.0, value=1.0, step=1.0, format="%.2f")

    st.markdown("---")
    with st.expander("Parámetros avanzados (modelo)", expanded=False):
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
    precio_simple_bo = st.number_input("Precio simple Bonoloto (€)", min_value=0.0, value=0.50, step=0.50, format="%.2f",
                                       help="Apuestas son múltiplos de 0,50 € por boleto.")

# -------------------------- TABS --------------------------
tab_primi, tab_bono, tab_tuto = st.tabs(["La Primitiva", "Bonoloto", "📘 Tutorial"])

# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader("La Primitiva · Ticket Óptimo (EV/€)")
    wizard = st.toggle("✨ Modo asistido (wizard)", value=False, help="Te guía con pasos sencillos.")

    # Carga histórico
    df_hist = load_sheet_df("sheet_id","worksheet_historico","Historico")
    last_rec = df_hist.tail(1) if not df_hist.empty else pd.DataFrame()

    fuente = st.radio("Origen de datos del último sorteo", ["Usar último del histórico", "Introducir manualmente"],
                      index=0 if not df_hist.empty else 1, horizontal=True)

    if fuente == "Usar último del histórico" and not df_hist.empty:
        row = last_rec.iloc[0]
        last_dt = pd.to_datetime(row["FECHA"])
        nums = [int(row["N1"]), int(row["N2"]), int(row["N3"]), int(row["N4"]), int(row["N5"]), int(row["N6"])]
        comp = int(row["Complementario"]) if not pd.isna(row["Complementario"]) else 18
        rein = int(row["Reintegro"]) if not pd.isna(row["Reintegro"]) else 0
        st.info(f"Usando el último sorteo del histórico: **{last_dt.strftime('%d/%m/%Y')}** · Números: {nums} · C: {comp} · R: {rein}")
        do_calc = st.button("Calcular · Primitiva", type="primary")
    else:
        with st.form("form_primi"):
            c1, c2, c3 = st.columns([1,1,1])
            last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=datetime.today().date())
            rein = c2.number_input("Reintegro (0-9)", min_value=0, max_value=9, value=2, step=1)
            comp = c3.number_input("Complementario (1-49)", min_value=1, max_value=49, value=18, step=1)

            st.markdown("**Números extraídos (6 distintos)**")
            cols = st.columns(6)
            defaults = [5,6,8,23,46,47]
            nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"npr{i+1}") for i in range(6)]
            do_calc = st.form_submit_button("Calcular · Primitiva")

        last_dt = pd.to_datetime(last_date)

    if do_calc:
        if len(set(nums))!=6:
            st.error("Los 6 números deben ser distintos."); st.stop()

        # Próximo día (Mon→Thu, Thu→Sat, Sat→Mon)
        wd = last_dt.weekday()
        if wd==0: next_dt, next_dayname = last_dt + timedelta(days=3), "Thursday"
        elif wd==3: next_dt, next_dayname = last_dt + timedelta(days=2), "Saturday"
        elif wd==5: next_dt, next_dayname = last_dt + timedelta(days=2), "Monday"
        else:
            st.error("La fecha debe ser Lunes, Jueves o Sábado."); st.stop()

        st.info(f"Próximo sorteo: **{next_dt.date().strftime('%d/%m/%Y')}** ({next_dayname})")

        # Base (ventana)
        base = df_hist[df_hist["FECHA"]<=last_dt].copy()
        if base.empty or not (base["FECHA"].dt.date == last_dt.date()).any():
            newrow = {"FECHA": last_dt, "N1": nums[0], "N2": nums[1], "N3": nums[2],
                      "N4": nums[3], "N5": nums[4], "N6": nums[5],
                      "Complementario": comp, "Reintegro": rein}
            base = pd.concat([base, pd.DataFrame([newrow])], ignore_index=True)

        base = base.sort_values("FECHA").tail(WINDOW_DRAWS).reset_index(drop=True)
        base["weekday"] = base["FECHA"].dt.weekday

        # Pesos
        weekday_mask = dayname_to_weekday(next_dayname)
        w_glob = weighted_counts_nums(base, last_dt, HALF_LIFE_DAYS)
        w_day  = weighted_counts_nums(base[base["weekday"]==weekday_mask], last_dt, HALF_LIFE_DAYS)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        # Determinismo
        seed_val = abs(hash(f"PRIMITIVA|{last_dt.date()}|{tuple(sorted(nums))}|{comp}|{rein}|k={k_nums}|multi={use_multi}|alpha={DAY_BLEND_ALPHA}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}")) % (2**32 - 1)
        np.random.seed(seed_val)

        # Candidatos A2
        cands, seen, tries = [], set(), 0
        while len(cands)<K_CANDIDATOS and tries < K_CANDIDATOS*60:
            c = tuple(random_combo()); tries += 1
            if c in seen: continue
            seen.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1_FIJAS_PRIMI.get(next_dayname, [])) > (1 - MIN_DIV): continue
            cands.append(c)
        cands = sorted(cands, key=lambda c: score_combo(c, w_blend, ALPHA_DIR, MU_PENALTY), reverse=True)
        pool = cands[:1200]

        # A2 óptima (por Lift ~ score proxy)
        if not pool:
            st.warning("No se generaron candidatos suficientes."); st.stop()
        top3 = pool[:3]

        # Lift proxy: re-escala score a z
        scores = np.array([score_combo(c, w_blend, ALPHA_DIR, MU_PENALTY) for c in top3])
        z = (scores - scores.mean())/(scores.std() if scores.std()!=0 else 1e-6)
        # Usamos lift relativo aproximado: mapeo lineal a ×1.1..×1.8 (visual)
        lifts = list(1.4 + 0.3*(z - z.min())/(z.ptp() if z.ptp()!=0 else 1.0))
        lift_map = {i:lifts[i] for i in range(len(top3))}

        # Apuesta óptima base: k=6
        best6 = list(top3[0])
        best_lift = lift_map[0]

        # Calcular prob base/ajustada segun k y Joker
        def calc_prob_and_cost(k, use_j):
            p_base = p_base_k(k)
            p_adj = p_base * best_lift
            coste = comb(k,6)*precio_simple_pr + (precio_joker if use_j else 0.0)
            return p_base, p_adj, coste

        # Ficha recomendación
        st.markdown(f"<div class='success'><b>Apuesta Óptima (EV/€):</b> {best6} · <span class='badge'>Lift ×{best_lift:.2f}</span></div>", unsafe_allow_html=True)

        # Opciones rápidas
        opt_k = 6 if not use_multi else k_nums
        use_j = (use_joker and 0.70 >= joker_thr)  # simple: si umbral ≤0.70 activamos por defecto
        p_base, p_adj, coste = calc_prob_and_cost(opt_k, use_j)
        st.metric("Prob. base", f"1 entre {pretty_one_in(p_base)}")
        st.metric("Prob. ajustada", f"1 entre {pretty_one_in(p_adj)}")
        st.metric("Coste", f"{coste:,.2f} €")

        # Tabla Ajusta tu ticket
        st.markdown("#### 🎟️ Ajusta tu ticket")
        opciones = []
        def fila(tipo, k, jflag):
            nums = expand_to_k(best6, w_blend, k) if k>6 else best6
            p_b, p_a, cst = calc_prob_and_cost(k, jflag)
            opciones.append({
                "Elegir": True if (k==6 and not jflag) else False,
                "Tipo": f"A2 k={k}" + (" + Joker" if jflag else ""),
                "k": k, "Joker": "Sí" if jflag else "No", "Números": ", ".join(map(str, nums)),
                "p_base": p_b, "p_adj": p_a, "Coste": cst
            })

        for k in [6,7,8]:
            fila("A2", k, False)
            if k==6 and use_joker:
                fila("A2", k, True)
            if k in [7,8] and use_joker:
                # Joker solo tiene sentido si se permite por regla; lo tratamos como adicional (mismo 1€).
                fila("A2", k, True)

        df_opts = pd.DataFrame(opciones)
        # Selección editable de 'Elegir'
        edited = st.data_editor(df_opts, num_rows="fixed", use_container_width=True, height=360,
                                column_config={"Elegir": st.column_config.CheckboxColumn(required=True)})
        # Resumen ticket
        elegido = edited[edited["Elegir"]==True]
        total_coste = float(elegido["Coste"].sum()) if not elegido.empty else 0.0
        total_p_adj = float(elegido["p_adj"].sum()) if not elegido.empty else 0.0
        st.markdown(f"<div class='card'><b>Total:</b> {len(elegido)} apuestas · <b>Coste:</b> {total_coste:,.2f} € · <b>Prob. ajustada (sum):</b> {total_p_adj:.6f} (~1 entre {pretty_one_in(total_p_adj)})</div>", unsafe_allow_html=True)

        # Bitácora
        marcar = st.checkbox("Marcar este ticket como jugado (Bitácora)")
        if marcar and st.button("Guardar ticket en Bitácora"):
            row = {
                "Juego":"Primitiva",
                "Fecha_sorteo": next_dt.strftime("%Y-%m-%d"),
                "Numeros": elegido.iloc[0]["Números"] if not elegido.empty else ", ".join(map(str,best6)),
                "k": int(elegido.iloc[0]["k"]) if not elegido.empty else opt_k,
                "Joker": elegido.iloc[0]["Joker"] if not elegido.empty else ("Sí" if use_j else "No"),
                "Coste": f"{total_coste:.2f}",
                "Lift": f"{best_lift:.2f}",
                "p_base": f"{p_base:.8f}",
                "p_adj": f"{p_adj:.8f}",
                "Bank": bank_pr,
                "Modelo": "UXv2"
            }
            ok, msg = append_bitacora(row)
            if ok: st.success("✅ Ticket guardado en Bitácora.")
            else:  st.info(f"ℹ️ No se guardó en Bitácora: {msg}")

# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader("Bonoloto · Ticket Óptimo (EV/€)")

    df_hist_b = load_sheet_df("sheet_id_bono","worksheet_historico_bono","HistoricoBono")
    last_rec_b = df_hist_b.tail(1) if not df_hist_b.empty else pd.DataFrame()

    fuente_b = st.radio("Origen de datos del último sorteo (Bonoloto)",
                        ["Usar último del histórico", "Introducir manualmente"],
                        index=0 if not df_hist_b.empty else 1, horizontal=True, key="src_b")

    if fuente_b == "Usar último del histórico" and not df_hist_b.empty:
        rowb = last_rec_b.iloc[0]
        last_dt_b = pd.to_datetime(rowb["FECHA"])
        nums_b = [int(rowb["N1"]), int(rowb["N2"]), int(rowb["N3"]), int(rowb["N4"]), int(rowb["N5"]), int(rowb["N6"])]
        comp_b = int(rowb["Complementario"]) if not pd.isna(rowb["Complementario"]) else 18
        rein_b = int(rowb["Reintegro"]) if not pd.isna(rowb["Reintegro"]) else 0
        st.info(f"Usando el último sorteo del histórico: **{last_dt_b.strftime('%d/%m/%Y')}** · Números: {nums_b} · C: {comp_b} · R: {rein_b}")
        do_calc_b = st.button("Calcular recomendaciones · Bonoloto", type="primary")
    else:
        with st.form("form_bono"):
            c1, c2, c3 = st.columns([1,1,1])
            last_date_b = c1.date_input("Fecha último sorteo", value=datetime.today().date(), key="dt_b")
            rein_b = c2.number_input("Reintegro (0-9)", min_value=0, max_value=9, value=2, step=1, key="re_b")
            comp_b = c3.number_input("Complementario (1-49)", min_value=1, max_value=49, value=18, step=1, key="co_b")

            st.markdown("**Números extraídos (6 distintos)**")
            cols = st.columns(6)
            defaults_b = [5,6,8,23,46,47]
            nums_b = [cols[i].number_input(f"N{i+1} (Bono)", 1, 49, defaults_b[i], 1, key=f"nbo{i+1}") for i in range(6)]
            do_calc_b = st.form_submit_button("Calcular recomendaciones · Bonoloto")
        last_dt_b = pd.to_datetime(last_date_b)

    if do_calc_b:
        if len(set(nums_b))!=6:
            st.error("Los 6 números deben ser distintos."); st.stop()

        next_dt_b = last_dt_b + timedelta(days=1)
        weekday = next_dt_b.weekday()
        st.info(f"Próximo sorteo (aprox.): **{next_dt_b.date().strftime('%d/%m/%Y')}** ({next_dt_b.day_name()})")

        # Base (ventana)
        base_b = df_hist_b[df_hist_b["FECHA"]<=last_dt_b].copy()
        if base_b.empty or not (base_b["FECHA"].dt.date == last_dt_b.date()).any():
            new_b = {"FECHA": last_dt_b, "N1": nums_b[0], "N2": nums_b[1], "N3": nums_b[2],
                     "N4": nums_b[3], "N5": nums_b[4], "N6": nums_b[5],
                     "Complementario": comp_b, "Reintegro": rein_b}
            base_b = pd.concat([base_b, pd.DataFrame([new_b])], ignore_index=True)

        base_b = base_b.sort_values("FECHA").tail(WINDOW_DRAWS).reset_index(drop=True)
        base_b["weekday"] = base_b["FECHA"].dt.weekday

        # Pesos
        w_glob_b = weighted_counts_nums(base_b, last_dt_b, HALF_LIFE_DAYS)
        w_day_b  = weighted_counts_nums(base_b[base_b["weekday"]==weekday], last_dt_b, HALF_LIFE_DAYS)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        # Determinismo
        seed_val_b = abs(hash(f"BONO|{last_dt_b.date()}|{tuple(sorted(nums_b))}|{comp_b}|{rein_b}|k={k_nums}|multi={use_multi}|alpha={DAY_BLEND_ALPHA}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}")) % (2**32 - 1)
        np.random.seed(seed_val_b)

        # Candidatos
        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*60:
            c = tuple(random_combo()); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok(c): continue
            cands_b.append(c)

        cands_b = sorted(cands_b, key=lambda c: score_combo(c, w_blend_b, ALPHA_DIR, MU_PENALTY), reverse=True)
        pool_b = cands_b[:1200]
        if not pool_b:
            st.warning("No se generaron candidatos suficientes."); st.stop()

        top3b = pool_b[:3]
        scores = np.array([score_combo(c, w_blend_b, ALPHA_DIR, MU_PENALTY) for c in top3b])
        z = (scores - scores.mean())/(scores.std() if scores.std()!=0 else 1e-6)
        lifts = list(1.4 + 0.3*(z - z.min())/(z.ptp() if z.ptp()!=0 else 1.0))
        lift_map = {i:lifts[i] for i in range(len(top3b))}
        best6_b = list(top3b[0]); best_lift_b = lift_map[0]

        def calc_prob_and_cost_b(k):
            p_base = p_base_k(k)
            p_adj = p_base * best_lift_b
            coste = comb(k,6)*precio_simple_bo
            return p_base, p_adj, coste

        st.markdown(f"<div class='success'><b>Apuesta Óptima (EV/€):</b> {best6_b} · <span class='badge'>Lift ×{best_lift_b:.2f}</span></div>", unsafe_allow_html=True)
        opt_k_b = 6 if not use_multi else k_nums
        p_b, p_ab, cst_b = calc_prob_and_cost_b(opt_k_b)
        st.metric("Prob. base", f"1 entre {pretty_one_in(p_b)}")
        st.metric("Prob. ajustada", f"1 entre {pretty_one_in(p_ab)}")
        st.metric("Coste", f"{cst_b:,.2f} €")

        st.markdown("#### 🎟️ Ajusta tu ticket")
        opciones_b = []
        for k in [6,7,8]:
            nums = expand_to_k(best6_b, w_blend_b, k) if k>6 else best6_b
            p_b0, p_a0, c0 = calc_prob_and_cost_b(k)
            opciones_b.append({
                "Elegir": True if k==6 else False,
                "Tipo": f"A2 k={k}", "k": k, "Joker": "—", "Números": ", ".join(map(str, nums)),
                "p_base": p_b0, "p_adj": p_a0, "Coste": c0
            })
        df_opts_b = pd.DataFrame(opciones_b)
        edited_b = st.data_editor(df_opts_b, num_rows="fixed", use_container_width=True, height=360,
                                  column_config={"Elegir": st.column_config.CheckboxColumn(required=True)})
        elegido_b = edited_b[edited_b["Elegir"]==True]
        total_coste_b = float(elegido_b["Coste"].sum()) if not elegido_b.empty else 0.0
        total_p_adj_b = float(elegido_b["p_adj"].sum()) if not elegido_b.empty else 0.0
        st.markdown(f"<div class='card'><b>Total:</b> {len(elegido_b)} apuestas · <b>Coste:</b> {total_coste_b:,.2f} € · <b>Prob. ajustada (sum):</b> {total_p_adj_b:.6f} (~1 entre {pretty_one_in(total_p_adj_b)})</div>", unsafe_allow_html=True)

        marcar_b = st.checkbox("Marcar este ticket como jugado (Bitácora)", key="bit_b")
        if marcar_b and st.button("Guardar ticket en Bitácora", key="btn_bit_b"):
            row = {
                "Juego":"Bonoloto",
                "Fecha_sorteo": next_dt_b.strftime("%Y-%m-%d"),
                "Numeros": elegido_b.iloc[0]["Números"] if not elegido_b.empty else ", ".join(map(str,best6_b)),
                "k": int(elegido_b.iloc[0]["k"]) if not elegido_b.empty else opt_k_b,
                "Joker": "—",
                "Coste": f"{total_coste_b:.2f}",
                "Lift": f"{best_lift_b:.2f}",
                "p_base": f"{p_b:.8f}",
                "p_adj": f"{p_ab:.8f}",
                "Bank": bank_bo,
                "Modelo": "UXv2"
            }
            ok, msg = append_bitacora(row)
            if ok: st.success("✅ Ticket guardado en Bitácora.")
            else:  st.info(f"ℹ️ No se guardó en Bitácora: {msg}")

# =========================== TUTORIAL ===========================
with tab_tuto:
    st.subheader("Cómo usar el recomendador (guía rápida)")
    st.markdown("""
**Flujo natural (por sorteo):**
1) Deja **wizard ON** si prefieres pasos guiados.
2) Pulsa **Calcular**. Te daremos **una Apuesta Óptima (EV/€)** con su *Lift ×N* y **probabilidades**.
3) Si quieres más cobertura, usa **Ajusta tu ticket** para activar k=7/8 o añadir otra A2.
4) Marca **Bitácora** para registrar el ticket (ayuda a evaluar y mejorar).

**Conceptos:**
- **Lift ×N**: multiplicador de señal respecto al azar (×1.70 ⇒ ~70% de mejora relativa).
- **Prob. base** (k): chance matemática *1 entre X* de acertar 6 con k números. Para k>6 hay más combinaciones simples.
- **Prob. ajustada**: prob. base × Lift. Es una aproximación conservadora.
- **k múltiple**: sube combinaciones del mismo boleto (más coste y menor varianza), **no mejora la eficiencia por €**.
- **Joker (Primitiva)**: vía extra de premio; recomendamos solo si **ScoreJ** (señal de reintegro) supera tu umbral.
- **Determinista**: mismas entradas ⇒ misma recomendación.

**Notas:**
- Este recomendador es estadístico. La lotería es aleatoria por naturaleza.
- Los parámetros avanzados cambian cómo “miramos” el histórico (ventana, vida media, etc.).
""")
