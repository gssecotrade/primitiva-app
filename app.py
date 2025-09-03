# app.py — Recomendador Primitiva & Bonoloto (UX simplificado · determinista)
# v2025-09-03 — Apuesta Óptima (EV/€) + Ajusta tu Ticket (k & Joker) + Bitácora
# Autor: Proyecto Primitiva-bonoloto

import math
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from datetime import datetime, timedelta

# ==== Google Sheets (opcional, fail-safe) ======================================
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_SHEETS = True
except Exception:
    HAS_SHEETS = False

# ==== Estilo ===================================================================
st.set_page_config(page_title="Recomendador Primitiva & Bonoloto", page_icon="🎯", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
html, body, [class*="css"]{font-family:'Poppins',sans-serif;}
.block-container{padding-top:1rem;}
.kpill{display:inline-block;background:#0ea5e9;color:#fff;padding:2px 8px;border-radius:99px;font-size:0.8rem;}
.badge{display:inline-block;background:#0f766e;color:#fff;padding:6px 10px;border-radius:8px;}
.small{color:#64748b;font-size:0.9rem;}
.card{background:#0b1220;border:1px solid #1f2937;border-radius:10px;padding:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎯 Recomendador Primitiva & Bonoloto")
st.caption("Optimización determinista con ventana móvil, mezcla por día, diversidad y recomendación simple. Lift ×N y probabilidad ajustada. Bitácora opcional en Google Sheets.")

# ==== Constantes / Parámetros por defecto =====================================
C496 = math.comb(49, 6)

# Modelo
WINDOW_DRAWS_DEF    = 24
HALF_LIFE_DAYS_DEF  = 60.0
DAY_BLEND_ALPHA_DEF = 0.30
ALPHA_DIR_DEF       = 0.30
MU_PENALTY_DEF      = 1.00
LAMBDA_DIVERSIDAD_DEF = 0.60
K_CANDIDATOS        = 3000
MIN_DIV             = 0.60  # mínima diversidad vs A1

# Reglas de #A2 recomendadas (para no abrumar la UI)
MAX_A2_TO_SHOW = 3

# A1 fijas por día (Primitiva)
A1_FIJAS_PRIMI = {
    "Monday":    [4,24,35,37,40,46],
    "Thursday":  [1,10,23,39,45,48],
    "Saturday":  [7,12,14,25,29,40],
}
REIN_FIJOS_PRIMI = {"Monday":1, "Thursday":8, "Saturday":0}

# A1 neutras por día (Bonoloto)
A1_FIJAS_BONO = {i: [4,24,35,37,40,46] for i in range(7)}

# ==== Helpers combinatorios / prob / coste ====================================
def comb(n,k):
    try:
        return math.comb(n,k)
    except Exception:
        from math import factorial
        return factorial(n)//(factorial(k)*factorial(n-k))

def p_base_k(k: int) -> float:
    if k < 6: return 0.0
    return comb(k,6) / C496

def coste_boleto(k: int, precio_simple: float, joker: bool, precio_joker: float) -> float:
    csim = comb(k,6) * float(precio_simple)
    cjok = float(precio_joker) if joker else 0.0
    return csim + cjok

def formato_1entre(p: float) -> str:
    if p <= 0: return "—"
    x = int(round(1.0/max(p,1e-18),0))
    return f"1 entre {x:,}".replace(",", ".")

# ==== Helpers modelo señal =====================================================
def dayname_to_weekday(dn: str) -> int:
    return {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}.get(dn,-1)

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
    if k<=6: return list(base6[:6])
    extras = [n for n in range(1,50) if n not in base6]
    extras_sorted = sorted(extras, key=lambda x: weights.get(x,0.0), reverse=True)
    add = extras_sorted[:max(0,k-6)]
    out = sorted(list(set(base6) | set(add)))
    return out[:k]

# ==== Lift / Joker (simple) ====================================================
def calc_lift_for_combo(combo, weights, alpha_dir):
    # Lift relativo ~ ratio de media de pesos del combo vs media global (proxy estable)
    allW = np.array([weights.get(i,0.0) for i in range(1,50)], dtype=float)
    gmean = float(allW.mean()) if allW.sum()>0 else 1.0
    cm = float(np.mean([weights.get(n,0.0) for n in combo])) if combo else gmean
    base = max(gmean, 1e-9)
    lift = max(0.5, min(3.0, cm/base))  # acotar visualmente 0.5x..3x
    return lift

def minmax_norm(x, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def joker_score(combo, weights, rein_dict=None):
    z = zscore_combo(combo, weights)
    zN = minmax_norm(z, -1.5, 1.5)
    return 0.6*zN + 0.4*0.7  # 0.7 fijo como proxy de reintegro (si no calculamos dinámico aquí)

# ==== Google Sheets (seguro) ===================================================
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
        raise RuntimeError("Credenciales no disponibles.")
    return Credentials.from_service_account_info(info, scopes=scopes)

@st.cache_data(ttl=600, show_spinner=False)
def load_sheet_df(sheet_id_key: str, worksheet_key: str, default_ws: str):
    if not HAS_SHEETS: return pd.DataFrame()
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get("gcp_service_account", {}) or {}).get(sheet_id_key) or st.secrets.get(sheet_id_key)
        wsn = (st.secrets.get("gcp_service_account", {}) or {}).get(worksheet_key, default_ws) or st.secrets.get(worksheet_key, default_ws)
        if not sid: return pd.DataFrame()
        sh = gc.open_by_key(sid); ws = sh.worksheet(wsn)
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
        df = pd.DataFrame(rows)
        expected = ["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
        for c in expected:
            if c not in df.columns: df[c]=np.nan
        df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
        for c in ["N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
        return df[expected]
    except Exception:
        return pd.DataFrame()

def append_bitacora(sheet_id_key, worksheet_key, default_ws, row_dict):
    if not HAS_SHEETS: return False, "Sheets no disponible"
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get("gcp_service_account", {}) or {}).get(sheet_id_key) or st.secrets.get(sheet_id_key)
        wsn = (st.secrets.get("gcp_service_account", {}) or {}).get(worksheet_key, default_ws) or st.secrets.get(worksheet_key, default_ws)
        sh = gc.open_by_key(sid); ws = sh.worksheet(wsn)
        new_row = [
            row_dict.get("FECHA", datetime.today().strftime("%d/%m/%Y")),
            row_dict.get("JUEGO",""),
            row_dict.get("RESUMEN",""),
            row_dict.get("COSTE",""),
            row_dict.get("PROB_AJUST",""),
        ]
        ws.append_row(new_row)
        return True, "OK"
    except Exception as e:
        return False, f"{e}"

# ==== Sidebar parámetros básicos ==============================================
with st.sidebar:
    st.subheader("Parámetros · Primitiva")
    bank_pr = st.number_input("Banco (€) · Primitiva", min_value=0, value=10, step=1)
    vol_pr  = st.selectbox("Volatilidad · Primitiva", ["Low","Medium","High"], index=1)
    precio_simple_pr = st.number_input("Precio por apuesta simple (€)", min_value=0.5, value=1.0, step=0.5, format="%.2f")

    st.markdown("---")
    st.subheader("Joker (Primitiva)")
    use_joker   = st.checkbox("Activar recomendaciones de Joker por apuesta", value=True)
    joker_thr   = st.slider("Umbral para recomendar Joker", 0.00, 1.00, 0.65, 0.01)
    precio_joker  = st.number_input("Precio Joker (€)", min_value=1.0, value=1.0, step=1.0, format="%.2f")

    st.markdown("---")
    st.subheader("Parámetros · Bonoloto")
    bank_bo = st.number_input("Banco (€) · Bonoloto", min_value=0, value=10, step=1)
    vol_bo  = st.selectbox("Volatilidad · Bonoloto", ["Low","Medium","High"], index=1, key="vol_bono")
    precio_simple_bo = st.number_input("Precio simple Bonoloto (€)", min_value=0.5, value=0.50, step=0.5, format="%.2f",
                                       help="Bonoloto: múltiplos de 0,50 € por apuesta.")

    st.markdown("---")
    with st.expander("Parámetros avanzados (modelo)", expanded=False):
        WINDOW_DRAWS    = st.slider("Ventana (nº sorteos)", 12, 120, WINDOW_DRAWS_DEF, 1)
        HALF_LIFE_DAYS  = float(st.slider("Vida media (días)", 15, 180, int(HALF_LIFE_DAYS_DEF), 1))
        DAY_BLEND_ALPHA = float(st.slider("Mezcla por día (α)", 0.0, 1.0, float(DAY_BLEND_ALPHA_DEF), 0.05))
        ALPHA_DIR       = float(st.slider("Suavizado Dirichlet (α)", 0.00, 1.00, float(ALPHA_DIR_DEF), 0.01))
        MU_PENALTY      = float(st.slider("Penalización 'popularidad'", 0.0, 2.0, float(MU_PENALTY_DEF), 0.1))
        LAMBDA_DIVERSIDAD = float(st.slider("Peso diversidad (λ)", 0.0, 2.0, float(LAMBDA_DIVERSIDAD_DEF), 0.1))

# ==== Tabs =====================================================================
tab_primi, tab_bono, tab_tutorial = st.tabs(["La Primitiva", "Bonoloto", "📘 Tutorial"])

# =========================== PRIMITIVA =========================================
with tab_primi:
    st.subheader("La Primitiva · Ticket Óptimo (EV/€)")
    df_hist = load_sheet_df("sheet_id","worksheet_historico","Historico")
    last_rec = df_hist.tail(1) if not df_hist.empty else pd.DataFrame()

    fuente = st.radio("Origen de datos del último sorteo", ["Usar último del histórico","Introducir manualmente"],
                      index=0 if not df_hist.empty else 1, horizontal=True)

    if fuente == "Usar último del histórico" and not df_hist.empty:
        row = last_rec.iloc[0]
        last_dt = pd.to_datetime(row["FECHA"])
        nums = [int(row["N1"]),int(row["N2"]),int(row["N3"]),int(row["N4"]),int(row["N5"]),int(row["N6"])]
        comp = int(row["Complementario"]) if not pd.isna(row["Complementario"]) else 18
        rein = int(row["Reintegro"]) if not pd.isna(row["Reintegro"]) else 0
        st.info(f"Usando el último sorteo del histórico: **{last_dt.strftime('%d/%m/%Y')}** · Números: {nums} · C: {comp} · R: {rein}")
        do_calc = st.button("Calcular · Primitiva", type="primary")
    else:
        with st.form("form_primi"):
            c1,c2,c3 = st.columns([1,1,1])
            last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=datetime.today().date())
            rein = c2.number_input("Reintegro (0-9)", 0, 9, 2, 1)
            comp = c3.number_input("Complementario (1-49)", 1, 49, 18, 1)
            st.markdown("**Números extraídos (6 distintos)**")
            cols = st.columns(6); defaults=[5,6,8,23,46,47]
            nums = [cols[i].number_input(f"N{i+1}",1,49,defaults[i],1) for i in range(6)]
            do_calc = st.form_submit_button("Calcular · Primitiva", type="primary")

        if do_calc and not df_hist.empty:
            target = pd.to_datetime(last_date).date()
            same = df_hist["FECHA"].dt.date == target
            if same.any():
                r = df_hist.loc[same].tail(1).iloc[0]
                last_dt = pd.to_datetime(r["FECHA"])
                nums = [int(r["N1"]),int(r["N2"]),int(r["N3"]),int(r["N4"]),int(r["N5"]),int(r["N6"])]
                comp = int(r["Complementario"]) if not pd.isna(r["Complementario"]) else 18
                rein = int(r["Reintegro"]) if not pd.isna(r["Reintegro"]) else 0
            else:
                last_dt = pd.to_datetime(last_date)
        elif do_calc and df_hist.empty:
            last_dt = pd.to_datetime(last_date)

    if do_calc:
        if len(set(nums))!=6:
            st.error("Los 6 números deben ser distintos."); st.stop()

        wd = last_dt.weekday()
        if wd==0: next_dt, next_dayname = last_dt + timedelta(days=3), "Thursday"
        elif wd==3: next_dt, next_dayname = last_dt + timedelta(days=2), "Saturday"
        elif wd==5: next_dt, next_dayname = last_dt + timedelta(days=2), "Monday"
        else:
            st.error("La fecha debe ser Lunes, Jueves o Sábado."); st.stop()
        st.info(f"Próximo sorteo: **{next_dt.strftime('%d/%m/%Y')}** ({next_dayname})")

        # Base histórica
        base = df_hist[df_hist["FECHA"]<=last_dt].copy() if not df_hist.empty else pd.DataFrame()
        if base.empty or not (base["FECHA"].dt.date == last_dt.date()).any():
            newrow = {"FECHA": last_dt, "N1": nums[0], "N2": nums[1], "N3": nums[2], "N4": nums[3], "N5": nums[4], "N6": nums[5],
                      "Complementario": comp, "Reintegro": rein}
            base = pd.concat([base, pd.DataFrame([newrow])], ignore_index=True)
        base = base.sort_values("FECHA").tail(WINDOW_DRAWS).reset_index(drop=True)
        base["weekday"] = base["FECHA"].dt.weekday

        # Determinismo
        seed_val = abs(hash(f"PRIMI|{last_dt.date()}|{tuple(sorted(nums))}|{comp}|{rein}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}|alpha={DAY_BLEND_ALPHA}"))%(2**32-1)
        np.random.seed(seed_val)

        # Pesos
        weekday_mask = dayname_to_weekday(next_dayname)
        w_glob = weighted_counts_nums(base, last_dt, HALF_LIFE_DAYS)
        w_day  = weighted_counts_nums(base[base["weekday"]==weekday_mask], last_dt, HALF_LIFE_DAYS)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        # A1 y candidatos A2
        A1_6 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])
        cands, seen, tries = [], set(), 0
        while len(cands)<K_CANDIDATOS and tries < K_CANDIDATOS*50:
            c = tuple(random_combo()); tries += 1
            if c in seen: continue
            seen.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1_6) > (1 - MIN_DIV): continue
            cands.append(c)

        cands = sorted(cands, key=lambda c: score_combo(c, w_blend, ALPHA_DIR, MU_PENALTY), reverse=True)
        pool = cands[:1200]
        if not pool:
            st.warning("No se generaron candidatos con las restricciones actuales."); st.stop()

        # Apuesta Óptima = top-1 por score (proxy de EV/€ al mantener coste lineal)
        best6 = list(pool[0])
        lift_best = calc_lift_for_combo(best6, w_blend, ALPHA_DIR)

        st.markdown('<div class="badge">Apuesta Óptima (EV/€)</div>', unsafe_allow_html=True)
        st.markdown(f"**{sorted(best6)}** &nbsp; <span class='kpill'>Lift ×{lift_best:.2f}</span>", unsafe_allow_html=True)

        # Métricas prob base/ajustada para k=6 por defecto
        pb = p_base_k(6); pa = pb * lift_best
        st.write(f"**Prob. base:** {formato_1entre(pb)}")
        st.write(f"**Prob. ajustada:** {formato_1entre(pa)}")

        # ================== 🎫 AJUSTA TU TICKET (una fila por A2) ==================
        st.markdown("### 🎫 Ajusta tu ticket")

        # Seleccionamos hasta 3 A2 (para no abrumar)
        A2s_6 = greedy_select(pool, w_blend, n=min(3, len(pool)), alpha_dir=ALPHA_DIR, mu_penalty=MU_PENALTY, lambda_div=LAMBDA_DIVERSIDAD)

        rows = []
        for i, a2_base6 in enumerate(A2s_6[:MAX_A2_TO_SHOW], start=1):
            lift_i = calc_lift_for_combo(a2_base6, w_blend, ALPHA_DIR)
            scoreJ = joker_score(a2_base6, w_blend, None) if use_joker else 0.0
            joker_reco = bool(use_joker and scoreJ >= joker_thr)
            rows.append({
                "Elegir": True if i==1 else False,
                "Tipo": f"A2 #{i}",
                "Números": ", ".join(map(str, sorted(a2_base6))),
                "k": 6,
                "Joker": joker_reco,
                "Lift": lift_i
            })

        df_ticket = pd.DataFrame(rows)
        edited = st.data_editor(
            df_ticket,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Elegir": st.column_config.CheckboxColumn("Elegir", help="Incluir esta A2 en tu ticket", default=True),
                "Tipo": st.column_config.TextColumn("Tipo", disabled=True),
                "Números": st.column_config.TextColumn("Números", help="Combinación sugerida", disabled=True),
                "k": st.column_config.SelectboxColumn("k", options=[6,7,8], help="Tamaño de apuesta"),
                "Joker": st.column_config.CheckboxColumn("Joker", help="Activar Joker (Primitiva)"),
                "Lift": None,
            },
            disabled=["Tipo","Números"],
            key="ticket_pri_editor"
        )

        eliges = edited[edited["Elegir"]==True].copy()
        tot_coste = 0.0
        tot_prob = 0.0
        for _, r in eliges.iterrows():
            k = int(r["k"]); jk = bool(r["Joker"]); lift_i = float(r.get("Lift",1.0))
            pb_i = p_base_k(k); pa_i = pb_i * max(1.0, lift_i)
            c_i  = coste_boleto(k, float(precio_simple_pr), jk, float(precio_joker))
            tot_coste += c_i
            tot_prob  += pa_i  # suma de probabilidades de eventos raros (aprox)

        st.info(f"**Total:** {len(eliges)} apuestas · **Coste:** {tot_coste:,.2f} € · **Prob. ajustada (suma):** {tot_prob:.6f}  \n"
                f"≈ *{formato_1entre(tot_prob)}*")

        # Bitácora
        log_it = st.checkbox("✅ Marcar este ticket como jugado (Bitácora)")
        if log_it and len(eliges)>0:
            resumen = "; ".join([f"{r['Tipo']} (k={r['k']}, Joker={'Sí' if r['Joker'] else 'No'}): [{r['Números']}]" for _,r in eliges.iterrows()])
            ok,msg = append_bitacora("sheet_id","worksheet_bitacora","Bitacora", {
                "FECHA": next_dt.strftime("%d/%m/%Y"),
                "JUEGO": "Primitiva",
                "RESUMEN": resumen,
                "COSTE": f"{tot_coste:.2f}",
                "PROB_AJUST": f"{tot_prob:.6f}"
            })
            if ok: st.success("Bitácora actualizada.")
            else:  st.info("No se pudo escribir en Bitácora (desactivado o sin permisos).")

# =========================== BONOLOTO ==========================================
with tab_bono:
    st.subheader("Bonoloto · Ticket Óptimo (EV/€)")
    df_b = load_sheet_df("sheet_id_bono","worksheet_historico_bono","HistoricoBono")
    last_rec_b = df_b.tail(1) if not df_b.empty else pd.DataFrame()

    fuente_b = st.radio("Origen de datos del último sorteo (Bonoloto)", ["Usar último del histórico","Introducir manualmente"],
                        index=0 if not df_b.empty else 1, horizontal=True, key="src_b")
    if fuente_b == "Usar último del histórico" and not df_b.empty:
        rb = last_rec_b.iloc[0]
        last_dt_b = pd.to_datetime(rb["FECHA"])
        nums_b = [int(rb["N1"]),int(rb["N2"]),int(rb["N3"]),int(rb["N4"]),int(rb["N5"]),int(rb["N6"])]
        comp_b = int(rb["Complementario"]) if not pd.isna(rb["Complementario"]) else 18
        rein_b = int(rb["Reintegro"]) if not pd.isna(rb["Reintegro"]) else 0
        st.info(f"Usando el último sorteo del histórico (Bono): **{last_dt_b.strftime('%d/%m/%Y')}** · Números: {nums_b} · C: {comp_b} · R: {rein_b}")
        do_calc_b = st.button("Calcular · Bonoloto", type="primary")
    else:
        with st.form("form_bono"):
            c1,c2,c3 = st.columns([1,1,1])
            last_date_b = c1.date_input("Fecha último sorteo (Bonoloto)", value=datetime.today().date(), key="dt_b")
            rein_b = c2.number_input("Reintegro (0-9)", 0, 9, 2, 1, key="re_b")
            comp_b = c3.number_input("Complementario (1-49)", 1, 49, 18, 1, key="co_b")
            st.markdown("**Números extraídos (6 distintos)**")
            cols = st.columns(6); defaults_b=[5,6,8,23,46,47]
            nums_b = [cols[i].number_input(f"N{i+1} (Bono)",1,49,defaults_b[i],1) for i in range(6)]
            do_calc_b = st.form_submit_button("Calcular · Bonoloto", type="primary")
        if do_calc_b and not df_b.empty:
            target_b = pd.to_datetime(last_date_b).date()
            same_b = df_b["FECHA"].dt.date == target_b
            if same_b.any():
                rb = df_b.loc[same_b].tail(1).iloc[0]
                last_dt_b = pd.to_datetime(rb["FECHA"])
                nums_b = [int(rb["N1"]),int(rb["N2"]),int(rb["N3"]),int(rb["N4"]),int(rb["N5"]),int(rb["N6"])]
                comp_b = int(rb["Complementario"]) if not pd.isna(rb["Complementario"]) else 18
                rein_b = int(rb["Reintegro"]) if not pd.isna(rb["Reintegro"]) else 0
            else:
                last_dt_b = pd.to_datetime(last_date_b)
        elif do_calc_b and df_b.empty:
            last_dt_b = pd.to_datetime(last_date_b)

    if do_calc_b:
        if len(set(nums_b))!=6:
            st.error("Los 6 números deben ser distintos."); st.stop()

        next_dt_b = last_dt_b + timedelta(days=1)
        wd_b = next_dt_b.weekday()
        st.info(f"Próximo sorteo (aprox.): **{next_dt_b.strftime('%d/%m/%Y')}** ({next_dt_b.day_name()})")

        base_b = df_b[df_b["FECHA"]<=last_dt_b].copy() if not df_b.empty else pd.DataFrame()
        if base_b.empty or not (base_b["FECHA"].dt.date == last_dt_b.date()).any():
            new_b = {"FECHA": last_dt_b, "N1": nums_b[0], "N2": nums_b[1], "N3": nums_b[2], "N4": nums_b[3], "N5": nums_b[4], "N6": nums_b[5],
                     "Complementario": comp_b, "Reintegro": rein_b}
            base_b = pd.concat([base_b, pd.DataFrame([new_b])], ignore_index=True)
        base_b = base_b.sort_values("FECHA").tail(WINDOW_DRAWS).reset_index(drop=True)
        base_b["weekday"] = base_b["FECHA"].dt.weekday

        seed_val_b = abs(hash(f"BONO|{last_dt_b.date()}|{tuple(sorted(nums_b))}|{comp_b}|{rein_b}|win={WINDOW_DRAWS}|hl={HALF_LIFE_DAYS}|alpha={DAY_BLEND_ALPHA}"))%(2**32-1)
        np.random.seed(seed_val_b)

        w_glob_b = weighted_counts_nums(base_b, last_dt_b, HALF_LIFE_DAYS)
        w_day_b  = weighted_counts_nums(base_b[base_b["weekday"]==wd_b], last_dt_b, HALF_LIFE_DAYS)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        A1b_6 = A1_FIJAS_BONO.get(wd_b, [4,24,35,37,40,46])

        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*50:
            c = tuple(random_combo()); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1b_6) > (1 - MIN_DIV): continue
            cands_b.append(c)
        cands_b = sorted(cands_b, key=lambda c: score_combo(c, w_blend_b, ALPHA_DIR, MU_PENALTY), reverse=True)
        pool_b = cands_b[:1200]
        if not pool_b:
            st.warning("No se generaron candidatos con las restricciones actuales."); st.stop()

        best6_b = list(pool_b[0])
        lift_best_b = calc_lift_for_combo(best6_b, w_blend_b, ALPHA_DIR)
        st.markdown('<div class="badge">Apuesta Óptima (EV/€)</div>', unsafe_allow_html=True)
        st.markdown(f"**{sorted(best6_b)}** &nbsp; <span class='kpill'>Lift ×{lift_best_b:.2f}</span>", unsafe_allow_html=True)
        pb_b = p_base_k(6); pa_b = pb_b * lift_best_b
        st.write(f"**Prob. base:** {formato_1entre(pb_b)}")
        st.write(f"**Prob. ajustada:** {formato_1entre(pa_b)}")

        # 🎫 Ajusta tu ticket (Bonoloto, sin Joker)
        st.markdown("### 🎫 Ajusta tu ticket")
        A2s_b_6 = greedy_select(pool_b, w_blend_b, n=min(3, len(pool_b)), alpha_dir=ALPHA_DIR, mu_penalty=MU_PENALTY, lambda_div=LAMBDA_DIVERSIDAD)

        rows_b = []
        for i, a2_base6 in enumerate(A2s_b_6[:MAX_A2_TO_SHOW], start=1):
            lift_i = calc_lift_for_combo(a2_base6, w_blend_b, ALPHA_DIR)
            rows_b.append({
                "Elegir": True if i==1 else False,
                "Tipo": f"A2 #{i}",
                "Números": ", ".join(map(str, sorted(a2_base6))),
                "k": 6,
                "Lift": lift_i
            })
        df_ticket_b = pd.DataFrame(rows_b)
        edited_b = st.data_editor(
            df_ticket_b,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Elegir": st.column_config.CheckboxColumn("Elegir", default=True),
                "Tipo": st.column_config.TextColumn("Tipo", disabled=True),
                "Números": st.column_config.TextColumn("Números", disabled=True),
                "k": st.column_config.SelectboxColumn("k", options=[6,7,8]),
                "Lift": None,
            },
            disabled=["Tipo","Números"],
            key="ticket_bono_editor"
        )
        eliges_b = edited_b[edited_b["Elegir"]==True].copy()
        tot_coste_b = 0.0; tot_prob_b=0.0
        for _, r in eliges_b.iterrows():
            k = int(r["k"]); lift_i = float(r.get("Lift",1.0))
            pb_i = p_base_k(k); pa_i = pb_i * max(1.0, lift_i)
            c_i  = coste_boleto(k, float(precio_simple_bo), False, 0.0)
            tot_coste_b += c_i; tot_prob_b += pa_i
        st.info(f"**Total:** {len(eliges_b)} apuestas · **Coste:** {tot_coste_b:,.2f} € · **Prob. ajustada (suma):** {tot_prob_b:.6f}  \n"
                f"≈ *{formato_1entre(tot_prob_b)}*")

        log_it_b = st.checkbox("✅ Marcar este ticket como jugado (Bitácora)", key="bitacora_b")
        if log_it_b and len(eliges_b)>0:
            resumen_b = "; ".join([f"{r['Tipo']} (k={r['k']}): [{r['Números']}]" for _,r in eliges_b.iterrows()])
            ok,msg = append_bitacora("sheet_id_bono","worksheet_bitacora_bono","BitacoraBono", {
                "FECHA": next_dt_b.strftime("%d/%m/%Y"),
                "JUEGO": "Bonoloto",
                "RESUMEN": resumen_b,
                "COSTE": f"{tot_coste_b:.2f}",
                "PROB_AJUST": f"{tot_prob_b:.6f}"
            })
            if ok: st.success("Bitácora actualizada.")
            else:  st.info("No se pudo escribir en Bitácora (desactivado o sin permisos).")

# =========================== TUTORIAL ==========================================
with tab_tutorial:
    st.subheader("📘 Cómo usar el recomendador (en llano)")
    st.markdown("""
**1) Pulsa “Calcular”** en el juego que quieras (Primitiva o Bonoloto).  
Te mostraremos **una Apuesta Óptima (EV/€)**: la combinación con mayor **Lift** (multiplicador vs azar) según nuestro modelo.

**2) 🎫 Ajusta tu ticket.**  
Verás **hasta 3 A2 recomendadas** (para no liarte). Marca las que quieras:
- **k**: tamaño de la apuesta (6/7/8). *Subir k* no mejora la **eficiencia por €**, solo cambia la **varianza** y la **comodidad** (más combinaciones en un solo boleto).
- **Joker (solo Primitiva)**: lo recomendamos si la señal (ScoreJ) supera el umbral. Añade una vía extra de premio con coste fijo.

**3) Total**  
Abajo verás **Coste total** y **Probabilidad ajustada (suma)** en formato **“1 entre X”**.

**4) Bitácora**  
Actívala si quieres guardar tu ticket en Google Sheets. Si no hay permisos o credenciales, la app te lo dirá sin romper el flujo.

---

### Conceptos clave
- **Lift ×N**: cuántas veces mejora tu combinación frente a jugar totalmente al azar (N=1.00 sería azar puro).
- **Prob. base** de acertar 6 con k: `C(k,6) / C(49,6)`. Con **Lift**, la prob. **ajustada = base × Lift**.
- **Determinismo**: con el mismo histórico y parámetros, verás siempre las mismas recomendaciones.

> Nota honesta: La lotería es aleatoria. Este recomendador prioriza **consistencia y claridad** para decisiones repetidas a largo plazo.
""")
