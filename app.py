# app.py — Primitiva · Recomendador A2 (Google Sheets Live, robusto)
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import gspread
from google.oauth2.service_account import Credentials

# ---------------- Page ----------------
st.set_page_config(page_title="Primitiva · Recomendador A2", page_icon="🎯", layout="centered")
st.title("🎯 Primitiva · Recomendador A2 (n dinámico)")
st.caption("Ventana 24 sorteos · t½=60d · ajuste por día (30%) · antipopularidad · diversidad · antiduplicados · fuente: Google Sheets")

# ---------------- Config (modelo) ----------------
A1_FIJAS = {"Monday":[4,24,35,37,40,46], "Thursday":[1,10,23,39,45,48], "Saturday":[7,12,14,25,29,40]}
REIN_FIJOS = {"Monday":1, "Thursday":8, "Saturday":0}

WINDOW_DRAWS    = 24
HALF_LIFE_DAYS  = 60.0
DAY_BLEND_ALPHA = 0.30
ALPHA_DIR       = 0.30
MU_PENALTY      = 1.00
K_CANDIDATOS    = 3000
MIN_DIV         = 0.60
LAMBDA_DIVERSIDAD = 0.60
THRESH_N = [
  {"z": 0.50, "n": 6},
  {"z": 0.35, "n": 4},
  {"z": 0.20, "n": 3},
  {"z": 0.10, "n": 2},
  {"z":-999,  "n": 1},
]

# ---------- helpers ----------
def get_secret_key(name, group="gcp_service_account"):
    """Devuelve un secret tanto si está en la raíz como dentro del bloque [gcp_service_account]."""
    try:
        if name in st.secrets:
            return st.secrets[name]
        if group in st.secrets and name in st.secrets[group]:
            return st.secrets[group][name]
    except Exception:
        pass
    return None

# ---------------- Google Sheets Loader ----------------
@st.cache_data(ttl=600, show_spinner=True)
def load_sheet_df():
    # 1) Credenciales (bloque gcp_service_account es obligatorio)
    if "gcp_service_account" not in st.secrets:
        st.error("No encuentro el bloque [gcp_service_account] en Secrets. Añádelo y pulsa Reboot.")
        return pd.DataFrame()

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    gc = gspread.authorize(creds)

    # 2) sheet_id / worksheet: acepta raíz o dentro del bloque
    sheet_id = get_secret_key("sheet_id")
    ws_name  = get_secret_key("worksheet_historico") or "Historico"
    if not sheet_id:
        st.error(
            "No encuentro `sheet_id` en Secrets.\n\n"
            "En Settings → Secrets añade (fuera de [gcp_service_account]):\n"
            'sheet_id = "TU_SHEET_ID"\nworksheet_historico = "Historico"\n'
            "Guarda y pulsa Reboot."
        )
        return pd.DataFrame()

    # 3) Leer hoja
    try:
        sh = gc.open_by_key(sheet_id)
        ws = sh.worksheet(ws_name)
    except Exception as e:
        st.error(f"No puedo abrir el Sheet/Worksheet. Revisa `sheet_id` y `worksheet_historico`. Detalle: {e}")
        return pd.DataFrame()

    rows = ws.get_all_records(numericise_ignore=["FECHA"])
    df = pd.DataFrame(rows)

    expected = ["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en la hoja: {missing}")
        return pd.DataFrame(columns=expected)

    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
    for c in ["N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ---------------- Utilidades modelo ----------------
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

def overlap_ratio(a,b): return len(set(a)&set(b))/6.0

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
    while len(selected)<n:
        bestC=None; bestVal=-1e9
        for c in sorted_pool:
            if any(tuple(c)==tuple(s) for s in selected): continue
            div_pen = sum(overlap_ratio(c,s) for s in selected)
            val = score_combo(c,weights) - LAMBDA_DIVERSIDAD*div_pen
            if val>bestVal: bestVal=val; bestC=c
        if bestC is None: break
        selected.append(bestC)
    return selected

def next_draw_from_last(last_date):
    wd = last_date.weekday()  # 0=Mon..6=Sun
    if wd==0: return last_date + pd.Timedelta(days=3), "Monday", "Thursday"
    if wd==3: return last_date + pd.Timedelta(days=2), "Thursday", "Saturday"
    if wd==5: return last_date + pd.Timedelta(days=2), "Saturday", "Monday"
    return None, "", ""

def to_js_day(dayname):
    return 1 if dayname=="Monday" else 4 if dayname=="Thursday" else 6 if dayname=="Saturday" else -1

def has_duplicate_row(df, last_dt, nums, comp, rein):
    if df.empty: return False, False
    same_date = df["FECHA"].dt.date == last_dt.date()
    if not same_date.any(): return False, False
    row = df.loc[same_date].tail(1)
    try:
        match = (int(row["N1"].values[0])==nums[0] and
                 int(row["N2"].values[0])==nums[1] and
                 int(row["N3"].values[0])==nums[2] and
                 int(row["N4"].values[0])==nums[3] and
                 int(row["N5"].values[0])==nums[4] and
                 int(row["N6"].values[0])==nums[5] and
                 int(row["Complementario"].values[0])==comp and
                 int(row["Reintegro"].values[0])==rein)
    except Exception:
        match = False
    return True, match

# ---------------- Carga desde Google Sheets ----------------
df_hist = load_sheet_df()
if df_hist.empty:
    st.stop()

# ---------------- Sidebar (parámetros) ----------------
with st.sidebar:
    bank = st.number_input("Banco disponible (€)", min_value=0, value=10, step=1)
    vol  = st.selectbox("Volatilidad objetivo", ["Low","Medium","High"], index=1,
                        help="Low: conservador · Medium: estándar · High: agresivo")

# ---------------- Formulario ----------------
with st.form("entrada"):
    c1, c2 = st.columns(2)
    last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=pd.Timestamp.today().date())
    rein = c2.number_input("Reintegro", min_value=0, max_value=9, value=2, step=1)
    comp = c2.number_input("Complementario", min_value=1, max_value=49, value=18, step=1)

    st.markdown("**Números extraídos (6 distintos)**")
    cols = st.columns(6)
    defaults = [5,6,8,23,46,47]
    nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"n{i+1}") for i in range(6)]

    do_calc = st.form_submit_button("Calcular recomendaciones")

# ---------------- Cálculo ----------------
if do_calc:
    if len(set(nums)) != 6:
        st.error("Los 6 números deben ser distintos.")
        st.stop()

    last_dt = pd.to_datetime(last_date)
    next_dt, _, next_dayname = next_draw_from_last(last_dt)
    if next_dt is None:
        st.error("La fecha debe ser Lunes, Jueves o Sábado.")
        st.stop()

    st.info(f"Próximo sorteo: **{next_dt.date().strftime('%d/%m/%Y')}** ({next_dayname})")

    base = df_hist.copy()
    base = base[base["FECHA"] <= last_dt].sort_values("FECHA")

    has_date, full_match = has_duplicate_row(base, last_dt, nums, comp, rein)
    if has_date and full_match:
        st.success("✅ Sorteo ya existe en el histórico con la misma combinación. No se añade (antiduplicado).")
        df_recent = base.tail(WINDOW_DRAWS)
    elif has_date and not full_match:
        st.warning("⚠️ Misma fecha con combinación distinta en el Sheet. Uso el Sheet (no añado la nueva). Revisa tu histórico.")
        df_recent = base.tail(WINDOW_DRAWS)
    else:
        row_now = pd.DataFrame([{
            "FECHA": last_dt, "N1": nums[0], "N2": nums[1], "N3": nums[2],
            "N4": nums[3], "N5": nums[4], "N6": nums[5],
            "Complementario": comp, "Reintegro": rein
        }])
        df_recent = pd.concat([base, row_now], ignore_index=True).sort_values("FECHA").tail(WINDOW_DRAWS)

    df_recent["weekday"] = df_recent["FECHA"].dt.weekday

    w_glob = weighted_counts_nums(df_recent, last_dt)
    w_day  = weighted_counts_nums(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
    w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

    A1 = A1_FIJAS.get(next_dayname, [4,24,35,37,40,46])

    cands, seen, tries = [], set(), 0
    while len(cands)<K_CANDIDATOS and tries < K_CANDIDATOS*50:
        c = tuple(random_combo()); tries += 1
        if c in seen: continue
        seen.add(c)
        if not terciles_ok(c): continue
        if overlap_ratio(c, A1) > (1 - MIN_DIV): continue
        cands.append(c)
    cands = sorted(cands, key=lambda c: score_combo(c, w_blend), reverse=True)
    pool = cands[:1000]

    bestA2 = list(pool[0]) if pool else []
    zA2 = zscore_combo(bestA2, w_blend) if bestA2 else 0.0
    n = pick_n(zA2, bank, vol); n = max(1, min(6, n))
    A2s = greedy_select(pool, w_blend, max(0, n-1))

    wr_glob = weighted_counts_rei(df_recent, last_dt)
    wr_day  = weighted_counts_rei(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
    rei_scores = {r: DAY_BLEND_ALPHA*wr_day.get(r,0.0) + (1-DAY_BLEND_ALPHA)*wr_glob.get(r,0.0) for r in range(10)}
    rein_sug = max(rei_scores, key=lambda r: rei_scores[r]) if rei_scores else 0

    joker = (zA2 >= 0.35) and (bank >= n+1) and (vol!="Low")

    st.subheader("Resultados")
    st.write(f"**A1 (fija)** {A1}  |  **n recomendado:** {n}")
    for i, c in enumerate(A2s, start=1):
        st.write(f"**A2 #{i}** {list(c)}")
    st.write(f"**Reintegro sugerido (informativo)**: {rein_sug}  ·  **Ref. día**: {REIN_FIJOS.get(next_dayname,'')}")
    st.write(f"**Joker recomendado**: {'Sí' if joker else 'No'}")

    rows = [{"Tipo":"A1", "N1":A1[0],"N2":A1[1],"N3":A1[2],"N4":A1[3],"N5":A1[4],"N6":A1[5]}]
    for i, c in enumerate(A2s, start=1):
        cl = list(c)
        rows.append({"Tipo":f"A2-{i}", "N1":cl[0],"N2":cl[1],"N3":cl[2],"N4":cl[3],"N5":cl[4],"N6":cl[5]})
    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True)
    st.download_button("Descargar combinaciones (CSV)", data=df_out.to_csv(index=False).encode("utf-8"),
                       file_name="primitiva_recomendaciones.csv", mime="text/csv")
