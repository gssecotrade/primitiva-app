# app.py — Primitiva & Bonoloto · Recomendador A2 (con UX + Ayuda)
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials

# ------------------------- Config & Styles -------------------------
st.set_page_config(
    page_title="Primitiva & Bonoloto · Recomendador A2",
    page_icon="🎯",
    layout="wide"
)

# Cargar estilos locales (Poppins, botones, tablas…)
def _load_css():
    css_path = Path("styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

_load_css()

# ------------------------- Utilidades varias -------------------------
def load_md(rel_path: str) -> str:
    """Lee un markdown del repo (para pestaña Ayuda)."""
    p = Path(rel_path)
    if not p.exists():
        return f"⚠️ No encuentro `{rel_path}` en el repositorio."
    return p.read_text(encoding="utf-8")

def get_secret_key(name, group="gcp_service_account"):
    """Obtiene un secret ya sea en la raíz o dentro de [gcp_service_account]."""
    try:
        if name in st.secrets:
            return st.secrets[name]
        if group in st.secrets and name in st.secrets[group]:
            return st.secrets[group][name]
    except Exception:
        pass
    return None

def get_gcp_credentials():
    """
    Construye credenciales de Google desde Secrets.
    Acepta private_key con saltos como '\n' o reales.
    Soporta dos modos:
      - Bloque [gcp_service_account] usual
      - Modo gcp_json (JSON entero entre triple comilla) si existiese
    """
    # 1) Modo JSON entero (opcional)
    try:
        gcp_json = st.secrets.get("gcp_json", None)
    except Exception:
        gcp_json = None

    if gcp_json:
        # Si el usuario pegó el JSON completo
        import json
        info = json.loads(gcp_json)
        pk = info.get("private_key", "")
        if isinstance(pk, str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n", "\n")
        info["private_key"] = info["private_key"].strip()
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        return Credentials.from_service_account_info(info, scopes=scopes)

    # 2) Modo bloque [gcp_service_account]
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Falta el bloque [gcp_service_account] en Secrets.")

    info = dict(st.secrets["gcp_service_account"])
    pk = info.get("private_key", "")
    if isinstance(pk, str) and "\\n" in pk:
        info["private_key"] = pk.replace("\\n", "\n")
    info["private_key"] = info["private_key"].strip()
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    return Credentials.from_service_account_info(info, scopes=scopes)

# ------------------------- Parámetros del modelo -------------------------
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

# A1 fijas por día para PRIMITIVA (calibradas)
A1_FIJAS_PRIMI = {
    "Monday":    [4, 24, 35, 37, 40, 46],
    "Thursday":  [1, 10, 23, 39, 45, 48],
    "Saturday":  [7, 12, 14, 25, 29, 40],
}
REIN_FIJOS_PRIMI = {"Monday": 1, "Thursday": 8, "Saturday": 0}

# A1 iniciales por día para BONOLOTO (neutras, se calibran con uso)
A1_FIJAS_BONO = {
    0: [4,24,35,37,40,46],  # Mon
    1: [4,24,35,37,40,46],  # Tue
    2: [4,24,35,37,40,46],  # Wed
    3: [4,24,35,37,40,46],  # Thu
    4: [4,24,35,37,40,46],  # Fri
    5: [4,24,35,37,40,46],  # Sat
    6: [4,24,35,37,40,46],  # Sun
}

# ------------------------- Lectura Google Sheets -------------------------
@st.cache_data(ttl=600, show_spinner=True)
def load_sheet_df_generic(sheet_id_key: str, worksheet_key: str, default_ws: str) -> pd.DataFrame:
    """
    Lee un DataFrame desde Google Sheets usando claves en Secrets.
    Necesita:
      sheet_id_key, p.ej. "sheet_id"
      worksheet_key, p.ej. "worksheet_historico"
    """
    creds = get_gcp_credentials()
    gc = gspread.authorize(creds)

    sid = get_secret_key(sheet_id_key)
    wsn = get_secret_key(worksheet_key) or default_ws
    if not sid:
        st.error(
            f"No encuentro `{sheet_id_key}` en Secrets. Añade:\n\n"
            f"{sheet_id_key} = \"TU_SHEET_ID\"\n{worksheet_key} = \"{default_ws}\"\n"
        )
        return pd.DataFrame()

    try:
        sh = gc.open_by_key(sid)
        ws = sh.worksheet(wsn)
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

def load_sheet_df_primi() -> pd.DataFrame:
    return load_sheet_df_generic("sheet_id", "worksheet_historico", "Historico")

def load_sheet_df_bono() -> pd.DataFrame:
    return load_sheet_df_generic("sheet_id_bono", "worksheet_historico_bono", "HistoricoBono")

# Guardar nueva fila si no existe (antiduplicados)
def append_if_new(sheet_id_key, worksheet_key, new_row: dict) -> bool:
    """
    Inserta una fila en el sheet si no existe ya una del mismo día con misma combinación.
    Devuelve True si inserta, False si no.
    """
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = get_secret_key(sheet_id_key)
        wsn = get_secret_key(worksheet_key)
        sh = gc.open_by_key(sid)
        ws = sh.worksheet(wsn)

        # Carga actual para comprobar
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
        df = pd.DataFrame(rows)
        if "FECHA" not in df.columns:
            return False
        df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")

        same_date = df["FECHA"].dt.date == pd.to_datetime(new_row["FECHA"]).date()
        if same_date.any():
            # Compara todos los números, complementario y reintegro
            row = df.loc[same_date].tail(1)
            try:
                match = (
                    int(row["N1"].values[0])==int(new_row["N1"]) and
                    int(row["N2"].values[0])==int(new_row["N2"]) and
                    int(row["N3"].values[0])==int(new_row["N3"]) and
                    int(row["N4"].values[0])==int(new_row["N4"]) and
                    int(row["N5"].values[0])==int(new_row["N5"]) and
                    int(row["N6"].values[0])==int(new_row["N6"]) and
                    int(row["Complementario"].values[0])==int(new_row["Complementario"]) and
                    int(row["Reintegro"].values[0])==int(new_row["Reintegro"])
                )
            except Exception:
                match = False
            if match:
                return False  # Ya está

        # Si no estaba, añadir al final
        ws.append_row([
            pd.to_datetime(new_row["FECHA"]).strftime("%d/%m/%Y"),
            int(new_row["N1"]), int(new_row["N2"]), int(new_row["N3"]),
            int(new_row["N4"]), int(new_row["N5"]), int(new_row["N6"]),
            int(new_row["Complementario"]), int(new_row["Reintegro"])
        ])
        return True
    except Exception:
        return False

# ------------------------- Utilidades de modelo -------------------------
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

def proxy_prob_at_least_k(k):
    """
    Proxy simple y homogénea para mostrar magnitudes relativas en la UI.
    NO es probabilidad real del juego completo; sirve para comparar señales.
    """
    # Cuanto mayor k, menor proxy.
    base = {1:0.25, 2:0.10, 3:0.03, 4:0.006, 5:0.0012, 6:0.0002}
    return base.get(k, 0.0)

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

def to_js_day(dayname):
    return 1 if dayname=="Monday" else 4 if dayname=="Thursday" else 6 if dayname=="Saturday" else -1

# ------------------------- UI -------------------------
st.title("🎯 Primitiva & Bonoloto · Recomendador A2 (n dinámico)")
st.caption("Ventana 24 sorteos · t½=60d · mezcla por día (30%) · antipopularidad · diversidad · antiduplicados · Google Sheets (live)")

tab_primi, tab_bono, tab_help = st.tabs(["La Primitiva", "Bonoloto", "Ayuda"])

# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader("La Primitiva · Recomendador A2")
    st.caption("A1 fija por día · A2 dinámica · Joker opcional")

    df_hist = load_sheet_df_primi()
    if df_hist.empty:
        st.stop()

    # Sidebar - parámetros específicos
    with st.sidebar:
        st.markdown("### Primitiva · Parámetros")
        bank = st.number_input("Banco disponible (€)", min_value=0, value=10, step=1, key="bank_primi")
        vol  = st.selectbox("Volatilidad objetivo", ["Low","Medium","High"], index=1, key="vol_primi",
                            help="Low: conservador · Medium: estándar · High: agresivo")

    # Formulario de entrada
    with st.form("entrada_primi"):
        c1, c2 = st.columns(2)
        last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=pd.Timestamp.today().date())
        rein = c2.number_input("Reintegro", min_value=0, max_value=9, value=2, step=1)
        comp = c2.number_input("Complementario", min_value=1, max_value=49, value=18, step=1)

        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults = [5,6,8,23,46,47]
        nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"npr{i+1}") for i in range(6)]

        save_hist = st.checkbox("Guardar en histórico (Primitiva) si es nuevo", value=False)
        do_calc = st.form_submit_button("Calcular recomendaciones · Primitiva")

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

        # Guardar en sheet si procede
        if save_hist:
            new_row = {
                "FECHA": last_dt,
                "N1": nums[0], "N2": nums[1], "N3": nums[2], "N4": nums[3], "N5": nums[4], "N6": nums[5],
                "Complementario": comp, "Reintegro": rein
            }
            inserted = append_if_new("sheet_id", "worksheet_historico", new_row)
            if inserted:
                st.success("✅ Añadido al histórico (Primitiva).")
                st.cache_data.clear()
                df_hist = load_sheet_df_primi()
            else:
                st.info("ℹ️ No se añadió: ya existe una fila igual para esa fecha o hubo un problema de acceso.")

        base = df_hist[df_hist["FECHA"] <= last_dt].sort_values("FECHA").copy()
        df_recent = base.tail(WINDOW_DRAWS)
        df_recent["weekday"] = df_recent["FECHA"].dt.weekday

        w_glob = weighted_counts_nums(df_recent, last_dt)
        w_day  = weighted_counts_nums(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        A1 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])

        # Candidatos A2
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

        st.subheader("Resultados · Primitiva")
        st.write(f"**A1 (fija)** {A1}  |  **n recomendado:** {n}")
        for i, c in enumerate(A2s, start=1):
            st.write(f"**A2 #{i}** {list(c)}")
        st.write(f"**Reintegro sugerido (informativo)**: {rein_sug}  ·  **Ref. día**: {REIN_FIJOS_PRIMI.get(next_dayname,'')}")
        st.write(f"**Joker recomendado**: {'Sí' if joker else 'No'}")

        # Métricas compactas
        with st.expander("📊 Métricas y proxy de probabilidades"):
            st.write(f"Señal (z): **{zA2:.3f}**")
            st.write("Proxy de p(≥k aciertos) — orientativa y homogénea:")
            df_proxy = pd.DataFrame({
                "k aciertos": [1,2,3,4,5,6],
                "p_proxy": [proxy_prob_at_least_k(k) for k in [1,2,3,4,5,6]]
            })
            st.table(df_proxy)

        # Tabla y descarga
        rows = [{"Tipo":"A1", "N1":A1[0],"N2":A1[1],"N3":A1[2],"N4":A1[3],"N5":A1[4],"N6":A1[5]}]
        for i, c in enumerate(A2s, start=1):
            cl = list(c)
            rows.append({"Tipo":f"A2-{i}", "N1":cl[0],"N2":cl[1],"N3":cl[2],"N4":cl[3],"N5":cl[4],"N6":cl[5]})
        df_out = pd.DataFrame(rows)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Descargar combinaciones · Primitiva (CSV)",
                           data=df_out.to_csv(index=False).encode("utf-8"),
                           file_name="primitiva_recomendaciones.csv", mime="text/csv")

        # Últimos sorteos (vista rápida)
        st.markdown("#### 🗂️ Últimos sorteos cargados (Primitiva)")
        st.dataframe(df_hist.tail(10), use_container_width=True)

# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader("Bonoloto · Recomendador A2")
    st.caption("A1 ancla inicial por día · A2 dinámica · sin Joker")

    df_bono = load_sheet_df_bono()
    if df_bono.empty:
        st.stop()

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Bonoloto · Parámetros")
        bank_b = st.number_input("Banco (€) · Bonoloto", min_value=0, value=10, step=1, key="bank_bono")
        vol_b  = st.selectbox("Volatilidad · Bonoloto", ["Low","Medium","High"], index=1, key="vol_bono")

    with st.form("entrada_bono"):
        c1, c2 = st.columns(2)
        last_date_b = c1.date_input("Fecha último sorteo (Bonoloto)", value=pd.Timestamp.today().date())
        rein_b = c2.number_input("Reintegro (0–9)", min_value=0, max_value=9, value=2, step=1)
        comp_b = c2.number_input("Complementario (1–49)", min_value=1, max_value=49, value=18, step=1)

        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults_b = [5,6,8,23,46,47]
        nums_b = [cols[i].number_input(f"N{i+1} (Bono)", 1, 49, defaults_b[i], 1, key=f"nbo{i+1}") for i in range(6)]

        save_hist_b = st.checkbox("Guardar en histórico (Bonoloto) si es nuevo", value=False)
        do_calc_b = st.form_submit_button("Calcular recomendaciones · Bonoloto")

    if do_calc_b:
        if len(set(nums_b)) != 6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        last_dt_b = pd.to_datetime(last_date_b)
        weekday = last_dt_b.weekday()  # 0=Mon..6=Sun
        next_dt_b = last_dt_b + pd.Timedelta(days=1)  # Bonoloto sortea a diario
        next_dayname_b = next_dt_b.day_name()

        st.info(f"Próximo sorteo (aprox.): **{next_dt_b.date().strftime('%d/%m/%Y')}** ({next_dayname_b})")

        if save_hist_b:
            new_row_b = {
                "FECHA": last_dt_b,
                "N1": nums_b[0], "N2": nums_b[1], "N3": nums_b[2],
                "N4": nums_b[3], "N5": nums_b[4], "N6": nums_b[5],
                "Complementario": comp_b, "Reintegro": rein_b
            }
            inserted_b = append_if_new("sheet_id_bono", "worksheet_historico_bono", new_row_b)
            if inserted_b:
                st.success("✅ Añadido al histórico (Bonoloto).")
                st.cache_data.clear()
                df_bono = load_sheet_df_bono()
            else:
                st.info("ℹ️ No se añadió: ya existe una fila igual para esa fecha o hubo un problema de acceso.")

        base_b = df_bono[df_bono["FECHA"] <= last_dt_b].sort_values("FECHA").copy()
        df_recent_b = base_b.tail(WINDOW_DRAWS)
        df_recent_b["weekday"] = df_recent_b["FECHA"].dt.weekday

        w_glob_b = weighted_counts_nums(df_recent_b, last_dt_b)
        w_day_b  = weighted_counts_nums(df_recent_b[df_recent_b["weekday"]==weekday], last_dt_b)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        A1b = A1_FIJAS_BONO.get((weekday+1) % 7, [4,24,35,37,40,46])

        # Candidatos
        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*50:
            c = tuple(random_combo()); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1b) > (1 - MIN_DIV): continue
            cands_b.append(c)
        cands_b = sorted(cands_b, key=lambda c: score_combo(c, w_blend_b), reverse=True)
        pool_b = cands_b[:1000]

        bestA2_b = list(pool_b[0]) if pool_b else []
        zA2_b = zscore_combo(bestA2_b, w_blend_b) if bestA2_b else 0.0

        def pick_n_b(z, bank, vol):
            adj = 0.05 if vol=="Low" else -0.05 if vol=="High" else 0.0
            for th in THRESH_N:
                if z >= th["z"] + adj:
                    n = min(th["n"], int(bank))
                    return max(1, n)
            return 1

        n_b = pick_n_b(zA2_b, bank_b, vol_b)
        n_b = max(1, min(6, n_b))

        def greedy_select_b(pool,w,n):
            if n<=0: return []
            sp=sorted(pool,key=lambda c:score_combo(c,w),reverse=True)
            sel=[sp[0]]
            while len(sel)<n:
                best=None; bestv=-1e9
                for c in sp:
                    if any(tuple(c)==tuple(s) for s in sel): continue
                    pen=sum(overlap_ratio(c,s) for s in sel)
                    v=score_combo(c,w)-LAMBDA_DIVERSIDAD*pen
                    if v>bestv: bestv=v; best=c
                if best is None: break
                sel.append(best)
            return sel

        A2s_b = greedy_select_b(pool_b, w_blend_b, max(0, n_b-1))

        # Reintegro sugerido (informativo)
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

        with st.expander("📊 Métricas y proxy de probabilidades"):
            st.write(f"Señal (z): **{zA2_b:.3f}**")
            df_proxy_b = pd.DataFrame({
                "k aciertos": [1,2,3,4,5,6],
                "p_proxy": [proxy_prob_at_least_k(k) for k in [1,2,3,4,5,6]]
            })
            st.table(df_proxy_b)

        rows_b = [{"Tipo":"A1","N1":A1b[0],"N2":A1b[1],"N3":A1b[2],"N4":A1b[3],"N5":A1b[4],"N6":A1b[5]}]
        for i, c in enumerate(A2s_b, start=1):
            cl = list(c)
            rows_b.append({"Tipo":f"A2-{i}","N1":cl[0],"N2":cl[1],"N3":cl[2],"N4":cl[3],"N5":cl[4],"N6":cl[5]})
        df_out_b = pd.DataFrame(rows_b)
        st.dataframe(df_out_b, use_container_width=True)
        st.download_button("Descargar combinaciones · Bonoloto (CSV)",
                           data=df_out_b.to_csv(index=False).encode("utf-8"),
                           file_name="bonoloto_recomendaciones.csv", mime="text/csv")

        st.markdown("#### 🗂️ Últimos sorteos cargados (Bonoloto)")
        st.dataframe(df_bono.tail(10), use_container_width=True)

# =========================== AYUDA ===========================
with tab_help:
    st.subheader("Centro de ayuda")

    # Vídeo opcional desde Secrets
    try:
        url = st.secrets.get("help_video_url", None)
    except Exception:
        url = None
    if url:
        st.video(url)

    st.markdown("---")
    st.markdown(load_md("assets/quickstart.md"))

    with st.expander("📘 Tutorial completo", expanded=False):
        st.markdown(load_md("assets/tutorial_full.md"))

    with st.expander("❓ Preguntas frecuentes (FAQ)", expanded=False):
        st.markdown(load_md("assets/faq.md"))
