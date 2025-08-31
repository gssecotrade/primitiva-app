# app.py — Primitiva & Bonoloto · Recomendador A2 (Google Sheets Live, robusto)
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import gspread
from google.oauth2.service_account import Credentials

# ---------------- Credenciales (normaliza private_key) ----------------
def get_gcp_credentials():
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Falta [gcp_service_account] en Secrets.")
    info = dict(st.secrets["gcp_service_account"])
    pk = info.get("private_key", "")
    if isinstance(pk, str) and "\\n" in pk:
        info["private_key"] = pk.replace("\\n", "\n")
    info["private_key"] = info["private_key"].strip()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    return Credentials.from_service_account_info(info, scopes=scopes)

# ---------------- Page ----------------
st.set_page_config(page_title="Primitiva & Bonoloto · Recomendador A2", page_icon="🎯", layout="centered")
st.title("🎯 Primitiva & Bonoloto · Recomendador A2 (n dinámico)")
st.caption("Ventana 24 sorteos · t½=60d · mezcla por día (30%) · antipopularidad · diversidad · antiduplicados · fuente: Google Sheets")

# ---------------- Parámetros del modelo ----------------
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

# A1 fijas por día para PRIMITIVA
A1_FIJAS_PRIMI = {
    "Monday":[4,24,35,37,40,46],
    "Thursday":[1,10,23,39,45,48],
    "Saturday":[7,12,14,25,29,40]
}
REIN_FIJOS_PRIMI = {"Monday":1, "Thursday":8, "Saturday":0}

# A1 neutras iniciales por día para BONOLOTO
A1_FIJAS_BONO = {0:[4,24,35,37,40,46],1:[4,24,35,37,40,46],2:[4,24,35,37,40,46],
                 3:[4,24,35,37,40,46],4:[4,24,35,37,40,46],5:[4,24,35,37,40,46],6:[4,24,35,37,40,46]}

# ---------------- Helpers de secrets ----------------
def get_secret_key(name, group="gcp_service_account"):
    try:
        if name in st.secrets:
            return st.secrets[name]
        if group in st.secrets and name in st.secrets[group]:
            return st.secrets[group][name]
    except Exception:
        pass
    return None

# ---------------- Lectura Google Sheets (con diagnóstico claro) ----------------
@st.cache_data(ttl=600, show_spinner=True)
def load_sheet_df_primi():
    return load_sheet_df_generic("sheet_id", "worksheet_historico", "Historico")

def load_sheet_df_generic(sheet_id_key: str, worksheet_key: str, default_ws: str):
    if "gcp_service_account" not in st.secrets:
        st.error("❌ Falta el bloque [gcp_service_account] en Settings → Secrets.")
        return pd.DataFrame()

    # Credenciales con diagnóstico claro
    try:
        creds = get_gcp_credentials()
    except Exception as e:
        st.error(
            "❌ La clave de servicio de Google está mal formateada en Secrets.\n"
            "Usa la `private_key` en **una sola línea** con `\\n` (como en el bloque que te pasé).\n"
            f"Detalle técnico: {type(e).__name__}"
        )
        return pd.DataFrame()

    # Autorizar gspread
    try:
        gc = gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ No puedo autorizar gspread con esas credenciales. Detalle: {type(e).__name__}: {e}")
        return pd.DataFrame()

    # Leer IDs de Secrets
    sid = get_secret_key(sheet_id_key)
    wsn = get_secret_key(worksheet_key) or default_ws
    if not sid:
        st.error(
            f"❌ No encuentro `{sheet_id_key}` en Secrets.\n"
            f"Añade:\n{sheet_id_key} = \"TU_SHEET_ID\"\n{worksheet_key} = \"{default_ws}\""
        )
        return pd.DataFrame()

    # Abrir hoja y worksheet
    try:
        sh = gc.open_by_key(sid)
        ws = sh.worksheet(wsn)
    except Exception as e:
        st.error(f"❌ No puedo abrir el Sheet/Worksheet ({sheet_id_key}/{worksheet_key}). Detalle: {e}")
        return pd.DataFrame()

    # Descargar y validar
    try:
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
        df = pd.DataFrame(rows)
    except Exception as e:
        st.error(f"❌ Error leyendo registros del Sheet. Detalle: {e}")
        return pd.DataFrame()

    expected = ["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"❌ Faltan columnas en la pestaña '{wsn}': {missing}")
        return pd.DataFrame(columns=expected)

    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
    for c in ["N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ---------------- Utilidades de modelo ----------------
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

def weekday_from_name(dayname: str) -> int:
    mapping = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
    return mapping.get(dayname, -1)

# ---------------- Pestañas ----------------
tab_primi, tab_bono = st.tabs(["La Primitiva", "Bonoloto"])

# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader("La Primitiva · Recomendador A2")
    st.caption("A1 fija por día · A2 dinámica · Joker opcional")

    df_hist = load_sheet_df_primi()
    if df_hist.empty:
        st.stop()

    with st.sidebar:
        bank = st.number_input("Banco disponible (€)", min_value=0, value=10, step=1, key="bank_primi")
        vol  = st.selectbox("Volatilidad objetivo", ["Low","Medium","High"], index=1, key="vol_primi")

    with st.form("entrada_primi"):
        c1, c2 = st.columns(2)
        last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)", value=pd.Timestamp.today().date())
        rein = c2.number_input("Reintegro", min_value=0, max_value=9, value=2, step=1)
        comp = c2.number_input("Complementario", min_value=1, max_value=49, value=18, step=1)
        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults = [5,6,8,23,46,47]
        nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"npr{i+1}") for i in range(6)]
        do_calc = st.form_submit_button("Calcular recomendaciones · Primitiva")

    if do_calc:
        if len(set(nums)) != 6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        last_dt = pd.to_datetime(last_date)
        wd = last_dt.weekday()

        if wd == 0:
            next_dt, next_dayname = last_dt + pd.Timedelta(days=3), "Thursday"
        elif wd == 3:
            next_dt, next_dayname = last_dt + pd.Timedelta(days=2), "Saturday"
        elif wd == 5:
            next_dt, next_dayname = last_dt + pd.Timedelta(days=2), "Monday"
        else:
            st.error("La fecha debe ser Lunes, Jueves o Sábado.")
            st.stop()

        st.info(f"Próximo sorteo: **{next_dt.date().strftime('%d/%m/%Y')}** ({next_dayname})")

        base = df_hist[df_hist["FECHA"] <= last_dt].sort_values("FECHA").copy()

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
        if has_date and full_match:
            st.success("✅ Sorteo ya existe en el histórico con la misma combinación. No se añade.")
            df_recent = base.tail(WINDOW_DRAWS)
        elif has_date and not full_match:
            st.warning("⚠️ Misma fecha con combinación distinta en el Sheet. Uso el Sheet (no añado).")
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
        next_wd = {"Monday":0,"Thursday":3,"Saturday":5}[next_dayname]
        w_day  = weighted_counts_nums(df_recent[df_recent["weekday"]==next_wd], last_dt)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        A1 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])

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
        n = max(1, min(6, pick_n(zA2, bank, vol)))
        A2s = greedy_select(pool, w_blend, max(0, n-1))

        wr_glob = weighted_counts_rei(df_recent, last_dt)
        wr_day  = weighted_counts_rei(df_recent[df_recent["weekday"]==next_wd], last_dt)
        rei_scores = {r: DAY_BLEND_ALPHA*wr_day.get(r,0.0) + (1-DAY_BLEND_ALPHA)*wr_glob.get(r,0.0) for r in range(10)}
        rein_sug = max(rei_scores, key=lambda r: rei_scores[r]) if rei_scores else 0

        joker = (zA2 >= 0.35) and (bank >= n+1) and (vol!="Low")

        st.subheader("Resultados · Primitiva")
        st.write(f"**A1 (fija)** {A1}  |  **n recomendado:** {n}")
        for i, c in enumerate(A2s, start=1):
            st.write(f"**A2 #{i}** {list(c)}")
        st.write(f"**Reintegro sugerido (informativo)**: {rein_sug}  ·  **Ref. día**: {REIN_FIJOS_PRIMI.get(next_dayname,'')}")
        st.write(f"**Joker recomendado**: {'Sí' if joker else 'No'}")

        rows = [{"Tipo":"A1", "N1":A1[0],"N2":A1[1],"N3":A1[2],"N4":A1[3],"N5":A1[4],"N6":A1[5]}]
        for i, c in enumerate(A2s, start=1):
            cl = list(c)
            rows.append({"Tipo":f"A2-{i}", "N1":cl[0],"N2":cl[1],"N3":cl[2],"N4":cl[3],"N5":cl[4],"N6":cl[5]})
        df_out = pd.DataFrame(rows)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Descargar combinaciones · Primitiva (CSV)",
                           data=df_out.to_csv(index=False).encode("utf-8"),
                           file_name="primitiva_recomendaciones.csv", mime="text/csv")

# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader("Bonoloto · Recomendador A2")
    st.caption("A1 ancla inicial por día · A2 dinámica · sin Joker")

    df_bono = load_sheet_df_generic("sheet_id_bono", "worksheet_historico_bono", "HistoricoBono")
    if df_bono.empty:
        st.stop()

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Bonoloto** · Parámetros")
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
        do_calc_b = st.form_submit_button("Calcular recomendaciones · Bonoloto")

    if do_calc_b:
        if len(set(nums_b)) != 6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        last_dt_b = pd.to_datetime(last_date_b)
        weekday = last_dt_b.weekday()
        next_dt_b = last_dt_b + pd.Timedelta(days=1)  # aproximación general
        next_dayname_b = next_dt_b.day_name()

        st.info(f"Próximo sorteo (aprox.): **{next_dt_b.date().strftime('%d/%m/%Y')}** ({next_dayname_b})")

        base_b = df_bono[df_bono["FECHA"] <= last_dt_b].sort_values("FECHA").copy()

        def has_dup(df, last_dt, nums, comp, rein):
            if df.empty: return False, False
            same = df["FECHA"].dt.date == last_dt.date()
            if not same.any(): return False, False
            row = df.loc[same].tail(1)
            try:
                match = (int(row["N1"].values[0])==nums[0] and int(row["N2"].values[0])==nums[1] and
                         int(row["N3"].values[0])==nums[2] and int(row["N4"].values[0])==nums[3] and
                         int(row["N5"].values[0])==nums[4] and int(row["N6"].values[0])==nums[5] and
                         int(row["Complementario"].values[0])==comp and int(row["Reintegro"].values[0])==rein)
            except Exception:
                match = False
            return True, match

        has_date_b, full_match_b = has_dup(base_b, last_dt_b, nums_b, comp_b, rein_b)
        if has_date_b and full_match_b:
            st.success("✅ Sorteo ya existe en histórico con la misma combinación. No se añade.")
            df_recent_b = base_b.tail(WINDOW_DRAWS)
        elif has_date_b and not full_match_b:
            st.warning("⚠️ Misma fecha con combinación distinta en el Sheet. Uso el Sheet (no añado).")
            df_recent_b = base_b.tail(WINDOW_DRAWS)
        else:
            row_now_b = pd.DataFrame([{
                "FECHA": last_dt_b, "N1": nums_b[0], "N2": nums_b[1], "N3": nums_b[2],
                "N4": nums_b[3], "N5": nums_b[4], "N6": nums_b[5],
                "Complementario": comp_b, "Reintegro": rein_b
            }])
            df_recent_b = pd.concat([base_b, row_now_b], ignore_index=True).sort_values("FECHA").tail(WINDOW_DRAWS)

        df_recent_b["weekday"] = df_recent_b["FECHA"].dt.weekday
        w_glob_b = weighted_counts_nums(df_recent_b, last_dt_b)
        w_day_b  = weighted_counts_nums(df_recent_b[df_recent_b["weekday"]==weekday], last_dt_b)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        A1b = A1_FIJAS_BONO.get((weekday+1) % 7, [4,24,35,37,40,46])

        def score_combo_b(c, w):
            return sum(np.log(w.get(n,0.0)+ALPHA_DIR) for n in c) - MU_PENALTY*popularity_penalty(c)

        def terciles_ok_b(c):
            return any(1 <= x <= 16 for x in c) and any(17 <= x <= 32 for x in c) and any(33 <= x <= 49 for x in c)

        def overlap_ratio_b(a,b):
            return len(set(a)&set(b))/6.0

        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*50:
            c = tuple(random_combo()); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok_b(c): continue
            if overlap_ratio_b(c, A1b) > (1 - MIN_DIV): continue
            cands_b.append(c)
        cands_b = sorted(cands_b, key=lambda c: score_combo_b(c, w_blend_b), reverse=True)
        pool_b = cands_b[:1000]

        def zscore_combo_b(c, w):
            allW = np.array([w.get(i,0.0) for i in range(1,50)])
            m=float(allW.mean()); sd=float(allW.std()) if allW.std()!=0 else 1e-6
            cm=float(np.mean([w.get(n,0.0) for n in c])) if c else 0.0
            return (cm-m)/sd

        bestA2_b = list(pool_b[0]) if pool_b else []
        zA2_b = zscore_combo_b(bestA2_b, w_blend_b) if bestA2_b else 0.0

        def pick_n_b(z, bank, vol):
            adj = 0.05 if vol=="Low" else -0.05 if vol=="High" else 0.0
            for th in THRESH_N:
                if z >= th["z"] + adj:
                    n = min(th["n"], int(bank))
                    return max(1, n)
            return 1

        n_b = max(1, min(6, pick_n_b(zA2_b, bank_b, vol_b)))

        def greedy_select_b(pool,w,n):
            if n<=0: return []
            sp=sorted(pool,key=lambda c:score_combo_b(c,w),reverse=True)
            sel=[sp[0]]
            while len(sel)<n:
                best=None; bestv=-1e9
                for c in sp:
                    if any(tuple(c)==tuple(s) for s in sel): continue
                    pen=sum(overlap_ratio_b(c,s) for s in sel)
                    v=score_combo_b(c,w)-LAMBDA_DIVERSIDAD*pen
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
        st.dataframe(df_out_b, use_container_width=True)
        st.download_button("Descargar combinaciones · Bonoloto (CSV)",
                           data=df_out_b.to_csv(index=False).encode("utf-8"),
                           file_name="bonoloto_recomendaciones.csv", mime="text/csv")
