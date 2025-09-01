# app.py — Primitiva & Bonoloto · Recomendador A2 (determinista + Joker + apuesta múltiple k)
# v3: recomendación Joker, k=6–8 (C(k,6)), coste correcto, métricas simples, UI por pestañas

import hashlib
from math import comb
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials


# -------------------------- Config & estilos --------------------------
st.set_page_config(page_title="Primitiva & Bonoloto · Recomendador A2", page_icon="🎯", layout="wide")

def _load_css():
    p = Path("styles.css")
    if p.exists():
        st.markdown(f"<style>{p.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

_load_css()


# -------------------------- Helpers generales --------------------------
def load_md(rel_path: str) -> str:
    p = Path(rel_path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")

def get_secret_key(name, group="gcp_service_account"):
    if name in st.secrets:
        return st.secrets[name]
    if group in st.secrets and name in st.secrets[group]:
        return st.secrets[group][name]
    return None

def get_gcp_credentials():
    # Permite gcp_json (opcional) o bloque [gcp_service_account]
    try:
        gcp_json = st.secrets.get("gcp_json", None)
    except Exception:
        gcp_json = None
    if gcp_json:
        import json
        info = json.loads(gcp_json)
        pk = info.get("private_key", "")
        if isinstance(pk, str) and "\\n" in pk:
            info["private_key"] = pk.replace("\\n", "\n")
        info["private_key"] = info["private_key"].strip()
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        return Credentials.from_service_account_info(info, scopes=scopes)

    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Falta el bloque [gcp_service_account] en Secrets.")

    info = dict(st.secrets["gcp_service_account"])
    pk = info.get("private_key", "")
    if isinstance(pk, str) and "\\n" in pk:
        info["private_key"] = pk.replace("\\n", "\n")
    info["private_key"] = info["private_key"].strip()
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    return Credentials.from_service_account_info(info, scopes=scopes)

@st.cache_data(ttl=600, show_spinner=True)
def load_sheet_df(sheet_id_key: str, worksheet_key: str, default_ws: str) -> pd.DataFrame:
    creds = get_gcp_credentials()
    gc = gspread.authorize(creds)
    sid = get_secret_key(sheet_id_key)
    wsn = get_secret_key(worksheet_key) or default_ws
    if not sid:
        st.error(f"No encuentro `{sheet_id_key}` en Secrets.")
        return pd.DataFrame()
    try:
        sh = gc.open_by_key(sid)
        ws = sh.worksheet(wsn)
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
    except Exception as e:
        st.error(f"No puedo abrir el Sheet/Worksheet ({sheet_id_key}/{worksheet_key}). Detalle: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    expected = ["FECHA","N1","N2","N3","N4","N5","N6","Complementario","Reintegro"]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FECHA"]).sort_values("FECHA").reset_index(drop=True)
    for c in expected[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[expected]

def append_if_new(sheet_id_key, worksheet_key, new_row: dict) -> bool:
    """Antiduplicado por fecha y combinación completa."""
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = get_secret_key(sheet_id_key); wsn = get_secret_key(worksheet_key)
        sh = gc.open_by_key(sid); ws = sh.worksheet(wsn)
        rows = ws.get_all_records(numericise_ignore=["FECHA"])
        df = pd.DataFrame(rows)
        if "FECHA" not in df.columns:
            return False
        df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
        same_date = df["FECHA"].dt.date == pd.to_datetime(new_row["FECHA"]).date()
        if same_date.any():
            last = df.loc[same_date].tail(1)
            try:
                match = (
                    int(last["N1"].values[0])==int(new_row["N1"]) and
                    int(last["N2"].values[0])==int(new_row["N2"]) and
                    int(last["N3"].values[0])==int(new_row["N3"]) and
                    int(last["N4"].values[0])==int(new_row["N4"]) and
                    int(last["N5"].values[0])==int(new_row["N5"]) and
                    int(last["N6"].values[0])==int(new_row["N6"]) and
                    int(last["Complementario"].values[0])==int(new_row["Complementario"]) and
                    int(last["Reintegro"].values[0])==int(new_row["Reintegro"])
                )
            except Exception:
                match = False
            if match:
                return False
        ws.append_row([
            pd.to_datetime(new_row["FECHA"]).strftime("%d/%m/%Y"),
            int(new_row["N1"]), int(new_row["N2"]), int(new_row["N3"]),
            int(new_row["N4"]), int(new_row["N5"]), int(new_row["N6"]),
            int(new_row["Complementario"]), int(new_row["Reintegro"])
        ])
        return True
    except Exception:
        return False


# -------------------------- Parámetros del modelo --------------------------
WINDOW_DRAWS    = 24
HALF_LIFE_DAYS  = 60.0
DAY_BLEND_ALPHA = 0.30
ALPHA_DIR       = 0.30

# Valores por defecto (ajustables en UI)
MU_PENALTY_DEF         = 1.00   # Aversión a patrones populares
LAMBDA_DIVERSIDAD_DEF  = 0.60   # Penalización por solape entre A2

THRESH_N = [
  {"z": 0.50, "n": 6},
  {"z": 0.35, "n": 4},
  {"z": 0.20, "n": 3},
  {"z": 0.10, "n": 2},
  {"z":-999,  "n": 1},
]

A1_FIJAS_PRIMI = {
    "Monday":   [4,24,35,37,40,46],
    "Thursday": [1,10,23,39,45,48],
    "Saturday": [7,12,14,25,29,40],
}
REIN_FIJOS_PRIMI = {"Monday":1, "Thursday":8, "Saturday":0}

A1_FIJAS_BONO = {i:[4,24,35,37,40,46] for i in range(7)}  # ancla inicial por día


# -------------------------- Utilidades de modelo --------------------------
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
    return 1.2*p_dates + 0.8*consec + 0.5*max(0, max_dec-2) + 0.5*max(0, max_unit-2) + 0.4*roundness

def score_combo(combo, weights, mu_penalty):
    return sum(np.log(weights.get(n,0.0) + ALPHA_DIR) for n in combo) - mu_penalty*popularity_penalty(combo)

def terciles_ok(combo):
    return any(1<=x<=16 for x in combo) and any(17<=x<=32 for x in combo) and any(33<=x<=49 for x in combo)

def overlap_ratio(a,b): return len(set(a)&set(b))/6.0

def zscore_combo(combo, weights):
    allW = np.array([weights.get(i,0.0) for i in range(1,50)], dtype=float)
    meanW = float(allW.mean()); sdW = float(allW.std()) if allW.std()!=0 else 1e-6
    comboMean = float(np.mean([weights.get(n,0.0) for n in combo])) if combo else 0.0
    return (comboMean - meanW)/sdW

def to_js_day(dayname):
    return 1 if dayname=="Monday" else 4 if dayname=="Thursday" else 6 if dayname=="Saturday" else -1

def get_rng(seed_str: str):
    # Semilla determinista a partir de texto estable (juego|fecha|nums|…)
    h = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    seed_int = int(h[:12], 16)
    return np.random.default_rng(seed_int)

def random_combo_rng(rng):
    pool = list(range(1,50)); out=[]
    while len(out)<6:
        i = rng.integers(0, len(pool))
        out.append(pool.pop(i))
    return sorted(out)


# -------------------------- UI --------------------------
st.title("🎯 Primitiva & Bonoloto · Recomendador A2 (determinista)")
st.caption("Ventana 24 sorteos · t½=60d · mezcla por día (30%) · antipopularidad · diversidad · apuesta múltiple k · Joker (Primitiva) · Google Sheets")

tab_primi, tab_bono, tab_help = st.tabs(["La Primitiva", "Bonoloto", "Ayuda"])


# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader("La Primitiva · Recomendador A2")
    st.caption("A1 fija por día · A2 dinámica · Joker opcional · Apuesta múltiple k=6–8 (coste = C(k,6) por boleto)")

    df_hist = load_sheet_df("sheet_id", "worksheet_historico", "Historico")
    if df_hist.empty:
        st.stop()

    # Sidebar — parámetros claros
    with st.sidebar:
        st.markdown("### Primitiva · Parámetros")
        c1, c2 = st.columns(2)
        presupuesto = c1.number_input("Presupuesto por sorteo (€)", min_value=0, value=10, step=1, key="bank_pri")
        vol = c2.selectbox("Ritmo de inversión", ["Bajo","Medio","Alto"], index=1,
                           help="Bajo: conservador · Medio: estándar · Alto: agresivo", key="vol_pri")
        precio_apuesta = c1.number_input("Precio por apuesta simple (€)", min_value=0.0, value=1.0, step=0.5)

        st.markdown("#### Apuesta múltiple (opcional)")
        apuesta_multiple_on = st.checkbox("Usar apuesta múltiple (k>6)", value=False,
                                          help="Permite jugar 7 u 8 números en una sola apuesta; el sistema genera todas las combinaciones de 6. Coste = C(k,6) × precio simple.")
        k_tamano = st.slider("Números por apuesta (k)", 6, 8, 6, help="k=6 (simple), k=7 (×7 coste), k=8 (×28 coste).")

        st.markdown("#### Joker")
        usar_joker = st.checkbox("Añadir Joker", value=False,
                                 help="Juego paralelo con coste fijo por boleto. Lo recomendaremos solo si hay señal y margen.")
        precio_joker = st.number_input("Precio Joker (€)", min_value=0.0, value=1.0, step=0.5)

        st.markdown("---")
        mu_penalty = st.slider("Aversión a patrones populares", 0.0, 2.0, MU_PENALTY_DEF, 0.05,
                               help="Más alto = más castigo a fechas/secuencias/sumas típicas.")
        lambda_div = st.slider("Diversificación entre A2", 0.0, 1.5, LAMBDA_DIVERSIDAD_DEF, 0.05,
                               help="Más alto = menos solapadas las A2.")
        modo_det = st.checkbox("Modo determinista (recomendado)", value=True,
                               help="Misma entrada ⇒ mismas recomendaciones.")

    # Formulario de entrada con botón "Cargar último del histórico"
    with st.form("entrada_primi"):
        c1, c2 = st.columns(2)
        last_from_hist = c1.form_submit_button("⬇️ Cargar último del histórico")
        if last_from_hist and not df_hist.empty:
            last_row = df_hist.tail(1).iloc[0]
            st.session_state["last_date_pri"] = pd.to_datetime(last_row["FECHA"]).date()
            st.session_state["rein_pri"] = int(last_row["Reintegro"])
            st.session_state["comp_pri"] = int(last_row["Complementario"])
            st.session_state["nums_pri"] = [int(last_row[f"N{i}"]) for i in range(1,7)]

        last_date = c1.date_input("Fecha último sorteo (Lun/Jue/Sáb)",
                                  value=st.session_state.get("last_date_pri", pd.Timestamp.today().date()))
        rein = c2.number_input("Reintegro (0–9)", 0, 9, st.session_state.get("rein_pri", 2), 1)
        comp = c2.number_input("Complementario (1–49)", 1, 49, st.session_state.get("comp_pri", 18), 1)

        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults = st.session_state.get("nums_pri", [5,6,8,23,46,47])
        nums = [cols[i].number_input(f"N{i+1}", 1, 49, defaults[i], 1, key=f"npri{i+1}") for i in range(6)]

        guardar = st.checkbox("Guardar en histórico si es nuevo", value=False)
        do_calc = st.form_submit_button("Calcular recomendaciones · Primitiva")

    if do_calc:
        if len(set(nums)) != 6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        # Persisto entradas en sesión para no volver a pedirlas
        st.session_state["last_date_pri"] = last_date
        st.session_state["rein_pri"] = rein
        st.session_state["comp_pri"] = comp
        st.session_state["nums_pri"] = nums

        last_dt = pd.to_datetime(last_date)
        wd = last_dt.weekday()
        if wd==0: next_dt, next_dayname = last_dt + pd.Timedelta(days=3), "Thursday"
        elif wd==3: next_dt, next_dayname = last_dt + pd.Timedelta(days=2), "Saturday"
        elif wd==5: next_dt, next_dayname = last_dt + pd.Timedelta(days=2), "Monday"
        else:
            st.error("La fecha debe ser Lunes, Jueves o Sábado.")
            st.stop()

        if guardar:
            inserted = append_if_new("sheet_id","worksheet_historico",{
                "FECHA": last_dt, "N1":nums[0],"N2":nums[1],"N3":nums[2],
                "N4":nums[3],"N5":nums[4],"N6":nums[5],"Complementario":comp,"Reintegro":rein
            })
            if inserted:
                st.success("✅ Añadido al histórico (Primitiva).")
                st.cache_data.clear()
                df_hist = load_sheet_df("sheet_id","worksheet_historico","Historico")
            else:
                st.info("ℹ️ No se añadió (duplicado o sin permisos).")

        # Ventana de referencia y pesos
        base = df_hist[df_hist["FECHA"] <= last_dt].sort_values("FECHA").copy()
        df_recent = pd.concat([base, pd.DataFrame([{
            "FECHA": last_dt, "N1":nums[0],"N2":nums[1],"N3":nums[2],
            "N4":nums[3],"N5":nums[4],"N6":nums[5],"Complementario":comp,"Reintegro":rein
        }])], ignore_index=True).sort_values("FECHA").tail(WINDOW_DRAWS)
        df_recent["weekday"] = df_recent["FECHA"].dt.weekday
        w_glob = weighted_counts_nums(df_recent, last_dt)
        w_day  = weighted_counts_nums(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)

        # Determinismo (semilla estable)
        seed_str = f"PRIMI|{str(last_dt.date())}|{nums}|{comp}|{rein}|{DAY_BLEND_ALPHA}|{mu_penalty}|{lambda_div}|{k_tamano}|{apuesta_multiple_on}"
        rng = get_rng(seed_str) if modo_det else np.random.default_rng()

        # A1 fija
        A1 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])

        # Candidatos A2 (deterministas)
        K_CANDIDATOS = 3000
        MIN_DIV = 0.60
        cands, seen, tries = [], set(), 0
        while len(cands)<K_CANDIDATOS and tries < K_CANDIDATOS*50:
            c = tuple(random_combo_rng(rng)); tries += 1
            if c in seen: continue
            seen.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1) > (1 - MIN_DIV): continue
            cands.append(c)
        cands = sorted(cands, key=lambda c: score_combo(c, w_blend, mu_penalty), reverse=True)

        # Tabs de salida
        tab_res, tab_ap, tab_met, tab_win = st.tabs(["Recomendación", "Apuestas", "Métricas", "Ventana de referencia"])

        with tab_res:
            pool = cands[:1000]
            bestA2 = list(pool[0]) if pool else []
            zA2 = zscore_combo(bestA2, w_blend) if bestA2 else 0.0

            # nº boletos (A1 + A2s) vía señal y ritmo
            def pick_n(z, presupuesto, ritmo):
                adj = 0.05 if ritmo=="Bajo" else -0.05 if ritmo=="Alto" else 0.0
                for th in THRESH_N:
                    if z >= th["z"] + adj:
                        n = min(th["n"], int(presupuesto))  # acotado por presupuesto entero
                        return max(1, n)
                return 1
            n_boletos_sugeridos = pick_n(zA2, presupuesto, vol)
            n_boletos_sugeridos = max(1, min(6, n_boletos_sugeridos))

            # Selección greedy con diversidad (n-1 A2 + A1)
            def greedy_select(pool, w, n, lam):
                if n<=0: return []
                sp = sorted(pool, key=lambda c: score_combo(c,w,mu_penalty), reverse=True)
                sel=[sp[0]]
                while len(sel)<n:
                    best=None; bestv=-1e9
                    for c in sp:
                        if any(tuple(c)==tuple(s) for s in sel): continue
                        pen=sum(overlap_ratio(c,s) for s in sel)
                        v=score_combo(c,w,mu_penalty) - lam*pen
                        if v>bestv: bestv=v; best=c
                    if best is None: break
                    sel.append(best)
                return sel
            A2s = greedy_select(pool, w_blend, max(0, n_boletos_sugeridos-1), lambda_div)

            # Reintegro (informativo)
            wr_glob = weighted_counts_rei(df_recent, last_dt)
            wr_day  = weighted_counts_rei(df_recent[df_recent["weekday"]==to_js_day(next_dayname)], last_dt)
            rei_scores = {r: DAY_BLEND_ALPHA*wr_day.get(r,0.0) + (1-DAY_BLEND_ALPHA)*wr_glob.get(r,0.0) for r in range(10)}
            rein_sug = max(rei_scores, key=lambda r: rei_scores[r]) if rei_scores else 0
            rein_ref = REIN_FIJOS_PRIMI.get(next_dayname, "")

            # Apuesta múltiple: coste real
            k_eff = k_tamano if (apuesta_multiple_on and k_tamano>6) else 6
            simples_por_boleto = comb(k_eff, 6)  # 1, 7, 28
            n_boletos = 1 + len(A2s)            # A1 + A2s
            coste_apuestas = n_boletos * simples_por_boleto * precio_apuesta
            coste = coste_apuestas + (precio_joker if usar_joker else 0.0)

            # Recomendación Joker (regla determinista)
            def recomendar_joker(z, presupuesto, coste_actual, ritmo):
                usar_riesgo = (ritmo in ["Medio","Alto"])
                margen = presupuesto - coste_actual
                return (z >= 0.35) and usar_riesgo and (margen >= precio_joker)

            joker_recomendado = recomendar_joker(zA2, presupuesto, coste_apuestas, vol)

            # Sugerencia automática de k (no forzamos, solo informamos)
            k_sugerida = 6
            if apuesta_multiple_on:
                if zA2 >= 0.60 and presupuesto >= 40*precio_apuesta:
                    k_sugerida = 8
                elif zA2 >= 0.45 and presupuesto >= 12*precio_apuesta:
                    k_sugerida = 7

            # Métricas cabecera
            conf = "Alta" if zA2>=0.45 else "Media" if zA2>=0.25 else "Baja"
            c1, c2, c3 = st.columns(3)
            c1.metric("Boletos (A1 + A2)", n_boletos)
            c2.metric("Coste estimado (€)", f"{coste:.2f}")
            c3.metric("Confianza (señal)", conf)

            st.markdown(f"**A1 (ancla fija)**: {A1}")
            for i, c in enumerate(A2s, start=1):
                st.write(f"**A2 #{i}**: {list(c)}")

            st.caption(f"Tamaño de apuesta (k): **{k_eff}** → {simples_por_boleto} combinaciones simples por boleto.")
            if apuesta_multiple_on:
                st.caption(f"Sugerencia automática para k (según señal/presupuesto): **{k_sugerida}**.")
            st.caption(f"Reintegro (info): sugerido {rein_sug} · referencia del día {rein_ref}.")

            # Joker: mensaje claro
            if usar_joker:
                st.write("**Joker**: añadido por decisión del usuario.")
            else:
                st.write(f"**Joker recomendado**: {'Sí' if joker_recomendado else 'No'} "
                         f"(señal {zA2:.2f}, ritmo {vol}, "
                         f"{'margen OK' if (presupuesto - coste_apuestas) >= precio_joker else 'sin margen'}).")

        with tab_ap:
            # Tabla y descarga
            rows = [{"Tipo":"A1", "N1":A1[0],"N2":A1[1],"N3":A1[2],"N4":A1[3],"N5":A1[4],"N6":A1[5]}]
            for i, c in enumerate(A2s, start=1):
                cl = list(c)
                rows.append({"Tipo":f"A2-{i}","N1":cl[0],"N2":cl[1],"N3":cl[2],"N4":cl[3],"N5":cl[4],"N6":cl[5]})
            df_out = pd.DataFrame(rows)
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Descargar combinaciones (CSV)",
                               data=df_out.to_csv(index=False).encode("utf-8"),
                               file_name="primitiva_recomendaciones.csv", mime="text/csv")

            # Escenarios rápidos
            st.markdown("#### 🎛️ Escenarios")
            n_esc = st.slider("Forzar número de boletos (1–6)", 1, 6, (1+len(A2s)))
            k_esc = st.slider("Forzar k por boleto (6–8)", 6, 8, k_eff)
            simples_esc = comb(k_esc, 6)
            coste_esc = n_esc*simples_esc*precio_apuesta + (precio_joker if usar_joker else 0.0)
            st.info(f"Si juegas **{n_esc}** boletos con **k={k_esc}** (→ {simples_esc} simples/bolet.), el coste sería **{coste_esc:.2f} €**.")

        with tab_met:
            st.markdown("#### 📊 Métricas (explicadas)")
            st.info(
                "- **Señal**: cuanto mayor, más concentración reciente de esos números.\n"
                "- **Cobertura (k)**: con 7 u 8 números, cada boleto incluye más combinaciones (×7 o ×28 el coste).\n"
                "- **Diversificación**: evitamos que las A2 se pisen entre sí.\n"
                "- **Joker**: premio independiente; lo recomendamos solo si hay señal y margen."
            )
            st.metric("Señal (z) A2 top", f"{zA2:.2f}")
            st.metric("Diversificación objetivo", f"{lambda_div:.2f}")
            st.metric("Aversión a populares", f"{mu_penalty:.2f}")

        with tab_win:
            st.markdown("#### 🧭 Ventana de referencia usada")
            dmin = df_recent["FECHA"].min().date() if not df_recent.empty else None
            dmax = df_recent["FECHA"].max().date() if not df_recent.empty else None
            st.write(f"Rango: **{dmin} → {dmax}**  ·  Sorteos: **{len(df_recent)}**")
            top = sorted(w_blend.items(), key=lambda kv: kv[1], reverse=True)[:10]
            st.table(pd.DataFrame([{"Número":k, "Peso":round(v,4)} for k, v in top]))


# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader("Bonoloto · Recomendador A2")
    st.caption("A1 ancla inicial por día · A2 dinámica · sin Joker · Apuesta múltiple k=6–8 (coste = C(k,6))")

    df_bono = load_sheet_df("sheet_id_bono","worksheet_historico_bono","HistoricoBono")
    if df_bono.empty:
        st.stop()

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Bonoloto · Parámetros")
        c1, c2 = st.columns(2)
        presupuesto_b = c1.number_input("Presupuesto por sorteo (€) · Bono", min_value=0, value=10, step=1)
        vol_b = c2.selectbox("Ritmo de inversión · Bono", ["Bajo","Medio","Alto"], index=1)
        precio_apuesta_b = c1.number_input("Precio por apuesta simple (€) · Bono", min_value=0.0, value=0.5, step=0.5)

        st.markdown("#### Apuesta múltiple (opcional) · Bono")
        apuesta_multiple_on_b = st.checkbox("Usar apuesta múltiple (k>6) · Bono", value=False)
        k_tamano_b = st.slider("Números por apuesta (k) · Bono", 6, 8, 6)

        st.markdown("---")
        mu_penalty_b = st.slider("Aversión a populares · Bono", 0.0, 2.0, MU_PENALTY_DEF, 0.05)
        lambda_div_b = st.slider("Diversificación entre A2 · Bono", 0.0, 1.5, LAMBDA_DIVERSIDAD_DEF, 0.05)
        modo_det_b = st.checkbox("Modo determinista · Bono", value=True)

    with st.form("entrada_bono"):
        c1, c2 = st.columns(2)
        last_from_hist_b = c1.form_submit_button("⬇️ Cargar último del histórico (Bono)")
        if last_from_hist_b and not df_bono.empty:
            last_row = df_bono.tail(1).iloc[0]
            st.session_state["last_date_b"] = pd.to_datetime(last_row["FECHA"]).date()
            st.session_state["rein_b"] = int(last_row["Reintegro"])
            st.session_state["comp_b"] = int(last_row["Complementario"])
            st.session_state["nums_b"] = [int(last_row[f"N{i}"]) for i in range(1,7)]

        last_date_b = c1.date_input("Fecha último sorteo (Bonoloto)",
                                    value=st.session_state.get("last_date_b", pd.Timestamp.today().date()))
        rein_b = c2.number_input("Reintegro (0–9)", 0, 9, st.session_state.get("rein_b", 2), 1)
        comp_b = c2.number_input("Complementario (1–49)", 1, 49, st.session_state.get("comp_b", 18), 1)

        st.markdown("**Números extraídos (6 distintos)**")
        cols = st.columns(6)
        defaults_b = st.session_state.get("nums_b", [10,13,17,30,41,44])
        nums_b = [cols[i].number_input(f"N{i+1} (Bono)", 1, 49, defaults_b[i], 1, key=f"nbo{i+1}") for i in range(6)]

        guardar_b = st.checkbox("Guardar en histórico (Bonoloto) si es nuevo", value=False)
        do_calc_b = st.form_submit_button("Calcular recomendaciones · Bonoloto")

    if do_calc_b:
        if len(set(nums_b)) != 6:
            st.error("Los 6 números deben ser distintos.")
            st.stop()

        st.session_state["last_date_b"] = last_date_b
        st.session_state["rein_b"] = rein_b
        st.session_state["comp_b"] = comp_b
        st.session_state["nums_b"] = nums_b

        last_dt_b = pd.to_datetime(last_date_b)
        next_dt_b = last_dt_b + pd.Timedelta(days=1)
        next_dayname_b = next_dt_b.day_name()

        if guardar_b:
            inserted_b = append_if_new("sheet_id_bono","worksheet_historico_bono",{
                "FECHA": last_dt_b, "N1":nums_b[0],"N2":nums_b[1],"N3":nums_b[2],
                "N4":nums_b[3],"N5":nums_b[4],"N6":nums_b[5],"Complementario":comp_b,"Reintegro":rein_b
            })
            if inserted_b:
                st.success("✅ Añadido al histórico (Bonoloto).")
                st.cache_data.clear()
                df_bono = load_sheet_df("sheet_id_bono","worksheet_historico_bono","HistoricoBono")
            else:
                st.info("ℹ️ No se añadió (duplicado o sin permisos).")

        base_b = df_bono[df_bono["FECHA"] <= last_dt_b].sort_values("FECHA").copy()
        df_recent_b = pd.concat([base_b, pd.DataFrame([{
            "FECHA": last_dt_b, "N1":nums_b[0],"N2":nums_b[1],"N3":nums_b[2],
            "N4":nums_b[3],"N5":nums_b[4],"N6":nums_b[5],"Complementario":comp_b,"Reintegro":rein_b
        }])], ignore_index=True).sort_values("FECHA").tail(WINDOW_DRAWS)
        df_recent_b["weekday"] = df_recent_b["FECHA"].dt.weekday

        w_glob_b = weighted_counts_nums(df_recent_b, last_dt_b)
        w_day_b  = weighted_counts_nums(df_recent_b[df_recent_b["weekday"]==last_dt_b.weekday()], last_dt_b)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        seed_str_b = f"BONO|{str(last_dt_b.date())}|{nums_b}|{comp_b}|{rein_b}|{DAY_BLEND_ALPHA}|{mu_penalty_b}|{lambda_div_b}|{k_tamano_b}|{apuesta_multiple_on_b}"
        rng_b = get_rng(seed_str_b) if modo_det_b else np.random.default_rng()

        A1b = A1_FIJAS_BONO.get((last_dt_b.weekday()+1)%7, [4,24,35,37,40,46])

        K_CANDIDATOS = 3000
        MIN_DIV = 0.60
        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*50:
            c = tuple(random_combo_rng(rng_b)); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1b) > (1 - MIN_DIV): continue
            cands_b.append(c)
        cands_b = sorted(cands_b, key=lambda c: score_combo(c, w_blend_b, mu_penalty_b), reverse=True)

        tab_res_b, tab_ap_b, tab_met_b, tab_win_b = st.tabs(["Recomendación", "Apuestas", "Métricas", "Ventana de referencia"])

        with tab_res_b:
            pool_b = cands_b[:1000]
            bestA2_b = list(pool_b[0]) if pool_b else []
            zA2_b = zscore_combo(bestA2_b, w_blend_b) if bestA2_b else 0.0

            def pick_n_b(z, presupuesto, ritmo):
                adj = 0.05 if ritmo=="Bajo" else -0.05 if ritmo=="Alto" else 0.0
                for th in THRESH_N:
                    if z >= th["z"] + adj:
                        n = min(th["n"], int(presupuesto))
                        return max(1, n)
                return 1
            n_b = pick_n_b(zA2_b, presupuesto_b, vol_b)
            n_b = max(1, min(6, n_b))

            def greedy_select_b(pool,w,n, lam):
                if n<=0: return []
                sp=sorted(pool,key=lambda c:score_combo(c,w,mu_penalty_b),reverse=True)
                sel=[sp[0]]
                while len(sel)<n:
                    best=None; bestv=-1e9
                    for c in sp:
                        if any(tuple(c)==tuple(s) for s in sel): continue
                        pen=sum(overlap_ratio(c,s) for s in sel)
                        v=score_combo(c,w,mu_penalty_b)-lam*pen
                        if v>bestv: bestv=v; best=c
                    if best is None: break
                    sel.append(best)
                return sel

            A2s_b = greedy_select_b(pool_b, w_blend_b, max(0, n_b-1), lambda_div_b)

            # Apuesta múltiple: coste real
            k_eff_b = k_tamano_b if (apuesta_multiple_on_b and k_tamano_b>6) else 6
            simples_por_boleto_b = comb(k_eff_b, 6)
            n_boletos_b = 1 + len(A2s_b)
            coste_b = n_boletos_b * simples_por_boleto_b * precio_apuesta_b

            conf_b = "Alta" if zA2_b>=0.45 else "Media" if zA2_b>=0.25 else "Baja"
            c1, c2, c3 = st.columns(3)
            c1.metric("Boletos (A1 + A2)", n_boletos_b)
            c2.metric("Coste estimado (€)", f"{coste_b:.2f}")
            c3.metric("Confianza (señal)", conf_b)

            st.markdown(f"**A1 (ancla)**: {A1b}")
            for i, c in enumerate(A2s_b, start=1):
                st.write(f"**A2 #{i}**: {list(c)}")

            st.caption(f"Tamaño de apuesta (k): **{k_eff_b}** → {simples_por_boleto_b} simples/bolet.")

        with tab_ap_b:
            rows_b = [{"Tipo":"A1","N1":A1b[0],"N2":A1b[1],"N3":A1b[2],"N4":A1b[3],"N5":A1b[4],"N6":A1b[5]}]
            for i, c in enumerate(A2s_b, start=1):
                cl = list(c)
                rows_b.append({"Tipo":f"A2-{i}","N1":cl[0],"N2":cl[1],"N3":cl[2],"N4":cl[3],"N5":cl[4],"N6":cl[5]})
            df_out_b = pd.DataFrame(rows_b)
            st.dataframe(df_out_b, use_container_width=True)
            st.download_button("Descargar combinaciones (CSV) · Bonoloto",
                               data=df_out_b.to_csv(index=False).encode("utf-8"),
                               file_name="bonoloto_recomendaciones.csv", mime="text/csv")

            n_esc_b = st.slider("Forzar nº de boletos (1–6) · Bono", 1, 6, (1+len(A2s_b)))
            k_esc_b = st.slider("Forzar k por boleto (6–8) · Bono", 6, 8, k_eff_b)
            simples_esc_b = comb(k_esc_b, 6)
            st.info(f"Si juegas **{n_esc_b}** boletos con **k={k_esc_b}** (→ {simples_esc_b} simples/bolet.), el coste sería **{n_esc_b*simples_esc_b*precio_apuesta_b:.2f} €**.")

        with tab_met_b:
            st.markdown("#### 📊 Métricas (explicadas)")
            st.info(
                "- **Señal**: mayor = más peso reciente.\n"
                "- **Cobertura (k)**: 7 u 8 números multiplican combinaciones y coste.\n"
                "- **Diversificación**: evita solapes entre A2."
            )
            st.metric("Señal (z) A2 top", f"{zA2_b:.2f}")
            st.metric("Diversificación objetivo", f"{lambda_div_b:.2f}")
            st.metric("Aversión a populares", f"{mu_penalty_b:.2f}")

        with tab_win_b:
            dmin = df_recent_b["FECHA"].min().date() if not df_recent_b.empty else None
            dmax = df_recent_b["FECHA"].max().date() if not df_recent_b.empty else None
            st.write(f"Rango: **{dmin} → {dmax}**  ·  Sorteos: **{len(df_recent_b)}**")
            top_b = sorted(w_blend_b.items(), key=lambda kv: kv[1], reverse=True)[:10]
            st.table(pd.DataFrame([{"Número":k, "Peso":round(v,4)} for k, v in top_b]))


# =========================== AYUDA ===========================
with tab_help:
    st.subheader("Centro de ayuda")
    st.markdown(load_md("assets/quickstart.md") or "Inicio rápido: introduce el último sorteo, calcula y descarga CSV.")
    with st.expander("📘 Tutorial completo"):
        st.markdown(load_md("assets/tutorial_full.md") or "Tutorial en preparación.")
    with st.expander("❓ Preguntas frecuentes"):
        st.markdown(load_md("assets/faq.md") or "FAQ en preparación.")
