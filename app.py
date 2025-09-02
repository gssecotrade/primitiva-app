# app.py — Francisco Cabrera · Predictor de La Primitiva & Bonoloto
# Recomendador determinista + Lift, Joker, Simulador, Tutorial
# NUEVO: Selección final del jugador + Bitácora (registro en Google Sheets) + Mini-informe

import math
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import io, csv, ast

import gspread
from google.oauth2.service_account import Credentials

# -------------------------- ESTILO / BRANDING --------------------------
st.set_page_config(page_title='Francisco Cabrera · Predictor de La Primitiva & Bonoloto',
                   page_icon='🎯', layout='wide')

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif !important; }
.block-container { padding-top: 1.0rem; }
h1, h2, h3 { font-weight: 600; }
.sidebar .sidebar-content { width: 360px; }
.small-muted { color: #94a3b8; font-size: 0.85rem; }
.kpill { display:inline-block; background:#0ea5e9; color:white; padding:2px 8px; border-radius:99px; font-size:0.8rem; }
.readonly { opacity: 0.85; }
</style>
''', unsafe_allow_html=True)

# Header (branding)
st.markdown('''
### **Francisco Cabrera · Predictor de La Primitiva & Bonoloto**
<span class="small-muted">Estrategia A1/A2 con ventana móvil, mezcla por día, diversidad, selección determinista y recomendación de Joker por apuesta. Fuente: Google Sheets.</span>
''', unsafe_allow_html=True)

# -------------------------- DETERMINISMO --------------------------
def stable_seed(*parts) -> int:
    m = hashlib.sha256()
    for p in parts:
        m.update(str(p).encode('utf-8'))
    return int.from_bytes(m.digest()[:8], 'big', signed=False)

# -------------------------- CONSTANTES MODELO --------------------------
WINDOW_DRAWS_DEF    = 24
HALF_LIFE_DAYS_DEF  = 60.0
DAY_BLEND_ALPHA_DEF = 0.30
ALPHA_DIR_DEF       = 0.30
MU_PENALTY_DEF      = 1.00
K_CANDIDATOS        = 1500
MIN_DIV             = 0.60
LAMBDA_DIVERSIDAD_DEF = 0.60
THRESH_N = [
  {'z': 0.50, 'n': 6},
  {'z': 0.35, 'n': 4},
  {'z': 0.20, 'n': 3},
  {'z': 0.10, 'n': 2},
  {'z':-999,  'n': 1},
]

A1_FIJAS_PRIMI = {'Monday':[4,24,35,37,40,46],'Thursday':[1,10,23,39,45,48],'Saturday':[7,12,14,25,29,40]}
REIN_FIJOS_PRIMI = {'Monday':1, 'Thursday':8, 'Saturday':0}
A1_FIJAS_BONO = {i:[4,24,35,37,40,46] for i in range(7)}

# -------------------------- HELPERS --------------------------
def comb(n, k):
    try: return math.comb(n, k)
    except Exception:
        from math import factorial
        return factorial(n)//(factorial(k)*factorial(n-k))

def dayname_to_weekday(dn: str) -> int:
    return {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}.get(dn, -1)

def weekday_to_dayname(w: int) -> str:
    return ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][w]

def time_weight(d, ref, half_life_days):
    delta = max(0, (ref - d).days)
    return float(np.exp(-np.log(2)/half_life_days * delta))

def weighted_counts_nums(df_in, ref, half_life_days):
    w = {i:0.0 for i in range(1,50)}
    for _, r in df_in.iterrows():
        tw = time_weight(r['FECHA'], ref, half_life_days)
        for c in ['N1','N2','N3','N4','N5','N6']:
            if not pd.isna(r[c]):
                w[int(r[c])] += tw
    return w

def weighted_counts_rei(df_in, ref, half_life_days):
    w = {i:0.0 for i in range(10)}
    if 'Reintegro' in df_in.columns:
        for _, r in df_in.dropna(subset=['Reintegro']).iterrows():
            tw = time_weight(r['FECHA'], ref, half_life_days)
            w[int(r['Reintegro'])] += tw
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

def random_combo(rng):
    pool = list(range(1,50)); out=[]
    while len(out)<6:
        i = int(rng.integers(0, len(pool)))
        out.append(pool.pop(i))
    return sorted(out)

def overlap_ratio(a,b): 
    return len(set(a) & set(b))/6.0

def zscore_combo(combo, weights):
    allW = np.array([weights.get(i,0.0) for i in range(1,50)], dtype=float)
    meanW = float(allW.mean()); sdW = float(allW.std()) if allW.std()!=0 else 1e-6
    comboMean = float(np.mean([weights.get(n,0.0) for n in combo])) if combo else 0.0
    return (comboMean - meanW)/sdW

def pick_n(z, bank, vol, thresh_table):
    adj = 0.05 if vol=='Low' else -0.05 if vol=='High' else 0.0
    for th in thresh_table:
        if z >= th['z'] + adj:
            n = min(th['n'], int(bank))
            return max(1, n)
    return 1

def greedy_select(pool, weights, n, alpha_dir, mu_penalty, lambda_div):
    if n<=0: return []
    sorted_pool = sorted(pool, key=lambda c: (score_combo(c,weights,alpha_dir,mu_penalty), tuple(c)), reverse=True)
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
    if k<=6: 
        return list(base6[:6])
    extras = [n for n in range(1,50) if n not in base6]
    extras_sorted = sorted(extras, key=lambda x: weights.get(x,0.0), reverse=True)
    add = extras_sorted[:max(0,k-6)]
    out = sorted(list(set(base6) | set(add)))
    return out[:k]

def conf_label(z):
    if z>=0.50: return 'Alta'
    if z>=0.20: return 'Media'
    return 'Baja'

# ---- Joker & Lift helpers ----
def minmax_norm(x, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def compute_rein_probs(df_recent, ref_dt, weekday_mask, half_life_days, alpha_day):
    wr_glob = weighted_counts_rei(df_recent, ref_dt, half_life_days)
    wr_day  = weighted_counts_rei(df_recent[df_recent['weekday']==weekday_mask], ref_dt, half_life_days)
    rr = {r: alpha_day*wr_day.get(r,0.0) + (1-alpha_day)*wr_glob.get(r,0.0) for r in range(10)}
    return rr

def joker_score(combo, weights, rein_dict):
    z = zscore_combo(combo, weights)
    zN = minmax_norm(z, -1.5, 1.5)
    if rein_dict:
        top = max(rein_dict.values())
        reinN = minmax_norm(top, 0.0, top if top>0 else 1.0)
    else:
        reinN = 0.0
    return 0.6*zN + 0.4*reinN

def random_baseline_scores(weights, alpha_dir, mu_penalty, rng, n=800):
    out = []
    tries = 0
    while len(out) < n and tries < n*20:
        c = tuple(random_combo(rng)); tries += 1
        if not terciles_ok(c): 
            continue
        out.append(score_combo(c, weights, alpha_dir, mu_penalty))
    if not out:
        return None, None
    arr = np.array(out, dtype=float)
    return arr, float(np.median(np.exp(arr)))

def lift_text_only(sc, baseline_median, pool_scores=None):
    if not baseline_median or baseline_median <= 0:
        return '—', '—'
    val = math.exp(sc)
    ratio = val / baseline_median
    ratio = max(min(ratio, 3.0), 0.33)
    lift_str = f'×{ratio:.2f}'
    if pool_scores is not None and len(pool_scores) > 0:
        exp_pool = np.exp(np.array(pool_scores, dtype=float))
        rank = (exp_pool < val).mean()
        top_pct = max(1.0 - rank, 0.0001) * 100.0
        pct_str = f'top {top_pct:.1f}% del azar'
    else:
        pct_str = '—'
    return lift_str, pct_str

def _parse_lift_num(txt):
    try:
        return float(str(txt).replace("×","").strip())
    except:
        return 1.0

def _parse_pct_num(txt):
    try:
        s=str(txt).lower().replace("top","").replace("%","").replace("del azar","").strip()
        val=float(s)
        return max(min(val,100.0),0.0)
    except:
        return None

def ev_proxy(p_adj, coste):
    try:
        coste = float(coste)
        return float(p_adj)/coste if coste>0 else 0.0
    except:
        return 0.0


# -------------------------- GOOGLE SHEETS --------------------------
def get_gcp_credentials():
    import json as _json
    scopes = ['https://www.googleapis.com/auth/spreadsheets']
    info = None
    if 'gcp_service_account' in st.secrets:
        info = dict(st.secrets['gcp_service_account'])
        pk = info.get('private_key','')
        if isinstance(pk,str) and '\\n' in pk:
            info['private_key']=pk.replace('\\\\n','\\n')
    elif 'gcp_json' in st.secrets:
        info = _json.loads(st.secrets['gcp_json'])
        if isinstance(info.get('private_key',''), str) and '\\n' in info['private_key']:
            info['private_key'] = info['private_key'].replace('\\\\n','\\n')
    else:
        raise RuntimeError('Faltan credenciales: añade [gcp_service_account] o gcp_json en Secrets.')
    return Credentials.from_service_account_info(info, scopes=scopes)

@st.cache_data(ttl=600, show_spinner=False)
def load_sheet_df(sheet_id_key: str, worksheet_key: str, default_ws: str):
    creds = get_gcp_credentials()
    gc = gspread.authorize(creds)
    sid = (st.secrets.get('gcp_service_account', {}) or {}).get(sheet_id_key) or st.secrets.get(sheet_id_key)
    wsn = (st.secrets.get('gcp_service_account', {}) or {}).get(worksheet_key, default_ws) or st.secrets.get(worksheet_key, default_ws)
    if not sid:
        return pd.DataFrame()
    try:
        sh = gc.open_by_key(sid); ws = sh.worksheet(wsn)
        rows = ws.get_all_records(numericise_ignore=['FECHA'])
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    expected = ['FECHA','N1','N2','N3','N4','N5','N6','Complementario','Reintegro']
    for c in expected:
        if c not in df.columns: df[c]=np.nan
    df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
    for c in ['N1','N2','N3','N4','N5','N6','Complementario','Reintegro']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['FECHA']).sort_values('FECHA').reset_index(drop=True)
    return df[expected]

def append_row_if_new(sheet_id_key, worksheet_key, default_ws, row_dict):
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get('gcp_service_account', {}) or {}).get(sheet_id_key) or st.secrets.get(sheet_id_key)
        wsn = (st.secrets.get('gcp_service_account', {}) or {}).get(worksheet_key, default_ws) or st.secrets.get(worksheet_key, default_ws)
        sh = gc.open_by_key(sid); ws = sh.worksheet(wsn)
        rows = ws.get_all_records(numericise_ignore=['FECHA'])
        df = pd.DataFrame(rows)
        if not df.empty:
            df['FECHA'] = pd.to_datetime(df['FECHA'], dayfirst=True, errors='coerce')
            same = df['FECHA'].dt.date == pd.to_datetime(row_dict['FECHA']).date()
            if same.any():
                last = df.loc[same].tail(1).to_dict('records')[0]
                keys = ['N1','N2','N3','N4','N5','N6','Complementario','Reintegro']
                match = all(int(last[k])==int(row_dict[k]) for k in keys if not pd.isna(row_dict[k]))
                if match: 
                    return False
        new_row = [
            pd.to_datetime(row_dict['FECHA']).strftime('%d/%m/%Y'),
            row_dict['N1'],row_dict['N2'],row_dict['N3'],row_dict['N4'],
            row_dict['N5'],row_dict['N6'],row_dict['Complementario'],row_dict['Reintegro']
        ]
        ws.append_row(new_row)
        return True
    except Exception:
        return False

def ensure_decisiones_sheet():
    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get('gcp_service_account', {}) or {}).get('sheet_id') or st.secrets.get('sheet_id')
        if not sid: 
            return None, None, None
        sh = gc.open_by_key(sid)
        try:
            ws = sh.worksheet('Decisiones')
        except Exception:
            ws = sh.add_worksheet(title='Decisiones', rows=2000, cols=16)
            ws.append_row(["TS","Juego","FechaSorteo","Tipo","k","Numeros","Simples","Lift","LiftNum","Score","Pct","PctNum","Joker","Coste(€)","A1","Rein_A1_ref","Rein_dyn","Window","HalfLife","AlphaDay","AlphaDir","MuPenalty","LambdaDiv","Bank","Vol","PrecioSimple","PrecioJoker","BaselineMedian","Seed","P_base","P_ajustada","EV_proxy","EV_proxy_Joker","Policy","Version"])
        return gc, sh, ws
    except Exception:
        return None, None, None

def append_decision(row):
    gc, sh, ws = ensure_decisiones_sheet()
    if ws is None:
        return False
    ws.append_row(row)
    return True

# -------------------------- SIDEBAR --------------------------
with st.sidebar:
    st.subheader('Parámetros · Primitiva')
    bank_pr = st.number_input('Banco (€) · Primitiva', min_value=0, value=10, step=1)
    vol_pr  = st.selectbox('Volatilidad · Primitiva', ['Low','Medium','High'], index=1)
    precio_simple = st.number_input('Precio por apuesta simple (€)', min_value=0.0, value=1.0, step=0.5, format='%.2f')

    st.markdown('---')
    st.subheader('Apuesta múltiple (opcional)')
    use_multi = st.checkbox('Usar apuesta múltiple (k>6)', value=True)
    k_nums    = st.slider('Números por apuesta (k)', min_value=6, max_value=8, value=8, step=1, disabled=not use_multi)

    st.markdown('---')
    st.subheader('Joker (Primitiva)')
    use_joker   = st.checkbox('Activar recomendaciones de Joker por apuesta', value=True)
    joker_thr   = st.slider('Umbral para recomendar Joker', 0.00, 1.00, 0.65, 0.01)
    precio_joker  = st.number_input('Precio Joker (€)', min_value=1.0, value=1.0, step=1.0, format='%.2f')

    st.markdown('---')
    with st.expander('Parámetros avanzados (simulación)', expanded=False):
        WINDOW_DRAWS    = st.slider('Ventana (nº de sorteos usados)', 12, 120, WINDOW_DRAWS_DEF, 1)
        HALF_LIFE_DAYS  = float(st.slider('Vida media temporal (días)', 15, 180, int(HALF_LIFE_DAYS_DEF), 1))
        DAY_BLEND_ALPHA = float(st.slider('Mezcla por día (α)', 0.0, 1.0, float(DAY_BLEND_ALPHA_DEF), 0.05))
        ALPHA_DIR       = float(st.slider('Suavizado pseudo-frecuencias (α_dir)', 0.00, 1.00, float(ALPHA_DIR_DEF), 0.01))
        MU_PENALTY      = float(st.slider('Penalización "popularidad"', 0.0, 2.0, float(MU_PENALTY_DEF), 0.1))
        LAMBDA_DIVERSIDAD = float(st.slider('Peso diversidad (λ)', 0.0, 2.0, float(LAMBDA_DIVERSIDAD_DEF), 0.1))

    st.markdown('---')
    st.subheader('Parámetros · Bonoloto')
    bank_bo = st.number_input('Banco (€) · Bonoloto', min_value=0, value=10, step=1, key='bank_bono')
    vol_bo  = st.selectbox('Volatilidad · Bonoloto', ['Low','Medium','High'], index=1, key='vol_bono')
    precio_simple_bono = st.number_input('Precio por apuesta simple (Bonoloto) €', min_value=0.0, value=0.50, step=0.5, format='%.2f',
                                         help='Bonoloto: múltiplos de 0,50€')

# -------------------------- TABS --------------------------
tab_primi, tab_bono, tab_sim, tab_help, tab_log = st.tabs(['La Primitiva', 'Bonoloto', '🧪 Simulador', '📘 Tutorial', '📒 Bitácora'])

# =========================== PRIMITIVA ===========================
with tab_primi:
    st.subheader(f'La Primitiva · Recomendador A2 · k={"múltiple" if (use_multi and k_nums>6) else "6"}')
    st.caption('1) Elige la fuente del último sorteo · 2) Pulsa **Calcular** · 3) Elige tu A2 final y confirma.')

    df_hist_full = load_sheet_df('sheet_id','worksheet_historico','Historico')
    last_rec = df_hist_full.tail(1) if not df_hist_full.empty else pd.DataFrame()

    fuente = st.radio('Origen de datos del último sorteo', ['Usar último del histórico', 'Introducir manualmente'],
                      index=0 if not df_hist_full.empty else 1, horizontal=True)

    if fuente == 'Usar último del histórico' and not df_hist_full.empty:
        row = last_rec.iloc[0]
        last_dt = pd.to_datetime(row['FECHA'])
        nums = [int(row['N1']), int(row['N2']), int(row['N3']), int(row['N4']), int(row['N5']), int(row['N6'])]
        comp = int(row['Complementario']) if not pd.isna(row['Complementario']) else 18
        rein = int(row['Reintegro']) if not pd.isna(row['Reintegro']) else 0
        st.info(f'Usando el último sorteo del histórico: **{last_dt.strftime("%d/%m/%Y")}**  ·  Números: {nums}  ·  C: {comp}  ·  R: {rein}')
        save_hist = False
        do_calc = st.button('Calcular recomendaciones · Primitiva', type='primary')
    else:
        with st.form('form_primi'):
            c1, c2, c3 = st.columns([1,1,1])
            last_date = c1.date_input('Fecha último sorteo (Lun/Jue/Sáb)', value=datetime.today().date())
            rein = c2.number_input('Reintegro (0-9)', min_value=0, max_value=9, value=2, step=1)
            comp = c3.number_input('Complementario (1-49)', min_value=1, max_value=49, value=18, step=1)

            st.markdown('**Números extraídos (6 distintos)**')
            cols = st.columns(6)
            defaults = [5,6,8,23,46,47]
            nums = [cols[i].number_input(f'N{i+1}', 1, 49, defaults[i], 1, key=f'npr{i+1}') for i in range(6)]

            save_hist = st.checkbox('Guardar en histórico (Primitiva) si es nuevo', value=True)
            do_calc = st.form_submit_button('Calcular recomendaciones · Primitiva')

        if do_calc:
            if df_hist_full.empty:
                last_dt = pd.to_datetime(last_date)
            else:
                target = pd.to_datetime(last_date).date()
                same = df_hist_full['FECHA'].dt.date == target
                if same.any():
                    r = df_hist_full.loc[same].tail(1).iloc[0]
                    last_dt = pd.to_datetime(r['FECHA'])
                    nums = [int(r['N1']), int(r['N2']), int(r['N3']), int(r['N4']), int(r['N5']), int(r['N6'])]
                    comp = int(r['Complementario']) if not pd.isna(r['Complementario']) else 18
                    rein = int(r['Reintegro']) if not pd.isna(r['Reintegro']) else 0
                    save_hist = False
                    st.info('La fecha ya estaba en el histórico. Se han usado los datos existentes y no se añadirá nada.')
                else:
                    last_dt = pd.to_datetime(last_date)

    if 'do_calc' in locals() and do_calc:
        if len(set(nums))!=6:
            st.error('Los 6 números deben ser distintos.'); st.stop()

        wd = last_dt.weekday()
        if wd==0: next_dt, next_dayname = last_dt + timedelta(days=3), 'Thursday'
        elif wd==3: next_dt, next_dayname = last_dt + timedelta(days=2), 'Saturday'
        elif wd==5: next_dt, next_dayname = last_dt + timedelta(days=2), 'Monday'
        else:
            st.error('La fecha debe ser Lunes, Jueves o Sábado.'); st.stop()
        st.info(f'Próximo sorteo: **{next_dt.date().strftime("%d/%m/%Y")}** ({next_dayname})')

        base = df_hist_full[df_hist_full['FECHA']<=last_dt].copy()
        if base.empty or not (base['FECHA'].dt.date == last_dt.date()).any():
            newrow = {'FECHA': last_dt, 'N1': nums[0], 'N2': nums[1], 'N3': nums[2],
                      'N4': nums[3], 'N5': nums[4], 'N6': nums[5],
                      'Complementario': comp, 'Reintegro': rein}
            base = pd.concat([base, pd.DataFrame([newrow])], ignore_index=True)
        base = base.sort_values('FECHA').tail(WINDOW_DRAWS).reset_index(drop=True)
        base['weekday'] = base['FECHA'].dt.weekday

        weekday_mask = dayname_to_weekday(next_dayname)
        w_glob = weighted_counts_nums(base, last_dt, HALF_LIFE_DAYS_DEF if 'HALF_LIFE_DAYS' not in st.session_state else st.session_state.get('HALF_LIFE_DAYS', HALF_LIFE_DAYS_DEF))
        w_day  = weighted_counts_nums(base[base['weekday']==weekday_mask], last_dt, HALF_LIFE_DAYS_DEF if 'HALF_LIFE_DAYS' not in st.session_state else st.session_state.get('HALF_LIFE_DAYS', HALF_LIFE_DAYS_DEF))
        HALF_LIFE_DAYS = st.session_state.get('HALF_LIFE_DAYS', HALF_LIFE_DAYS_DEF)
        DAY_BLEND_ALPHA = st.session_state.get('DAY_BLEND_ALPHA', DAY_BLEND_ALPHA_DEF)
        ALPHA_DIR = st.session_state.get('ALPHA_DIR', ALPHA_DIR_DEF)
        MU_PENALTY = st.session_state.get('MU_PENALTY', MU_PENALTY_DEF)
        LAMBDA_DIVERSIDAD = st.session_state.get('LAMBDA_DIVERSIDAD', LAMBDA_DIVERSIDAD_DEF)

        w_blend = blend(w_day, w_glob, alpha=DAY_BLEND_ALPHA)
        rein_dict = compute_rein_probs(base, last_dt, weekday_mask, HALF_LIFE_DAYS, DAY_BLEND_ALPHA)
        rein_sug_dynamic = max(rein_dict, key=lambda r: rein_dict[r]) if rein_dict else 0
        rein_sug_A1_ref  = REIN_FIJOS_PRIMI.get(next_dayname, rein_sug_dynamic)

        A1_6 = A1_FIJAS_PRIMI.get(next_dayname, [4,24,35,37,40,46])
        A1_k = expand_to_k(A1_6, w_blend, k_nums if (use_multi and k_nums>6) else 6)

        seed_val = stable_seed('PRIMITIVA', last_dt.date(), tuple(sorted(nums)), comp, rein, k_nums, use_multi, DAY_BLEND_ALPHA, WINDOW_DRAWS, HALF_LIFE_DAYS)
        rng = np.random.default_rng(seed_val)

        cands, seen, tries = [], set(), 0
        while len(cands)<K_CANDIDATOS and tries < K_CANDIDATOS*30:
            c = tuple(random_combo(rng)); tries += 1
            if c in seen: continue
            seen.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1_6) > (1 - MIN_DIV): continue
            cands.append(c)

        cands = sorted(cands, key=lambda c: (score_combo(c, w_blend, ALPHA_DIR, MU_PENALTY), tuple(c)), reverse=True)
        pool = cands[:800]
        best6 = list(pool[0]) if pool else []
        zA2 = zscore_combo(best6, w_blend) if best6 else 0.0
        n_sugerido = pick_n(zA2, bank_pr, vol_pr, THRESH_N)

        A2s_6 = greedy_select(pool, w_blend, n_sugerido, ALPHA_DIR, MU_PENALTY, LAMBDA_DIVERSIDAD)
        A2s_k = [expand_to_k(a2, w_blend, k_nums) if (use_multi and k_nums>6) else a2 for a2 in A2s_6]

        rng_base = np.random.default_rng(stable_seed('BASELINE','PRIMITIVA', last_dt.date()))
        pool_scores_pr, baseline_median_pr = random_baseline_scores(w_blend, ALPHA_DIR, MU_PENALTY, rng_base, n=800)

        rows = []
        total_simples = 0
        joker_count = 0
        rows.append({'Tipo':'A1','Números': A1_k if (use_multi and k_nums>6) else A1_6,
                     'k': k_nums if (use_multi and k_nums>6) else 6,
                     'Simples': comb(k_nums,6) if (use_multi and k_nums>6) else 1,
                     'Joker':'—','ScoreJ':'—','Score':'—','Lift':'—','Pct':'—'})
        total_simples += (comb(k_nums,6) if (use_multi and k_nums>6) else 1)

        for i, a2 in enumerate(A2s_k, start=1):
            base6 = A2s_6[i-1]
            sc_joker = joker_score(base6, w_blend, rein_dict) if use_joker else 0.0
            flag = (use_joker and sc_joker >= joker_thr)
            sc_val = score_combo(base6, w_blend, ALPHA_DIR, MU_PENALTY)
            lift_txt, pct_txt = lift_text_only(sc_val, baseline_median_pr, pool_scores_pr)
            simples_this = comb(k_nums,6) if (use_multi and k_nums>6) else 1
            rows.append({'Tipo': f'A2 #{i}' + (f' (k={k_nums})' if (use_multi and k_nums>6) else ''),
                         'Números': a2,'k': k_nums if (use_multi and k_nums>6) else 6,'Simples': simples_this,
                         'Joker': '⭐' if flag else '—','ScoreJ': f'{sc_joker:.2f}','Score': f'{sc_val:.2f}',
                         'Lift': lift_txt,'Pct': pct_txt})
            if flag: joker_count += 1
            total_simples += simples_this

        coste_total = total_simples * float(precio_simple) + joker_count * float(precio_joker)

        subtab1, subtab2, subtab3, subtab4 = st.tabs(['Recomendación', 'Apuestas', 'Métricas', 'Ventana'])

        with subtab1:
            st.caption('**Cómo leer**: Lift ×1.40 = 1.4× mejor que azar. ⭐ indica Joker recomendado.')
            cA, cB, cC = st.columns([1,1,1])
            cA.metric('Boletos (A1 + A2)', 1 + len(A2s_k))
            cB.metric('Coste estimado (€)', f'{coste_total:,.2f}')
            cC.metric('Confianza (señal)', conf_label(zA2))
            st.write(f'**A1**: {rows[0]["Números"]}')
            for r in rows[1:]:
                star = ' — ⭐ Joker' if r['Joker']=='⭐' else ''
                st.write(f'**{r["Tipo"]}**: {list(r["Números"])}{star}  ·  ScoreJ={r["ScoreJ"]}  ·  Score={r["Score"]}  ·  Lift: {r["Lift"]}  ·  {r.get("Pct","")}')


            # --- Recomendación Óptima (EV/€) ---
            st.markdown("### 🏆 Recomendación Óptima (EV/€)")
            # Selecciona A2 con mayor Lift (determinista)
            if len(rows) > 1:
                # rows[1:] are A2s with fields including Lift and ScoreJ
                def _lift_num(txt):
                    try: return float(str(txt).replace('×','').strip())
                    except: return 1.0
                best = max(rows[1:], key=lambda r: _lift_num(r.get('Lift','×1.00')))
                # k recomendado: por claridad, k=6 (eficiencia por € es independiente de k). Si vol alta y multi, sugerimos k actual.
                k_opt = 6 if vol_pr=='Low' or not (use_multi and k_nums>6) else k_nums
                # coste por boleto (sin Joker)
                from math import comb as C
                simples_opt = C(k_opt,6) if k_opt>6 else 1
                coste_opt = simples_opt * float(precio_simple)
                # probas
                p_base_opt = C(k_opt,6)/C(49,6)
                lift_val = _lift_num(best.get('Lift','×1.00'))
                p_adj_opt = p_base_opt * lift_val
                # Joker (adaptativo)
                scj = 0.0
                try: scj = float(best.get('ScoreJ','0'))
                except: pass
                if scj >= max(0.60, float(joker_thr)): joker_msg = "⭐ Sí"
                elif scj >= 0.45: joker_msg = "Opcional"
                else: joker_msg = "No"
                ev_no_j = ev_proxy(p_adj_opt, coste_opt)
                ev_con_j = ev_proxy(p_adj_opt, coste_opt + float(precio_joker)) if joker_msg!='No' else ev_no_j
                st.write(f"**Números**: {list(best['Números'])}")
                st.write(f"**Lift**: {best['Lift']}  ·  **Score**: {best['Score']}  ·  **k recomendado**: {k_opt}")
                st.write(f"**Prob. base**: 1 entre ~{int(round(1/p_base_opt)):,}  ·  **Prob. ajustada**: 1 entre ~{int(round(1/p_adj_opt)):,}")
                st.write(f"**Joker**: {joker_msg}")
                st.caption(f"EV/€ proxy sin Joker: {ev_no_j:.3e}  ·  con Joker: {ev_con_j:.3e} (la eficiencia por € es independiente de k; k>6 reduce varianza).")
                if st.button("Confirmar Óptima (Primitiva)"):
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    usar_j = (joker_msg.startswith('⭐'))
                    coste_final = coste_opt + (float(precio_joker) if usar_j else 0.0)
                    # registrar
                    ok = append_decision([ts,"Primitiva", next_dt.strftime("%d/%m/%Y"), best["Tipo"], int(k_opt),
                                          ", ".join(map(str, best["Números"])), int(simples_opt), best["Lift"],
                                          _parse_lift_num(best["Lift"]), best["Score"], best.get("Pct","—"), _parse_pct_num(best.get("Pct","")), "SI" if usar_j else "NO",
                                          float(round(coste_final,2)),
                                          ", ".join(map(str, A1_k if (use_multi and k_nums>6) else A1_6)), rein_sug_A1_ref, rein_sug_dynamic,
                                          int(WINDOW_DRAWS), float(HALF_LIFE_DAYS), float(DAY_BLEND_ALPHA), float(ALPHA_DIR), float(MU_PENALTY), float(LAMBDA_DIVERSIDAD),
                                          int(bank_pr), vol_pr, float(precio_simple), float(precio_joker), float(baseline_median_pr) if baseline_median_pr else 0.0,
                                          int(seed_val), float(p_base_opt), float(p_adj_opt), float(ev_no_j), float(ev_con_j), "opt", "2025-09-02-lift-det-bitacora-v3"])
                    if ok: st.success("✅ Óptima confirmada y registrada en Google Sheets (Decisiones).")
                    else:  st.success("✅ Óptima confirmada (no se pudo registrar en Sheets).")

            # --- Selección final del jugador (Primitiva) ---
            st.markdown("### ✅ Mi selección final")
            catalogo = []
            for r in rows[1:]:
                label = f"{r['Tipo']} · k={r['k']} · Lift {r['Lift']} · {r.get('Pct','')}"
                catalogo.append((label, r))
            if catalogo:
                labels = [c[0] for c in catalogo]
                sel_label = st.selectbox("Elige tu apuesta a jugar", labels, index=0, key='pick_pr')
                chosen = next(p for (lbl, p) in catalogo if lbl == sel_label)
                activar_joker = st.checkbox("Añadir Joker a mi apuesta", value=(chosen["Joker"]=='⭐'), key='joker_pr')

                if 'selecciones' not in st.session_state:
                    st.session_state['selecciones'] = []

                if st.button("Confirmar mi apuesta (Primitiva)"):
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    coste_bolet = chosen["Simples"] * float(precio_simple)
                    coste_tot = coste_bolet + (float(precio_joker) if activar_joker else 0.0)
                    entry = {
                        "ts": ts, "juego": "Primitiva",
                        "fecha_sorteo": next_dt.date().strftime("%d/%m/%Y"),
                        "tipo": chosen["Tipo"],
                        "numeros": list(chosen["Números"]),
                        "k": int(chosen["k"]),
                        "simples": int(chosen["Simples"]),
                        "lift_txt": chosen["Lift"],
                        "score": chosen["Score"],
                        "pct": chosen.get("Pct","—"),
                        "joker": bool(activar_joker),
                        "coste": float(round(coste_tot,2))
                    }
                    st.session_state['selecciones'].append(entry)
                    ok = append_decision([ts,"Primitiva", next_dt.strftime("%d/%m/%Y"), chosen["Tipo"], int(chosen["k"]),
                                          ", ".join(map(str, chosen["Números"])), int(chosen["Simples"]), chosen["Lift"],
                                          _parse_lift_num(chosen["Lift"]), chosen["Score"], chosen.get("Pct","—"), _parse_pct_num(chosen.get("Pct","")), "SI" if activar_joker else "NO",
                                          float(round(coste_tot,2)),
                                          ", ".join(map(str, A1_k if (use_multi and k_nums>6) else A1_6)), rein_sug_A1_ref, rein_sug_dynamic,
                                          int(WINDOW_DRAWS), float(HALF_LIFE_DAYS), float(DAY_BLEND_ALPHA), float(ALPHA_DIR), float(MU_PENALTY), float(LAMBDA_DIVERSIDAD),
                                          int(bank_pr), vol_pr, float(precio_simple), float(precio_joker), float(baseline_median_pr) if baseline_median_pr else 0.0,
                                          int(seed_val), float(p_base), float(p_adj), float(ev_proxy(p_adj, coste_bolet)), float(ev_proxy(p_adj, coste_bolet + (float(precio_joker) if activar_joker else 0.0))), "manual", "2025-09-02-lift-det-bitacora-v3"])
                    if ok: st.success("✅ Apuesta confirmada y registrado en Google Sheets (Decisiones).")
                    else:  st.success("✅ Apuesta confirmada (no se pudo registrar en Sheets).")

                    from math import comb as C
                    def parse_lift_multiplier(lift_txt):
                        try: return float(str(lift_txt).replace('×','').strip())
                        except: return 1.0
                    k_val = entry["k"]
                    p_base = C(k_val, 6) / C(49, 6)
                    lift_num = parse_lift_multiplier(entry["lift_txt"])
                    p_adj = p_base * lift_num
                    st.markdown("### 🧾 Mini-informe")
                    colA, colB, colC = st.columns(3)
                    colA.metric("Juego", "Primitiva")
                    colB.metric("k elegido", k_val)
                    colC.metric("Coste (€)", f"{entry['coste']:.2f}")
                    st.write(f"**Números**: {entry['numeros']}")
                    st.write(f"**Fecha sorteo**: {entry['fecha_sorteo']}  ·  **Tipo**: {entry['tipo']}")
                    st.write(f"**Lift**: {entry['lift_txt']}  ·  **Score**: {entry['score']}  ·  **Percentil**: {entry['pct']}")
                    st.write(f"**Prob. base** (k={k_val}): 1 entre ~{int(round(1/p_base)):,}")
                    st.write(f"**Prob. ajustada (Lift)**: 1 entre ~{int(round(1/p_adj)):,}")
                    if entry["joker"]:
                        st.write("**Joker**: Sí  · vía adicional ≈ +10% (orientativo)")
                    txt = []
                    txt.append(f"TS: {entry['ts']}")
                    txt.append(f"Juego: Primitiva")
                    txt.append(f"FechaSorteo: {entry['fecha_sorteo']}")
                    txt.append(f"Tipo: {entry['tipo']}")
                    txt.append(f"k: {entry['k']}  |  Simples: {entry['simples']}")
                    txt.append(f"Números: {', '.join(map(str,entry['numeros']))}")
                    txt.append(f"Lift: {entry['lift_txt']}  |  Score: {entry['score']}  |  Pct: {entry['pct']}")
                    txt.append(f"Prob. base (k={k_val}): 1 / ~{int(round(1/p_base))}")
                    txt.append(f"Prob. ajustada (Lift): 1 / ~{int(round(1/p_adj))}")
                    txt.append(f"Joker: {'SI' if entry['joker'] else 'NO'}")
                    txt.append(f"Coste (€): {entry['coste']:.2f}")
                    txt_data = "\n".join(txt).encode("utf-8")
                    st.download_button("💾 Descargar mini-informe (TXT)", data=txt_data,
                                       file_name=f"mi_apuesta_primitiva_{ts.replace(':','-')}.txt", mime="text/plain")

        with subtab2:
            df_out = pd.DataFrame([{'Tipo':rows[0]['Tipo'], 'k':rows[0]['k'], 'Simples':rows[0]['Simples'],
                'Números': ', '.join(map(str, rows[0]['Números'])), 'Joker': rows[0]['Joker'],
                'ScoreJ': rows[0]['ScoreJ'], 'Score': rows[0]['Score'], 'Lift': rows[0]['Lift'], 'Pct': rows[0]['Pct']}] + 
                [{'Tipo':r['Tipo'], 'k':r['k'], 'Simples':r['Simples'],
                'Números': ', '.join(map(str, r['Números'])), 'Joker': r['Joker'],
                'ScoreJ': r['ScoreJ'], 'Score': r['Score'], 'Lift': r['Lift'], 'Pct': r['Pct']} for r in rows[1:]])
            st.dataframe(df_out, use_container_width=True, height=320)
            st.download_button('Descargar combinaciones · Primitiva (CSV)',
                               data=df_out.to_csv(index=False).encode('utf-8'),
                               file_name='primitiva_recomendaciones.csv', mime='text/csv')

        with subtab3:
            st.markdown('**Señal media A2 (z-score):** {:.3f}'.format(zA2))
            base_w = np.array([w_blend.get(i,0.0) for i in range(1,50)])
            p_norm = base_w / (base_w.sum() if base_w.sum()>0 else 1.0)
            p_top6 = np.sort(p_norm)[-6:].mean()
            st.markdown(f'**Intensidad media de pesos (top-6):** {p_top6:.3%}')

        with subtab4:
            st.dataframe(base[['FECHA','N1','N2','N3','N4','N5','N6','Complementario','Reintegro']].tail(min(24, len(base))),
                         use_container_width=True, height=280)

# =========================== BONOLOTO ===========================
with tab_bono:
    st.subheader(f'Bonoloto · Recomendador A2 · k={"múltiple" if (use_multi and k_nums>6) else "6"}')
    st.caption('1) Elige la fuente del último sorteo · 2) Pulsa **Calcular** · 3) Elige tu A2 final y confirma.')

    df_b_full = load_sheet_df('sheet_id_bono','worksheet_historico_bono','HistoricoBono')
    last_rec_b = df_b_full.tail(1) if not df_b_full.empty else pd.DataFrame()

    fuente_b = st.radio('Origen de datos del último sorteo (Bonoloto)',
                        ['Usar último del histórico', 'Introducir manualmente'],
                        index=0 if not df_b_full.empty else 1, horizontal=True, key='src_b')

    if fuente_b == 'Usar último del histórico' and not df_b_full.empty:
        rowb = last_rec_b.iloc[0]
        last_dt_b = pd.to_datetime(rowb['FECHA'])
        nums_b = [int(rowb['N1']), int(rowb['N2']), int(rowb['N3']), int(rowb['N4']), int(rowb['N5']), int(rowb['N6'])]
        comp_b = int(rowb['Complementario']) if not pd.isna(rowb['Complementario']) else 18
        rein_b = int(rowb['Reintegro']) if not pd.isna(rowb['Reintegro']) else 0
        st.info(f'Usando el último sorteo del histórico (Bonoloto): **{last_dt_b.strftime("%d/%m/%Y")}**  ·  Números: {nums_b}  ·  C: {comp_b}  ·  R: {rein_b}')
        save_hist_b = False
        do_calc_b = st.button('Calcular recomendaciones · Bonoloto', type='primary')
    else:
        with st.form('form_bono'):
            c1, c2, c3 = st.columns([1,1,1])
            last_date_b = c1.date_input('Fecha último sorteo (Bonoloto)', value=datetime.today().date(), key='dt_b')
            rein_b = c2.number_input('Reintegro (0-9)', min_value=0, max_value=9, value=2, step=1, key='re_b')
            comp_b = c3.number_input('Complementario (1-49)', min_value=1, max_value=49, value=18, step=1, key='co_b')

            st.markdown('**Números extraídos (6 distintos)**')
            cols = st.columns(6)
            defaults_b = [5,6,8,23,46,47]
            nums_b = [cols[i].number_input(f'N{i+1} (Bono)', 1, 49, defaults_b[i], 1, key=f'nbo{i+1}') for i in range(6)]

            save_hist_b = st.checkbox('Guardar en histórico (Bonoloto) si es nuevo', value=True)
            do_calc_b = st.form_submit_button('Calcular recomendaciones · Bonoloto')

        if do_calc_b:
            if df_b_full.empty:
                last_dt_b = pd.to_datetime(last_date_b)
            else:
                target_b = pd.to_datetime(last_date_b).date()
                same_b = df_b_full['FECHA'].dt.date == target_b
                if same_b.any():
                    rb = df_b_full.loc[same_b].tail(1).iloc[0]
                    last_dt_b = pd.to_datetime(rb['FECHA'])
                    nums_b = [int(rb['N1']), int(rb['N2']), int(rb['N3']), int(rb['N4']), int(rb['N5']), int(rb['N6'])]
                    comp_b = int(rb['Complementario']) if not pd.isna(rb['Complementario']) else 18
                    rein_b = int(rb['Reintegro']) if not pd.isna(rb['Reintegro']) else 0
                    save_hist_b = False
                    st.info('La fecha ya estaba en el histórico (Bonoloto). Se han usado los datos existentes.')

    if 'do_calc_b' in locals() and do_calc_b:
        if len(set(nums_b))!=6:
            st.error('Los 6 números deben ser distintos.'); st.stop()

        next_dt_b = last_dt_b + timedelta(days=1)
        weekday = next_dt_b.weekday()
        st.info(f'Próximo sorteo (aprox.): **{next_dt_b.date().strftime("%d/%m/%Y")}** ({next_dt_b.day_name()})')

        base_b = df_b_full[df_b_full['FECHA']<=last_dt_b].copy()
        if base_b.empty or not (base_b['FECHA'].dt.date == last_dt_b.date()).any():
            new_b = {'FECHA': last_dt_b, 'N1': nums_b[0], 'N2': nums_b[1], 'N3': nums_b[2],
                     'N4': nums_b[3], 'N5': nums_b[4], 'N6': nums_b[5],
                     'Complementario': comp_b, 'Reintegro': rein_b}
            base_b = pd.concat([base_b, pd.DataFrame([new_b])], ignore_index=True)

        base_b = base_b.sort_values('FECHA').tail(WINDOW_DRAWS).reset_index(drop=True)
        base_b['weekday'] = base_b['FECHA'].dt.weekday

        HALF_LIFE_DAYS = st.session_state.get('HALF_LIFE_DAYS', HALF_LIFE_DAYS_DEF)
        DAY_BLEND_ALPHA = st.session_state.get('DAY_BLEND_ALPHA', DAY_BLEND_ALPHA_DEF)
        ALPHA_DIR = st.session_state.get('ALPHA_DIR', ALPHA_DIR_DEF)
        MU_PENALTY = st.session_state.get('MU_PENALTY', MU_PENALTY_DEF)
        LAMBDA_DIVERSIDAD = st.session_state.get('LAMBDA_DIVERSIDAD', LAMBDA_DIVERSIDAD_DEF)

        w_glob_b = weighted_counts_nums(base_b, last_dt_b, HALF_LIFE_DAYS)
        w_day_b  = weighted_counts_nums(base_b[base_b['weekday']==weekday], last_dt_b, HALF_LIFE_DAYS)
        w_blend_b = blend(w_day_b, w_glob_b, alpha=DAY_BLEND_ALPHA)

        A1b_6 = A1_FIJAS_BONO.get(weekday, [4,24,35,37,40,46])
        A1b_k = expand_to_k(A1b_6, w_blend_b, k_nums if (use_multi and k_nums>6) else 6)

        seed_val_b = stable_seed('BONOLOTO', last_dt_b.date(), tuple(sorted(nums_b)), comp_b, rein_b, k_nums, use_multi, DAY_BLEND_ALPHA, WINDOW_DRAWS, HALF_LIFE_DAYS)
        rng_b = np.random.default_rng(seed_val_b)

        cands_b, seen_b, tries_b = [], set(), 0
        while len(cands_b)<K_CANDIDATOS and tries_b < K_CANDIDATOS*30:
            c = tuple(random_combo(rng_b)); tries_b += 1
            if c in seen_b: continue
            seen_b.add(c)
            if not terciles_ok(c): continue
            if overlap_ratio(c, A1b_6) > (1 - MIN_DIV): continue
            cands_b.append(c)

        cands_b = sorted(cands_b, key=lambda c: (score_combo(c, w_blend_b, ALPHA_DIR, MU_PENALTY), tuple(c)), reverse=True)
        pool_b = cands_b[:800]
        best6_b = list(pool_b[0]) if pool_b else []
        zA2_b = zscore_combo(best6_b, w_blend_b) if best6_b else 0.0
        n_b = pick_n(zA2_b, bank_bo, vol_bo, THRESH_N)

        A2s_b_6 = greedy_select(pool_b, w_blend_b, n_b, ALPHA_DIR, MU_PENALTY, LAMBDA_DIVERSIDAD)
        A2s_b_k = [expand_to_k(a2, w_blend_b, k_nums) if (use_multi and k_nums>6) else a2 for a2 in A2s_b_6]

        rng_base_b = np.random.default_rng(stable_seed('BASELINE','BONOLOTO', last_dt_b.date()))
        pool_scores_bo, baseline_median_bo = random_baseline_scores(w_blend_b, ALPHA_DIR, MU_PENALTY, rng_base_b, n=800)

        combos_por_boleto_b = comb(k_nums,6) if (use_multi and k_nums>6) else 1
        coste_total_b = (1 + len(A2s_b_k)) * combos_por_boleto_b * float(precio_simple_bono)
        coste_total_b = round(coste_total_b + 1e-9, 2)

        subB1, subB2, subB3, subB4 = st.tabs(['Recomendación', 'Apuestas', 'Métricas', 'Ventana'])

        with subB1:
            st.caption('**Cómo leer**: Lift ×1.30 = 1.3× mejor que azar.')
            cA, cB, cC = st.columns([1,1,1])
            cA.metric('Boletos (A1 + A2)', 1 + len(A2s_b_k))
            cB.metric('Coste estimado (€)', f'{coste_total_b:,.2f}')
            cC.metric('Confianza (señal)', conf_label(zA2_b))
            st.write(f'**A1**: {A1b_k if (use_multi and k_nums>6) else A1b_6}')
            for i, a2 in enumerate(A2s_b_k, start=1):
                base6_b = A2s_b_6[i-1]
                sc_val_b = score_combo(base6_b, w_blend_b, ALPHA_DIR, MU_PENALTY)
                lift_txt_b, pct_txt_b = lift_text_only(sc_val_b, baseline_median_bo, pool_scores_bo)
                st.write(f'**A2 #{i}**: {list(a2)}  ·  Score={sc_val_b:.2f}  ·  Lift: {lift_txt_b}  ·  {pct_txt_b}')


            # --- Recomendación Óptima (EV/€) ---
            st.markdown("### 🏆 Recomendación Óptima (EV/€)")
            if A2s_b_k:
                def _lift_num(txt):
                    try: return float(str(txt).replace('×','').strip())
                    except: return 1.0
                # construir lista con lift/score para elegir el mejor
                cand_list = []
                for idx, a2 in enumerate(A2s_b_k, start=1):
                    base6_b = A2s_b_6[idx-1]
                    sc_val_b = score_combo(base6_b, w_blend_b, ALPHA_DIR, MU_PENALTY)
                    lift_txt_b, pct_txt_b = lift_text_only(sc_val_b, baseline_median_bo, pool_scores_bo)
                    cand_list.append({"Tipo": f"A2 #{idx}", "Números": a2, "Lift": lift_txt_b, "Score": f"{sc_val_b:.2f}", "Pct": pct_txt_b})
                best_b = max(cand_list, key=lambda r: _lift_num(r['Lift']))
                # k recomendado
                k_opt_b = 6 if vol_bo=='Low' or not (use_multi and k_nums>6) else k_nums
                from math import comb as C
                simples_opt_b = C(k_opt_b,6) if k_opt_b>6 else 1
                coste_opt_b = simples_opt_b * float(precio_simple_bono)
                p_base_opt_b = C(k_opt_b,6)/C(49,6)
                lift_val_b = _lift_num(best_b['Lift'])
                p_adj_opt_b = p_base_opt_b * lift_val_b
                ev_no_j_b = ev_proxy(p_adj_opt_b, coste_opt_b)
                st.write(f"**Números**: {list(best_b['Números'])}")
                st.write(f"**Lift**: {best_b['Lift']}  ·  **Score**: {best_b['Score']}  ·  **k recomendado**: {k_opt_b}")
                st.write(f"**Prob. base**: 1 entre ~{int(round(1/p_base_opt_b)):,}  ·  **Prob. ajustada**: 1 entre ~{int(round(1/p_adj_opt_b)):,}")
                st.caption(f"EV/€ proxy: {ev_no_j_b:.3e} (independiente de k; k>6 reduce varianza).")
                if st.button("Confirmar Óptima (Bonoloto)"):
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    coste_final_b = coste_opt_b
                    ok2 = append_decision([ts,"Bonoloto", next_dt_b.strftime("%d/%m/%Y"), best_b["Tipo"], int(k_opt_b),
                                          ", ".join(map(str, best_b["Números"])), int(simples_opt_b), best_b["Lift"],
                                          _parse_lift_num(best_b["Lift"]), best_b["Score"], best_b.get("Pct","—"), _parse_pct_num(best_b.get("Pct","")), "NO",
                                          float(round(coste_final_b,2)),
                                          ", ".join(map(str, A1b_k if (use_multi and k_nums>6) else A1b_6)), "-", "-",
                                          int(WINDOW_DRAWS), float(HALF_LIFE_DAYS), float(DAY_BLEND_ALPHA), float(ALPHA_DIR), float(MU_PENALTY), float(LAMBDA_DIVERSIDAD),
                                          int(bank_bo), vol_bo, float(precio_simple_bono), 0.0, float(baseline_median_bo) if baseline_median_bo else 0.0,
                                          int(seed_val_b), float(p_base_opt_b), float(p_adj_opt_b), float(ev_no_j_b), float(ev_no_j_b), "opt", "2025-09-02-lift-det-bitacora-v3"])
                    if ok2: st.success("✅ Óptima confirmada y registrada en Google Sheets (Decisiones).")
                    else:   st.success("✅ Óptima confirmada (no se pudo registrar en Sheets).")

            # --- Selección final del jugador (Bonoloto) ---
            st.markdown("### ✅ Mi selección final")
            catalogo_b = []
            for idx, a2 in enumerate(A2s_b_k, start=1):
                base6_b = A2s_b_6[idx-1]
                sc_val_b = score_combo(base6_b, w_blend_b, ALPHA_DIR, MU_PENALTY)
                lift_txt_b, pct_txt_b = lift_text_only(sc_val_b, baseline_median_bo, pool_scores_bo)
                label = f"A2 #{idx} · k={k_nums if (use_multi and k_nums>6) else 6} · Lift {lift_txt_b} · {pct_txt_b}"
                catalogo_b.append((label, {
                    "Tipo": f"A2 #{idx}",
                    "Números": a2,
                    "k": k_nums if (use_multi and k_nums>6) else 6,
                    "Simples": comb(k_nums,6) if (use_multi and k_nums>6) else 1,
                    "Lift": lift_txt_b, "Score": f"{sc_val_b:.2f}", "Pct": pct_txt_b
                }))
            if catalogo_b:
                labels_b = [c[0] for c in catalogo_b]
                sel_label_b = st.selectbox("Elige tu apuesta a jugar", labels_b, index=0, key='pick_bo')
                chosen_b = next(p for (lbl, p) in catalogo_b if lbl == sel_label_b)

                if 'selecciones' not in st.session_state:
                    st.session_state['selecciones'] = []

                if st.button("Confirmar mi apuesta (Bonoloto)"):
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    coste_bolet_b = chosen_b["Simples"] * float(precio_simple_bono)
                    entry = {
                        "ts": ts, "juego": "Bonoloto",
                        "fecha_sorteo": next_dt_b.date().strftime("%d/%m/%Y"),
                        "tipo": chosen_b["Tipo"],
                        "numeros": list(chosen_b["Números"]),
                        "k": int(chosen_b["k"]),
                        "simples": int(chosen_b["Simples"]),
                        "lift_txt": chosen_b["Lift"],
                        "score": chosen_b["Score"],
                        "pct": chosen_b.get("Pct","—"),
                        "joker": False,
                        "coste": float(round(coste_bolet_b,2))
                    }
                    st.session_state['selecciones'].append(entry)
                    ok2 = append_decision([ts,"Bonoloto", next_dt_b.strftime("%d/%m/%Y"), chosen_b["Tipo"], int(chosen_b["k"]),
                                          ", ".join(map(str, chosen_b["Números"])), int(chosen_b["Simples"]), chosen_b["Lift"],
                                          _parse_lift_num(chosen_b["Lift"]), chosen_b["Score"], chosen_b.get("Pct","—"), _parse_pct_num(chosen_b.get("Pct","")), "NO", float(round(coste_bolet_b,2)),
                                          ", ".join(map(str, A1b_k if (use_multi and k_nums>6) else A1b_6)), "-", "-",
                                          int(WINDOW_DRAWS), float(HALF_LIFE_DAYS), float(DAY_BLEND_ALPHA), float(ALPHA_DIR), float(MU_PENALTY), float(LAMBDA_DIVERSIDAD),
                                          int(bank_bo), vol_bo, float(precio_simple_bono), 0.0, float(baseline_median_bo) if baseline_median_bo else 0.0,
                                          int(seed_val_b), float(p_base), float(p_adj), float(ev_proxy(p_adj, coste_bolet_b)), float(ev_proxy(p_adj, coste_bolet_b)), "manual", "2025-09-02-lift-det-bitacora-v3"])
                    if ok2: st.success("✅ Apuesta confirmada y registrada en Google Sheets (Decisiones).")
                    else:   st.success("✅ Apuesta confirmada (no se pudo registrar en Sheets).")

                    from math import comb as C
                    def parse_lift_multiplier(lift_txt):
                        try: return float(str(lift_txt).replace('×','').strip())
                        except: return 1.0
                    k_val = entry["k"]
                    p_base = C(k_val, 6) / C(49, 6)
                    lift_num = parse_lift_multiplier(entry["lift_txt"])
                    p_adj = p_base * lift_num
                    st.markdown("### 🧾 Mini-informe")
                    colA, colB, colC = st.columns(3)
                    colA.metric("Juego", "Bonoloto")
                    colB.metric("k elegido", k_val)
                    colC.metric("Coste (€)", f"{entry['coste']:.2f}")
                    st.write(f"**Números**: {entry['numeros']}")
                    st.write(f"**Fecha sorteo**: {entry['fecha_sorteo']}  ·  **Tipo**: {entry['tipo']}")
                    st.write(f"**Lift**: {entry['lift_txt']}  ·  **Score**: {entry['score']}  ·  **Percentil**: {entry['pct']}")
                    st.write(f"**Prob. base** (k={k_val}): 1 entre ~{int(round(1/p_base)):,}")
                    st.write(f"**Prob. ajustada (Lift)**: 1 entre ~{int(round(1/p_adj)):,}")

        with subB2:
            filas_b = [{
                'Tipo':'A1','k': k_nums if (use_multi and k_nums>6) else 6,
                'Simples': comb(k_nums,6) if (use_multi and k_nums>6) else 1,
                'Números': ', '.join(map(str, A1b_k if (use_multi and k_nums>6) else A1b_6)),
                'Score': '—', 'Lift': '—', 'Pct': '—'
            }]
            for i, a2 in enumerate(A2s_b_k, start=1):
                base6_b = A2s_b_6[i-1]
                sc_val_b = score_combo(base6_b, w_blend_b, ALPHA_DIR, MU_PENALTY)
                lift_txt_b, pct_txt_b = lift_text_only(sc_val_b, baseline_median_bo, pool_scores_bo)
                filas_b.append({
                    'Tipo':f'A2-{i}','k': k_nums if (use_multi and k_nums>6) else 6,
                    'Simples': comb(k_nums,6) if (use_multi and k_nums>6) else 1,
                    'Números':', '.join(map(str,a2)),
                    'Score': f'{sc_val_b:.2f}',
                    'Lift': lift_txt_b,
                    'Pct': pct_txt_b
                })
            df_out_b = pd.DataFrame(filas_b)
            st.dataframe(df_out_b, use_container_width=True, height=320)
            st.download_button('Descargar combinaciones · Bonoloto (CSV)',
                               data=df_out_b.to_csv(index=False).encode('utf-8'),
                               file_name='bonoloto_recomendaciones.csv', mime='text/csv')

        with subB3:
            st.markdown('**Señal media A2 (z-score):** {:.3f}'.format(zA2_b))
            base_wb = np.array([w_blend_b.get(i,0.0) for i in range(1,50)])
            p_normb = base_wb / (base_wb.sum() if base_wb.sum()>0 else 1.0)
            p_top6b = np.sort(p_normb)[-6:].mean()
            st.markdown(f'**Intensidad media de pesos (top-6):** {p_top6b:.3%}')

        with subB4:
            st.dataframe(base_b[['FECHA','N1','N2','N3','N4','N5','N6','Complementario','Reintegro']].tail(min(24, len(base_b))),
                         use_container_width=True, height=280)

# =========================== 🧪 SIMULADOR ===========================
with tab_sim:
    st.subheader('🧪 Simulador — escenarios rápidos por juego')
    st.caption('Elige juego, preset y parámetros básicos para estimar coste y nº de boletos. No usa histórico.')

    juego_sim = st.radio('Juego', ['Primitiva','Bonoloto'], horizontal=True)

    presets = {
        'Conservador': {'k':6, 'vol':'Low', 'bank':6},
        'Equilibrado': {'k':7, 'vol':'Medium', 'bank':10},
        'Agresivo': {'k':8, 'vol':'High', 'bank':15},
        'Personalizado': None
    }
    preset = st.selectbox('Preset', list(presets.keys()), index=1)

    cfg = presets[preset]
    k_sim = st.slider('k por boleto (sim)', 6, 8, (cfg['k'] if cfg else 8), 1)
    vol_sim = st.selectbox('Volatilidad (sim)', ['Low','Medium','High'],
                           index=(['Low','Medium','High'].index(cfg['vol']) if cfg else 1))
    bank_sim = st.number_input('Bank (nº máximo de A2)', 0, 999, (cfg['bank'] if cfg else 10), 1)

    if juego_sim=='Primitiva':
        precio_sim = st.number_input('Precio simple (Primitiva) €', 0.0, 10.0, float(1.0), 0.5, format='%.2f')
    else:
        precio_sim = st.number_input('Precio simple (Bonoloto) €', 0.0, 10.0, float(0.50), 0.5, format='%.2f',
                                     help='Bonoloto: múltiplos de 0,50€')

    def estimate_n_from_vol(vol, bank):
        z_proxy = {'Low':0.55, 'Medium':0.30, 'High':0.15}[vol]
        table = THRESH_N
        adj = 0.05 if vol=='Low' else -0.05 if vol=='High' else 0.0
        for th in table:
            if z_proxy >= th['z'] + adj:
                return min(th['n'], int(bank)) if bank>0 else 0
        return 0

    n_here = estimate_n_from_vol(vol_sim, bank_sim)
    simples_por_boleto = comb(k_sim, 6) if k_sim>6 else 1
    coste_est = (1 + n_here) * simples_por_boleto * float(precio_sim)

    c1,c2,c3 = st.columns(3)
    c1.metric('Boletos (A1 + A2)', 1 + n_here)
    c2.metric('Simples por boleto', simples_por_boleto)
    c3.metric('Coste estimado (€)', f'{coste_est:,.2f}')
    st.caption('Más k ⇒ más combinaciones por boleto. Más volatilidad ⇒ más A2 posibles.')

# =========================== 📘 TUTORIAL ===========================
with tab_help:
    st.subheader('📘 Tutorial — cómo usar el recomendador (explicado fácil)')
    st.markdown('''
**¿Qué hace este sistema?**  
Te propone **apuestas recomendadas (A2)** que **mejoran** frente a jugar al azar, ponderando más lo reciente y mezclando con el **día del sorteo**. Evitamos patrones muy **populares** para no compartir premio.

### Conceptos clave
- **A1**: boleto ancla del día. Asegura **diversidad**.
- **A2**: boletos sugeridos por el modelo.
- **k**: tamaño del boleto (6..8). Si **k>6**, un boleto contiene **varias** combinaciones simples.
- **Score**: señal de la combinación (log-pesos) – penalización por popularidad. **Cuanto más alto, mejor**.
- **Lift**: `×ratio` frente al azar (misma estructura). `×1.40` = 1.4 veces mejor que aleatorio.
- **Pct**: posición frente a una muestra aleatoria (p.e. **top 6% del azar**).
- **ScoreJ (Primitiva)**: mezcla de señal de la A2 y fuerza del **Reintegro**; si **≥ umbral** ⇒ ⭐ activar Joker.

### Pasos para usarlo
1. **Fuente del último sorteo** (hoja o manual).  
2. Pulsa **Calcular**.  
3. En **Recomendación**, **elige tu A2 final** y confirma. Se guardará en **Bitácora**.

### Consejos
- Bank pequeño ⇒ `k=6`, volatilidad **Low**.  
- Más cobertura ⇒ `k=7/8`, volatilidad **Medium/High**.  
- **Bonoloto**: precio en **múltiplos de 0,50€**.
''')

# =========================== 📒 BITÁCORA ===========================
with tab_log:
    st.subheader('📒 Bitácora de decisiones')
    st.caption('Se registran tus apuestas confirmadas en Google Sheets (worksheet "Decisiones").')

    try:
        creds = get_gcp_credentials()
        gc = gspread.authorize(creds)
        sid = (st.secrets.get('gcp_service_account', {}) or {}).get('sheet_id') or st.secrets.get('sheet_id')
        if sid:
            sh = gc.open_by_key(sid)
            try:
                ws = sh.worksheet('Decisiones')
                rows = ws.get_all_records()
                df_dec = pd.DataFrame(rows)
            except Exception:
                df_dec = pd.DataFrame()
        else:
            df_dec = pd.DataFrame()
    except Exception:
        df_dec = pd.DataFrame()

    if df_dec.empty:
        st.info("Aún no hay decisiones registradas.")
    else:
        st.dataframe(df_dec, use_container_width=True, height=340)

        try:
            df_dec['Coste(€)'] = pd.to_numeric(df_dec.get('Coste(€)', 0), errors='coerce').fillna(0.0)
            total_coste = df_dec['Coste(€)'].sum()
            n_dec = len(df_dec)
            c1,c2 = st.columns(2)
            c1.metric("Total decisiones", n_dec)
            c2.metric("Gasto total (€)", f"{total_coste:,.2f}")
        except Exception:
            pass

        st.markdown("#### Evaluación vs resultados (si el sorteo ya está en el histórico)")
        def parse_nums(s):
            if isinstance(s, list): return s
            try:
                if "[" in str(s):
                    return list(map(int, ast.literal_eval(s)))
                return list(map(int, str(s).replace(" ","").split(",")))
            except Exception:
                return []
        def hits(a, b):
            return len(set(a) & set(b))

        df_h_pr = load_sheet_df('sheet_id','worksheet_historico','Historico')
        df_h_bo = load_sheet_df('sheet_id_bono','worksheet_historico_bono','HistoricoBono')

        eval_rows = []
        for _, r in df_dec.iterrows():
            juego = r.get('Juego','')
            fecha_txt = r.get('FechaSorteo','')
            nums_sel = parse_nums(r.get('Numeros',''))
            if not fecha_txt or not nums_sel:
                continue
            try:
                f = pd.to_datetime(fecha_txt, dayfirst=True).date()
            except Exception:
                continue
            if juego=='Primitiva' and not df_h_pr.empty:
                same = df_h_pr['FECHA'].dt.date == f
                if same.any():
                    row = df_h_pr.loc[same].iloc[0]
                    reales = [int(row[c]) for c in ['N1','N2','N3','N4','N5','N6']]
                    eval_rows.append([fecha_txt, juego, hits(nums_sel, reales)])
            if juego=='Bonoloto' and not df_h_bo.empty:
                same = df_h_bo['FECHA'].dt.date == f
                if same.any():
                    row = df_h_bo.loc[same].iloc[0]
                    reales = [int(row[c]) for c in ['N1','N2','N3','N4','N5','N6']]
                    eval_rows.append([fecha_txt, juego, hits(nums_sel, reales)])

        if eval_rows:
            df_eval = pd.DataFrame(eval_rows, columns=['Fecha','Juego','Aciertos'])
            st.dataframe(df_eval, use_container_width=True, height=220)
            rate_3p = (df_eval['Aciertos']>=3).mean()
            rate_4p = (df_eval['Aciertos']>=4).mean()
            rate_5p = (df_eval['Aciertos']>=5).mean()
            rate_6p = (df_eval['Aciertos']>=6).mean()
            d1,d2,d3,d4 = st.columns(4)
            d1.metric("≥3 aciertos", f"{rate_3p:.2%}")
            d2.metric("≥4 aciertos", f"{rate_4p:.2%}")
            d3.metric("≥5 aciertos", f"{rate_5p:.3%}")
            d4.metric("6 aciertos",  f"{rate_6p:.4%}")
        else:
            st.caption("Cuando se publiquen los resultados de esos sorteos en los históricos, verás los aciertos aquí.")
