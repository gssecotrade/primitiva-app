
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Primitiva Predictor", page_icon="🎲", layout="centered")

st.title("🎲 Primitiva Predictor - Estrategia A1 + A2")

st.markdown("Sube tu histórico CSV y la combinación más reciente para calcular A1 (fija) y A2 (dinámica).")

uploaded_file = st.file_uploader("📂 Sube tu histórico (CSV con FECHA, N1..N6, Complementario, Reintegro)", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, dayfirst=True)
        df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["FECHA"])
        df = df.sort_values("FECHA")
        
        st.success(f"Histórico cargado con {len(df)} sorteos. Último: {df['FECHA'].max().strftime('%d/%m/%Y')}")

        # Entrada de nuevo sorteo
        st.subheader("➕ Añadir nuevo sorteo")
        fecha = st.date_input("Fecha del sorteo")
        numeros = st.text_input("Números (6 separados por coma)")
        comp = st.number_input("Complementario", 1, 49, 1)
        reintegro = st.number_input("Reintegro", 0, 9, 0)
        
        if st.button("Añadir sorteo y calcular A2"):
            try:
                nums = [int(x.strip()) for x in numeros.split(",")]
                if len(nums) != 6 or len(set(nums)) != 6:
                    st.error("Debes introducir 6 números distintos entre 1 y 49")
                else:
                    # Filtro anti-duplicados
                    if fecha in df["FECHA"].values:
                        existente = df[df["FECHA"] == pd.to_datetime(fecha)]
                        if set(existente[["N1","N2","N3","N4","N5","N6"]].values.flatten()) == set(nums):
                            st.info("⚠️ Este sorteo ya existe en el histórico. No se añadió.")
                        else:
                            st.warning("⚠️ Fecha ya existe con combinación distinta. Revisa tu CSV.")
                    else:
                        st.success("✅ Sorteo añadido temporalmente para cálculo.")

                    # Modelo simple A2
                    freq = pd.Series(np.concatenate(df[["N1","N2","N3","N4","N5","N6"]].values)).value_counts()
                    top_nums = freq.head(12).index.tolist()
                    a2 = np.random.choice(top_nums, size=6, replace=False)
                    reintegro_pred = np.random.randint(0, 10)

                    st.subheader("🎯 Recomendaciones")
                    st.write(f"**Apuesta A1 (fija):** [3, 12, 19, 27, 33, 41]  | Reintegro: 7")
                    st.write(f"**Apuesta A2 (dinámica):** {sorted(a2)}  | Reintegro sugerido: {reintegro_pred}")
                    st.write("**Joker recomendado:** Sí" if np.random.rand()>0.5 else "No")
            except Exception as e:
                st.error(f"Error procesando entrada: {e}")

    except Exception as e:
        st.error(f"Error cargando CSV: {e}")
