# 🚀 Inicio rápido

**Objetivo**: obtener recomendaciones A1 (ancla fija) y A2 (dinámicas) para **Primitiva** y **Bonoloto**.

## 1) Conecta Google Sheets (una vez)
1. En [Google Cloud Console] crea una **cuenta de servicio** y descarga su JSON.
2. Copia el JSON en **Streamlit Cloud → App → Settings → Secrets** como bloque `[gcp_service_account]`.
3. Añade también:
   ```toml
   sheet_id = "ID_PRIMITIVA"
   worksheet_historico = "Historico"
   sheet_id_bono = "ID_BONOLOTO"
   worksheet_historico_bono = "HistoricoBono"
