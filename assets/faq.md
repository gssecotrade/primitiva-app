# ❓ Preguntas frecuentes

**¿Puedo usar mi propio Sheet?**  
Sí. Cambia los IDs en *Secrets* y respeta los nombres de pestaña y columnas.

**¿Cómo sé si se ha guardado un sorteo?**  
Verás un mensaje ✅ y aparecerá en la tabla “Últimos sorteos cargados”.

**¿Por qué no veo los estilos?**  
Asegúrate de tener `styles.css` en la raíz y que la app lo carga tras `st.set_page_config`.

**¿Puedo cambiar la ventana de 24 sorteos?**  
Sí, en `app.py` → `WINDOW_DRAWS`.

**¿Qué significa la probabilidad mostrada?**  
Es **proxy comparativa** basada en pesos y combinatoria, útil para **ordenar** combinaciones, no para estimar prob. real de premio final.
