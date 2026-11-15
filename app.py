import streamlit as st
import cv2
import tempfile
import time
import os
import numpy as np
from PIL import Image

from detector_epp import EvaluadorEPP
from base_datos import GestorIncidentes

st.set_page_config(
    page_title="Monitor SST - Detección de EPP con IA",
    page_icon="🦺",
    layout="wide"
)

@st.cache_resource
def cargar_motor(umbral):
    return EvaluadorEPP(ruta_modelo="best.pt", umbral_confianza=umbral)

db = GestorIncidentes()

# Barra lateral
st.sidebar.title("⚙️ Panel de Control SST")
umbral_conf = st.sidebar.slider(
    "Umbral de Confianza del Modelo",
    min_value=0.20, max_value=0.90, value=0.40, step=0.05
)
motor = cargar_motor(umbral_conf)

modo_entrada = st.sidebar.radio(
    "Fuente de Inspección en Vivo:",
    [
        "🖼️ Imagen de Prueba",
        "🎥 Video de Obra (.mp4)",
        "📱 Cámara del Celular (IP Webcam)",
        "📸 Captura Rápida Cámara Web"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("**Estado del Sistema:** En línea\n\n**Base de Datos:** SQLite Activa")

# Encabezado principal
st.title("🦺 Sistema Inteligente de Supervisión de EPP")
st.markdown("Plataforma de Visión Artificial con **YOLO11n** para cumplimiento de Seguridad y Salud en el Trabajo (SST).")

# Pestañas principales
tab_monitor, tab_historial, tab_metricas = st.tabs([
    "🔍 Monitor en Vivo",
    "📋 Registro de Incidentes (SQLite)",
    "📊 Métricas del Modelo IA"
])

# ==============================================================================
# PESTAÑA 1: MONITOR EN VIVO
# ==============================================================================
with tab_monitor:
    col_estado, col_trabajadores, col_cascos, col_chalecos = st.columns(4)
    slot_estado = col_estado.empty()
    slot_trabajadores = col_trabajadores.empty()
    slot_cascos = col_cascos.empty()
    slot_chalecos = col_chalecos.empty()

    def actualizar_tarjetas(estado, conteo, infracciones):
        if estado == "ALERTA":
            slot_estado.error(f"🚨 **ALERTA:** {', '.join(infracciones)}")
        elif estado == "SEGURO":
            slot_estado.success("✅ **CUMPLIMIENTO OK**")
        else:
            slot_estado.info("⏳ **En espera de personal...**")

        slot_trabajadores.metric("👷 Trabajadores", conteo.get("Person", 0))
        slot_cascos.metric("🪖 Cascos", conteo.get("helmet", 0))
        slot_chalecos.metric("🦺 Chalecos", conteo.get("vest", 0))

    if modo_entrada == "🖼️ Imagen de Prueba":
        archivo_img = st.file_uploader("Sube una fotografía de trabajadores (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if archivo_img is not None:
            imagen_pil = Image.open(archivo_img).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

            frame_proc, estado, infracciones, conteo = motor.procesar_fotograma(frame_bgr)
            actualizar_tarjetas(estado, conteo, infracciones)

            if estado == "ALERTA":
                db.registrar_incidente("Imagen Subida", infracciones, conteo.get("Person", 0))
                st.toast("Incidente registrado en la base de datos SQLite", icon="📝")

            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Resultado de Inspección IA", use_container_width=True)

    elif modo_entrada == "🎥 Video de Obra (.mp4)":
        archivo_vid = st.file_uploader("Sube un video de supervisión (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if archivo_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(archivo_vid.read())

            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            ultimo_registro = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (800, 480))
                frame_proc, estado, infracciones, conteo = motor.procesar_fotograma(frame)
                actualizar_tarjetas(estado, conteo, infracciones)

                if estado == "ALERTA" and (time.time() - ultimo_registro > 4):
                    db.registrar_incidente("Video MP4", infracciones, conteo.get("Person", 0))
                    ultimo_registro = time.time()

                frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

            cap.release()
            st.success("Análisis de video finalizado.")

    elif modo_entrada == "📱 Cámara del Celular (IP Webcam)":
        url_ip = st.text_input("Ingresa la URL de transmisión de IP Webcam:", value="http://192.168.1.15:8080/video")
        iniciar_stream = st.checkbox("▶️ Iniciar transmisión en vivo")

        if iniciar_stream:
            cap = cv2.VideoCapture(url_ip)
            st_frame = st.empty()
            ultimo_registro = 0

            if not cap.isOpened():
                st.error("No se pudo conectar a la IP del celular. Verifica que ambos estén en el mismo Wi-Fi.")
            else:
                while iniciar_stream and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Se perdió la señal de video.")
                        break

                    frame = cv2.resize(frame, (640, 480))
                    frame_proc, estado, infracciones, conteo = motor.procesar_fotograma(frame)
                    actualizar_tarjetas(estado, conteo, infracciones)

                    if estado == "ALERTA" and (time.time() - ultimo_registro > 4):
                        db.registrar_incidente("Cámara Móvil IP", infracciones, conteo.get("Person", 0))
                        ultimo_registro = time.time()

                    frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

                cap.release()

    elif modo_entrada == "📸 Captura Rápida Cámara Web":
        foto_camara = st.camera_input("Toma una foto con la cámara de tu dispositivo")
        if foto_camara is not None:
            imagen_pil = Image.open(foto_camara).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

            frame_proc, estado, infracciones, conteo = motor.procesar_fotograma(frame_bgr)
            actualizar_tarjetas(estado, conteo, infracciones)

            if estado == "ALERTA":
                db.registrar_incidente("Cámara Navegador", infracciones, conteo.get("Person", 0))

            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Inspección en Tiempo Real", use_container_width=True)

# ==============================================================================
# PESTAÑA 2: HISTORIAL DE INCIDENTES Y EXPORTACIÓN CSV
# ==============================================================================
with tab_historial:
    st.subheader("📑 Auditoría de Infracciones Registradas")
    df_historial = db.obtener_historial(limite=200)

    col_kpi1, col_kpi2, col_btn1, col_btn2 = st.columns([1.5, 1.5, 1.5, 1])
    col_kpi1.metric("Total de Alertas Guardadas", len(df_historial))
    if not df_historial.empty:
        col_kpi2.metric("Último Incidente", df_historial.iloc[0]["Fecha y Hora"])

        csv_data = df_historial.to_csv(index=False).encode("utf-8-sig")
        col_btn1.download_button(
            label="📥 Descargar Reporte CSV (Excel)",
            data=csv_data,
            file_name="reporte_incidentes_epp.csv",
            mime="text/csv"
        )

        if col_btn2.button("🗑️ Limpiar BD"):
            db.limpiar_historial()
            st.rerun()

        st.dataframe(df_historial, use_container_width=True, hide_index=True)
    else:
        st.info("Aún no hay infracciones registradas. Sube una imagen o video con alertas en el Monitor en Vivo.")

# ==============================================================================
# PESTAÑA 3: MÉTRICAS Y ARQUITECTURA DEL MODELO
# ==============================================================================
with tab_metricas:
    st.subheader("🧠 Rendimiento Técnico del Modelo Entrenado (YOLO11n)")
    st.markdown(
        "Modelo entrenado mediante **Transfer Learning** sobre **2,046 imágenes** industriales "
        "(101 capas, 2.59M parámetros, peso final: **5.5 MB**)."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("mAP@50 Global", "72.6%")
    m2.metric("mAP@50 Cascos", "94.8%")
    m3.metric("mAP@50 Chalecos", "90.1%")
    m4.metric("Velocidad Inferencia", "7.7 ms / img")

    st.markdown("---")
    st.markdown("#### Evidencias de Entrenamiento y Validación")
    c_img1, c_img2 = st.columns(2)

    if os.path.exists("assets/confusion_matrix.png"):
        c_img1.image("assets/confusion_matrix.png", caption="Matriz de Confusión Normalizada", use_container_width=True)
    else:
        c_img1.info("Coloca 'confusion_matrix.png' en la carpeta 'assets/' para visualizarla aquí.")

    if os.path.exists("assets/results.png"):
        c_img2.image("assets/results.png", caption="Curvas de Pérdida y Precisión (35 Épocas)", use_container_width=True)
    else:
        c_img2.info("Coloca 'results.png' en la carpeta 'assets/' para visualizarla aquí.")