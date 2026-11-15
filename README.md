# 🦺 Sistema Inteligente de Supervisión de EPP con Visión Artificial (YOLO11)

Aplicación web full-stack en Python para la detección automatizada en tiempo real de **Equipos de Protección Personal (EPP)** y auditoría de seguridad industrial (SST).

## 🚀 Arquitectura y Stack Tecnológico
- **Modelo de Deep Learning:** YOLO11n (Transfer Learning / Fine-Tuning sobre 2,046 imágenes industriales).
- **Rendimiento:** `mAP@50` de **94.8%** en cascos, **90.1%** en chalecos y velocidad de inferencia de **7.7 ms/frame**.
- **Backend & Lógica SST:** Python, OpenCV, reglas de cumplimiento por conteo cruzado y persistencia en **SQLite**.
- **Frontend Web:** Streamlit con soporte para imágenes, video MP4, IP Webcam y exportación de reportes en CSV/Excel.

## 💻 Ejecución Local
```bash
pip install -r requirements.txt
streamlit run app.py