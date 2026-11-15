import cv2
from ultralytics import YOLO

class EvaluadorEPP:
    def __init__(self, ruta_modelo="best.pt", umbral_confianza=0.45):
        self.model = YOLO(ruta_modelo)
        self.conf = umbral_confianza

        self.colores = {
            "helmet": (0, 210, 0),       # Verde
            "vest": (0, 210, 0),         # Verde
            "goggles": (255, 190, 0),    # Celeste
            "Person": (230, 230, 0),     # Cian
            "no-helmet": (0, 0, 255),    # Rojo intenso
            "no-vest": (0, 85, 255),     # Naranja fuerte
            "no-goggles": (0, 165, 255)  # Ámbar
        }

    def procesar_fotograma(self, frame):
        resultados = self.model.predict(source=frame, conf=self.conf, verbose=False)[0]
        frame_anotado = frame.copy()

        conteo = {nombre: 0 for nombre in self.model.names.values()}
        infracciones = []

        for box in resultados.boxes:
            cls_id = int(box.cls[0])
            nombre_clase = self.model.names[cls_id]
            confianza = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conteo[nombre_clase] = conteo.get(nombre_clase, 0) + 1
            color = self.colores.get(nombre_clase, (200, 200, 200))

            # Dibujar caja principal
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color, 2)

            # Dibujar etiqueta con fondo sólido para máxima legibilidad
            etiqueta = f"{nombre_clase} {confianza:.0%}"
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_texto = max(y1 - 6, th + 6)
            cv2.rectangle(frame_anotado, (x1, y_texto - th - 4), (x1 + tw + 4, y_texto + 2), color, -1)
            cv2.putText(frame_anotado, etiqueta, (x1 + 2, y_texto - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Reglas de negocio SST
        if conteo.get("no-helmet", 0) > 0:
            infracciones.append("Trabajador SIN CASCO")
        if conteo.get("no-vest", 0) > 0:
            infracciones.append("Trabajador SIN CHALECO")
        if conteo.get("no-goggles", 0) > 0:
            infracciones.append("Trabajador SIN LENTES")

        if conteo.get("Person", 0) > conteo.get("helmet", 0) and "Trabajador SIN CASCO" not in infracciones:
            infracciones.append("Posible trabajador SIN CASCO (conteo desigual)")

        if len(infracciones) > 0:
            estado_global = "ALERTA"
        elif conteo.get("Person", 0) > 0 or conteo.get("helmet", 0) > 0:
            estado_global = "SEGURO"
        else:
            estado_global = "SIN TRABAJADORES"

        return frame_anotado, estado_global, infracciones, conteo