import sqlite3
from datetime import datetime
import pandas as pd

class GestorIncidentes:
    def __init__(self, db_path="incidentes_epp.db"):
        self.db_path = db_path
        self._inicializar_tabla()

    def _inicializar_tabla(self):
        """Crea la tabla de incidentes si aún no existe."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidentes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha_hora TEXT NOT NULL,
                    origen TEXT NOT NULL,
                    tipo_infraccion TEXT NOT NULL,
                    trabajadores_detectados INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def registrar_incidente(self, origen, infracciones, num_personas):
        """Guarda una nueva alerta de seguridad en la base de datos."""
        if not infracciones:
            return
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detalle = " | ".join(infracciones)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incidentes (fecha_hora, origen, tipo_infraccion, trabajadores_detectados)
                VALUES (?, ?, ?, ?)
            """, (fecha_actual, origen, detalle, num_personas))
            conn.commit()

    def obtener_historial(self, limite=100):
        """Retorna los últimos incidentes registrados como un DataFrame de Pandas."""
        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT id AS ID,
                       fecha_hora AS 'Fecha y Hora',
                       origen AS 'Fuente de Video',
                       tipo_infraccion AS 'Infracción Detectada',
                       trabajadores_detectados AS 'Trabajadores en Escena'
                FROM incidentes
                ORDER BY id DESC
                LIMIT {limite}
            """
            return pd.read_sql_query(query, conn)

    def limpiar_historial(self):
        """Borra todos los registros (útil para reiniciar pruebas antes de una demo)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM incidentes")
            conn.commit()