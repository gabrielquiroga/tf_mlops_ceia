"""
Frontend Streamlit para API de Clasificación Estelar.
Permite ingrsar datos de objetos estelares y obtener predicciones.
"""

import os
import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any

LOW_RANGE_EXTENSION = 0.5
HIGH_RANGE_EXTENSION = 1.5


# Configuración de rangos (extendidos 50% respecto a min/max del dataset)
RANGES = {
    # Variables obligatorias
    "alpha": (0.005528 * LOW_RANGE_EXTENSION, 359.999810 * HIGH_RANGE_EXTENSION),
    "delta": (-18.785328 * HIGH_RANGE_EXTENSION, 83.000519 * HIGH_RANGE_EXTENSION),
    "u": (10.996230 * LOW_RANGE_EXTENSION, 32.781390 * HIGH_RANGE_EXTENSION),
    "g": (10.498200 * LOW_RANGE_EXTENSION, 31.602240 * HIGH_RANGE_EXTENSION),
    "r": (9.822070 * LOW_RANGE_EXTENSION, 29.571860 * HIGH_RANGE_EXTENSION),
    "i": (9.469903 * LOW_RANGE_EXTENSION, 32.141470 * HIGH_RANGE_EXTENSION),
    "z": (9.612333 * LOW_RANGE_EXTENSION, 29.383740 * HIGH_RANGE_EXTENSION),
    "redshift": (-0.009971 * HIGH_RANGE_EXTENSION, 7.011245 * HIGH_RANGE_EXTENSION),
    # Variables opcionales
    "obj_ID": (1.237646e+18 * LOW_RANGE_EXTENSION, 1.237681e+18 * HIGH_RANGE_EXTENSION),
    "run_ID": (109.0 * LOW_RANGE_EXTENSION, 8162.0 * HIGH_RANGE_EXTENSION),
    "rerun_ID": (301.0 * LOW_RANGE_EXTENSION, 301.0 * HIGH_RANGE_EXTENSION),
    "cam_col": (1.0 * LOW_RANGE_EXTENSION, 6.0 * HIGH_RANGE_EXTENSION),
    "field_ID": (11.0 * LOW_RANGE_EXTENSION, 989.0 * HIGH_RANGE_EXTENSION),
    "spec_obj_ID": (2.995191e+17 * LOW_RANGE_EXTENSION, 1.412694e+19 * HIGH_RANGE_EXTENSION),
    "plate": (266.0 * LOW_RANGE_EXTENSION, 12547.0 * HIGH_RANGE_EXTENSION),
    "MJD": (51608.0 * LOW_RANGE_EXTENSION, 58932.0 * HIGH_RANGE_EXTENSION),
    "fiber_ID": (1.0 * LOW_RANGE_EXTENSION, 1000.0 * HIGH_RANGE_EXTENSION),
}

# Descripción de variables
DESCRIPTIONS = {
    "alpha": "Ascensión recta (grados)",
    "delta": "Declinación (grados)",
    "u": "Magnitud ultravioleta",
    "g": "Magnitud verde",
    "r": "Magnitud roja",
    "i": "Magnitud infrarroja cercana",
    "z": "Magnitud infrarroja",
    "redshift": "Corrimiento al rojo",
    "obj_ID": "ID del objeto",
    "run_ID": "ID de ejecución",
    "rerun_ID": "ID de re-ejecución",
    "cam_col": "Columna de cámara",
    "field_ID": "ID de campo",
    "spec_obj_ID": "ID de objeto espectroscópico",
    "plate": "Placa",
    "MJD": "Fecha juliana modificada",
    "fiber_ID": "ID de fibra",
}

REQUIRED_FIELDS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
OPTIONAL_FIELDS = [
    "obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", 
    "spec_obj_ID", "plate", "MJD", "fiber_ID"
]


def validate_and_get_inputs() -> Dict[str, Any]:
    """
    Crea los inputs del formulario y retorna los valores validados.
    """
    st.subheader("Variables obligatorias")
    
    inputs = {}

    # Crear los inputs para campos obligatorios en dos columnas
    col1, col2 = st.columns(2)

    for idx, field in enumerate(REQUIRED_FIELDS):
        col = col1 if idx % 2 == 0 else col2
        min_val, max_val = RANGES[field]

        with col:
            inputs[field] = st.number_input(
                label=f"{field} - {DESCRIPTIONS[field]}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                format="%.6f",
                help=f"Rango válido: [{min_val:.2f}, {max_val:.2f}]"
            )

    return inputs

def make_prediction(api_url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza la petición  POST a la API de predicción.

    Args:
        api_url: URL base de la API
        data: Diccionario con los datos de entrada

    Returns:
        Respuesta de la API
    """
    try:
        # Se espera una lista
        payload = [data]
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        response.raise_for_status()
        return {"success": True, "data": response.json()}

    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "No se pudo conectar con la API. Verifica que esté corriendo."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "La petición tardó demasiado tiempo."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"Error HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": f"Error inesperado: {str(e)}"}

def display_prediction(result: Dict[str, Any]):
    """
    Muestra el resultado de la predicción.
    """
    if result["success"]:
        predictions = result["data"]
        st.success("Predicción exitosa")

        if isinstance(predictions, list) and len(predictions) > 0:
            prediction = predictions[0]

            st.markdown("### Resultado")
            class_names = {
                "STAR": "⭐ Estrella",
                "GALAXY": "🌌 Galaxia",
                "QSO": "💫 Cuásar (QSO)"
            }
            
            # Mostrar predicción con estilo
            pred_class = prediction.get("prediction", prediction)
            display_name = class_names.get(pred_class, pred_class)
            
            st.markdown(f"## {display_name}")
            
            # Si hay información adicional (ej: probabilidades)
            if isinstance(prediction, dict):
                st.json(prediction)
        else:
            st.warning("La API no devolvió predicciones")
    else:
        st.error(f"Error: {result['error']}")

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="Clasificador Estelar",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Clasificador Estelar")
    st.markdown("""
        Esta aplicación permite clasificar objetos astronómicos en:
        - Estrellas
        - Galaxias
        - Cuásares

        Según componentes de sus ondas electromagnéticas.
        **Ingresá los datos del objeto y obtené tu predicción.*
    """)

    # Sidebar
    with st.sidebar:
        st.header("Configuración")

        default_api_url = os.getenv("API_BASE_URL", "http://localhost:8000")

        api_url = st.text_input(
            "URL de la API",
            value=default_api_url,
            help="URL base donde está corriendo la API de predicción"
        )

        st.markdown("---")
        st.markdown("### Información")
        st.info("""
        **Variables obligatorias:**
        - alpha, delta (coordenadas)
        - u, g, r, i, z (magnitudes)
        - redshift
        
        **Variables opcionales:**
        - IDs y metadatos del telescopio
        """)
        
        st.markdown("---")
        st.markdown("### Enlaces")
        st.markdown("[Documentación SDSS](https://www.sdss.org/)")

    # Formulario principal
    with st.form("prediction_form"):
        inputs = validate_and_get_inputs()

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button(
                "Clasificar",
                use_container_width=True
            )

    # Procesar predicción
    if submit_button:
        with st.spinner("Consultando la API..."):
            result = make_prediction(api_url, inputs)

        st.markdown("---")
        display_prediction(result)

        with st.expander("Ver datos enviados"):
            df = pd.DataFrame([inputs])
            st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()