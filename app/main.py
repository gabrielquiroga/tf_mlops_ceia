from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ml.predict import predict

# --- Modelo de entrada ---
# Define exactamente qué campos espera recibir el endpoint.
# Pydantic valida automáticamente que los tipos sean correctos.
class StarInput(BaseModel):
    alpha: float
    delta: float
    u: float
    g: float
    r: float
    i: float
    z: float
    redshift: float

    # Opcionales (con valores por defecto None)
    run_ID: Optional[int] = None
    cam_col: Optional[int] = None
    field_ID: Optional[int] = None
    plate: Optional[int] = None
    MJD: Optional[int] = None
    fiber_ID: Optional[int] = None
    obj_ID: Optional[float] = None
    rerun_ID: Optional[int] = None
    spec_obj_ID: Optional[float] = None


# --- App ---
app = FastAPI(
    title="Stellar Classifier API",
    description="Clasifica objetos astronómicos en GALAXY, STAR o QSO usando el modelo champion.",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Endpoint para verificar que la API está levantada."""
    return {"status": "ok"}


@app.post("/predict")
def predict_class(stars: List[StarInput]):
    """
    Recibe los datos de un objeto astronómico y devuelve su clasificación.
    La clasificación puede ser: GALAXY, STAR o QSO.
    """
    try:
        # Convertimos la lista de objetos a DataFrame
        input_df = pd.DataFrame([star.model_dump() for star in stars])
        results = predict(input_df)
        # Retornamos lista de predicciones
        return [{"prediction": pred} for pred in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
