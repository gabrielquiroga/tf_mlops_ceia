from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
    run_ID: int
    cam_col: int
    field_ID: int
    plate: int
    MJD: int
    fiber_ID: int


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
def predict_class(star: StarInput):
    """
    Recibe los datos de un objeto astronómico y devuelve su clasificación.
    La clasificación puede ser: GALAXY, STAR o QSO.
    """
    try:
        # Convertimos el objeto recibido a un DataFrame de una sola fila,
        # que es el formato que espera la función predict() de ml/predict.py
        input_df = pd.DataFrame([star.model_dump()])
        result = predict(input_df)
        return {"prediction": result[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
