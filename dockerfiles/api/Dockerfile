
FROM python:3.11-slim
 
WORKDIR /app
 
# Instalar curl para healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiamos e instalamos las dependencias
COPY dockerfiles/api/requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt
 
# Copiamos el código de la app y el paquete ml
COPY app/ ./app/
COPY ml/ ./ml/
 
# Levantamos la API en el puerto 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
 