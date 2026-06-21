FROM python:3.11-slim

WORKDIR /app

# Instalar curl para healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Instalar dependencias
COPY dockerfiles/frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY frontend/app.py .

# Exponer puerto
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
