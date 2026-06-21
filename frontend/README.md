# 🌟 Frontend para Clasificador de Objetos Estelares

Frontend simple con Streamlit para interactuar con la API de clasificación estelar.

## 🚀 Instalación

### Opción 1: Instalación local

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar la aplicación
```bash
streamlit run app.py
```

3. Abrir en el navegador
http://localhost:8501

### Opción 2: Con Docker

1. Crear/Verificar el `Dockerfile`:
```docker
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY app.py .

# Exponer puerto
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Construir y ejecutar:
```bash
docker build -t stellar-classifier-frontend .
docker run -p 8501:8501 stellar-classifier-frontend
```

### 📖 Uso

1. Configurar URL de la API: En el sidebar, ingresa la URL donde está corriendo tu API (ej: http://localhost:8000 o http://api:8000 si está en Docker)

2. Ingresar variables obligatorias: Completa los 8 campos obligatorios (alpha, delta, u, g, r, i, z, redshift)

3. Agregar variables opcionales (opcional): Expande la sección y marca los campos que quieras incluir

4. Realizar predicción: Click en "🚀 Realizar Predicción"

5. Ver resultado: La aplicación mostrará si el objeto es una Estrella, Galaxia o Cuásar

### 🔧 Configuración

#### Variables de entorno

Se puede configurar la URL por defecto creando un archivo `config.toml`:
```bash
[server]
port = 8501
address = "0.0.0.0"
```

#### Personalización

- Rangos: Edita el diccionario RANGES en app.py
- Descripciones: Edita el diccionario DESCRIPTIONS
- Estilos: Streamlit usa temas personalizables en `config.toml`

### 🐛 Troubleshooting

**Error de conexión**: Verifica que la API esté corriendo y accesible desde donde corre Streamlit

**CORS**: Si la API rechaza peticiones, asegúrate de configurar CORS correctamente en FastAPI

**Timeout**: Si las predicciones toman mucho tiempo, aumenta el timeout en la función make_prediction()