# Trabajo Final MLOps - Stellar Classification Pipeline

Trabajo en el que se aplican los conceptos aprendidos en la cátedra MLOps de la Carrera de Especialización en Inteligencia Artificial (FIUBA).

**Integrantes:**
- Quiroga, Martin Gabriel (a2412)
- García, Andrés Pablo (a2406)
- Jauffroy, Mariano (a2407)

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto es diseñar e implementar una **arquitectura MLOps *end-to-end*** robusta, escalable y automatizada. Se busca abarcar todo el ciclo de vida de un modelo de Machine Learning orientado a la clasificación estelar (Stellar Classification): desde la orquestación de la ingesta de datos, pasando por el entrenamiento, evaluación, versionado y registro (Tracking/Registry), hasta el despliegue del modelo campeón ("Champion") a través de una API y una interfaz gráfica interactiva.

## 🚀 Características y Tecnologías Utilizadas

Este proyecto adopta las mejores prácticas de la industria, utilizando las siguientes herramientas:

- **Containerización:** `Docker` y `Docker Compose` para el despliegue de toda la infraestructura.
- **Orquestación de Pipelines:** `Apache Airflow` (con CeleryExecutor y Redis) para programar y monitorear los flujos de trabajo (DAGs).
- **Tracking & Model Registry:** `MLflow` para llevar el registro de los experimentos, métricas, hiperparámetros y versionar los modelos generados.
- **Object Storage:** `MinIO` (compatible con S3) como repositorio central de artefactos (datasets y modelos binarios).
- **Base de Datos Relacional:** `PostgreSQL` para almacenar los metadatos de MLflow y Airflow.
- **Message Broker:** `Redis` (Valkey) encargado de gestionar la cola de mensajes entre los workers y el scheduler de Airflow.
- **API de Serving:** `FastAPI` / Python para exponer el modelo campeón y realizar predicciones en tiempo real.
- **Frontend / UI:** `Streamlit` para proveer una interfaz web intuitiva a los usuarios finales.
- **Gestión de Paquetes:** `uv`, el package manager ultrarrápido para gestionar las dependencias del proyecto en Python 3.12.

## 📂 Estructura del Proyecto

El repositorio está organizado en los siguientes directorios principales:

```text
tf_mlops_ceia/
├── airflow/        # DAGs, plugins y configuraciones de Apache Airflow
├── app/            # Código fuente de la API REST (Serving del modelo)
├── dataset/        # Carpeta local compartida con los servicios (ej. MinIO y Airflow)
├── dockerfiles/    # Imágenes Docker personalizadas para cada servicio
├── frontend/       # Interfaz gráfica de usuario desarrollada con Streamlit
├── ml/             # Código fuente de entrenamiento, validación y definición de los modelos (Trainers)
├── docker-compose.yml  # Definición de la infraestructura completa
└── pyproject.toml  # Definición de las dependencias gestionadas con uv
```

## 🛠️ Cómo clonar y ejecutar el proyecto

### Prerrequisitos
- Docker y Docker Compose
- Python 3.12 o superior
- Administrador de dependencias `uv`

### Instrucciones paso a paso

1. **Clonar el repositorio**
   ```bash
   git clone <url_de_tu_repositorio>
   cd tf_mlops_ceia
   ```

2. **Configurar las variables de entorno**
   Copia el archivo de ejemplo de configuración y ajusta las variables según sea necesario:
   ```bash
   cp .env.example .env
   # Editar .env para configurar credenciales y accesos
   ```

3. **Iniciar la infraestructura con Docker Compose**
   Levanta todos los servicios en segundo plano:
   ```bash
   docker-compose up -d
   ```
   *(Nota: La primera ejecución tomará unos minutos mientras construye las imágenes e inicializa las bases de datos).*

4. **Instalar dependencias locales (opcional para desarrollo local)**
   ```bash
   uv sync
   ```

5. **Ejecutar los DAGs en Airflow manualmente si se desea.**
   Es necesario para tener un primer champion luego de levantar el proyecto.

6. **Hacer una solicitud a la API**
   - Puede ser via UI en [http://localhost:8501](http://localhost:8501)
   - Puede ser via Postman o Curl
   ```bash
   curl -X POST http://localhost:8000/predict \
   -H "Content-Type: application/json" \
   -d '[{
      "alpha": 135.689,
      "delta": 32.494,
      "u": 23.87882,
      "g": 22.27530,
      "r": 20.39501,
      "i": 19.16573,
      "z": 18.79371,
      "redshift": 0.6347
   }]'
   ```


## 🌐 Acceso a las Interfaces de los Servicios

Una vez que los contenedores estén corriendo de forma saludable (healthy), se puede acceder a las siguientes interfaces web directamente desde el navegador:

- **Frontend (Streamlit)**: [http://localhost:8501](http://localhost:8501) - *Interfaz de usuario para predicciones.*
- **API Serving (Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs) - *Documentación Swagger de la API.*
- **Apache Airflow UI**: [http://localhost:8080](http://localhost:8080) - *Monitoreo de DAGs y tareas.*
- **MLflow Tracking**: [http://localhost:5000](http://localhost:5000) - *Visualización de experimentos y métricas.*
- **MinIO Console**: [http://localhost:9001](http://localhost:9001) - *Explorador de buckets (datasets y artefactos).*

## 👨‍💻 Development

### Ejecutando la aplicación de entrenamiento localmente
Para ejecutar el flujo completo de entrenamiento, evaluación y logueo a MLflow de forma manual, se utiliza el comando:
```bash
python -m ml.train <model_type> <source>
```
Donde `<model_type>` debe ser uno de los *trainers* habilitados en `ml/trainers/` (ej: `stellar-classifier`) y `<source>` es la ubicación del dataset a usar.

### Agregar dependencias
Usa `uv` para agregar las dependencias consistentemente en el proyecto:
```bash
uv add <package>
```

### Cómo agregar un nuevo modelo para entrenar
Por un lado, debe especificarse la configuración de los hiperparámetros del modelo y los parámetros del entrenamiento en un archivo `.yaml` dentro de la carpeta `/ml/config/`, por ejemplo:
```yaml
model:
  name: "stellar-classifier"
  type: "RandomForestClassifier"

features: ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']

hyperparameters:
  n_estimators: 94
  max_depth: 21
  min_samples_split: 2
  min_samples_leaf: 1
  random_state: 42
  n_jobs: -1

training:
  test_size: 0.2
  random_state: 42
```

Por otro lado, se debe crear el *trainer* del modelo en la carpeta `/ml/trainers/`, heredando de la clase `BaseTrainer`. En este *trainer* se debe definir el Pipeline (sklearn) para entrenar, que incluye el escalado (opcional), la codificación (opcional) y el modelo.
```python
class RandomForestTrainer(BaseTrainer):
    @property
    def model_name(self) -> str:
        return "stellar-classifier"

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(**self.config.hyperparameters)),
        ])
```

## 📝 Convención de commits
Con el objetivo de mantener la historia limpia y trackeable, se definen algunas convenciones para redactar los commits de git. El commit base tiene la siguiente forma:
```bash
git commit -m "<type>(<scope>): <description>"
```
- **\<type>** Puede ser: `feat`, `fix`, `refactor`, `test`, `docs`, `ops`.
- **\<scope>** Contexto de la modificación (módulo o archivo, estandarizado, en minúsculas).
- **\<description>** Descripción detallada del cambio para que cualquier usuario la comprenda.

