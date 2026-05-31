# Trabajo Final MLOps

Trabajo en el que se aplican los conceptos aprendidos en la cátedra MLOps de la Carrera de Especialización en Inteligencia Artificial (FIUBA).
MLOps refiere al conjunto de prácticas dedicadas a automatizar y optimizar el ciclo de vida de un proyecto de Machine Learning, asegurando un despliegue y mantenimiento eficiente del modelo.

Integrantes:
- Quiroga, Martin Gabriel (a2412)
- García, Andrés Pablo (a2406)
- Jauffroy, Mariano (a2407)

## Setup

### Prerrequisitos

- Docker y Docker Compose
- Python 3.12 o superior
- uv package manager

### Instalación

1. Clonar el repositorio
```
git clone <url>
cd tf_mlops_ceia
```

2. Configurar las variables de entorno
```
copy .env.example .env
# Editar y coonfigurar las variables que correspondan.
```

3. Iniciar los servicios de Docker
```
docker-compose up -d
```

4. Instalar las dependencias del proyecto (uv)
```
uv sync
```

## Development

### Ejecutando la aplicación

Una vez que los servicios de docker estén levantados se puede acceder a las UI de MLflow y MinIO en:
- MLflow → http://localhost:5000
- MinIO → http://localhost:9000

Para ejecutar el flujo completo de entrenamiento, evaluación y logueo a MLflow, usar el comando:
```
python -m ml.train <model_type> <source>
```

donde `<model_type>` debe ser uno de los *trainers* habilitados en `ml/trainers/` y `<source>` es la ubicación del dataset a usar.

### Agregar dependencias

Usar `uv`para agregar las dependencias consistentemente en el proyecto.
```
uv add <package>
```

## Servidor API

## Convención de commits
Con el objetivo de mantener la historia limpia y trackeable, se definen algunas convenciones para redactar los commits de git. El commit base tiene la siguiente forma:
``` bash
git commit -m "<type>(<scope>):
    <description>"
```

donde:
- **\<type>** Indica qué tipo de cambio provee el commit. Puede ser:
    - `feat` -> Agrega nuevas características al proyecto
    - `fix` -> Soluciona algun problema
    - `refactor` -> Reescribe o reestructura alguna parte del proyecto
    - `test` -> Agrega tests o los corrige
    - `docs` -> Afecta únicamente documentación
    - `ops` -> Afecta la estructura operacional como infraestructura, deployment scripts, CI/CD pipelines, Docker, monitoreo, etc
- **\<scope>** Provee información del contexto de la modificación. Debe ser el nombre de una capacidad, módulo o archivo específicos del proyecto, estandarizados (No puede referirse al mismo contexto con nombres distintos ni por una mayúscula). Es opcional.
- **\<description>** Es la descripción detallada del proyecto y debe explayarse en la misma. Se debe evitar el uso frases abreviadas y poco explícitas. Un usuario externo al proyecto debe ser capaz de comprender los cambios de manera general leyendo el commit.

## Cómo agregar un nuevo modelo para entrenar

Por un lado, debe especificarse la configuración de los hiperparámetros del modelo y los parámetros del entrenamiento en un archivo .yaml dentro de la carpeta `/ml/config/`, por ejemplo:
```yaml
model:
  name: "stellar-classifier"
  type: "RandomForestClassifier"

# Features para el modelo
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

Por otro lado, se debe crear el *trainer* del modelo en la carpeta /ml/trainers/, heredando de la clase `BaseTrainer`. En este *trainer* se debe definir el Pipeline (sklearn) para entrenar, que incluye el escalado (opcional), la codificación (opcional) y el modelo.
```python
class RandomForestTrainer(BaseTrainer):
    
    @property
    def model_name(self) -> str:
        return "stellar-classifier"

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifer", RandomForestClassifier(
              **self.config.hyperparamenters
            )),
        ])
```
