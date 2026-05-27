# Trabajo Final MLOps

Introducción

Indicar cómo levantar la app

Instrucciones de deployment

Features

Etcétera

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
Por un lado, debe especificar la configuración de los hiperparámetros del modelo y los parámetros del entrenamiento en un archivo .yaml dentro de la carpeta `/ml/config/`, por ejemplo:
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
            ("classifer", RandomForestClassifier(**self.config.hyperparamenters)),
        ])
```