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