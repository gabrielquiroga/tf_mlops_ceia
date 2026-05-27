# Registro del proyecto

En siguiente documento registra las acciones y decisiones que se toman a medida que se consolida este proyecto, intentando seguir las buenas prácticas de desarrollo.

## Inicialización

El proyecto se inicializó de manera simple mediante `uv`, ejecutando los siguientes comandos. Las dependencias agregadas son aquellas que se considera a priori que van a ser utilizadas en el proyecto. A futuro pueden cambiar y probablemente lo hagan. Esto Queda reflejado en el archivo `pyproject.toml`.

``` bash
uv init tf_mlops_ceia
cd tf_mlops_ceia
uv add fastapi uvicorn[standard] pandas scikit-learn mlflow pydantic-settings python-dotenv
uv add --dev pytest httpx ruff
```

Se crea también un archivo `.env.example` con las variables de entorno a configurar en donde sea que vaya a funcionar esa aplicación; y una clase `Settings` que las lee.

## Configuración de Machine Learning

Se migra el preprocesamiento del dataset a funciones en `ml/preprocessing`, refactorizando segun corresponde para seguir las buenas prácticas.
Se especifica una interfaz de configuración para los modelos, tratando de mantener el preprocesamiento agnóstico, tal que la configuración del modelo y sus entrenamiento se debe indicar en un archivo `.yaml`.