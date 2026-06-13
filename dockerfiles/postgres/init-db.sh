#!/bin/bash
set -e

# Este script se ejecuta automáticamente la primera vez que postgres arranca
# (solo si el volumen está vacío)

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE ${PG_DATABASE_MLFLOW:-mlflow_db};
    CREATE DATABASE ${PG_DATABASE_AIRFLOW:-airflow_db};
EOSQL

echo "Bases de datos creadas: ${PG_DATABASE_MLFLOW:-mlflow_db}, ${PG_DATABASE_AIRFLOW:-airflow_db}"
