#!/bin/bash
set -e

pg_isready -p 5432 -U "$POSTGRES_USER" -d postgres || exit 1

psql -U "$POSTGRES_USER" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname='${PG_DATABASE_MLFLOW:-mlflow_db}'" \
  | grep -q 1 || exit 1

psql -U "$POSTGRES_USER" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname='${PG_DATABASE_AIRFLOW:-airflow_db}'" \
  | grep -q 1 || exit 1
