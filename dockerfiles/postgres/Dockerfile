FROM postgres:15

# Copiar el script de inicialización
# Postgres ejecuta automáticamente todos los .sh en /docker-entrypoint-initdb.d/
COPY init-db.sh /docker-entrypoint-initdb.d/

# Dar permisos de ejecución
RUN chmod +x /docker-entrypoint-initdb.d/init-db.sh
