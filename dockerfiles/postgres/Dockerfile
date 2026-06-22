FROM postgres:15

# Copiar el script de inicialización
# Postgres ejecuta automáticamente todos los .sh en /docker-entrypoint-initdb.d/
COPY init-db.sh /docker-entrypoint-initdb.d/
COPY healthcheck.sh /healthcheck.sh

# Convertir line endings a LF (por si fueron editados en Windows) y dar permisos
RUN sed -i 's/\r//' /docker-entrypoint-initdb.d/init-db.sh \
    && sed -i 's/\r//' /healthcheck.sh \
    && chmod +x /docker-entrypoint-initdb.d/init-db.sh /healthcheck.sh
