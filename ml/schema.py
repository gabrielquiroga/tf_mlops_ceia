# Metadata del dataset SDSS. Refleja la realidad del CSV, no decisiones de experimento.

# Columnas de identificación (nunca features)
ID_COLS = ['obj_ID', 'spec_obj_ID']

# Columna de agrupación para evitar data leakage en el split
GROUP_COL = 'obj_ID'

# Variable objetivo
TARGET = 'class'
TARGET_CLASSES = ['GALAXY', 'STAR', 'QSO']

# Valor centinela que indica error de medición
OUTLIER_SENTINEL = -9999
OUTLIER_COLS = ['u', 'g', 'z']  # columnas donde puede aparecer

# Catálogo de features disponibles, organizadas por tipo
PHYSICAL_FEATURES = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
INSTRUMENTAL_FEATURES = ['run_ID', 'cam_col', 'field_ID', 'plate', 'MJD', 'fiber_ID']
ALL_FEATURES = PHYSICAL_FEATURES + INSTRUMENTAL_FEATURES