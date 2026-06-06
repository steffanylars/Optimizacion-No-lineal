"""
Geometría única del refrigerador OXXO.

Fuente de verdad compartida por `modelos.py` y `visualizacion.py` para evitar
duplicar (y desincronizar) las constantes del refri. Sin dependencias pesadas
(no importa pulp/lalo) para que la visualización pueda usarlo sin arrastrarlas.

Convención de índices (igual que en lalo_metaheuristicas):
  - jx : columna/puerta, 0..NX-1
  - jy : nivel,          0..NY-1  (0 = ABAJO, NY-1 = ARRIBA)
"""

NX, NY = 3, 6                                      # 3 puertas × 6 niveles = 18 charolas
ALTURAS = [42.0, 42.0, 31.5, 31.5, 28.0, 25.0]    # cm disponibles por nivel jy (0=abajo)
ANCHO_CHAROLA = 55.0                               # cm de ancho útil por charola (default)
K_DEFAULT = 8                                      # frentes (espacios) por charola (default)
