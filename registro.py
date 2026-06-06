"""
Persistencia de la tabla de soluciones (registros) del planograma OXXO.

Acumula —entre sesiones— un registro por modelo (PuLP / SA / GRASP) de cada
corrida en un JSON guardado junto a este archivo (`soluciones.json`). Cada
corrida ("CORRER LOS 3 MODELOS") comparte un mismo `id_solucion` y produce 3
registros. Columnas de la tabla de salida:

    id_solucion · modelo · charolas · frentes · niveles · desviacion_no_lineal · timestamp

- charolas:  número de charolas (jx, jy) ocupadas por la solución.
- frentes:   número total de frentes colocados (filas del sol_df).
- niveles:   número de niveles (jy) distintos ocupados.
- desviacion_no_lineal: el `fitness` que reporta cada solver en `metrics`
  (en SA/GRASP es el fitness penalizado R1-R13 del motor; en PuLP es el
  objetivo lineal de CBC — distinta escala, ver app/CLAUDE.md).
"""
import os
import re
import json
from datetime import datetime

import pandas as pd

# JSON junto a este archivo => persiste en el mismo directorio del proyecto.
RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soluciones.json')

# Orden canónico de columnas de la tabla de salida.
COLUMNAS = ['id_solucion', 'modelo', 'charolas', 'frentes', 'niveles',
            'desviacion_no_lineal', 'timestamp']

MODELOS = ['pulp', 'sa', 'grasp']
MODELO_NOMBRE = {'pulp': 'PuLP MIQP', 'sa': 'Simulated Annealing', 'grasp': 'GRASP + LS'}


def cargar_tabla(ruta=RUTA_JSON):
    """Lee la lista de registros del JSON. Devuelve [] si no existe o está corrupto."""
    if not os.path.exists(ruta):
        return []
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        return datos if isinstance(datos, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def guardar_tabla(registros, ruta=RUTA_JSON):
    """Escribe la lista completa de registros al JSON (UTF-8, indentado)."""
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)


def _siguiente_id(registros):
    """Siguiente id_solucion tipo 'SOL-0001' a partir del máximo existente."""
    maxn = 0
    for r in registros:
        m = re.match(r'SOL-(\d+)$', str(r.get('id_solucion', '')))
        if m:
            maxn = max(maxn, int(m.group(1)))
    return f'SOL-{maxn + 1:04d}'


def _metricas_modelo(sol_df, metrics):
    """(charolas, frentes, niveles, desviacion_no_lineal) de una solución."""
    if sol_df is None or len(sol_df) == 0:
        charolas = frentes = niveles = 0
    else:
        charolas = int(sol_df.groupby(['jx', 'jy']).ngroups)
        frentes = int(len(sol_df))
        niveles = int(sol_df['jy'].nunique())
    fitness = metrics.get('fitness') if metrics else None
    desv = round(float(fitness), 2) if fitness is not None else None
    return charolas, frentes, niveles, desv


def construir_registros(resultados, id_solucion, timestamp=None):
    """Construye la lista de registros (uno por modelo) de una corrida.

    Función pura (sin I/O). `resultados` es el dict de app.py:
    {'pulp': {'sol': df, 'metrics': {...}}, 'sa': {...}, 'grasp': {...}}.
    """
    timestamp = timestamp or datetime.now().isoformat(timespec='seconds')
    registros = []
    for k in MODELOS:
        if k not in resultados:
            continue
        charolas, frentes, niveles, desv = _metricas_modelo(
            resultados[k].get('sol'), resultados[k].get('metrics'))
        registros.append({
            'id_solucion': id_solucion,
            'modelo': MODELO_NOMBRE.get(k, k),
            'charolas': charolas,
            'frentes': frentes,
            'niveles': niveles,
            'desviacion_no_lineal': desv,
            'timestamp': timestamp,
        })
    return registros


def registrar_corrida(resultados, ruta=RUTA_JSON):
    """Carga la tabla, agrega los registros de esta corrida y la guarda.

    Devuelve (id_solucion, registros_nuevos, tabla_completa).
    """
    tabla = cargar_tabla(ruta)
    id_solucion = _siguiente_id(tabla)
    nuevos = construir_registros(resultados, id_solucion)
    tabla.extend(nuevos)
    guardar_tabla(tabla, ruta)
    return id_solucion, nuevos, tabla


def tabla_como_df(ruta=RUTA_JSON):
    """Tabla completa como DataFrame con las columnas en orden canónico."""
    registros = cargar_tabla(ruta)
    if not registros:
        return pd.DataFrame(columns=COLUMNAS)
    df = pd.DataFrame(registros)
    for c in COLUMNAS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNAS]
