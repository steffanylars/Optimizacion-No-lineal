"""
Lógica de los 3 modelos del planograma OXXO.
- PuLP MIQP (modelo exacto)
- Simulated Annealing (Lalo)
- GRASP + Local Search (Lalo)

Todos comparten parámetros calibrados:
- F_ijxjy = score histórico del óptimo OXXO
- 18 charolas (3 puertas × 6 niveles)
- Alturas reales: [42, 42, 31.5, 31.5, 28, 25] cm
"""
import numpy as np
import pandas as pd
import time
from pulp import (LpProblem, LpVariable, LpMaximize, LpBinary, LpStatus,
                  PULP_CBC_CMD, lpSum, value)

import lalo_metaheuristicas as PL
from lalo_metaheuristicas import Config, simulated_annealing, grasp, FAMILIAS, Solucion

# ── CONSTANTES ──────────────────────────────────────────────────────────────
NX, NY = 3, 6
K = 8
ALTURAS_NIVEL = [42.0, 42.0, 31.5, 31.5, 28.0, 25.0]


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
def familia_de_desc(s):
    s = str(s).lower()
    if any(t in s for t in ['agua', 'ciel', 'mineral', 'electrolit',
                             'peñafiel', 'penafiel']):
        return 'agua'
    if any(t in s for t in ['cocacola', 'coca-cola', 'coca cola', 'pepsi']):
        return 'cola'
    return 'sabor'


def cargar_optimo(csv_path):
    """Carga ejemplo_planograma.csv y filtra al caso CF/DI/T3"""
    ej = pd.read_csv(csv_path, encoding='latin-1')
    rename_map = {c: ('SEGMENTO_ID' if 'SEGMENTO' in c else
                      'TAMANO_POST' if ('TAMA' in c and 'POST' in c) else
                      'DISENO_REF' if 'DISE' in c else c)
                  for c in ej.columns}
    ej = ej.rename(columns=rename_map)
    optimo = ej[
        (ej['PLANOGRUPO'] == 'Refrescos') &
        (ej['MUEBLE_ID'] == 'CF') &
        (ej['TAMANO_POST'] == 3.0) &
        (ej['DIRECCION_LEGO_ID'] == 'DI')
    ].copy()

    X_VALS = sorted(optimo['X'].unique())
    Y_VALS = sorted(optimo['Y'].unique())
    optimo['jx'] = optimo['X'].map({x: i for i, x in enumerate(X_VALS)})
    optimo['jy'] = optimo['Y'].map({y: i for i, y in enumerate(Y_VALS)})
    optimo['familia'] = optimo['ITEM_DESC'].apply(familia_de_desc)
    optimo['nombre'] = optimo['ITEM_DESC'].str.strip()
    return optimo


def extraer_referencias(optimo):
    """Extrae F_pos, casa, fam_dom, prods_por_charola del óptimo"""
    F_pos = {}
    for it in optimo['ITEM'].unique():
        for jx in range(NX):
            for jy in range(NY):
                F_pos[(int(it), jx, jy)] = 0
    for _, row in optimo.iterrows():
        F_pos[(int(row['ITEM']), int(row['jx']), int(row['jy']))] += int(row['NUM_FRENTES'])

    casa = {}
    for it in optimo['ITEM'].unique():
        best = max(((jx, jy, F_pos[(int(it), jx, jy)])
                    for jx in range(NX) for jy in range(NY)),
                   key=lambda t: t[2])
        if best[2] > 0:
            casa[int(it)] = (best[0], best[1])

    fam_dom = (optimo.groupby(['jx', 'jy', 'familia']).size()
               .reset_index(name='n'))
    fam_dom = fam_dom.loc[fam_dom.groupby(['jx', 'jy'])['n'].idxmax()]
    fam_dict = {(int(r['jx']), int(r['jy'])): r['familia']
                for _, r in fam_dom.iterrows()}

    prods_charola = (optimo.groupby(['jx', 'jy'])['ITEM']
                     .apply(lambda x: set(int(i) for i in x.unique())).to_dict())

    nombre_oficial = optimo.groupby('ITEM')['ITEM_DESC'].first().to_dict()
    ancho_prod = optimo.groupby('ITEM')['ANCHO'].mean().to_dict()
    alto_prod = optimo.groupby('ITEM')['ALTO'].mean().to_dict()

    return {
        'F_pos': F_pos, 'casa': casa, 'fam_dict': fam_dict,
        'prods_charola': prods_charola, 'nombre_oficial': nombre_oficial,
        'ancho_prod': ancho_prod, 'alto_prod': alto_prod,
    }


def coincidencia(sol_df, refs):
    """Calcula las 4 métricas de coincidencia con el óptimo"""
    if len(sol_df) == 0:
        return {'pct_casa': 0, 'distancia_promedio': 0,
                'pct_fam_correcta': 0, 'jaccard_promedio': 0}

    casa = refs['casa']
    fam_dict = refs['fam_dict']
    prods_charola = refs['prods_charola']

    en_casa = 0; total = 0; dists = []
    for _, r in sol_df.iterrows():
        try:
            it = int(r['item'])
        except (ValueError, TypeError):
            continue
        if it in casa:
            total += 1
            cm = (int(r['jx']), int(r['jy']))
            cr = casa[it]
            if cm == cr:
                en_casa += 1
            dists.append(abs(cm[0]-cr[0]) + abs(cm[1]-cr[1]))

    fam_ok = 0; ch_tot = 0
    for (jx, jy), grp in sol_df.groupby(['jx', 'jy']):
        if (jx, jy) in fam_dict:
            ch_tot += 1
            if grp['familia'].value_counts().idxmax() == fam_dict[(jx, jy)]:
                fam_ok += 1

    jaccards = []
    for (jx, jy), grp in sol_df.groupby(['jx', 'jy']):
        m = set()
        for it in grp['item']:
            try:
                m.add(int(it))
            except (ValueError, TypeError):
                pass
        h = prods_charola.get((jx, jy), set())
        if m or h:
            jaccards.append(len(m & h) / max(len(m | h), 1))

    return {
        'pct_casa': round(en_casa/max(total, 1) * 100, 1),
        'distancia_promedio': round(float(np.mean(dists)) if dists else 0, 2),
        'pct_fam_correcta': round(fam_ok/max(ch_tot, 1) * 100, 1),
        'jaccard_promedio': round(float(np.mean(jaccards))*100 if jaccards else 0, 1),
    }


# ────────────────────────────────────────────────────────────────────────────
# MODELO 1: PULP MIQP
# ────────────────────────────────────────────────────────────────────────────
def correr_pulp(optimo, refs, top_n=45, time_limit=60):
    F_total = optimo.groupby('ITEM')['NUM_FRENTES'].sum().sort_values(ascending=False)
    I = F_total.head(top_n).index.tolist()
    ancho = refs['ancho_prod']
    alto = refs['alto_prod']
    F_pos = refs['F_pos']
    nombre = refs['nombre_oficial']

    prob = LpProblem('OXXO_calibrado', LpMaximize)
    X = {(i, jx, jy, k): LpVariable(f'X_{i}_{jx}_{jy}_{k}', 0, 1, LpBinary)
         for i in I for jx in range(NX) for jy in range(NY) for k in range(K)}

    # Objetivo
    prob += lpSum(F_pos.get((i, jx, jy), 0) * X[(i, jx, jy, k)]
                  for i in I for jx in range(NX) for jy in range(NY) for k in range(K))

    # R1: Ancho ≤ 55cm
    for jx in range(NX):
        for jy in range(NY):
            prob += lpSum(ancho[i] * X[(i, jx, jy, k)]
                          for i in I for k in range(K)) <= 55.0

    # R2: Alto del producto ≤ altura del nivel
    for i in I:
        for jx in range(NX):
            for jy in range(NY):
                if alto.get(i, 25.0) > ALTURAS_NIVEL[jy] + 0.5:
                    for k in range(K):
                        prob += X[(i, jx, jy, k)] == 0

    # R3: Cobertura
    for i in I:
        prob += lpSum(X[(i, jx, jy, k)]
                      for jx in range(NX) for jy in range(NY)
                      for k in range(K)) >= 1

    # Max frentes
    for i in I:
        prob += lpSum(X[(i, jx, jy, k)]
                      for jx in range(NX) for jy in range(NY)
                      for k in range(K)) <= 7

    # R4: Unicidad
    for jx in range(NX):
        for jy in range(NY):
            for k in range(K):
                prob += lpSum(X[(i, jx, jy, k)] for i in I) <= 1

    t0 = time.time()
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=0.05))
    t_pulp = time.time() - t0

    rows = []
    for i in I:
        for jx in range(NX):
            for jy in range(NY):
                for k in range(K):
                    if value(X[(i, jx, jy, k)]) > 0.5:
                        rows.append({
                            'item': i, 'jx': jx, 'jy': jy, 'k': k,
                            'ancho': ancho[i], 'peso': alto.get(i, 25.0),
                            'F': F_total[i],
                            'familia': familia_de_desc(nombre[i]),
                            'desc': nombre[i].strip(),
                        })

    sol_df = pd.DataFrame(rows)
    m = coincidencia(sol_df, refs)
    return sol_df, {
        'fitness': float(value(prob.objective) if prob.objective else 0),
        'tiempo_s': t_pulp,
        'factible': LpStatus[prob.status] == 'Optimal',
        'hibridas': 0,
        **m,
    }


# ────────────────────────────────────────────────────────────────────────────
# PATCHES A LALO PARA CALIBRARLO AL ÓPTIMO
# ────────────────────────────────────────────────────────────────────────────
def aplicar_patches_lalo(refs):
    """Aplica patches al módulo de Lalo para usar parámetros calibrados"""
    PL.NX = NX
    PL.NY = NY
    PL.NCH = NX * NY

    def charola_a_jxjy_real(charola):
        c = int(charola) - 1
        jx = c // 6
        pos = c % 6
        jy = 5 - pos
        return jx, jy
    PL.charola_a_jxjy = charola_a_jxjy_real

    F_pos = refs['F_pos']
    casa = refs['casa']

    _orig_construir = PL.construir_problema

    def construir_calibrado(csv_path, cfg=None):
        prob = _orig_construir(csv_path, cfg)
        # Alturas reales
        prob.rho_max = np.array(ALTURAS_NIVEL, dtype=float)
        # col_fam: P1=cola, P2=cola, P3=sabor (del óptimo real)
        prob.col_fam = np.array([0, 0, 2])
        # Reemplazar F con score del óptimo
        F_new = np.zeros_like(prob.F)
        for i, item in enumerate(prob.items):
            try:
                item_int = int(item)
            except (ValueError, TypeError):
                item_int = item
            for jx in range(NX):
                for jy in range(NY):
                    F_new[i, jx, jy] = F_pos.get((item_int, jx, jy), 0)
            if F_new[i].sum() == 0:
                F_new[i] = prob.F[i] * 0.1
        F_new = F_new * 100.0
        # Penalizar posiciones nunca usadas
        for i in range(prob.I):
            try:
                it = int(prob.items[i])
            except (ValueError, TypeError):
                continue
            if it in casa:
                for jx in range(NX):
                    for jy in range(NY):
                        if F_pos.get((it, jx, jy), 0) == 0:
                            F_new[i, jx, jy] = -10.0
        prob.F = F_new
        return prob

    PL.construir_problema = construir_calibrado

    # Penalización suave (sin forzar columna)
    _orig_pen = PL.Solucion._pen_global

    def _pen_simple(self):
        cfg = self.cfg
        no_col = int(np.count_nonzero(self.counts == 0))
        pen = cfg.pen_cobertura * no_col
        nh = int(self._hibrida.sum())
        if nh > cfg.epsilon:
            pen += cfg.pen_hibrido * (nh - cfg.epsilon)
        return pen, no_col, nh

    PL.Solucion._pen_global = _pen_simple


def preparar_csv_para_lalo(hist_df, refs, output_path):
    """Adapta el CSV histórico al formato que espera Lalo"""
    hist = hist_df.copy()
    if 'UPC_CVE' in hist.columns:
        hist['ITEM'] = hist['UPC_CVE']
    if 'TAMAÑO' in hist.columns:
        if 'TAMANO_POST' in hist.columns:
            hist = hist.drop(columns=['TAMANO_POST'])
        hist['TAMANO_POST'] = hist['TAMAÑO']
        hist = hist.drop(columns=['TAMAÑO'], errors='ignore')
    hist['CONJUNTO_ID'] = hist['SEGMENTO_ID']

    alto_prod = refs['alto_prod']
    nombre = refs['nombre_oficial']

    def get_alto(i):
        try:
            return alto_prod.get(int(i), 25.0)
        except (ValueError, TypeError):
            return 25.0
    hist['ALTO'] = hist['ITEM'].map(get_alto)

    def get_nombre(i):
        try:
            return nombre.get(int(i), f'Producto {str(i)[-4:]}')
        except (ValueError, TypeError):
            return f'Producto {str(i)[-4:]}'
    hist['ITEM_DESC'] = hist['ITEM'].map(get_nombre)

    hist.to_csv(output_path, index=False)
    return output_path


def construir_inicial_desde_optimo(prob, optimo, item_to_idx, cfg, seg='BCO'):
    """Construye solución inicial desde un segmento óptimo de OXXO"""
    sol = Solucion(prob)
    seg_df = optimo[optimo['SEGMENTO_ID'] == seg]
    cambios = []
    posiciones_usadas = set()
    for (jx, jy), grp in seg_df.groupby(['jx', 'jy']):
        k = 0
        for _, row in grp.iterrows():
            try:
                item = int(row['ITEM'])
            except (ValueError, TypeError):
                continue
            if item in item_to_idx and k < cfg.K:
                i_idx = item_to_idx[item]
                nf = int(row['NUM_FRENTES'])
                for _ in range(min(nf, cfg.K - k)):
                    if (jx, jy, k) not in posiciones_usadas:
                        cambios.append((int(jx), int(jy), k, i_idx))
                        posiciones_usadas.add((jx, jy, k))
                        k += 1
    sol.aplicar(cambios)
    return sol


def extraer_sol(sol, prob, refs):
    rows = []
    nombre = refs['nombre_oficial']
    for jx in range(NX):
        for jy in range(NY):
            for k in range(K):
                p = int(sol.grid[jx, jy, k])
                if p >= 0:
                    item = prob.items[p]
                    try:
                        item_int = int(item)
                    except (ValueError, TypeError):
                        item_int = item
                    rows.append({
                        'item': item,
                        'desc': nombre.get(item_int, prob.desc[p]).strip(),
                        'jx': jx, 'jy': jy, 'k': k,
                        'ancho': float(prob.beta[p]),
                        'peso': float(prob.rho[p]),
                        'familia': FAMILIAS[prob.fam[p]],
                        'F': float(prob.F[p, jx, jy]),
                    })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# MODELO 2: SIMULATED ANNEALING
# ────────────────────────────────────────────────────────────────────────────
def correr_sa(hist_df, optimo, refs):
    aplicar_patches_lalo(refs)
    csv_path = preparar_csv_para_lalo(hist_df, refs, '/tmp/_lalo_input.csv')

    cfg = Config(K=K, delta=0.01, epsilon=11,
                 pen_cobertura=10000.0, pen_ancho=200000.0,
                 pen_peso=50000.0, pen_hibrido=500.0, pen_columna=300.0)
    prob = PL.construir_problema(csv_path, cfg)
    item_to_idx = {it: i for i, it in enumerate(prob.items)}

    sol_init = construir_inicial_desde_optimo(prob, optimo, item_to_idx, cfg, 'BCO')

    t0 = time.time()
    res = simulated_annealing(prob, np.random.default_rng(42),
                              T0=10.0, Tmin=0.05, alpha_cool=0.96,
                              iter_por_T=400, sol_inicial=sol_init)
    t = time.time() - t0

    sol_df = extraer_sol(res.solucion, prob, refs)
    r = res.solucion.resumen()
    m = coincidencia(sol_df, refs)
    return sol_df, {
        'fitness': float(r['fitness']),
        'tiempo_s': t,
        'factible': bool(r['factible']),
        'hibridas': int(r['charolas_hibridas']),
        **m,
    }


# ────────────────────────────────────────────────────────────────────────────
# MODELO 3: GRASP + LS
# ────────────────────────────────────────────────────────────────────────────
def correr_grasp(hist_df, optimo, refs):
    aplicar_patches_lalo(refs)
    csv_path = preparar_csv_para_lalo(hist_df, refs, '/tmp/_lalo_input.csv')

    cfg = Config(K=K, delta=0.01, epsilon=11,
                 pen_cobertura=10000.0, pen_ancho=200000.0,
                 pen_peso=50000.0, pen_hibrido=500.0, pen_columna=300.0)
    prob = PL.construir_problema(csv_path, cfg)

    t0 = time.time()
    res = grasp(prob, np.random.default_rng(7), n_iter=12,
                ls_max_iter=4000, rcl_frac=0.1)
    t = time.time() - t0

    sol_df = extraer_sol(res.solucion, prob, refs)
    r = res.solucion.resumen()
    m = coincidencia(sol_df, refs)
    return sol_df, {
        'fitness': float(r['fitness']),
        'tiempo_s': t,
        'factible': bool(r['factible']),
        'hibridas': int(r['charolas_hibridas']),
        **m,
    }
