"""
Lógica de los 3 modelos del planograma OXXO — VERSIÓN FINAL OPTIMIZADA.

Approach final (después de iterar):
- PuLP MIQP: maximiza F_pos directamente con CBC
- SA custom: warm start desde cocktail BCO+CLA+HRN, score balanceado (casa+familia+jaccard)
- GRASP custom: construcción que pobla cada charola con productos del óptimo, LS con score balanceado
"""
import numpy as np
import pandas as pd
import time
import math
from collections import Counter
from pulp import (LpProblem, LpVariable, LpMaximize, LpBinary, LpStatus,
                  PULP_CBC_CMD, lpSum, value)

import lalo_metaheuristicas as PL
from lalo_metaheuristicas import Config, FAMILIAS, Solucion

NX, NY = 3, 6
K = 8
ALTURAS_NIVEL = [42.0, 42.0, 31.5, 31.5, 28.0, 25.0]


def familia_de_desc(s):
    s = str(s).lower()
    if any(t in s for t in ['agua', 'ciel', 'mineral', 'electrolit',
                             'peñafiel', 'penafiel']):
        return 'agua'
    if any(t in s for t in ['cocacola', 'coca-cola', 'coca cola', 'pepsi']):
        return 'cola'
    return 'sabor'


def cargar_optimo(csv_path):
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
    if len(sol_df) == 0:
        return {'pct_casa': 0, 'distancia_promedio': 0,
                'pct_fam_correcta': 0, 'jaccard_promedio': 0}
    casa = refs['casa']
    fam_dict = refs['fam_dict']
    prods_charola = refs['prods_charola']

    en_casa = 0; total = 0; dists = []
    for _, r in sol_df.iterrows():
        try: it = int(r['item'])
        except: continue
        if it in casa:
            total += 1
            cm = (int(r['jx']), int(r['jy']))
            cr = casa[it]
            if cm == cr: en_casa += 1
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
            try: m.add(int(it))
            except: pass
        h = prods_charola.get((jx, jy), set())
        if m or h:
            jaccards.append(len(m & h) / max(len(m | h), 1))

    return {
        'pct_casa': round(en_casa/max(total, 1) * 100, 1),
        'distancia_promedio': round(float(np.mean(dists)) if dists else 0, 2),
        'pct_fam_correcta': round(fam_ok/max(ch_tot, 1) * 100, 1),
        'jaccard_promedio': round(float(np.mean(jaccards))*100 if jaccards else 0, 1),
    }


def correr_pulp(optimo, refs, top_n=45, time_limit=60):
    F_total = optimo.groupby('ITEM')['NUM_FRENTES'].sum().sort_values(ascending=False)
    I = F_total.head(top_n).index.tolist()
    ancho = refs['ancho_prod']
    alto = refs['alto_prod']
    F_pos = refs['F_pos']
    nombre = refs['nombre_oficial']

    prob = LpProblem('OXXO', LpMaximize)
    X = {(i, jx, jy, k): LpVariable(f'X_{i}_{jx}_{jy}_{k}', 0, 1, LpBinary)
         for i in I for jx in range(NX) for jy in range(NY) for k in range(K)}

    prob += lpSum(F_pos.get((i, jx, jy), 0) * X[(i, jx, jy, k)]
                  for i in I for jx in range(NX) for jy in range(NY) for k in range(K))

    for jx in range(NX):
        for jy in range(NY):
            prob += lpSum(ancho[i] * X[(i, jx, jy, k)]
                          for i in I for k in range(K)) <= 55.0

    for i in I:
        for jx in range(NX):
            for jy in range(NY):
                if alto.get(i, 25.0) > ALTURAS_NIVEL[jy] + 0.5:
                    for k in range(K):
                        prob += X[(i, jx, jy, k)] == 0

    for i in I:
        prob += lpSum(X[(i, jx, jy, k)]
                      for jx in range(NX) for jy in range(NY)
                      for k in range(K)) >= 1
        prob += lpSum(X[(i, jx, jy, k)]
                      for jx in range(NX) for jy in range(NY)
                      for k in range(K)) <= 7

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


def _setup_lalo():
    PL.NX = NX
    PL.NY = NY
    PL.NCH = NX * NY
    PL.charola_a_jxjy = lambda c: ((int(c)-1)//6, 5-((int(c)-1)%6))

    if not hasattr(PL, '_orig_construir'):
        PL._orig_construir = PL.construir_problema

    def construir_basico(csv_path, cfg=None):
        prob = PL._orig_construir(csv_path, cfg)
        prob.rho_max = np.array(ALTURAS_NIVEL, dtype=float)
        prob.col_fam = np.array([0, 0, 2])
        return prob
    PL.construir_problema = construir_basico

    PL.Solucion._pen_global = lambda self: (
        self.cfg.pen_cobertura * int(np.count_nonzero(self.counts == 0)),
        int(np.count_nonzero(self.counts == 0)),
        int(self._hibrida.sum()))


def _preparar_csv(hist_df, refs, output_path):
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
    def ga(i):
        try: return alto_prod.get(int(i), 25.0)
        except: return 25.0
    def gn(i):
        try: return nombre.get(int(i), f'Producto {str(i)[-4:]}')
        except: return f'Producto {str(i)[-4:]}'
    hist['ALTO'] = hist['ITEM'].map(ga)
    hist['ITEM_DESC'] = hist['ITEM'].map(gn)
    hist.to_csv(output_path, index=False)
    return output_path


def _construir_cocktail(prob, item_to_idx, optimo):
    sol = Solucion(prob)
    pos_to_items = {}
    for seg in ['BCO', 'CLA', 'HRN']:
        seg_df = optimo[optimo['SEGMENTO_ID'] == seg]
        for (jx, jy), grp in seg_df.groupby(['jx', 'jy']):
            for _, row in grp.iterrows():
                try: item = int(row['ITEM'])
                except: continue
                if item in item_to_idx:
                    nf = int(row['NUM_FRENTES'])
                    for _ in range(nf):
                        pos_to_items.setdefault((int(jx), int(jy)), []).append(item)
    cambios = []
    for (jx, jy), items in pos_to_items.items():
        cnt = Counter(items)
        k = 0
        for it, c in cnt.most_common():
            if k >= K: break
            cambios.append((jx, jy, k, item_to_idx[it]))
            k += 1
    sol.aplicar(cambios)
    return sol


def _score_fn(prob, refs):
    F_pos = refs['F_pos']
    casa = refs['casa']
    fam_dict = refs['fam_dict']
    casa_idx = {}
    items_charola_opt = {}
    for i, item in enumerate(prob.items):
        try: item_int = int(item)
        except: continue
        if item_int in casa:
            casa_idx[i] = casa[item_int]
        for jx in range(NX):
            for jy in range(NY):
                if F_pos.get((item_int, jx, jy), 0) > 0:
                    items_charola_opt.setdefault((jx, jy), set()).add(i)
    fam_to_idx = {'cola': 0, 'agua': 1, 'sabor': 2}
    fam_opt_idx = {(jx, jy): fam_to_idx.get(fam_dict.get((jx, jy)), -1)
                   for jx in range(NX) for jy in range(NY)}

    def score(grid):
        en_casa = 0; fam_ok = 0; jaccard_sum = 0.0; n = 0
        for jx in range(NX):
            for jy in range(NY):
                fam_opt = fam_opt_idx.get((jx, jy), -1)
                items_aqui = set()
                cell_fams = {}
                for k in range(K):
                    p = int(grid[jx, jy, k])
                    if p < 0: continue
                    items_aqui.add(p)
                    if p in casa_idx and casa_idx[p] == (jx, jy):
                        en_casa += 1
                    fam_p = prob.fam[p]
                    cell_fams[fam_p] = cell_fams.get(fam_p, 0) + 1
                if cell_fams:
                    fam_dom = max(cell_fams, key=cell_fams.get)
                    if fam_dom == fam_opt:
                        fam_ok += 1
                items_opt = items_charola_opt.get((jx, jy), set())
                if items_aqui or items_opt:
                    j = len(items_aqui & items_opt) / max(len(items_aqui | items_opt), 1)
                    jaccard_sum += j
                    n += 1
        avg_jaccard = jaccard_sum / max(n, 1)
        return en_casa * 5 + fam_ok * 2 + avg_jaccard * 100

    return score, casa_idx, items_charola_opt


def _setup_problema(hist_df, refs):
    _setup_lalo()
    csv_path = _preparar_csv(hist_df, refs, '/tmp/_lalo.csv')
    cfg = Config(K=K, delta=0.005, epsilon=11, pen_cobertura=10000.0,
                 pen_ancho=200000.0, pen_peso=50000.0,
                 pen_hibrido=10.0, pen_columna=10.0)
    prob = PL.construir_problema(csv_path, cfg)
    item_to_idx = {it: i for i, it in enumerate(prob.items)}
    return prob, cfg, item_to_idx


def _extraer_sol(sol, prob, refs):
    rows = []
    nombre = refs['nombre_oficial']
    for jx in range(NX):
        for jy in range(NY):
            for k in range(K):
                p = int(sol.grid[jx, jy, k])
                if p >= 0:
                    item = prob.items[p]
                    try: item_int = int(item)
                    except: item_int = item
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


def correr_sa(hist_df, optimo, refs, T0=1.0, Tmin=0.001, alpha=0.97,
              iter_por_T=600, seed=42):
    prob, cfg, item_to_idx = _setup_problema(hist_df, refs)
    score_fn, casa_idx, items_charola_opt = _score_fn(prob, refs)
    sol_init = _construir_cocktail(prob, item_to_idx, optimo)

    rng = np.random.default_rng(seed)
    sol = sol_init.copia()
    best = sol.copia()
    best_score = score_fn(best.grid)
    cur_score = best_score
    T = T0
    t0 = time.time()

    while T > Tmin:
        for _ in range(iter_por_T):
            tipo = rng.integers(0, 4)
            jx = int(rng.integers(0, NX))
            jy = int(rng.integers(0, NY))
            k = int(rng.integers(0, K))
            if tipo == 0:
                jx2 = int(rng.integers(0, NX))
                jy2 = int(rng.integers(0, NY))
                k2 = int(rng.integers(0, K))
                pa = int(sol.grid[jx, jy, k])
                pb = int(sol.grid[jx2, jy2, k2])
                cambios = [(jx, jy, k, pb), (jx2, jy2, k2, pa)]
            elif tipo == 1:
                actual = int(sol.grid[jx, jy, k])
                if actual >= 0 and actual in casa_idx:
                    cjx, cjy = casa_idx[actual]
                    if (cjx, cjy) != (jx, jy):
                        k2 = int(rng.integers(0, K))
                        pb = int(sol.grid[cjx, cjy, k2])
                        cambios = [(jx, jy, k, pb), (cjx, cjy, k2, actual)]
                    else:
                        continue
                else:
                    continue
            elif tipo == 2:
                items_opt = items_charola_opt.get((jx, jy), set())
                if items_opt:
                    p = int(rng.choice(list(items_opt)))
                    cambios = [(jx, jy, k, p)]
                else:
                    continue
            else:
                cambios = [(jx, jy, k, -1)]

            inv = sol.aplicar(cambios)
            new_score = score_fn(sol.grid)
            d = new_score - cur_score
            if d >= 0 or rng.random() < math.exp(d / max(T, 0.001)):
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = sol.copia()
            else:
                sol.aplicar(inv)
        T *= alpha

    t = time.time() - t0
    sol_df = _extraer_sol(best, prob, refs)
    m = coincidencia(sol_df, refs)
    r = best.resumen()
    return sol_df, {
        'fitness': float(r['fitness']),
        'tiempo_s': t,
        'factible': bool(r['factible']),
        'hibridas': int(r['charolas_hibridas']),
        **m,
    }


def correr_grasp(hist_df, optimo, refs, n_iter=10, ls_iter=3000, seed=7):
    prob, cfg, item_to_idx = _setup_problema(hist_df, refs)
    score_fn, casa_idx, items_charola_opt = _score_fn(prob, refs)
    F_pos = refs['F_pos']

    rng = np.random.default_rng(seed)
    best_overall = None
    best_score = -np.inf
    t0 = time.time()

    for itr in range(n_iter):
        sol = Solucion(prob)
        cambios = []
        ocupados = set()
        ancho_charola = {(jx, jy): 0.0 for jx in range(NX) for jy in range(NY)}

        for jx in range(NX):
            for jy in range(NY):
                items_opt = items_charola_opt.get((jx, jy), set())
                if not items_opt:
                    continue
                items_scored = []
                for i in items_opt:
                    try: item_int = int(prob.items[i])
                    except: continue
                    freq = F_pos.get((item_int, jx, jy), 0)
                    items_scored.append((i, freq))
                items_scored.sort(key=lambda x: -x[1])
                if len(items_scored) > 3:
                    cutoff = max(1, len(items_scored) // 3)
                    top = items_scored[:cutoff]
                    rng.shuffle(top)
                    items_scored = top + items_scored[cutoff:]
                k = 0
                for i, freq in items_scored:
                    if k >= K: break
                    if ancho_charola[(jx, jy)] + prob.beta[i] > 55.5: continue
                    if prob.theta[i] > prob.rho_max[jy] + 0.5: continue
                    if (jx, jy, k) in ocupados:
                        k += 1; continue
                    cambios.append((jx, jy, k, i))
                    ocupados.add((jx, jy, k))
                    ancho_charola[(jx, jy)] += prob.beta[i]
                    k += 1
        sol.aplicar(cambios)

        cur_score = score_fn(sol.grid)
        for _ in range(ls_iter):
            jx = int(rng.integers(0, NX))
            jy = int(rng.integers(0, NY))
            k = int(rng.integers(0, K))
            jx2 = int(rng.integers(0, NX))
            jy2 = int(rng.integers(0, NY))
            k2 = int(rng.integers(0, K))
            pa = int(sol.grid[jx, jy, k])
            pb = int(sol.grid[jx2, jy2, k2])
            cs = [(jx, jy, k, pb), (jx2, jy2, k2, pa)]
            inv = sol.aplicar(cs)
            new_score = score_fn(sol.grid)
            if new_score > cur_score:
                cur_score = new_score
            else:
                sol.aplicar(inv)

        if cur_score > best_score:
            best_score = cur_score
            best_overall = sol.copia()

    t = time.time() - t0
    sol_df = _extraer_sol(best_overall, prob, refs)
    m = coincidencia(sol_df, refs)
    r = best_overall.resumen()
    return sol_df, {
        'fitness': float(r['fitness']),
        'tiempo_s': t,
        'factible': bool(r['factible']),
        'hibridas': int(r['charolas_hibridas']),
        **m,
    }
