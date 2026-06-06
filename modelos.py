"""
Lógica de los 3 modelos del planograma OXXO — VERSIÓN FINAL OPTIMIZADA.

Approach final (después de iterar):
- PuLP MIQP: maximiza F_pos directamente con CBC
- SA custom: warm start desde cocktail BCO+CLA+HRN, score balanceado (casa+familia+jaccard)
- GRASP custom: construcción que pobla cada charola con productos del óptimo, LS con score balanceado
"""
import os
import tempfile
import numpy as np
import pandas as pd
import time
import math
from collections import Counter
from dataclasses import dataclass
from pulp import (LpProblem, LpVariable, LpMaximize, LpInteger, LpStatus,
                  PULP_CBC_CMD, lpSum, value)

import lalo_metaheuristicas as PL
from lalo_metaheuristicas import Config, FAMILIAS, Solucion
from geometria import NX, NY, ALTURAS as ALTURAS_NIVEL, ANCHO_CHAROLA, K_DEFAULT


@dataclass
class ParamsOXXO:
    """Parámetros configurables desde la UI.

    Los defaults reproducen el comportamiento previo (K=8, ancho=55cm,
    demanda = capacidad total, top-N=45, etc.), así que correr sin tocar nada
    da resultados equivalentes a la versión anterior.
    """
    K: int = K_DEFAULT                    # frentes (espacios) por charola
    ancho_charola: float = ANCHO_CHAROLA  # cm de ancho útil por charola
    demanda_global: int | None = None     # None => capacidad total (NX*NY*K)
    pulp_top_n: int = 45                  # productos considerados por PuLP
    pulp_time_limit: int = 60             # segundos de CBC
    sa_iter_por_T: int = 600              # iteraciones por temperatura (SA)
    grasp_n_iter: int = 10                # multi-arranques de GRASP
    grasp_ls_iter: int = 3000            # iteraciones de búsqueda local (GRASP)
    seed: int = 42                        # semilla aleatoria (SA/GRASP)
    epsilon: int = 11                     # máx. charolas híbridas (R9)
    delta: float = 0.005                  # peso de la sinergia δ·G (fitness)

    def demanda(self) -> int:
        """Frentes totales objetivo en el refri (default = capacidad total)."""
        if self.demanda_global is not None:
            return int(self.demanda_global)
        return NX * NY * self.K


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


def _fitness_motor(sol_df, prob, item_to_idx):
    """Fitness penalizado R1-R13 del motor para una solución dada como `sol_df`.

    Mapea cada fila (item, jx, jy) al índice del motor (`item_to_idx`) y arma una
    `Solucion` para leer su `.fitness` — el MISMO criterio que SA/GRASP. Las filas
    cuyo item no está en el universo del motor (`prob.items`, derivado del
    histórico) se omiten, igual que SA/GRASP, que solo colocan productos de ese
    universo. El `k` se reasigna correlativo por charola (el fitness no depende
    del orden dentro de la charola) respetando el tope `K`.
    """
    sol = Solucion(prob)
    K = prob.cfg.K
    cambios = []
    k_por_celda = {}
    for _, r in sol_df.iterrows():
        try:
            idx = item_to_idx[int(r['item'])]
        except (KeyError, ValueError, TypeError):
            continue
        jx, jy = int(r['jx']), int(r['jy'])
        k = k_por_celda.get((jx, jy), 0)
        if k >= K:
            continue
        cambios.append((jx, jy, k, idx))
        k_por_celda[(jx, jy)] = k + 1
    sol.aplicar(cambios)
    return sol


def correr_pulp(hist_df, optimo, refs, params):
    top_n = params.pulp_top_n
    time_limit = params.pulp_time_limit
    Kp = params.K
    ancho_max = params.ancho_charola
    F_total = optimo.groupby('ITEM')['NUM_FRENTES'].sum().sort_values(ascending=False)
    I = F_total.head(top_n).index.tolist()
    ancho = refs['ancho_prod']
    alto = refs['alto_prod']
    F_pos = refs['F_pos']
    nombre = refs['nombre_oficial']
    # Demanda objetivo global (frentes totales en el refri). Se acota a >= |I|
    # porque R3 exige >=1 frente por producto del top-N (si D<|I| el modelo
    # sería infactible). La UI ya limita el mínimo a top-N.
    D = max(params.demanda(), len(I))

    prob = LpProblem('OXXO', LpMaximize)
    # n[i,jx,jy] = número de frentes del producto i en la charola (jx, jy).
    # Equivale a las K binarias por celda del modelo previo (el objetivo no
    # depende de k), con ~8x menos variables y restricciones.
    n = {(i, jx, jy): LpVariable(f'n_{i}_{jx}_{jy}', 0, Kp, LpInteger)
         for i in I for jx in range(NX) for jy in range(NY)}

    # Objetivo: maximizar coincidencia ponderada por F_pos
    prob += lpSum(F_pos.get((i, jx, jy), 0) * n[(i, jx, jy)]
                  for i in I for jx in range(NX) for jy in range(NY))

    for jx in range(NX):
        for jy in range(NY):
            # R1 ancho: suma de anchos por charola <= ancho_charola
            prob += lpSum(ancho[i] * n[(i, jx, jy)] for i in I) <= ancho_max
            # R4 capacidad: a lo más K frentes por charola
            prob += lpSum(n[(i, jx, jy)] for i in I) <= Kp

    # Demanda global: a lo más D frentes en todo el refrigerador
    prob += lpSum(n[(i, jx, jy)]
                  for i in I for jx in range(NX) for jy in range(NY)) <= D

    # R2 alto: producto que no cabe en el nivel queda prohibido ahí
    for i in I:
        for jx in range(NX):
            for jy in range(NY):
                if alto.get(i, 25.0) > ALTURAS_NIVEL[jy] + 0.5:
                    prob += n[(i, jx, jy)] == 0

    # R3 cobertura: cada producto entre 1 y 7 frentes en total
    for i in I:
        total_i = lpSum(n[(i, jx, jy)]
                        for jx in range(NX) for jy in range(NY))
        prob += total_i >= 1
        prob += total_i <= 7

    t0 = time.time()
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=0.05))
    t_pulp = time.time() - t0

    rows = []
    k_por_celda = {}
    for i in I:
        for jx in range(NX):
            for jy in range(NY):
                cnt = int(round(value(n[(i, jx, jy)]) or 0))
                for _ in range(cnt):
                    k = k_por_celda.get((jx, jy), 0)
                    k_por_celda[(jx, jy)] = k + 1
                    rows.append({
                        'item': i, 'jx': jx, 'jy': jy, 'k': k,
                        'ancho': ancho[i], 'alto': alto.get(i, 25.0),
                        'peso': alto.get(i, 25.0),
                        'F': F_total[i],
                        'familia': familia_de_desc(nombre[i]),
                        'desc': nombre[i].strip(),
                    })

    sol_df = pd.DataFrame(rows)
    m = coincidencia(sol_df, refs)

    # Desviación no lineal homogénea con SA/GRASP: evaluar la solución de PuLP
    # con el MISMO fitness penalizado R1-R13 del motor (antes aquí se reportaba
    # el objetivo lineal de CBC `value(prob.objective)`, en otra escala). Se
    # construye el Problema del motor (derivado del histórico) y se mapea el
    # sol_df a una Solucion para leer su .fitness.
    prob_motor, _, item_to_idx = _setup_problema(hist_df, refs, params)
    fitness_motor = float(_fitness_motor(sol_df, prob_motor, item_to_idx).fitness)

    return sol_df, {
        'fitness': fitness_motor,
        'objetivo_lineal': float(value(prob.objective) if prob.objective else 0),
        'tiempo_s': t_pulp,
        'factible': LpStatus[prob.status] == 'Optimal',
        'hibridas': 0,
        **m,
    }


def _setup_lalo(ancho_charola=ANCHO_CHAROLA):
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
        # Límite de ancho plano = ancho_charola en TODAS las charolas, idéntico al
        # que usan PuLP y GRASP (antes era una alpha derivada del histórico, que
        # hacía que cada solver topara el ancho de forma distinta).
        prob.alpha = np.full((NX, NY), float(ancho_charola))
        # Unificar la clasificación de familia con familia_de_desc (la misma que
        # usa la referencia y coincidencia), para que SA/GRASP optimicen y se
        # midan contra el MISMO criterio. FAMILIAS == ['cola', 'agua', 'sabor'].
        fam_idx = {'cola': 0, 'agua': 1, 'sabor': 2}
        prob.fam = np.array([fam_idx[familia_de_desc(d)] for d in prob.desc],
                            dtype=int)
        prob.C = np.zeros((prob.I, len(fam_idx)))
        prob.C[np.arange(prob.I), prob.fam] = 1.0
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


def _construir_cocktail(prob, item_to_idx, optimo, demanda=None):
    sol = Solucion(prob)
    K = prob.cfg.K
    if demanda is None:
        demanda = NX * NY * K
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
    colocados = 0
    for (jx, jy), items in pos_to_items.items():
        if colocados >= demanda:
            break
        cnt = Counter(items)
        k = 0
        for it, c in cnt.most_common():
            if k >= K or colocados >= demanda:
                break
            cambios.append((jx, jy, k, item_to_idx[it]))
            k += 1
            colocados += 1
    sol.aplicar(cambios)
    return sol


class _ScorerIncremental:
    """Acceptance score (similitud al óptimo) con evaluación incremental.

    El score es separable por charola, así que se mantienen la contribución de
    cada celda y sumas vivas; un movimiento que toca un conjunto de celdas solo
    recomputa esas celdas (O(|celdas|*K)) en vez de todo el grid (O(NX*NY*K)).
    La fórmula es idéntica a la evaluación por grid completo:
        5*Σ en_casa + 2*Σ fam_ok + 100*(Σ jaccard)/max(N, 1)
    """

    def __init__(self, prob, refs):
        self.prob = prob
        self.K = prob.cfg.K
        F_pos = refs['F_pos']
        casa = refs['casa']
        fam_dict = refs['fam_dict']
        self.casa_idx = {}
        self.items_charola_opt = {}
        for i, item in enumerate(prob.items):
            try:
                item_int = int(item)
            except (TypeError, ValueError):
                continue
            if item_int in casa:
                self.casa_idx[i] = casa[item_int]
            for jx in range(NX):
                for jy in range(NY):
                    if F_pos.get((item_int, jx, jy), 0) > 0:
                        self.items_charola_opt.setdefault((jx, jy), set()).add(i)
        fam_to_idx = {'cola': 0, 'agua': 1, 'sabor': 2}
        self.fam_opt_idx = {(jx, jy): fam_to_idx.get(fam_dict.get((jx, jy)), -1)
                            for jx in range(NX) for jy in range(NY)}
        # caché por celda y sumas vivas
        self._encasa = {}
        self._famok = {}
        self._jacc = {}
        self._counted = {}
        self._S_encasa = 0
        self._S_famok = 0
        self._S_jacc = 0.0
        self._N = 0

    def _cell_contrib(self, grid, jx, jy):
        prob = self.prob
        items_aqui = set()
        cell_fams = {}
        en_casa = 0
        for k in range(self.K):
            p = int(grid[jx, jy, k])
            if p < 0:
                continue
            items_aqui.add(p)
            if p in self.casa_idx and self.casa_idx[p] == (jx, jy):
                en_casa += 1
            fam_p = prob.fam[p]
            cell_fams[fam_p] = cell_fams.get(fam_p, 0) + 1
        fam_ok = 0
        if cell_fams:
            fam_dom = max(cell_fams, key=cell_fams.get)
            if fam_dom == self.fam_opt_idx.get((jx, jy), -1):
                fam_ok = 1
        items_opt = self.items_charola_opt.get((jx, jy), set())
        if items_aqui or items_opt:
            jacc = len(items_aqui & items_opt) / max(len(items_aqui | items_opt), 1)
            counted = 1
        else:
            jacc = 0.0
            counted = 0
        return en_casa, fam_ok, jacc, counted

    def total(self):
        avg_jaccard = self._S_jacc / max(self._N, 1)
        return self._S_encasa * 5 + self._S_famok * 2 + avg_jaccard * 100

    def recompute_full(self, grid):
        """Recalcula todo el grid desde cero (al inicio o tras un reinicio)."""
        self._S_encasa = self._S_famok = self._N = 0
        self._S_jacc = 0.0
        for jx in range(NX):
            for jy in range(NY):
                e, f, j, c = self._cell_contrib(grid, jx, jy)
                self._encasa[(jx, jy)] = e
                self._famok[(jx, jy)] = f
                self._jacc[(jx, jy)] = j
                self._counted[(jx, jy)] = c
                self._S_encasa += e
                self._S_famok += f
                self._S_jacc += j
                self._N += c
        return self.total()

    def update(self, grid, celdas):
        """Recalcula solo `celdas` (tuplas (jx, jy)) y devuelve el score total."""
        for (jx, jy) in celdas:
            e, f, j, c = self._cell_contrib(grid, jx, jy)
            self._S_encasa += e - self._encasa[(jx, jy)]
            self._S_famok += f - self._famok[(jx, jy)]
            self._S_jacc += j - self._jacc[(jx, jy)]
            self._N += c - self._counted[(jx, jy)]
            self._encasa[(jx, jy)] = e
            self._famok[(jx, jy)] = f
            self._jacc[(jx, jy)] = j
            self._counted[(jx, jy)] = c
        return self.total()


def _setup_problema(hist_df, refs, params):
    _setup_lalo(params.ancho_charola)
    csv_path = _preparar_csv(hist_df, refs,
                             os.path.join(tempfile.gettempdir(), '_lalo.csv'))
    cfg = Config(K=params.K, delta=params.delta, epsilon=params.epsilon,
                 pen_cobertura=10000.0, pen_ancho=200000.0, pen_peso=50000.0,
                 pen_hibrido=10.0, pen_columna=10.0)
    prob = PL.construir_problema(csv_path, cfg)
    item_to_idx = {it: i for i, it in enumerate(prob.items)}
    return prob, cfg, item_to_idx


def _extraer_sol(sol, prob, refs):
    rows = []
    nombre = refs['nombre_oficial']
    for jx in range(NX):
        for jy in range(NY):
            for k in range(prob.cfg.K):
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
                        'alto': float(prob.theta[p]),
                        'peso': float(prob.rho[p]),
                        'familia': FAMILIAS[prob.fam[p]],
                        'F': float(prob.F[p, jx, jy]),
                    })
    return pd.DataFrame(rows)


def correr_sa(hist_df, optimo, refs, params, T0=1.0, Tmin=0.001, alpha=0.97):
    prob, cfg, item_to_idx = _setup_problema(hist_df, refs, params)
    K = prob.cfg.K
    iter_por_T = params.sa_iter_por_T
    seed = params.seed
    scorer = _ScorerIncremental(prob, refs)
    casa_idx = scorer.casa_idx
    items_charola_opt = scorer.items_charola_opt
    sol_init = _construir_cocktail(prob, item_to_idx, optimo, params.demanda())

    rng = np.random.default_rng(seed)
    sol = sol_init.copia()
    best = sol.copia()
    cur_score = scorer.recompute_full(sol.grid)
    best_score = cur_score
    # Demanda global: SA no debe superar D frentes (igual que PuLP y GRASP).
    # Sin este tope, los movimientos de inserción llenarían el refri hasta la
    # capacidad porque el score de similitud premia frentes que coinciden.
    D = params.demanda()
    cur_occupied = int(np.count_nonzero(sol.grid >= 0))
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
            celdas = {(c[0], c[1]) for c in cambios}
            new_score = scorer.update(sol.grid, celdas)
            # cambio neto en frentes ocupados (solo la inserción puede subirlo)
            delta_occ = (sum(1 for c in cambios if c[3] >= 0)
                         - sum(1 for c in inv if c[3] >= 0))
            new_occupied = cur_occupied + delta_occ
            d = new_score - cur_score
            if new_occupied <= D and (d >= 0 or rng.random() < math.exp(d / max(T, 0.001))):
                cur_score = new_score
                cur_occupied = new_occupied
                if new_score > best_score:
                    best_score = new_score
                    best = sol.copia()
            else:
                sol.aplicar(inv)
                scorer.update(sol.grid, celdas)   # restaurar caché del scorer
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


def correr_grasp(hist_df, optimo, refs, params):
    prob, cfg, item_to_idx = _setup_problema(hist_df, refs, params)
    K = prob.cfg.K
    n_iter = params.grasp_n_iter
    ls_iter = params.grasp_ls_iter
    seed = params.seed
    demanda = params.demanda()
    scorer = _ScorerIncremental(prob, refs)
    items_charola_opt = scorer.items_charola_opt
    F_pos = refs['F_pos']

    rng = np.random.default_rng(seed)
    best_overall = None
    best_score = -np.inf
    t0 = time.time()

    for itr in range(n_iter):
        sol = Solucion(prob)
        cambios = []
        ocupados = set()
        colocados = 0   # frentes colocados (tope = demanda global)
        ancho_charola = {(jx, jy): 0.0 for jx in range(NX) for jy in range(NY)}

        for jx in range(NX):
            if colocados >= demanda:
                break
            for jy in range(NY):
                if colocados >= demanda:
                    break
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
                    if k >= K or colocados >= demanda: break
                    if ancho_charola[(jx, jy)] + prob.beta[i] > prob.alpha[jx, jy]: continue
                    if prob.theta[i] > prob.rho_max[jy] + 0.5: continue
                    if (jx, jy, k) in ocupados:
                        k += 1; continue
                    cambios.append((jx, jy, k, i))
                    ocupados.add((jx, jy, k))
                    ancho_charola[(jx, jy)] += prob.beta[i]
                    k += 1
                    colocados += 1
        sol.aplicar(cambios)

        cur_score = scorer.recompute_full(sol.grid)
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
            celdas = {(jx, jy), (jx2, jy2)}
            new_score = scorer.update(sol.grid, celdas)
            if new_score > cur_score:
                cur_score = new_score
            else:
                sol.aplicar(inv)
                scorer.update(sol.grid, celdas)   # restaurar caché del scorer

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
